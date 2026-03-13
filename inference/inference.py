import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from inference.structured_output import AssistantResponse

log = logging.getLogger(__name__)


class InferenceEngine:
    """Shared inference engine for all benchmarks."""

    def __init__(self, model_config):
        self.config = model_config
        self.model: Optional[AsyncOpenAI] = None
        self.backend = model_config.backend.lower()

    async def initialize(self) -> None:
        """Initialize the model asynchronously within the active event loop."""
        if self.backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.model = AsyncOpenAI(api_key=api_key)

        elif self.backend == "vllm-service":
            if not self.config.api_base:
                raise ValueError("api_base must be set for vllm-service")

            # Using a persistent client with proper connection pooling for vLLM
            self.model = AsyncOpenAI(
                base_url=self.config.api_base,
                api_key="EMPTY",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=100, max_keepalive_connections=20
                    ),
                    timeout=httpx.Timeout(600.0, connect=10.0),
                ),
            )
        log.info(f"Backend {self.backend} initialized for {self.config.name}")

    async def run_inference(
        self,
        contexts: List[List[Dict[str, Any]]],
        batch_size: int,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.config.model_in_the_loop:
            raise NotImplementedError("model_in_the_loop=True is not yet implemented.")

        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if self.backend == "openai":
            return await self._run_openai_batch(
                contexts,
                stop_sequences,
                tools,
                structured_labels,
            )

        elif self.backend == "vllm-service":
            return await self._run_openai_async(
                contexts,
                batch_size,
                stop_sequences,
                tools,
                structured_labels,
            )

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    async def _run_openai_async(
        self,
        contexts: List[List[Dict[str, Any]]],
        batch_size: int,
        stop_sequences: Optional[List[str]],
        tools: Optional[List[Dict[str, Any]]],
        structured_labels: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        assert self.model is not None
        assert isinstance(self.model, AsyncOpenAI)

        client: AsyncOpenAI = self.model
        semaphore = asyncio.Semaphore(batch_size)

        def _build_kwargs(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
            kwargs: Dict[str, Any] = {
                "model": self.config.name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            if stop_sequences:
                kwargs["stop"] = stop_sequences

            if self.config.use_structured:
                kwargs["extra_body"] = {
                    "structured_outputs": {
                        "json": AssistantResponse.model_json_schema(),
                        "backend": "xgrammer",
                    },
                }
            elif tools and self.config.template_has_tool_token:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return kwargs

        def _message_to_dict(message) -> Dict[str, Any]:
            """Convert an API response message into a plain result dict."""
            if message.tool_calls:
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }

            content = message.content or ""

            if structured_labels and content.strip().startswith("{"):
                try:
                    parsed = json.loads(content)
                    if "tool_calls" in parsed and parsed["tool_calls"]:
                        tc = parsed["tool_calls"]
                        return {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc["args"]),
                                    },
                                }
                            ],
                        }
                    if "content" in parsed:
                        content = parsed["content"] or ""

                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

            return {"role": "assistant", "content": content}

        async def _infer_single(context: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Send one context to the model and return a result dict."""
            async with semaphore:
                try:
                    response = await client.chat.completions.create(
                        **_build_kwargs(context)
                    )
                    return _message_to_dict(response.choices[0].message)
                except Exception as e:
                    return {"role": "assistant", "error": str(e)}

        return list(
            await tqdm_asyncio.gather(
                *(_infer_single(ctx) for ctx in contexts),
                desc="Inference",
                unit="sample",
                colour="cyan",
            )
        )

    async def _run_openai_batch(
        self,
        contexts: List[List[Dict[str, Any]]],
        stop_sequences: Optional[List[str]],
        tools: Optional[List[Dict[str, Any]]],
        structured_labels: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        assert self.model is not None
        client: AsyncOpenAI = self.model

        import tempfile

        def _build_body(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
            body: Dict[str, Any] = {
                "model": self.config.name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            if stop_sequences:
                body["stop"] = stop_sequences

            if tools and self.config.template_has_tool_token:
                body["tools"] = tools
                body["tool_choice"] = "auto"

            if self.config.use_structured:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "classification": {
                                    "type": "string",
                                    "enum": structured_labels,
                                }
                            },
                            "required": ["classification"],
                            "additionalProperties": False,
                        },
                    },
                }

            return body

        # Build JSONL batch file
        batch_requests = []
        for i, context in enumerate(contexts):
            batch_requests.append(
                {
                    "custom_id": f"sample-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": _build_body(context),
                }
            )

        # Write temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
            temp_path = f.name

        # Upload file
        batch_file = await client.files.create(
            file=open(temp_path, "rb"),
            purpose="batch",
        )

        # Create batch job
        batch_job = await client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        log.info(f"Batch job created: {batch_job.id}")

        # Poll for completion
        while True:
            job = await client.batches.retrieve(batch_job.id)
            if job.status in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(10)

        if job.status != "completed":
            raise RuntimeError(f"Batch failed: {job.status}")

        log.info("Batch completed. Downloading results...")

        # Download results
        if job.output_file_id is None:
            raise RuntimeError("Batch completed but no output_file_id_found")

        file_response = await client.files.content(job.output_file_id)
        file_bytes = file_response.read()
        lines = file_bytes.decode("utf-8").splitlines()

        outputs: List[Dict[str, Any]] = [{} for _ in range(len(contexts))]

        for line in lines:
            record = json.loads(line)
            idx = int(record["custom_id"].split("-")[1])

            message = record["response"]["body"]["choices"][0]["message"]

            tool_calls_raw = message.get("tool_calls") or []
            content_raw = message.get("content") or ""

            non_empty_tool_call = any(
                tc.get("function") and tc["function"].get("name")
                for tc in tool_calls_raw
            )

            if non_empty_tool_call:
                outputs[idx] = {
                    "role": "assistant",
                    "tool_calls": tool_calls_raw,
                }

            else:
                content = content_raw

                if structured_labels and content.startswith("{"):
                    try:
                        parsed = json.loads(content)
                        content = parsed.get("classification", content)
                    except Exception:
                        pass
                outputs[idx] = {
                    "role": "assistant",
                    "content": content,
                }

        return outputs

    def cleanup(self) -> None:
        """Clean up model resources to free GPU memory."""
        if self.model is None:
            log.info("No model to cleanup")
            return

        if self.backend in ("openai", "vllm-service"):
            self.model = None
            log.info(f"{self.backend} client cleared")
