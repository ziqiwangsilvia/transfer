import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)


class InferenceEngine:
    """Shared inference engine for all benchmarks."""

    def __init__(self, model_config):
        self.config = model_config
        self.model: Optional[Union[LLM, AsyncOpenAI, PreTrainedModel]] = None
        self.backend = model_config.backend.lower()
        self.model = None

    def initialize(self) -> None:
        """Initialize the model based on backend."""
        if self.backend == "openai":
            self._initialize_openai()
        elif self.backend == "vllm-service":
            self._initialize_vllm_service()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _initialize_openai(self) -> None:
        """Initialize OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.model = AsyncOpenAI(api_key=api_key)
        log.info(f"OpenAI client initialized: {self.config.name}")

    def _initialize_vllm_service(self) -> None:
        """Initialize OpenAI-compatible client for a running vLLM server."""
        if not self.config.api_base:
            raise ValueError(
                "api_base must be set for vllm-service backend "
                '(e.g. "http://localhost:8000/v1")'
            )

        self.model = AsyncOpenAI(base_url=self.config.api_base, api_key="EMPTY", max_retries=0)
        log.info(
            f"vllm-service client initialized: model={self.config.name}, "
            f"base_url={self.config.api_base}"
        )


    def run_inference(
        self,
        prompts: List[List[str]],
        batch_size: int,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_labels: Optional[List[str]] = None,
        system_message: Optional[str] = None,
        verbose: bool = False,
        log_interval: int = 10,
    ) -> List[str]:
        """Run inference on prompts using the configured backend."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if self.backend in ("openai", "vllm-service"):
            return self._run_openai_inference(
                prompts,
                batch_size,
                stop_sequences,
                tools,
                structured_labels,
                system_message,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        """Run vLLM inference."""

        assert self.model is not None
        assert isinstance(self.model, LLM)

        # Build sampling params
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if stop_sequences:
            params["stop"] = stop_sequences

        # Add structured outputs if specified
        if structured_labels and self.config.use_structured:
            try:
                from vllm.sampling_params import StructuredOutputsParams

                structured_outputs = StructuredOutputsParams(choice=structured_labels)
                params["structured_outputs"] = structured_outputs
                log.info(f"Using vLLM structured outputs: {structured_labels}")
            except (ImportError, TypeError) as e:
                log.warning(f"Structured outputs not available: {e}")

        sampling_params = SamplingParams(**params)

        all_outputs = []

        # Format prompts for tools if provided
        if tools:
            chat_prompts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                chat_prompts.append(messages)

            tokenizer = self.model.get_tokenizer()

            if template_path := getattr(self.config, "custom_chat_template", None):
                with open(template_path) as f:
                    tokenizer.chat_template = f.read()  # type: ignore[attr-defined]
                log.info(f"Loaded custom Gemma3 chat template from {template_path}")

            for i in range(0, len(chat_prompts), batch_size):
                batch_messages = chat_prompts[i : i + batch_size]

                if verbose and i % (log_interval * batch_size) == 0:
                    log.info(
                        f"Processing batch {i // batch_size + 1}/"
                        f"{(len(chat_prompts) - 1) // batch_size + 1}"
                    )

                try:
                    batch_formatted = []
                    for messages in batch_messages:
                        formatted = tokenizer.apply_chat_template(
                            messages,
                            tools=tools,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        batch_formatted.append(formatted)

                    outputs = self.model.generate(batch_formatted, sampling_params)
                    batch_outputs = [output.outputs[0].text for output in outputs]
                    all_outputs.extend(batch_outputs)

                except Exception as e:
                    log.warning(f"Tool calling format failed, falling back: {e}")
                    batch_prompts_text = [msg[0]["content"] for msg in batch_messages]
                    outputs = self.model.generate(batch_prompts_text, sampling_params)
                    batch_outputs = [output.outputs[0].text for output in outputs]
                    all_outputs.extend(batch_outputs)

        else:
            # Standard inference without tools
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]

                if verbose and i % (log_interval * batch_size) == 0:
                    log.info(
                        f"Processing batch {i // batch_size + 1}/"
                        f"{(len(prompts) - 1) // batch_size + 1}"
                    )

                outputs = self.model.generate(batch_prompts, sampling_params)
                batch_outputs = [output.outputs[0].text for output in outputs]
                all_outputs.extend(batch_outputs)

        return all_outputs


    def _run_openai_inference(
        self,
        prompts: List[List[str]],
        batch_size: int,
        stop_sequences: Optional[List[str]],
        tools: Optional[List[Dict[str, Any]]],
        structured_labels: Optional[List[str]],
        system_message: Optional[str],
    ) -> List[str]:
        """Run OpenAI inference."""
        return asyncio.run(
            self._run_openai_async(
                prompts,
                batch_size,
                stop_sequences,
                tools,
                structured_labels,
                system_message,
            )
        )

    async def _run_openai_async(
        self,
        prompts: List[List[str]],
        batch_size: int,
        stop_sequences: Optional[List[str]],
        tools: Optional[List[Dict[str, Any]]],
        structured_labels: Optional[List[str]],
        system_message: Optional[str],
    ) -> List[str]:
        """Run OpenAI inference asynchronously."""
        assert self.model is not None
        assert isinstance(self.model, AsyncOpenAI)
        client: AsyncOpenAI = self.model

        semaphore = asyncio.Semaphore(batch_size)

        async def process_single_chat(chat_turns: List[str]) -> List[str]:
            async with semaphore:
                messages = []
                results = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                for i, turn in enumerate(chat_turns):
                    messages.append({"role": "user", "content": turn})

                    kwargs: Dict[str, Any] = {
                        "model": self.config.name,
                        "messages": messages,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    }

                    if stop_sequences:
                        kwargs["stop"] = stop_sequences

                    if tools:
                        kwargs["tools"] = tools
                        kwargs["tool_choice"] = "auto"

                    if structured_labels and self.config.use_structured:
                        kwargs["response_format"] = {
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

                    try:
                        response = await client.chat.completions.create(**kwargs)
                        message = response.choices[0].message

                        if message.tool_calls:
                            results.append(
                                json.dumps(
                                    {
                                        "tool_calls": [
                                            {
                                                "type": "function",
                                                "function": {
                                                    "name": tc.function.name,
                                                    "arguments": tc.function.arguments,
                                                },
                                            }
                                            for tc in message.tool_calls
                                        ]
                                    }
                                )
                            )
                            for tc in message.tool_calls:
                                messages.append(
                                    {"role": "tool", "content": tc.function}
                                )

                        content = message.content or ""
                        messages.append({"role": "assistant", "content": content})

                        # Parse structured output if needed
                        if structured_labels and content.startswith("{"):
                            try:
                                parsed = json.loads(content)
                                results.append(parsed.get("classification", content))
                            except Exception:
                                pass

                        results.append(content)

                    except Exception as e:
                        log.error(f"OpenAI API error: {e}")
                        results.append("")
                return results

        tasks = [process_single_chat(chat) for chat in prompts]
        return await asyncio.gather(*tasks)

    def run_tool_calling_inference(
        self,
        dataset_path: str,
        tool_config: Any,
        prompt_sections: List[str],
        batch_size: int = 32,
        stop_sequences: list[str] | None = None,
        max_samples: int | None = None,
        verbose: bool = False,
        log_interval: int = 10,
    ) -> Dict[str, Any]:
        import evaluator.prompts as prompt_module
        from evaluator.tool_calling.tool_wrappers import (
            format_schemas_for_vllm,
            load_schemas_from_json,
        )
        from evaluator.tool_calling.utils import (
            convert_filter_format,
            create_system_prompt,
            format_tools_description,
            load_dataset,
            load_tools,
        )

        log.info("Loading dataset")
        dataset = load_dataset(path=dataset_path, max_samples=max_samples)
        log.info(f"Loaded {len(dataset)} samples")

        log.info("Converting ground truth in datasets for simplified schema")
        if tool_config.schema_path and "_simplified" in tool_config.schema_path:
            dataset = convert_filter_format(dataset)

        log.info("Loading tools")
        raw_tools = load_tools(tool_config)
        tool_schemas = load_schemas_from_json(raw_tools)
        tools = format_schemas_for_vllm(tool_schemas)
        log.info(f"Loaded {len(tools)} tools")

        tools_description = format_tools_description(tools)

        log.info("Preparing prompts from sections")
        section_strings = []
        for section_name in prompt_sections:
            if not hasattr(prompt_module, section_name):
                raise ValueError(f"Unknown prompt section: '{section_name}'")
            section_text = getattr(prompt_module, section_name)
            section_text = section_text.format(
                tools=json.dumps(tools, indent=4), tools_description=tools_description
            )
            section_strings.append(section_text)

        conversations = []
        for chat in dataset:
            prompts = []
            for item in chat:
                query = item.get("query", "")
                prompt = create_system_prompt(query=query, sections=section_strings)
                prompts.append(prompt)
            conversations.append(prompts)

        log.info(f"Prepared {len(prompts)} prompts")

        log.info(f"Running inference on {len(prompts)} prompts")
        outputs = self.run_inference(
            prompts=conversations,
            batch_size=batch_size,
            stop_sequences=stop_sequences,
            tools=tools,
            verbose=verbose,
            log_interval=log_interval,
        )

        dataset = [item for sublist in dataset for item in sublist]
        outputs = [item for sublist in outputs for item in sublist]
        prompts = [item for sublist in conversations for item in sublist]

        return {
            "dataset": dataset,
            "prompts": prompts,
            "outputs": outputs,
        }

    def cleanup(self) -> None:
        """Clean up model resources to free GPU memory."""
        if self.model is None:
            log.info("No model to cleanup")
            return

        if self.backend in ("openai", "vllm-service"):
            self.model = None
            log.info(f"{self.backend} client cleared")