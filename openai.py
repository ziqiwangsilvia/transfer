import asyncio
import json
import logging
import os
import httpx
from typing import Any, Dict, List, Optional, Union

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

class InferenceEngine:
    """Shared inference engine for all benchmarks using Option 2 (Single Event Loop)."""

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
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                    timeout=httpx.Timeout(600.0, connect=10.0) 
                )
            )
        log.info(f"Backend {self.backend} initialized for {self.config.name}")

    async def run_inference(
        self,
        prompts: List[List[str]],
        batch_size: int,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_labels: Optional[List[str]] = None,
        system_message: Optional[str] = None,
    ) -> List[str]:
        """Entry point for inference. Ensure initialize() was called or call it here."""
        if self.model is None:
            await self.initialize()
            
        return await self._run_openai_async(
            prompts,
            batch_size,
            stop_sequences,
            tools,
            structured_labels,
            system_message,
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
        """Run OpenAI inference asynchronously without calling asyncio.run() internally."""
        client: AsyncOpenAI = self.model
        semaphore = asyncio.Semaphore(batch_size)

        async def process_single_chat(chat_turns: List[str]) -> List[str]:
            async with semaphore:
                messages = []
                chat_results = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                
                for turn in chat_turns:
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
                                        "classification": {"type": "string", "enum": structured_labels}
                                    },
                                    "required": ["classification"],
                                    "additionalProperties": False,
                                },
                            },
                        }

                    try:
                        response = await client.chat.completions.create(**kwargs)
                        message = response.choices[0].message

                        # --- Original Tool Call Logic ---
                        if message.tool_calls:
                            tool_results = {
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    } for tc in message.tool_calls
                                ]
                            }
                            chat_results.append(json.dumps(tool_results))
                            
                            # Note: In real tool-use, you'd execute tools here. 
                            # Adding assistant's tool_calls to history for context.
                            messages.append({"role": "assistant", "tool_calls": message.tool_calls})

                        # --- Original Content & Structured Output Logic ---
                        content = message.content or ""
                        
                        if structured_labels and content.startswith("{"):
                            try:
                                parsed = json.loads(content)
                                chat_results.append(parsed.get("classification", content))
                            except Exception:
                                chat_results.append(content)
                        else:
                            chat_results.append(content)

                        # Update history for next turn
                        messages.append({"role": "assistant", "content": content})

                    except Exception as e:
                        log.error(f"OpenAI API error: {e}")
                        chat_results.append("")
                
                return chat_results

        # Execute all chat sequences concurrently
        tasks = [process_single_chat(chat) for chat in prompts]   
        tasks_results = await asyncio.gather(*tasks)
        
        # Flatten List[List[str]] -> List[str]
        return [item for sublist in tasks_results for item in sublist]

# --- Usage Example ---
# async def main():
#     engine = InferenceEngine(my_config)
#     results = await engine.run_inference(prompts=[["hi"], ["hello"]], batch_size=32)
#     print(results)
#
# if __name__ == "__main__":
#     asyncio.run(main())
