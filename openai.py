async def _run_openai_async(
        self,
        prompts: List[str],
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

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        async def process_single(prompt: str) -> str:
            async with semaphore:
                # messages = []
                messages.append({"role": "user", "content": prompt})
                print([item["role"] for item in messages])

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
                        return json.dumps(
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
                        for tc in message.tool_calls:
                            messages.append({"role": "tool", "content": tc.function})

                    content = message.content or ""
                    messages.append({"role": "assistant", "content": content})
                    print("after answer", content)

                    # Parse structured output if needed
                    if structured_labels and content.startswith("{"):
                        try:
                            parsed = json.loads(content)
                            return parsed.get("classification", content)
                        except Exception:
                            pass

                    return content

                except Exception as e:
                    log.error(f"OpenAI API error: {e}")
                    messages.append({"role": "assistant", "content": ""})
                    return ""

        tasks = [process_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
