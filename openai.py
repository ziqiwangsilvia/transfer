async def _run_openai_async(
    self,
    chats: List[List[str]], # List of conversations (each a list of turns)
    batch_size: int,
    stop_sequences: Optional[List[str]],
    tools: Optional[List[Dict[str, Any]]],
    structured_labels: Optional[List[str]],
    system_message: Optional[str],
) -> List[List[str]]:
    """Run OpenAI inference asynchronously with multi-turn context support."""
    assert self.model is not None
    assert isinstance(self.model, AsyncOpenAI)
    client: AsyncOpenAI = self.model

    semaphore = asyncio.Semaphore(batch_size)

    async def process_chat(turns: List[str]) -> List[str]:
        async with semaphore:
            # FIX: Each chat gets its own private message history
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            chat_responses = []

            # Process turns sequentially within this specific chat
            for prompt in turns:
                messages.append({"role": "user", "content": prompt})
                
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
                        # STEP 1: Append the EXACT message object from the response (contains tool_calls)
                        messages.append(message) 

                        # STEP 2: You MUST append a "tool" role message for EVERY tool_call returned
                        for tc in message.tool_calls:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id, # CRITICAL: vLLM needs this ID to match
                                "name": tc.function.name,
                                "content": "Tool result goes here" # Even if dummy, this must exist
                            })

                        # Format the string to return for your evaluation results
                        tool_json = json.dumps({
                            "tool_calls": [
                                {"name": tc.function.name, "arguments": tc.function.arguments}
                                for tc in message.tool_calls
                            ]
                        })
                        chat_responses.append(tool_json)
                        continue # Move to the next user turn

                        # Standard Content Handling
                        content = message.content or ""
                        messages.append({"role": "assistant", "content": content})
                        
                        # Parse structured output if needed
                        final_output = content
                        if structured_labels and content.startswith("{"):
                            try:
                                parsed = json.loads(content)
                                final_output = parsed.get("classification", content)
                            except Exception:
                                pass

                        chat_responses.append(final_output)

                except Exception as e:
                    log.error(f"OpenAI API error: {e}")
                    # Keep history consistent even on error
                    messages.append({"role": "assistant", "content": "Error processing request"})
                    chat_responses.append("")
            
            return chat_responses

    # Run all chats in parallel
    tasks = [process_chat(chat_turns) for chat_turns in chats]
    return await asyncio.gather(*tasks)
