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
                    
                    # Tool Handling Logic
                    if message.tool_calls:
                        # Append the Assistant's Tool Call to history
                        messages.append(message)
                        
                        tool_output = json.dumps({
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
                        })
                        
                        # Prepare for next turn by adding tool results (dummy content if not executing)
                        for tc in message.tool_calls:
                            messages.append({
                                "role": "tool", 
                                "content": "Tool executed successfully", 
                                "tool_call_id": tc.id
                            })
                        
                        chat_responses.append(tool_output)
                        continue # Move to next user turn

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
