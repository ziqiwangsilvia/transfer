from openai.types.chat import ChatCompletionMessageParam

async def _run_openai_async(
    self,
    # Prompts is now a list of full histories: [[{"role": "user", "content": "hi"}], [...]]
    conversations: List[List[ChatCompletionMessageParam]], 
    batch_size: int,
    stop_sequences: Optional[List[str]],
    tools: Optional[List[Dict[str, Any]]],
    structured_labels: Optional[List[str]],
    system_message: Optional[str],
) -> List[str]:
    assert isinstance(self.model, AsyncOpenAI)
    client: AsyncOpenAI = self.model
    semaphore = asyncio.Semaphore(batch_size)

    async def process_single_chat(history: List[ChatCompletionMessageParam]) -> str:
        async with semaphore:
            # 1. Build the initial message list for this specific chat
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.extend(history) # Add the existing multi-turn history

            kwargs: Dict[str, Any] = {
                "model": self.config.name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            if stop_sequences: kwargs["stop"] = stop_sequences
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            # response_format only applies to the final classification output
            if structured_labels and self.config.use_structured:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"classification": {"type": "string", "enum": structured_labels}},
                            "required": ["classification"],
                            "additionalProperties": False,
                        },
                    },
                }

            try:
                # Loop to handle one or more tool-calling turns
                while True:
                    response = await client.chat.completions.create(**kwargs)
                    message = response.choices[0].message
                    
                    if not message.tool_calls:
                        # No more tools needed, this is the final assistant response
                        content = message.content or ""
                        
                        # Handle structured JSON parsing
                        if structured_labels and content.strip().startswith("{"):
                            try:
                                return json.loads(content).get("classification", content)
                            except: pass
                        return content

                    # --- If tool_calls exist, we must alternate roles correctly ---
                    # 1. Append the assistant's tool request to history
                    messages.append(message) 

                    # 2. Execute tools and append 'tool' role results
                    for tc in message.tool_calls:
                        # TODO: Replace with your actual tool execution logic
                        tool_result = f"Output from {tc.function.name}" 
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result
                        })
                    
                    # 3. The loop continues to send the tool results back to OpenAI
                    # to get the final (or next) response.
                    kwargs["messages"] = messages

            except Exception as e:
                log.error(f"OpenAI API error: {e}")
                return ""

    tasks = [process_single_chat(chat) for chat in conversations]
    return await asyncio.gather(*tasks)
