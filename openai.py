class InferenceEngine:
    def __init__(self, model_config):
        self.config = model_config
        self.model: Optional[AsyncOpenAI] = None
        self.backend = model_config.backend.lower()

    async def initialize(self) -> None:
        """Initialize the model asynchronously."""
        if self.backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            self.model = AsyncOpenAI(api_key=api_key)
        elif self.backend == "vllm-service":
            # Using one persistent client for the duration of the loop
            self.model = AsyncOpenAI(
                base_url=self.config.api_base,
                api_key="EMPTY",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                    timeout=httpx.Timeout(600.0, connect=10.0)
                )
            )
        log.info(f"Backend {self.backend} initialized.")

    async def run_inference(
        self,
        prompts: List[List[str]],
        batch_size: int,
        **kwargs
    ) -> List[str]:
        """Now an async method. No asyncio.run() here."""
        if self.model is None:
            await self.initialize()
            
        return await self._run_openai_async(prompts, batch_size, **kwargs)

    async def _run_openai_async(
        self,
        prompts: List[List[str]],
        batch_size: int,
        stop_sequences: Optional[List[str]],
        tools: Optional[List[Dict[str, Any]]],
        structured_labels: Optional[List[str]],
        system_message: Optional[str],
    ) -> List[str]:
        client: AsyncOpenAI = self.model
        semaphore = asyncio.Semaphore(batch_size)

        async def process_single_chat(chat_turns: List[str]) -> List[str]:
            async with semaphore:
                messages = []
                turn_results = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                
                for turn in chat_turns:
                    messages.append({"role": "user", "content": turn})
                    
                    # Prepare arguments (omitted for brevity, keep your original logic)
                    request_kwargs = {
                        "model": self.config.name,
                        "messages": messages,
                        "tools": tools if tools else None,
                        # ... other original kwargs ...
                    }

                    try:
                        response = await client.chat.completions.create(**request_kwargs)
                        message = response.choices[0].message
                        content = message.content or ""
                        
                        # Update conversation history for next turn
                        messages.append({"role": "assistant", "content": content})
                        turn_results.append(content)
                    except Exception as e:
                        log.error(f"OpenAI API error: {e}")
                        turn_results.append("")
                
                return turn_results

        # Create tasks for all chat sequences
        tasks = [process_single_chat(chat) for chat in prompts]   
        tasks_results = await asyncio.gather(*tasks)
        
        # Flatten List[List[str]] -> List[str]
        return [item for sublist in tasks_results for item in sublist]
