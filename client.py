import httpx
from openai import AsyncOpenAI

# Create a client that does not keep connections alive
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    http_client=httpx.AsyncClient(
        # Set max_keepalive_connections to 0 to force fresh connections
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=0),
        # Ensure the timeout is high enough for vLLM prefill
        timeout=httpx.Timeout(600.0, connect=10.0) 
    )
)
