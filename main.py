from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os
import httpx
import json

# Load environment variables from .env file
load_dotenv()

# The single OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Please set it in your .env file.")

app = FastAPI(
    title="OpenAI Proxy API",
    description="A simple FastAPI proxy to route requests to OpenAI's API using a single key.",
    version="1.0.0",
)

# CORS configuration to allow requests from all origins
# This is crucial for local web apps or apps hosted on different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Shared HTTP client for making requests to OpenAI
client = httpx.AsyncClient(timeout=30.0)
OPENAI_BASE_URL = "https://api.openai.com/v1"

class Message(BaseModel):
    """Pydantic model for a single chat message."""
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    """Pydantic model for a chat completion request."""
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_completion_tokens: Optional[int] = 150

# New streaming generator function
async def proxy_stream_generator(request_data: Dict[str, Any]):
    """
    Asynchronously streams the response from the OpenAI chat completions endpoint
    and yields chunks directly to the client.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    # We use a try-except block to gracefully handle potential HTTP errors
    try:
        async with client.stream(
            "POST",
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=request_data
        ) as response:
            response.raise_for_status() # Raise an exception for bad status codes
            async for chunk in response.aiter_bytes():
                yield chunk
    except httpx.HTTPStatusError as e:
        # If an error occurs, yield a JSON-formatted error message
        error_msg = json.dumps({"error": f"API Error: {e.response.status_code} - {e.response.text}"})
        yield f"data: {error_msg}\n\n".encode()

@app.post("/v1/chat/completions")
async def chat_completions(request_body: ChatCompletionRequest):
    """
    Proxy endpoint for OpenAI's chat completions.
    Forwards the request to OpenAI, handling both streaming and non-streaming responses.
    """
    payload = request_body.dict(exclude_none=True)
    
    if payload.get("stream"):
        # If streaming is requested, return a StreamingResponse with our generator
        return StreamingResponse(proxy_stream_generator(payload), media_type="text/event-stream")
    
    # If not streaming, handle as a regular request
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        response = await client.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

