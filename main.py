from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os
import httpx
import json
import asyncio

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
    # Renamed the parameter to match the new API spec for GPT-5 models
    max_completion_tokens: Optional[int] = 150

class ImageGenerationRequest(BaseModel):
    """Pydantic model for an image generation request."""
    model: str
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = "standard"

async def stream_openai_response(request_data: Dict[str, Any]):
    """
    Asynchronously streams the response from the OpenAI chat completions endpoint.
    This function handles Server-Sent Events (SSE) for real-time output.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    # Send the request to OpenAI's chat completions endpoint
    async with client.stream(
        "POST",
        f"{OPENAI_BASE_URL}/chat/completions",
        headers=headers,
        json=request_data
    ) as response:
        # Re-raise HTTP errors to be caught by the FastAPI exception handler
        response.raise_for_status()
        
        # Iterate over the response stream and yield chunks as SSE
        async for chunk in response.aiter_bytes():
            yield chunk

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, request_body: ChatCompletionRequest):
    """
    Proxy endpoint for OpenAI's chat completions.
    Forwards the request to OpenAI, handling both streaming and non-streaming responses.
    """
    # Create the payload to be sent to OpenAI
    payload = request_body.dict(exclude_none=True)
    
    if request_body.stream:
        # If streaming is requested, return a StreamingResponse
        return StreamingResponse(stream_openai_response(payload), media_type="text/event-stream")
    
    try:
        # If not streaming, send a standard POST request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # Make the request to OpenAI
        response = await client.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        # Return the response directly
        return JSONResponse(response.json())
        
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors from OpenAI
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        # Handle other potential errors
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/images/generations")
async def image_generations(request: Request, request_body: ImageGenerationRequest):
    """
    Proxy endpoint for OpenAI's image generation.
    Forwards the request to OpenAI and returns the image data.
    """
    # Create the payload to be sent to OpenAI
    payload = request_body.dict(exclude_none=True)
    
    try:
        # Send the POST request to OpenAI
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        response = await client.post(f"{OPENAI_BASE_URL}/images/generations", headers=headers, json=payload)
        response.raise_for_status()
        
        # Return the response directly
        return JSONResponse(response.json())
        
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors from OpenAI
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        # Handle other potential errors
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
