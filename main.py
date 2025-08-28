from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os
import httpx

# Load environment variables from .env file
load_dotenv()

# The single OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Please set it in your .env file.")

app = FastAPI(
    title="OpenAI Proxy API",
    description="A simple FastAPI proxy to route requests to OpenAI's API using a single key.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = httpx.AsyncClient(timeout=30.0)
OPENAI_BASE_URL = "https://api.openai.com/v1"

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_completion_tokens: Optional[int] = 150

# A generator that reads from the upstream response and yields chunks
async def openai_stream_generator(response):
    # This generator now only yields chunks and does not handle exceptions
    async for chunk in response.aiter_bytes():
        yield chunk

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, request_body: ChatCompletionRequest):
    payload = request_body.dict(exclude_none=True)
    
    if not payload.get("stream"):
        raise HTTPException(status_code=400, detail="This proxy only supports streaming.")

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # Make the streaming request to OpenAI
        # This is where the error will be caught if a bad status is returned
        upstream_response = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        # We raise the exception here, before trying to access the response body
        upstream_response.raise_for_status()

        # Stream the response back to the client
        return StreamingResponse(
            openai_stream_generator(upstream_response),
            media_type="text/event-stream"
        )
    
    except httpx.HTTPStatusError as e:
        # We can safely raise an HTTPException here with the status code
        # because we are no longer trying to read the body.
        raise HTTPException(status_code=e.response.status_code, detail=f"Upstream API Error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
