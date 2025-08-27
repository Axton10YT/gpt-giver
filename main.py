from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import openai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import asyncio
import aiohttp

# Load environment variables
load_dotenv()

app = FastAPI(
    title="OpenAI Proxy API",
    description="Free access to GPT models and DALL-E via OpenAI API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai.api_key = OPENAI_API_KEY

# Pydantic models for request validation
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
    stream: Optional[bool] = False

class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = "standard"
    response_format: Optional[str] = "url"

class ImageVariationRequest(BaseModel):
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"

class ImageEditRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"

@app.get("/")
async def root():
    return {
        "message": "OpenAI Proxy API",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "image_generation": "/v1/images/generations",
            "image_variations": "/v1/images/variations",
            "image_edits": "/v1/images/edits",
            "models": "/v1/models"
        }
    }

@app.get("/v1/models")
async def list_models():
    """List available models"""
    try:
        response = await asyncio.to_thread(openai.Model.list)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint supporting both streaming and non-streaming"""
    try:
        # Convert Pydantic models to dict for OpenAI API
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stream": request.stream
        }
        
        if request.max_tokens:
            params["max_tokens"] = request.max_tokens
        
        if request.stream:
            # Handle streaming response
            async def generate():
                try:
                    response = await asyncio.to_thread(
                        openai.ChatCompletion.create,
                        **params
                    )
                    for chunk in response:
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_response = {
                        "error": {
                            "message": str(e),
                            "type": "api_error"
                        }
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Handle non-streaming response
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                **params
            )
            return response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat completion: {str(e)}")

@app.post("/v1/images/generations")
async def generate_image(request: ImageGenerationRequest):
    """Generate images using DALL-E"""
    try:
        response = await asyncio.to_thread(
            openai.Image.create,
            prompt=request.prompt,
            n=request.n,
            size=request.size,
            response_format=request.response_format
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.post("/v1/images/variations")
async def create_image_variation(file: bytes, request: ImageVariationRequest):
    """Create variations of an image"""
    try:
        response = await asyncio.to_thread(
            openai.Image.create_variation,
            image=file,
            n=request.n,
            size=request.size,
            response_format=request.response_format
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating image variation: {str(e)}")

@app.post("/v1/images/edits")
async def edit_image(image: bytes, mask: Optional[bytes], request: ImageEditRequest):
    """Edit an image using DALL-E"""
    try:
        params = {
            "image": image,
            "prompt": request.prompt,
            "n": request.n,
            "size": request.size,
            "response_format": request.response_format
        }
        
        if mask:
            params["mask"] = mask
            
        response = await asyncio.to_thread(
            openai.Image.create_edit,
            **params
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error editing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "OpenAI Proxy API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
