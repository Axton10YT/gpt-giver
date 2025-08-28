from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from typing import Optional, Literal
from fastapi.responses import JSONResponse

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="OpenAI API Wrapper",
    description="FastAPI app to access OpenAI GPT models (including GPT-5 family",
    version="1.1.0"
)

# Initialize OpenAI client with API key from .env
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models for request validation
class ChatRequest(BaseModel):
    prompt: str
    model: Literal["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"] = "gpt-4o"
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

class ImageRequest(BaseModel):
    prompt: str
    model: Literal["dall-e-3", "dall-e-2"] = "dall-e-3"
    size: Literal["1024x1024", "512x512", "256x256"] = "1024x1024"
    quality: Literal["standard", "hd"] = "standard"
    n: Optional[int] = 1

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the OpenAI API Wrapper. Use /chat for GPT models (including GPT-5 family) or /image for DALL-E."}

# Chat endpoint for GPT models
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

# Image generation endpoint for DALL-E
@app.post("/image")
async def generate_image(request: ImageRequest):
    try:
        response = await client.images.generate(
            model=request.model,
            prompt=request.prompt,
            size=request.size,
            quality=request.quality,
            n=request.n
        )
        return {"images": [image.url for image in response.data]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}
