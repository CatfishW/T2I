import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ============================================================================
# Configuration
# ============================================================================

# URL of the text-to-image server (the machine hosting the model)
# Note: If running both gateway and server on same machine, use different ports:
# - Gateway: port 21115 (default)
# - Server: port 8010 (run with: python text2image_server.py --port 8010)
TEXT2IMAGE_SERVER_URL = os.getenv(
    "TEXT2IMAGE_SERVER_URL",
    "http://localhost:8010"  # Default port 8010, matching start-production.sh
)

# Request timeout in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes

# HTTP client settings
HTTP_TIMEOUT = httpx.Timeout(REQUEST_TIMEOUT, connect=10.0)
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

# Global HTTP client
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup HTTP client"""
    global http_client
    
    # Get port from environment or use default
    backend_port = int(os.getenv("BACKEND_PORT", "21115"))
    
    # Startup
    print("=" * 60)
    print("Text-to-Image API Gateway")
    print("=" * 60)
    print(f"Forwarding to: {TEXT2IMAGE_SERVER_URL}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print("=" * 60)
    
    # Create HTTP client with connection pooling
    http_client = httpx.AsyncClient(
        base_url=TEXT2IMAGE_SERVER_URL,
        timeout=HTTP_TIMEOUT,
        limits=httpx.Limits(
            max_keepalive_connections=MAX_CONCURRENT_REQUESTS,
            max_connections=MAX_CONCURRENT_REQUESTS * 2,
        ),
    )
    
    # Test connection to text2image server
    try:
        response = await http_client.get("/health")
        if response.status_code == 200:
            print("✓ Connected to text-to-image server")
            server_info = response.json()
            print(f"  Model loaded: {server_info.get('model_loaded', 'unknown')}")
            print(f"  CUDA available: {server_info.get('cuda_available', 'unknown')}")
        else:
            print(f"⚠ Warning: Text-to-image server returned status {response.status_code}")
    except Exception as e:
        print(f"⚠ Warning: Could not connect to text-to-image server: {e}")
        print(f"  Make sure {TEXT2IMAGE_SERVER_URL} is running")
    
    print(f"\nServer ready! Listening on http://0.0.0.0:{backend_port}")
    print("=" * 60 + "\n")
    
    yield
    
    # Shutdown
    print("\nShutting down API gateway...")
    if http_client:
        await http_client.aclose()
        print("✓ HTTP client closed")


app = FastAPI(
    title="Text-to-Image API Gateway",
    description="API Gateway that forwards requests to a remote text-to-image generation server",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware - allow all origins (configure as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Semaphore to limit concurrent requests to the text2image server
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt for image generation")
    negative_prompt: str = Field(default="", max_length=2000, description="Negative prompt")
    height: int = Field(default=1024, ge=256, le=2048, description="Image height")
    width: int = Field(default=1024, ge=256, le=2048, description="Image width")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    num_inference_steps: int = Field(default=9, ge=6, le=12, description="Number of inference steps")
    guidance_scale: float = Field(default=0.0, description="Guidance scale (fixed at 0.0 for Turbo models)")


class GenerateResponse(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded PNG image with data URI prefix")
    seed: int = Field(..., description="Seed used for generation")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    image_id: Optional[str] = Field(None, description="Optional saved image ID")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Text-to-Image API Gateway",
        "version": "2.0.0",
        "status": "running",
        "text2image_server_url": TEXT2IMAGE_SERVER_URL,
        "mode": "gateway",
    }


@app.get("/health")
async def health():
    """Health check endpoint - also checks upstream server"""
    if http_client is None:
        return {
            "status": "unhealthy",
            "gateway": "not_initialized",
            "text2image_server": {
                "status": "unknown",
                "model_loaded": False,
            },
        }
    
    try:
        # Check upstream text2image server
        response = await http_client.get("/health", timeout=5.0)
        if response.status_code == 200:
            server_health = response.json()
            server_status = server_health.get("status", "unknown")
            model_loaded = server_health.get("model_loaded", False)
            cuda_available = server_health.get("cuda_available", False)
            
            # Consider healthy if:
            # 1. Server status is "healthy" (primary check), OR
            # 2. Model is loaded and CUDA is available (detailed check)
            is_healthy = (
                server_status == "healthy" or 
                (model_loaded and cuda_available)
            )
            
            return {
                "status": "healthy" if is_healthy else "degraded",
                "gateway": "running",
                "text2image_server": {
                    "status": server_status,
                    "model_loaded": model_loaded,
                    "cuda_available": cuda_available,
                    "current_queue_size": server_health.get("current_queue_size", 0),
                    "active_generations": server_health.get("active_generations", 0),
                    "available_slots": server_health.get("available_slots", 0),
                },
            }
        else:
            return {
                "status": "degraded",
                "gateway": "running",
                "text2image_server": {
                    "status": f"error_{response.status_code}",
                    "model_loaded": False,
                    "cuda_available": False,
                },
            }
    except httpx.ConnectError as e:
        return {
            "status": "degraded",
            "gateway": "running",
            "text2image_server": {
                "status": "unreachable",
                "model_loaded": False,
                "cuda_available": False,
                "error": f"Cannot connect to {TEXT2IMAGE_SERVER_URL}: {str(e)}",
            },
        }
    except httpx.TimeoutException:
        return {
            "status": "degraded",
            "gateway": "running",
            "text2image_server": {
                "status": "timeout",
                "model_loaded": False,
                "cuda_available": False,
                "error": "Health check timed out",
            },
        }
    except Exception as e:
        return {
            "status": "degraded",
            "gateway": "running",
            "text2image_server": {
                "status": "error",
                "model_loaded": False,
                "cuda_available": False,
                "error": str(e),
            },
        }


@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """Generate an image by forwarding request to text-to-image server"""
    if http_client is None:
        raise HTTPException(status_code=503, detail="API Gateway not initialized")
    
    # Acquire semaphore to limit concurrent requests
    async with request_semaphore:
        try:
            start_time = time.time()
            
            # Forward request to text2image server
            response = await http_client.post(
                "/generate",
                json={
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "height": request.height,
                    "width": request.width,
                    "seed": request.seed,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                },
            )
            
            # Handle response
            if response.status_code == 200:
                server_response = response.json()
                
                # Map response from text2image_server to our format
                # text2image_server returns: image_base64, seed, generation_time_ms, queue_wait_ms, width, height, image_id
                # We return: image_base64, seed, generation_time_ms, width, height, image_id
                return GenerateResponse(
                    image_base64=server_response.get("image_base64"),
                    seed=server_response.get("seed"),
                    generation_time_ms=server_response.get("generation_time_ms", 0),
                    width=server_response.get("width", request.width),
                    height=server_response.get("height", request.height),
                    image_id=server_response.get("image_id"),
                )
            elif response.status_code == 503:
                # Service unavailable from upstream
                error_detail = response.json().get("detail", "Text-to-image server unavailable")
                raise HTTPException(status_code=503, detail=error_detail)
            elif response.status_code == 504:
                # Timeout from upstream
                error_detail = response.json().get("detail", "Request timed out on text-to-image server")
                raise HTTPException(status_code=504, detail=error_detail)
            else:
                # Other errors
                try:
                    error_detail = response.json().get("detail", response.text)
                except:
                    error_detail = f"Text-to-image server returned status {response.status_code}"
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Text-to-image server error: {error_detail}"
                )
                
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=f"Request to text-to-image server timed out after {REQUEST_TIMEOUT} seconds"
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to text-to-image server at {TEXT2IMAGE_SERVER_URL}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Error communicating with text-to-image server: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )


@app.post("/generate-stream")
async def generate_image_stream(request: GenerateRequest):
    """Generate an image with Server-Sent Events for progress streaming"""
    if http_client is None:
        raise HTTPException(status_code=503, detail="API Gateway not initialized")
    
    async def event_generator():
        """Generator function for SSE events"""
        async with request_semaphore:
            try:
                # Forward streaming request to text2image server
                async with http_client.stream(
                    "POST",
                    "/generate-stream",
                    json={
                        "prompt": request.prompt,
                        "negative_prompt": request.negative_prompt,
                        "height": request.height,
                        "width": request.width,
                        "seed": request.seed,
                        "num_inference_steps": request.num_inference_steps,
                        "guidance_scale": request.guidance_scale,
                    },
                ) as response:
                    if response.status_code == 404:
                        # Endpoint not found - provide helpful error message
                        error_message = f"Endpoint /generate-stream not found on text-to-image server. The server may need to be restarted to load this endpoint."
                        yield f"data: {json.dumps({'type': 'error', 'message': error_message})}\n\n"
                        return
                    elif response.status_code != 200:
                        # Try to read error message from response
                        error_message = f"Upstream server returned status {response.status_code}"
                        try:
                            # Try to read as text first (for SSE error messages)
                            async for chunk in response.aiter_text():
                                if chunk:
                                    # Check if it's an SSE error message
                                    for line in chunk.split("\n"):
                                        if line.startswith("data: "):
                                            try:
                                                data = json.loads(line[6:])
                                                if data.get("type") == "error":
                                                    error_message = data.get("message", error_message)
                                                    break
                                            except:
                                                pass
                            # If no SSE format, try to read as JSON
                            if error_message.startswith("Upstream"):
                                try:
                                    error_data = await response.aread()
                                    error_json = json.loads(error_data)
                                    error_message = error_json.get("detail", error_message)
                                except:
                                    pass
                        except Exception as e:
                            error_message = f"Error reading response: {str(e)}"
                        
                        yield f"data: {json.dumps({'type': 'error', 'message': error_message})}\n\n"
                        return
                    
                    # Stream the response from upstream server
                    async for chunk in response.aiter_text():
                        if chunk:
                            yield chunk
                            
            except httpx.TimeoutException:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Request to text-to-image server timed out after {REQUEST_TIMEOUT} seconds'})}\n\n"
            except httpx.ConnectError:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Cannot connect to text-to-image server at {TEXT2IMAGE_SERVER_URL}'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/metrics")
async def get_metrics():
    """Get metrics from upstream text-to-image server"""
    if http_client is None:
        raise HTTPException(status_code=503, detail="API Gateway not initialized")
    
    try:
        response = await http_client.get("/metrics", timeout=5.0)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail="Could not retrieve metrics from text-to-image server"
            )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error retrieving metrics: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable, default to 21115 (matching start-production.sh)
    backend_port = int(os.getenv("BACKEND_PORT", "21115"))
    uvicorn.run(app, host="0.0.0.0", port=backend_port)
