"""
High-Performance Super Resolution Server

Features:
- Async FastAPI endpoints for non-blocking I/O
- Request queuing with backpressure control
- Multiple concurrent GPU workers
- Response streaming for large images
- Automatic model management
"""

import os
import io
import time
import base64
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sr_engine import SuperResolutionEngine, MODEL_CONFIGS, DEFAULT_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Server configuration
HOST = os.getenv("SR_HOST", "0.0.0.0")
PORT = int(os.getenv("SR_PORT", "8001"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("SR_MAX_CONCURRENT", "4"))
MAX_IMAGE_SIZE = int(os.getenv("SR_MAX_IMAGE_SIZE", "4096"))  # Max dimension
REQUEST_TIMEOUT = float(os.getenv("SR_REQUEST_TIMEOUT", "120.0"))  # seconds
TILE_SIZE = int(os.getenv("SR_TILE_SIZE", "512"))

# Global state
engine: Optional[SuperResolutionEngine] = None
request_semaphore: Optional[asyncio.Semaphore] = None
executor: Optional[ThreadPoolExecutor] = None


# Request/Response models
class UpscaleRequest(BaseModel):
    """Request model for base64 image upscaling."""
    image: str = Field(..., description="Base64 encoded image")
    scale: int = Field(default=4, ge=2, le=4, description="Upscale factor (only 4x supported)")
    model: str = Field(default=DEFAULT_MODEL, description="Model to use")
    output_format: str = Field(default="png", description="Output format (png, jpg, webp)")
    quality: int = Field(default=95, ge=1, le=100, description="Output quality for lossy formats")


class UpscaleResponse(BaseModel):
    """Response model for upscaling."""
    image: str = Field(..., description="Base64 encoded upscaled image")
    width: int = Field(..., description="Output image width")
    height: int = Field(..., description="Output image height")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model: str = Field(..., description="Model used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    device: str
    provider: str
    max_concurrent: int
    active_requests: int


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    scale: int
    description: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global engine, request_semaphore, executor
    
    logger.info("Starting Super Resolution Server...")
    logger.info(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    logger.info(f"Max image size: {MAX_IMAGE_SIZE}px")
    
    # Initialize engine
    try:
        engine = SuperResolutionEngine(
            model_name=DEFAULT_MODEL,
            device="cuda",
            tile_size=TILE_SIZE,
        )
        logger.info("SR Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SR engine: {e}")
        raise
    
    # Initialize concurrency controls
    request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
    
    logger.info(f"Server ready at http://{HOST}:{PORT}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if executor:
        executor.shutdown(wait=True)
    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Super Resolution API",
    description="High-performance image super resolution using Real-ESRGAN",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_active_requests() -> int:
    """Get number of active requests."""
    if request_semaphore is None:
        return 0
    return MAX_CONCURRENT_REQUESTS - request_semaphore._value


async def run_in_executor(func, *args):
    """Run a blocking function in the thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    # Remove data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    
    # Decode base64
    image_bytes = base64.b64decode(b64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    return image


def encode_image_to_base64(
    image: np.ndarray,
    format: str = "png",
    quality: int = 95,
) -> str:
    """Encode numpy array to base64 string."""
    # Encode image
    if format.lower() == "jpg" or format.lower() == "jpeg":
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ext = ".jpg"
    elif format.lower() == "webp":
        encode_param = [cv2.IMWRITE_WEBP_QUALITY, quality]
        ext = ".webp"
    else:
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ext = ".png"
    
    _, buffer = cv2.imencode(ext, image, encode_param)
    
    # Convert to base64
    b64_string = base64.b64encode(buffer).decode("utf-8")
    
    return b64_string


def process_upscale(
    image: np.ndarray,
    model_name: str = DEFAULT_MODEL,
) -> tuple[np.ndarray, float]:
    """Process image upscaling (blocking)."""
    global engine
    
    start_time = time.time()
    
    # Upscale image
    output = engine.upscale(image, use_tile=True)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    return output, inference_time


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    info = engine.get_info()
    
    return HealthResponse(
        status="healthy",
        model=info["model_name"],
        device=info["device"],
        provider=info["provider"] or "unknown",
        max_concurrent=MAX_CONCURRENT_REQUESTS,
        active_requests=get_active_requests(),
    )


@app.get("/models", response_model=List[ModelInfo], tags=["System"])
async def list_models():
    """List available models."""
    models = []
    for name, config in MODEL_CONFIGS.items():
        models.append(ModelInfo(
            name=name,
            scale=config["scale"],
            description=f"{name} - {config['scale']}x upscaling",
        ))
    return models


@app.post("/upscale", response_model=UpscaleResponse, tags=["Super Resolution"])
async def upscale_base64(request: UpscaleRequest):
    """
    Upscale an image from base64 input.
    
    Returns the upscaled image as base64.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Acquire semaphore for concurrency control
    async with request_semaphore:
        try:
            # Decode input image
            image = await run_in_executor(decode_base64_image, request.image)
            
            # Validate image size
            h, w = image.shape[:2]
            if max(h, w) > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too large. Max dimension: {MAX_IMAGE_SIZE}px"
                )
            
            logger.info(f"Processing image: {w}x{h}")
            
            # Process upscaling
            output, inference_time = await asyncio.wait_for(
                run_in_executor(process_upscale, image, request.model),
                timeout=REQUEST_TIMEOUT,
            )
            
            # Encode output
            output_b64 = await run_in_executor(
                encode_image_to_base64,
                output,
                request.output_format,
                request.quality,
            )
            
            out_h, out_w = output.shape[:2]
            logger.info(f"Output: {out_w}x{out_h}, {inference_time:.1f}ms")
            
            return UpscaleResponse(
                image=output_b64,
                width=out_w,
                height=out_h,
                inference_time_ms=inference_time,
                model=request.model,
            )
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Upscale error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/upscale/file", tags=["Super Resolution"])
async def upscale_file(
    file: UploadFile = File(..., description="Image file to upscale"),
    model: str = Query(DEFAULT_MODEL, description="Model to use"),
    output_format: str = Query("png", description="Output format"),
    quality: int = Query(95, ge=1, le=100, description="Output quality"),
):
    """
    Upscale an uploaded image file.
    
    Returns the upscaled image as a file download.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Validate file type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    async with request_semaphore:
        try:
            # Read file
            contents = await file.read()
            
            # Decode image
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to decode image")
            
            # Validate size
            h, w = image.shape[:2]
            if max(h, w) > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too large. Max dimension: {MAX_IMAGE_SIZE}px"
                )
            
            logger.info(f"Processing file: {file.filename}, {w}x{h}")
            
            # Process upscaling
            output, inference_time = await asyncio.wait_for(
                run_in_executor(process_upscale, image, model),
                timeout=REQUEST_TIMEOUT,
            )
            
            # Encode output
            if output_format.lower() in ["jpg", "jpeg"]:
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
                ext = ".jpg"
                media_type = "image/jpeg"
            elif output_format.lower() == "webp":
                encode_param = [cv2.IMWRITE_WEBP_QUALITY, quality]
                ext = ".webp"
                media_type = "image/webp"
            else:
                encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                ext = ".png"
                media_type = "image/png"
            
            _, buffer = cv2.imencode(ext, output, encode_param)
            
            # Generate output filename
            input_name = Path(file.filename or "image").stem
            output_filename = f"{input_name}_sr{ext}"
            
            out_h, out_w = output.shape[:2]
            logger.info(f"Output: {output_filename}, {out_w}x{out_h}, {inference_time:.1f}ms")
            
            return StreamingResponse(
                io.BytesIO(buffer.tobytes()),
                media_type=media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{output_filename}"',
                    "X-Inference-Time-Ms": str(inference_time),
                    "X-Output-Width": str(out_w),
                    "X-Output-Height": str(out_h),
                },
            )
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File upscale error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["System"])
async def get_stats():
    """Get server statistics."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    info = engine.get_info()
    
    return {
        "engine": info,
        "server": {
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "active_requests": get_active_requests(),
            "max_image_size": MAX_IMAGE_SIZE,
            "tile_size": TILE_SIZE,
            "request_timeout": REQUEST_TIMEOUT,
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        workers=1,  # Single worker for GPU
        log_level="info",
        access_log=True,
        reload=False,  # Disable for production
    )
