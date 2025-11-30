#!/usr/bin/env python3
"""
High-Performance Text-to-Image Server
Based on Test.py with enhanced concurrent request support

Features:
- Configurable concurrent generation limit
- Request queue management
- Model compilation for faster inference
- Request metrics and monitoring
- Optimized memory management
- Support for high concurrent load
"""

import asyncio
import base64
import io
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

import torch
import yaml
from diffusers import ZImagePipeline
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    
    # Default configuration
    default_config = {
        "concurrency": {
            "max_concurrent": 4,
            "max_queue": 50,
            "request_timeout": 300
        },
        "model": {
            "name": "Tongyi-MAI/Z-Image-Turbo",
            "torch_dtype": "bfloat16",
            "enable_compilation": False,
            "enable_cpu_offload": False,
            "enable_flash_attention": False
        },
        "storage": {
            "images_dir": "images",
            "save_images": False
        }
    }
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
            # Merge with defaults
            config = default_config.copy()
            if "concurrency" in user_config:
                config["concurrency"].update(user_config["concurrency"])
            if "model" in user_config:
                config["model"].update(user_config["model"])
            if "storage" in user_config:
                config["storage"].update(user_config["storage"])
            return config
        except Exception as e:
            print(f"Warning: Failed to load config.yaml: {e}. Using defaults.")
            return default_config
    else:
        # Create default config file if it doesn't exist
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            print(f"Created default config file: {config_file}")
        except Exception as e:
            print(f"Warning: Could not create config file: {e}")
        return default_config

# Check for Triton availability (required for model compilation)
def check_triton_available() -> bool:
    """Check if Triton is available for model compilation"""
    try:
        import triton
        return True
    except ImportError:
        return False

# Load configuration
config = load_config()

# Concurrency settings
MAX_CONCURRENT_GENERATIONS = config["concurrency"]["max_concurrent"]
MAX_QUEUE_SIZE = config["concurrency"]["max_queue"]
REQUEST_TIMEOUT = config["concurrency"]["request_timeout"]

# Model settings
MODEL_NAME = config["model"]["name"]
dtype_str = config["model"]["torch_dtype"].lower()
if dtype_str == "bfloat16":
    TORCH_DTYPE = torch.bfloat16
elif dtype_str == "float16":
    TORCH_DTYPE = torch.float16
elif dtype_str == "float32":
    TORCH_DTYPE = torch.float32
else:
    TORCH_DTYPE = torch.bfloat16

# Check if compilation is requested and if Triton is available
ENABLE_COMPILATION_REQUESTED = config["model"]["enable_compilation"]
TRITON_AVAILABLE = check_triton_available()
ENABLE_COMPILATION = ENABLE_COMPILATION_REQUESTED and TRITON_AVAILABLE

if ENABLE_COMPILATION_REQUESTED and not TRITON_AVAILABLE:
    print("=" * 60)
    print("⚠ WARNING: Model compilation requested but Triton is not installed.")
    print("  Compilation will be disabled automatically.")
    print("  To enable compilation, install Triton:")
    print("    pip install triton")
    print("  Note: Triton installation can be complex and may require")
    print("        specific CUDA versions. Compilation is optional.")
    print("=" * 60)

ENABLE_CPU_OFFLOAD = config["model"]["enable_cpu_offload"]
ENABLE_FLASH_ATTENTION = config["model"]["enable_flash_attention"]

# Storage
IMAGES_DIR = Path(config["storage"]["images_dir"])
IMAGES_DIR.mkdir(exist_ok=True)
SAVE_IMAGES = config["storage"]["save_images"]

# ============================================================================
# Global State
# ============================================================================

pipe: Optional[ZImagePipeline] = None
generation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

# Metrics
@dataclass
class ServerMetrics:
    total_requests: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    queue_wait_times: List[float] = field(default_factory=list)
    generation_times: List[float] = field(default_factory=list)
    current_queue_size: int = 0
    active_generations: int = 0
    start_time: datetime = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()

metrics = ServerMetrics()

# ============================================================================
# Model Loading
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global pipe, ENABLE_COMPILATION
    
    print("=" * 60)
    print("Loading High-Performance Text-to-Image Server")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max Concurrent Generations: {MAX_CONCURRENT_GENERATIONS}")
    print(f"Max Queue Size: {MAX_QUEUE_SIZE}")
    print(f"Model Compilation: {ENABLE_COMPILATION}")
    print(f"CPU Offloading: {ENABLE_CPU_OFFLOAD}")
    print(f"Flash Attention: {ENABLE_FLASH_ATTENTION}")
    print("=" * 60)
    
    # Startup
    print("\nLoading model...")
    try:
        pipe = ZImagePipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")
        print("✓ Model loaded on CUDA")
        
        # Optional optimizations
        if ENABLE_FLASH_ATTENTION:
            try:
                # Try Flash Attention 3 first, fallback to 2
                try:
                    pipe.transformer.set_attention_backend("_flash_3")
                    print("✓ Flash Attention 3 enabled")
                except:
                    pipe.transformer.set_attention_backend("flash")
                    print("✓ Flash Attention 2 enabled")
            except Exception as e:
                print(f"⚠ Flash Attention not available: {e}")
        
        if ENABLE_COMPILATION:
            print("Compiling model (first run will be slower)...")
            try:
                pipe.transformer.compile()
                print("✓ Model compilation enabled")
            except Exception as e:
                print(f"⚠ Model compilation failed: {e}")
                print("  Continuing without compilation...")
                # Disable compilation for future runs
                ENABLE_COMPILATION = False
        
        if ENABLE_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
            print("✓ CPU offloading enabled")
        
        # Warm up the model with a dummy generation
        # print("\nWarming up model...")
        # try:
        #     with torch.inference_mode():
        #         with torch.no_grad():
        #             _ = pipe(
        #                 prompt="warmup",
        #                 height=512,
        #                 width=512,
        #                 num_inference_steps=4,
        #                 guidance_scale=0.0,
        #                 generator=torch.Generator("cuda").manual_seed(0),
        #             )
        #     print("✓ Model warmed up")
        # except Exception as e:
        #     print(f"⚠ Warmup failed (non-critical): {e}")
        
        print("\n" + "=" * 60)
        print("Server ready! Listening on http://0.0.0.0:8010")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        raise
    
    yield
    
    # Shutdown
    print("\nShutting down server...")
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
        print("✓ Model unloaded, CUDA cache cleared")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="High-Performance Text-to-Image API",
    description="High-concurrency text-to-image generation server using Z-Image-Turbo",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    height: int = Field(default=1024, ge=256, le=2048)
    width: int = Field(default=1024, ge=256, le=2048)
    seed: int = Field(default=-1, ge=-1)
    num_inference_steps: int = Field(default=9, ge=6, le=12)
    guidance_scale: float = Field(default=0.0)

class GenerateResponse(BaseModel):
    image_base64: str
    seed: int
    generation_time_ms: int
    queue_wait_ms: int
    width: int
    height: int
    image_id: Optional[str] = None

class MetricsResponse(BaseModel):
    total_requests: int
    successful_generations: int
    failed_generations: int
    current_queue_size: int
    active_generations: int
    average_generation_time_ms: float
    average_queue_wait_ms: float
    uptime_seconds: float
    requests_per_minute: float

# ============================================================================
# Generation Function
# ============================================================================

async def generate_image_async(
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    queue_start_time: float,
) -> Dict:
    """Generate image asynchronously"""
    generation_start = time.time()
    queue_wait_ms = int((generation_start - queue_start_time) * 1000)
    
    try:
        # Generate random seed if not provided
        if seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Run generation in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        
        def _generate():
            with torch.inference_mode():
                with torch.no_grad():
                    try:
                        return pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        ).images[0]
                    except Exception as e:
                        # If compilation error occurs during generation
                        error_str = str(e).lower()
                        if "triton" in error_str or "compile" in error_str:
                            # Model was compiled but Triton is not available at runtime
                            # We can't easily uncompile, so provide helpful error
                            raise RuntimeError(
                                "Model compilation requires Triton but it's not available. "
                                "Either install Triton (pip install triton) or disable compilation "
                                "in config.yaml by setting enable_compilation: false. "
                                "Then restart the server."
                            ) from e
                        raise
        
        # Execute in thread pool
        image = await loop.run_in_executor(None, _generate)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{image_base64}"
        
        # Optionally save to disk
        image_id = None
        if SAVE_IMAGES:
            image_id = str(uuid.uuid4())
            image_path = IMAGES_DIR / f"{image_id}.png"
            image.save(image_path)
        
        generation_time_ms = int((time.time() - generation_start) * 1000)
        
        # Update metrics
        metrics.successful_generations += 1
        metrics.generation_times.append(generation_time_ms)
        metrics.queue_wait_times.append(queue_wait_ms)
        
        # Keep only last 1000 metrics
        if len(metrics.generation_times) > 1000:
            metrics.generation_times = metrics.generation_times[-1000:]
        if len(metrics.queue_wait_times) > 1000:
            metrics.queue_wait_times = metrics.queue_wait_times[-1000:]
        
        return {
            "image_base64": image_data_uri,
            "seed": seed,
            "generation_time_ms": generation_time_ms,
            "queue_wait_ms": queue_wait_ms,
            "width": width,
            "height": height,
            "image_id": image_id,
        }
        
    except Exception as e:
        metrics.failed_generations += 1
        raise e
    finally:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        metrics.active_generations = MAX_CONCURRENT_GENERATIONS - generation_semaphore._value

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "High-Performance Text-to-Image API",
        "version": "2.0.0",
        "status": "running",
        "model": MODEL_NAME,
        "max_concurrent": MAX_CONCURRENT_GENERATIONS,
        "max_queue": MAX_QUEUE_SIZE,
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "current_queue_size": request_queue.qsize(),
        "active_generations": metrics.active_generations,
        "available_slots": MAX_CONCURRENT_GENERATIONS - metrics.active_generations,
    }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get server metrics"""
    uptime = (datetime.now() - metrics.start_time).total_seconds()
    avg_gen_time = sum(metrics.generation_times) / len(metrics.generation_times) if metrics.generation_times else 0
    avg_queue_wait = sum(metrics.queue_wait_times) / len(metrics.queue_wait_times) if metrics.queue_wait_times else 0
    rpm = (metrics.total_requests / uptime * 60) if uptime > 0 else 0
    
    return MetricsResponse(
        total_requests=metrics.total_requests,
        successful_generations=metrics.successful_generations,
        failed_generations=metrics.failed_generations,
        current_queue_size=request_queue.qsize(),
        active_generations=metrics.active_generations,
        average_generation_time_ms=round(avg_gen_time, 2),
        average_queue_wait_ms=round(avg_queue_wait, 2),
        uptime_seconds=round(uptime, 2),
        requests_per_minute=round(rpm, 2),
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """Generate an image from a text prompt with high concurrency support"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA not available")
    
    # Check queue size
    if request_queue.full():
        raise HTTPException(
            status_code=503,
            detail=f"Request queue is full (max {MAX_QUEUE_SIZE}). Please try again later."
        )
    
    metrics.total_requests += 1
    queue_start_time = time.time()
    
    # Acquire semaphore to limit concurrent generations
    try:
        async with generation_semaphore:
            # Wrap generation in timeout (compatible with Python < 3.11)
            result = await asyncio.wait_for(
                generate_image_async(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    height=request.height,
                    width=request.width,
                    seed=request.seed,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    queue_start_time=queue_start_time,
                ),
                timeout=REQUEST_TIMEOUT
            )
            return GenerateResponse(**result)
    except asyncio.TimeoutError:
        metrics.failed_generations += 1
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {REQUEST_TIMEOUT} seconds"
        )
    except Exception as e:
        metrics.failed_generations += 1
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High-Performance Text-to-Image Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (use 1 for GPU sharing)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    
    args = parser.parse_args()
    
    if args.workers > 1:
        print(f"⚠ Warning: Multiple workers ({args.workers}) may cause GPU memory issues.")
        print("  Each worker loads its own model instance.")
        print("  Recommended: Use 1 worker with higher MAX_CONCURRENT_GENERATIONS")
    
    uvicorn.run(
        "text2image_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
        timeout_keep_alive=300,
    )

