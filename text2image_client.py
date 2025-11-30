#!/usr/bin/env python3
"""
Text2Image Server Client
Simple client to test the text2image_server.py API

Example Commands:
    # Check server health
    python text2image_client.py --health
    python text2image_client.py --health --url http://localhost:8000

    # Get server metrics
    python text2image_client.py --metrics
    python text2image_client.py --metrics --url http://localhost:8000

    # Generate a simple image
    python text2image_client.py --prompt "a beautiful sunset over mountains"

    # Generate with custom dimensions
    python text2image_client.py --prompt "a cat wearing sunglasses" --width 512 --height 512

    # Generate with specific seed for reproducibility
    python text2image_client.py --prompt "futuristic city" --seed 42

    # Generate with negative prompt
    python text2image_client.py --prompt "a serene landscape" --negative-prompt "people, buildings, cars"

    # Generate with custom inference steps and guidance
    python text2image_client.py --prompt "abstract art" --steps 20 --guidance 7.5

    # Generate and save to specific file
    python text2image_client.py --prompt "space exploration" --output ./images/space.png

    # Generate with all parameters
    python text2image_client.py --prompt "cyberpunk cityscape" \
        --negative-prompt "blurry, low quality" \
        --width 1024 --height 1024 \
        --seed 123 \
        --steps 15 \
        --guidance 5.0 \
        --output ./output/cyberpunk.png

    # Connect to remote server
    python text2image_client.py --prompt "test image" --url http://192.168.1.100:8000
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests


class Text2ImageClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> dict:
        """Check server health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check failed: {e}")
            return {}
    
    def get_metrics(self) -> dict:
        """Get server metrics"""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get metrics: {e}")
            return {}
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 1024,
        width: int = 1024,
        seed: int = -1,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        save_path: Optional[str] = None,
    ) -> dict:
        """Generate an image from a text prompt"""
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        print(f"🔄 Generating image: '{prompt}'...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            
            elapsed = time.time() - start_time
            
            # Extract base64 image
            image_data_uri = result.get("image_base64", "")
            if image_data_uri.startswith("data:image"):
                # Remove data URI prefix
                image_base64 = image_data_uri.split(",", 1)[1]
            else:
                image_base64 = image_data_uri
            
            # Decode and save if path provided
            if save_path:
                image_bytes = base64.b64decode(image_base64)
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(image_bytes)
                print(f"✅ Image saved to: {save_path}")
            
            print(f"✅ Generation completed in {elapsed:.2f}s")
            print(f"   Seed: {result.get('seed')}")
            print(f"   Generation time: {result.get('generation_time_ms')}ms")
            print(f"   Queue wait: {result.get('queue_wait_ms')}ms")
            print(f"   Dimensions: {result.get('width')}x{result.get('height')}")
            
            return result
            
        except requests.exceptions.Timeout:
            print("❌ Request timed out (server may be busy)")
            return {}
        except requests.exceptions.RequestException as e:
            print(f"❌ Generation failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Error details: {e.response.text}")
            return {}


def main():
    parser = argparse.ArgumentParser(description="Text2Image Server Client")
    parser.add_argument("--url", default="http://localhost:8010", help="Server URL")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--steps", type=int, default=9, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=0.0, help="Guidance scale")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--health", action="store_true", help="Check server health only")
    parser.add_argument("--metrics", action="store_true", help="Show server metrics only")
    
    args = parser.parse_args()
    
    client = Text2ImageClient(args.url)
    
    # Health check
    if args.health:
        print("🏥 Checking server health...")
        health = client.health_check()
        if health:
            print(json.dumps(health, indent=2))
        sys.exit(0 if health else 1)
    
    # Metrics
    if args.metrics:
        print("📊 Server metrics:")
        metrics = client.get_metrics()
        if metrics:
            print(json.dumps(metrics, indent=2))
        sys.exit(0 if metrics else 1)
    
    # Generate image
    output_path = args.output or f"generated_{int(time.time())}.png"
    result = client.generate_image(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        save_path=output_path,
    )
    
    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()

