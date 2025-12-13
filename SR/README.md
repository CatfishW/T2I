# Super Resolution Server

High-performance image super resolution API using Real-ESRGAN with ONNX Runtime.

## Features

- **4x upscaling** with Real-ESRGAN x4plus model
- **ONNX Runtime GPU** for fast inference (4-5x faster than PyTorch)
- **Tile-based processing** for large images (avoids GPU OOM)
- **High concurrency** support with async FastAPI
- **Auto-download** models from HuggingFace

## Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Start server
python server.py
```

Server runs at `http://localhost:8001`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upscale` | Base64 image input |
| POST | `/upscale/file` | File upload |
| GET | `/health` | Health check |
| GET | `/models` | List models |
| GET | `/docs` | Swagger UI |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SR_PORT` | 8001 | Server port |
| `SR_MAX_CONCURRENT` | 4 | Max concurrent requests |
| `SR_MAX_IMAGE_SIZE` | 4096 | Max input dimension (px) |
| `SR_TILE_SIZE` | 512 | Tile size for processing |

## Example Usage

### Python
```python
import requests
import base64

# Read image
with open("input.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Upscale
response = requests.post(
    "http://localhost:8001/upscale",
    json={"image": image_b64}
)

# Save output
output_b64 = response.json()["image"]
with open("output.png", "wb") as f:
    f.write(base64.b64decode(output_b64))
```

### cURL
```bash
curl -X POST "http://localhost:8001/upscale/file" \
  -F "file=@input.jpg" \
  -o output.png
```

## Performance

- ~50-100ms per image (512x512) on RTX 3090
- Supports images up to 4096x4096 with tile processing
- 4 concurrent requests by default
