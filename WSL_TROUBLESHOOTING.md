# WSL Troubleshooting Guide

This guide helps troubleshoot local model loading issues in Windows Subsystem for Linux (WSL).

## Common Issues

### Issue: Local model not found in WSL

**Symptoms:**
- Server starts but downloads model from HuggingFace instead of using local model
- Error messages about model path not found

**Solutions:**

1. **Check the path in config.yaml:**
   ```yaml
   model:
     local_path: "./models--Tongyi-MAI--Z-Image-Turbo"
   ```
   
   The path should be relative to where `text2image_server.py` is located, OR use an absolute path.

2. **Use absolute path (recommended for WSL):**
   ```yaml
   model:
     local_path: "/mnt/d/Text2Image/models--Tongyi-MAI--Z-Image-Turbo"
   ```
   
   Or if using Windows path format:
   ```yaml
   model:
     local_path: "D:/Text2Image/models--Tongyi-MAI--Z-Image-Turbo"
   ```

3. **Verify the model folder exists:**
   ```bash
   # In WSL, check if folder exists
   ls -la models--Tongyi-MAI--Z-Image-Turbo
   
   # Or check absolute path
   ls -la /mnt/d/Text2Image/models--Tongyi-MAI--Z-Image-Turbo
   ```

4. **Check folder structure:**
   The model folder should contain:
   - `config.json` (required)
   - Model files (`*.safetensors` or `*.bin`)
   
   OR it might be in HuggingFace cache format:
   - `snapshots/` subdirectory
   - With a hash-named subdirectory inside

5. **Check file permissions in WSL:**
   ```bash
   ls -la models--Tongyi-MAI--Z-Image-Turbo/
   ```
   
   Make sure the folder is readable. If needed:
   ```bash
   chmod -R 755 models--Tongyi-MAI--Z-Image-Turbo
   ```

6. **Use script's directory:**
   The script automatically searches in multiple locations:
   - Script directory (where `text2image_server.py` is)
   - Current working directory
   - Parent directory
   - HuggingFace cache directory

## Debugging Steps

1. **Enable verbose output:**
   The server now prints detailed path search information on startup. Look for:
   ```
   Searching for local model (explicit path: ...)...
   ℹ Running in WSL - Using enhanced path resolution
   ```

2. **Check the startup logs:**
   The server will show:
   - Where it's searching
   - What paths it checked
   - Whether it found the model or not

3. **Manual path verification:**
   ```python
   from pathlib import Path
   import os
   
   # Check script location
   script_dir = Path(__file__).parent.resolve()
   print(f"Script directory: {script_dir}")
   
   # Check if model exists
   model_path = script_dir / "models--Tongyi-MAI--Z-Image-Turbo"
   print(f"Model path: {model_path}")
   print(f"Exists: {model_path.exists()}")
   print(f"Is dir: {model_path.is_dir()}")
   ```

## Path Format Examples

### Relative Path (from script directory):
```yaml
local_path: "./models--Tongyi-MAI--Z-Image-Turbo"
# or
local_path: "models--Tongyi-MAI--Z-Image-Turbo"
```

### Absolute Path (Linux/WSL):
```yaml
local_path: "/home/username/models--Tongyi-MAI--Z-Image-Turbo"
# or
local_path: "/mnt/d/Text2Image/models--Tongyi-MAI--Z-Image-Turbo"
```

### Absolute Path (Windows format in WSL):
```yaml
local_path: "D:/Text2Image/models--Tongyi-MAI--Z-Image-Turbo"
```

## Auto-Detection

If `local_path` is not set or is `null`, the server will automatically search:
1. Script directory for `models--Tongyi-MAI--Z-Image-Turbo` folders
2. Current working directory
3. HuggingFace cache directory
4. `./models/` subdirectory

## Still Having Issues?

1. **Check startup logs** - The server prints detailed path information
2. **Try absolute path** - Most reliable in WSL
3. **Check file permissions** - WSL can have permission issues with Windows filesystems
4. **Verify folder structure** - Make sure `config.json` exists in the model folder
5. **Check WSL mount** - Ensure Windows drive is properly mounted in WSL

## Example Working Config

```yaml
model:
  name: "Tongyi-MAI/Z-Image-Turbo"
  local_path: "/mnt/d/Text2Image/models--Tongyi-MAI--Z-Image-Turbo"
  # Or leave as null for auto-detection:
  # local_path: null
```

