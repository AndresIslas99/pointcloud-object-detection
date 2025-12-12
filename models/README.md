# Models Directory

Place your TensorRT engine files here.

## Download Pre-trained Models

```bash
# Using the download script
python3 ../scripts/download_models.py --output . --precision fp16

# Or manually:
# 1. Download ONNX model from NVIDIA CUDA-PointPillars releases
# 2. Convert to TensorRT:
trtexec --onnx=pointpillars.onnx \
    --saveEngine=pointpillars.engine \
    --fp16 \
    --workspace=4096
```

## Expected Files

- `pointpillars.engine` - TensorRT FP16 engine for Jetson AGX Orin
- `pointpillars.onnx` - ONNX model (optional, for re-conversion)

## File Sizes

- ONNX: ~20 MB
- TensorRT Engine: ~30-50 MB (varies by platform)
