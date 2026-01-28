# SwinIR for NVIDIA Jetson Orin Nano Super

This guide explains how to run SwinIR super-resolution on the **NVIDIA Jetson Orin Nano Super** (8GB unified memory).

## 🔴 Common Error: NVMapMemHandleAlloc Error 12

If you see this error:
```
NVMapMemHandleAlloc error 12
INTERNAL ASSERT FAILED at ../c10/cuda/CUDACachingAllocator.cpp:1113
```

**This means GPU memory is exhausted.** The Jetson Orin Nano Super has only 8GB of unified memory shared between CPU and GPU, which is far less than desktop GPUs.

## 🛠️ Optimizations Applied

The Jetson-optimized scripts include these memory management techniques:

### 1. PyTorch CUDA Allocator Configuration
```python
PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6'
```
- **expandable_segments**: Reduces memory fragmentation
- **max_split_size_mb**: Prevents large block fragmentation (128MB limit)
- **garbage_collection_threshold**: Triggers cleanup at 60% memory usage

### 2. cuDNN Settings
```python
torch.backends.cudnn.benchmark = False  # Disable algorithm caching
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 on Ampere cores
```

### 3. Memory Management
- Model loaded to CPU first, then moved to GPU
- Aggressive garbage collection after each image
- Explicit `torch.cuda.empty_cache()` calls
- Lazy tensor allocation in TensorRT

### 4. FP32 Inference
Using FP32 (not FP16) for maximum stability and compatibility.

## 🚀 Quick Start

All scripts now use **absolute paths** automatically. Just run from any directory:

### For TensorRT (Fastest, Lowest Memory)
```bash
cd d:/TensorRT_ISR2.0/pythonProject/PnP_Nystra/run_swinIR
python trt_jetson.py
```

### For PyTorch
```bash
cd d:/TensorRT_ISR2.0/pythonProject/PnP_Nystra/run_swinIR
python run_swinir_jetson.py
```

### For Benchmarking
```bash
cd d:/TensorRT_ISR2.0/pythonProject/PnP_Nystra/run_swinIR
python benchmark_jetson.py --runtime trt --iterations 5
```

## 📁 Jetson-Optimized Files

| File | Description |
|------|-------------|
| `jetson_utils.py` | Core utilities: memory management, path resolution, environment config |
| `run_swinir_jetson.py` | PyTorch inference optimized for Jetson |
| `trt_jetson.py` | TensorRT inference with minimal memory |
| `benchmark_jetson.py` | Safe benchmarking (one runtime at a time) |

## 🔧 Build TensorRT Engine on Jetson

**IMPORTANT:** TensorRT engines are NOT portable between different GPU architectures. You must rebuild on your Jetson:

```bash
cd d:/TensorRT_ISR2.0/pythonProject/PnP_Nystra

# Using the Python script (recommended)
python converter_codes/build_engine_from_onnx.py --workspace 256

# Or using trtexec directly
/usr/src/tensorrt/bin/trtexec \
    --onnx=swinir_pnp2.onnx \
    --saveEngine=swinir_pnp_fp32.engine \
    --workspace=256
```

## ⚙️ Environment Variables

Set these before running scripts:
```bash
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6'
export TRT_MAX_WORKSPACE_SIZE=268435456
```

(The Jetson scripts set these automatically when imported)

## 🆘 If Still Getting OOM Errors

### 1. Add Swap Space
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 2. Reduce System Memory Load
```bash
# Disable X11/GUI temporarily
sudo systemctl stop gdm3

# Or switch to text mode
sudo systemctl set-default multi-user.target
sudo reboot
```

### 3. Process One Image at a Time
```bash
python run_swinir_jetson.py --sample_k 1
```

### 4. Use Tiled Processing
```bash
python run_swinir_jetson.py --use_tile --tile 128
```

### 5. Monitor Memory in Real-Time
```bash
# In another terminal
tegrastats
```

## 📊 Expected Performance (FP32)

| Runtime | Inference Time (224x224) | Memory Usage |
|---------|-------------------------|--------------|
| TensorRT | ~25-40ms | ~800MB-1GB |
| PyTorch | ~80-150ms | ~2-3GB |
| ONNX Runtime | ~30-60ms | ~800MB |

*Note: Actual performance varies with model configuration and system load.*

## 📂 Directory Structure

All paths are now resolved automatically from the project root:
```
PnP_Nystra/
├── converter_codes/
│   ├── build_engine_from_onnx.py  # Build TRT engine
│   └── pth_to_onnx.py             # Export to ONNX
├── input_images/                   # Place your images here
├── model_weights/
│   └── x2.pth                     # PyTorch weights
├── results/                        # Output directory
├── swinir_pnp2.onnx               # ONNX model
├── swinir_pnp_fp32.engine         # TRT engine (build on Jetson!)
└── run_swinIR/
    ├── jetson_utils.py            # Jetson utilities
    ├── run_swinir_jetson.py       # PyTorch inference
    ├── trt_jetson.py              # TRT inference
    └── benchmark_jetson.py        # Benchmarking
```

## ❓ FAQ

**Q: Why is FP32 used instead of FP16?**
A: The user requested FP32 for maximum stability. FP16 can be enabled by modifying the `use_fp16` parameter if your model supports it.

**Q: The script says "No images found"?**
A: Place your input images in `PnP_Nystra/input_images/` or specify with `--image` / `--folder_lq`.

**Q: Engine file not found?**
A: You need to build the TensorRT engine on your Jetson. See "Build TensorRT Engine" section above.

**Q: How do I know if I'm running out of memory?**
A: The scripts print memory status at each stage. Look for "[Memory]" lines in the output.
