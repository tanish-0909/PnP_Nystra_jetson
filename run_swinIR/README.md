# run_swinIR

This directory contains inference scripts for running SwinIR image super-resolution models with support for multiple backend implementations: PyTorch, ONNX, and TensorRT.

## Overview

The `run_swinIR` module provides three different inference pipelines for the SwinIR model:

- **PyTorch Backend** (`run_swinir.py`): Native PyTorch inference with support for both original window attention and PnP-Nystra mechanisms
- **ONNX Backend** (`run_onnx_swinir.py`): ONNX Runtime inference for cross-platform deployment
- **TensorRT Backend** (`run_trt_swinir.py`): NVIDIA TensorRT inference for GPU-optimized performance

## Scripts

### `run_swinir.py`
PyTorch-based inference script for SwinIR super-resolution models.

**Features:**
- Support for original window attention and PnP-Nystra mechanisms
- Configurable Nyström parameters (landmarks, iterations)
- PSNR/SSIM evaluation metrics
- Batch image processing with optional sampling

**Usage:**
```bash
python run_swinir.py --input_dir <path_to_images> --result_dir <path_to_results> \
  --model_path <path_to_model.pth> --scale <2|4|8> --mech <original|pnp_nystra> \
  --device cuda
```

### `run_onnx_swinir.py`
ONNX Runtime-based inference for optimized cross-platform deployment.

**Features:**
- ONNX model execution
- Support for both original and PnP-Nystra mechanisms
- Performance metrics (PSNR/SSIM)
- Image batch processing

**Usage:**
```bash
python run_onnx_swinir.py --input_dir <path_to_images> --result_dir <path_to_results> \
  --model_path <path_to_model.onnx> --scale <2|4|8> --mech <original|pnp_nystra>
```

### `run_trt_swinir.py`
NVIDIA TensorRT-based inference for GPU-accelerated performance.

**Features:**
- TensorRT engine execution for maximum GPU throughput
- Support for both original and PnP-Nystra mechanisms
- Integrated timing analysis
- Optimized inference pipeline

**Usage:**
```bash
python run_trt_swinir.py --input_dir <path_to_images> --result_dir <path_to_results> \
  --engine_path <path_to_model.engine> --scale <2|4|8> --mech <original|pnp_nystra>
```

## Testing Utilities

### `test_trt_model.py`
Helper module for TensorRT model inference and validation.

**Contents:**
- `TRTWrapperTorch`: Wrapper class for TensorRT engine execution with PyTorch tensor compatibility
- I/O tensor binding and mapping
- Engine loading and validation

### `test_onnx_model.py`
Validation script for comparing ONNX and PyTorch model outputs.

**Purpose:**
- Verify ONNX model conversion accuracy
- Compare outputs between PyTorch and ONNX implementations
- Measure output differences for quality assurance

## Common Parameters

All inference scripts support the following key arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `--input_dir` | str | Directory containing input images for processing |
| `--result_dir` | str | Output directory for processed images |
| `--model_path` / `--engine_path` | str | Path to the model file (`.pth`, `.onnx`, or `.engine`) |
| `--scale` | int | Super-resolution scale factor (2, 4, or 8) |
| `--mech` | str | Attention mechanism: `original` or `pnp_nystra` |
| `--device` | str | Compute device: `cuda` or `cpu` (PyTorch only) |
| `--num_landmarks` | int | Number of Nyström landmarks (PnP-Nystra only) |
| `--iters` | int | Number of iterations (PnP-Nystra only) |

## Model Files

Ensure the following model files are available in the parent directory:

- `swinir_pnp.engine` - TensorRT model (PnP-Nystra version)
- `swinir_pnp_fp32.engine` - TensorRT model (FP32 precision)
- `swinir_pnp.onnx` - ONNX model
- `swinir_pnp2.onnx` - Alternative ONNX model

## Performance Comparison

The three backends offer different trade-offs:

- **PyTorch**: Maximum flexibility and ease of use, best for development
- **ONNX**: Cross-platform compatibility, good performance portability
- **TensorRT**: Maximum GPU performance, optimized inference throughput

## Requirements

- PyTorch
- ONNX Runtime (for `run_onnx_swinir.py`)
- TensorRT (for `run_trt_swinir.py`)
- OpenCV
- NumPy

Install dependencies using the main project's `requirements.txt`.

## Output

All scripts generate:
- **Processed images**: Upscaled images saved in the result directory
- **Evaluation metrics**: PSNR and SSIM values for each image
- **Timing information**: Inference timing for performance analysis

## Notes

- The PnP-Nystra mechanism provides a training-free optimization achieving 2-4× speedup with minimal quality loss
- TensorRT inference requires NVIDIA GPU and CUDA Toolkit
- Image paths are sorted alphabetically for consistent processing order
- Random seed is fixed (42) for reproducibility
