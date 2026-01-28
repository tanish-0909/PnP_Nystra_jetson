"""
Jetson Orin Nano Super - Optimized Utilities
=============================================
This module provides memory-optimized utilities specifically designed for 
NVIDIA Jetson devices with limited unified memory (8GB).

Key Optimizations:
- Aggressive memory management
- FP32 inference (stable, compatible)
- CUDA memory growth limiting
- Expandable segments allocator
- Max split size limiting
- cuDNN optimization settings
"""

import os
import gc
import torch
import numpy as np
import warnings

# ============================================================================
# PATH UTILITIES
# ============================================================================
def get_project_root():
    """Get the absolute path to the PnP_Nystra project root."""
    # This file is in run_swinIR/, so parent is PnP_Nystra/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def get_absolute_path(relative_path):
    """Convert a relative path to absolute path from project root.
    
    Args:
        relative_path: Path relative to PnP_Nystra/ (e.g., 'model_weights/x2.pth')
        
    Returns:
        Absolute path string
    """
    project_root = get_project_root()
    # Handle both forward and back slashes
    relative_path = relative_path.replace('\\', os.sep).replace('/', os.sep)
    # Remove leading ../ or ..\\ if present
    relative_path = relative_path.lstrip('.').lstrip(os.sep).lstrip('.').lstrip(os.sep)
    return os.path.join(project_root, relative_path)


# ============================================================================
# ENVIRONMENT CONFIGURATION (Must be set before importing other CUDA libs)
# ============================================================================
def configure_jetson_environment():
    """Configure PyTorch for optimal Jetson performance.
    
    MUST be called at the very beginning of your script, before any CUDA operations.
    """
    # === MEMORY ALLOCATOR CONFIGURATION ===
    # Enable expandable segments to reduce fragmentation
    # Set max_split_size_mb to prevent large block fragmentation
    # garbage_collection_threshold triggers cleanup at 60% memory usage
    alloc_conf = ','.join([
        'expandable_segments:True',
        'max_split_size_mb:128',
        'garbage_collection_threshold:0.6'
    ])
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = alloc_conf
    
    # === TENSORRT CONFIGURATION ===
    # Reduce TensorRT workspace size (in bytes) - 256MB for Jetson
    os.environ['TRT_MAX_WORKSPACE_SIZE'] = str(256 * 1024 * 1024)
    
    # === CUDNN CONFIGURATION ===
    # Disable cuDNN benchmark to reduce memory overhead from algorithm caching
    # (benchmark mode tries multiple algorithms and caches the best one)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Allow TF32 on Ampere+ GPUs (Jetson Orin has Ampere cores)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # === CUDA MEMORY MANAGEMENT ===
    # Force synchronous memory operations for better cleanup
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # === DEBUG OPTIONS (uncomment for debugging) ===
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous for better errors
    # os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Device-side assertions
    
    print("[Jetson] Environment configured for low-memory FP32 operation")
    print(f"[Jetson] PYTORCH_CUDA_ALLOC_CONF = {alloc_conf}")


def get_optimal_dtype():
    """Return the optimal dtype for Jetson inference (FP32 for stability)."""
    return torch.float32


def get_device():
    """Get the CUDA device with proper error handling."""
    if not torch.cuda.is_available():
        warnings.warn("[Jetson] CUDA not available, falling back to CPU")
        return torch.device('cpu')
    return torch.device('cuda')


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================
def clear_memory(aggressive=True):
    """Clear GPU and CPU memory aggressively.
    
    Args:
        aggressive: If True, performs multiple gc.collect() passes
    """
    # Clear Python garbage
    if aggressive:
        for _ in range(3):
            gc.collect()
    else:
        gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_info():
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0}
    
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    
    # Jetson uses unified memory, estimate free based on reserved
    # Assume 7GB usable (leaving 1GB for system)
    usable_mb = 7 * 1024
    free = usable_mb - reserved
    
    return {
        "allocated": f"{allocated:.1f}MB",
        "reserved": f"{reserved:.1f}MB",
        "estimated_free": f"{max(0, free):.1f}MB"
    }


def print_memory_status(prefix=""):
    """Print current memory status."""
    info = get_memory_info()
    print(f"[Memory] {prefix} Allocated: {info['allocated']}, "
          f"Reserved: {info['reserved']}, Est. Free: {info['estimated_free']}")


# ============================================================================
# TENSOR UTILITIES
# ============================================================================
def preprocess_image_jetson(img_bgr, target_size=224, use_fp16=False):
    """Preprocess an image for inference with optimal memory usage.
    
    Args:
        img_bgr: Input BGR image from cv2.imread
        target_size: Target size for resizing (default 224)
        use_fp16: Use FP16 for reduced memory
        
    Returns:
        torch.Tensor: Preprocessed tensor (1, 3, H, W) on GPU
    """
    import cv2
    
    # Store original dimensions
    h_old, w_old = img_bgr.shape[:2]
    
    # Normalize and resize
    img = img_bgr.astype(np.float32) / 255.0
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    # BGR -> RGB, HWC -> CHW
    img = img[:, :, [2, 1, 0]].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    
    # Convert to tensor
    dtype = torch.float16 if use_fp16 else torch.float32
    tensor = torch.from_numpy(img).unsqueeze(0).to(dtype=dtype)
    
    # Move to GPU (in-place to avoid creating CPU copy)
    tensor = tensor.cuda(non_blocking=True)
    
    return tensor, (h_old, w_old)


def postprocess_output_jetson(output_tensor, original_size, scale=2):
    """Postprocess model output with minimal memory overhead.
    
    Args:
        output_tensor: Model output tensor
        original_size: Tuple (h_old, w_old)
        scale: Upscaling factor
        
    Returns:
        np.ndarray: BGR uint8 image
    """
    import cv2
    
    h_old, w_old = original_size
    
    # Move to CPU and convert (avoid in-place operations on GPU memory)
    output_np = output_tensor.float().squeeze().clamp(0, 1).cpu().numpy()
    
    # CHW -> HWC, RGB -> BGR
    output_np = np.transpose(output_np, (1, 2, 0))
    output_np = output_np[:, :, [2, 1, 0]]
    output_np = (output_np * 255.0).round().astype(np.uint8)
    
    # Final resize
    final_output = cv2.resize(
        output_np, 
        (w_old * scale, h_old * scale), 
        interpolation=cv2.INTER_CUBIC
    )
    
    # Clear intermediate
    del output_np
    
    return final_output


# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================
def load_model_to_gpu_safe(model, use_fp16=False):
    """Safely load a model to GPU with memory checks.
    
    Args:
        model: PyTorch model
        use_fp16: Convert to FP16 for memory savings
        
    Returns:
        model: Model on GPU
    """
    clear_memory()
    print_memory_status("Before model load:")
    
    model.eval()
    
    if use_fp16:
        model = model.half()
        print("[Jetson] Model converted to FP16")
    
    model = model.cuda()
    print_memory_status("After model load:")
    
    return model


# ============================================================================
# INFERENCE CONTEXT MANAGER
# ============================================================================
class JetsonInferenceContext:
    """Context manager for memory-safe inference on Jetson.
    
    Usage:
        with JetsonInferenceContext() as ctx:
            output = model(input)
    """
    def __init__(self, clear_before=True, clear_after=True):
        self.clear_before = clear_before
        self.clear_after = clear_after
    
    def __enter__(self):
        if self.clear_before:
            clear_memory()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.clear_after:
            clear_memory()
        return False


# ============================================================================
# INITIALIZATION
# ============================================================================
# Auto-configure when module is imported
configure_jetson_environment()
