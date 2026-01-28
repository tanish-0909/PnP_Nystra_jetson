"""
Jetson Orin Nano Super - Optimized PyTorch SwinIR Inference
=============================================================
Memory-efficient PyTorch inference for SwinIR on Jetson devices.

Key Optimizations:
- FP32 inference (stable, compatible)
- Single model load (not per-image)
- Tile-based processing for large images
- Aggressive memory cleanup
"""

import argparse
import cv2
import glob
import numpy as np
import os
import sys
import torch
import random
import gc

random.seed(42)

# Configure Jetson environment FIRST
from jetson_utils import (
    configure_jetson_environment,
    clear_memory,
    print_memory_status,
    load_model_to_gpu_safe,
    JetsonInferenceContext,
    get_device,
    get_absolute_path
)

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # PnP_Nystra/
sys.path.insert(0, PROJECT_ROOT)

from models.swinir import SwinIR as net


def define_model_jetson(args, mech, num_landmarks, iters, window_size, img_size, use_fp16=False):
    """Create SwinIR model optimized for Jetson.
    
    Args:
        args: Arguments with model_path
        mech: Mechanism type ('original' or 'pnp_nystra')
        num_landmarks: Number of landmarks for pnp_nystra
        iters: Iterations for pnp_nystra
        window_size: Window size
        img_size: Image size
        use_fp16: Use FP16 for memory savings
        
    Returns:
        model: Loaded and configured model
    """
    clear_memory()
    print_memory_status("Before model creation:")
    
    # Create model with smaller parameters if memory is critical
    model = net(
        mech=mech,
        num_landmarks=num_landmarks,
        iters=iters,
        upscale=args.scale,
        in_chans=3,
        img_size=img_size,
        window_size=window_size,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    
    # Load weights
    print(f'[Jetson] Loading model from: {args.model_path}')
    
    # Load to CPU first to avoid GPU memory spike
    pretrained_model = torch.load(args.model_path, map_location='cpu')
    
    param_key_g = 'params'
    state_dict = pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model
    
    # Filter state dict
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            if value.size() == model_state_dict[key].size():
                filtered_state_dict[key] = value
    
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    
    # Free pretrained weights immediately
    del pretrained_model, state_dict, filtered_state_dict, model_state_dict
    gc.collect()
    
    # Move to GPU with FP16
    model = load_model_to_gpu_safe(model, use_fp16=use_fp16)
    
    return model


def test_jetson(img_lq, model, args, window_size, use_tile=False, tile_size=128):
    """Run inference with optional tiling for memory management.
    
    Tiling is recommended for images larger than 256x256 on Jetson.
    
    Args:
        img_lq: Input tensor (1, C, H, W)
        model: SwinIR model
        args: Arguments
        window_size: Window size
        use_tile: Whether to use tile-based inference
        tile_size: Tile size (must be multiple of window_size)
        
    Returns:
        output: Output tensor
    """
    if not use_tile or args.tile is None:
        # Direct inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(img_lq)
        return output
    
    # Tile-based inference for large images
    b, c, h, w = img_lq.size()
    tile = min(args.tile if args.tile else tile_size, h, w)
    
    # Ensure tile is multiple of window_size
    tile = (tile // window_size) * window_size
    if tile < window_size:
        tile = window_size
    
    tile_overlap = min(args.tile_overlap, tile // 2)
    sf = args.scale
    stride = tile - tile_overlap
    
    h_idx_list = list(range(0, h - tile, stride)) + [max(0, h - tile)]
    w_idx_list = list(range(0, w - tile, stride)) + [max(0, w - tile)]
    
    # Allocate on CPU to save GPU memory, transfer results incrementally
    E = torch.zeros(b, c, h * sf, w * sf, device='cpu', dtype=torch.float32)
    W = torch.zeros_like(E)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch = model(in_patch)
                    
                    # Move to CPU immediately
                    out_patch_cpu = out_patch.cpu().float()
                    del out_patch
                    
                    out_patch_mask = torch.ones_like(out_patch_cpu)
                    
                    E[..., h_idx * sf:(h_idx + tile) * sf, 
                      w_idx * sf:(w_idx + tile) * sf].add_(out_patch_cpu)
                    W[..., h_idx * sf:(h_idx + tile) * sf, 
                      w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
                    
                    del out_patch_cpu, out_patch_mask
                    clear_memory()
    
    output = E.div_(W)
    del E, W
    
    return output.cuda()


def process_jetson(
    args,
    window_size,
    mech,
    device,
    num_landmarks=None,
    iters=None,
    sample_k=None,
    use_fp16=False,
    use_tile=False
):
    """Process images with Jetson-optimized pipeline.
    
    Key difference from original: Model is loaded ONCE, not per-image.
    """
    # Setup directories with absolute paths
    save_dir = get_absolute_path(f'results/swinir_x{args.scale}_jetson')
    folder = args.folder_lq
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect image paths
    all_paths = sorted(glob.glob(os.path.join(folder, "*")))
    print(f"[Jetson] Found {len(all_paths)} images in {folder}")
    
    if sample_k is not None:
        paths_to_process = sorted(random.sample(all_paths, min(sample_k, len(all_paths))))
    else:
        paths_to_process = all_paths
    
    if len(paths_to_process) == 0:
        print("[Jetson] No images to process!")
        return "no images"
    
    # CRITICAL: Load model ONCE outside the loop
    clear_memory()
    print_memory_status("Before model load:")
    
    # Use fixed 224 size (matching ONNX/TRT models)
    img_size = 224
    
    if mech == "original":
        model = define_model_jetson(args, mech, 16, 1, window_size, img_size, use_fp16)
    else:
        model = define_model_jetson(args, mech, num_landmarks, iters, window_size, img_size, use_fp16)
    
    print_memory_status("After model load:")
    
    # Process each image
    for idx, path in enumerate(paths_to_process):
        print(f"\n[Jetson] Processing image {idx + 1}/{len(paths_to_process)}: {os.path.basename(path)}")
        
        with JetsonInferenceContext():
            try:
                # Read image
                imgname = os.path.splitext(os.path.basename(path))[0]
                img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                
                if img_bgr is None:
                    print(f"[ERROR] Could not read: {path}")
                    continue
                
                h_old, w_old = img_bgr.shape[:2]
                
                # Preprocess - resize to 224x224
                img_lq = img_bgr.astype(np.float32) / 255.0
                img_lq = cv2.resize(img_lq, (224, 224), interpolation=cv2.INTER_CUBIC)
                
                # BGR -> RGB, HWC -> CHW
                img_lq = img_lq[:, :, [2, 1, 0]]
                img_lq = np.transpose(img_lq, (2, 0, 1))
                img_lq = np.ascontiguousarray(img_lq)
                
                # Convert to tensor
                dtype = torch.float16 if use_fp16 else torch.float32
                img_tensor = torch.from_numpy(img_lq).float().unsqueeze(0)
                img_tensor = img_tensor.to(dtype=dtype, device=device)
                
                # Free numpy arrays
                del img_bgr, img_lq
                gc.collect()
                
                print_memory_status("Before inference:")
                
                # Inference
                import time
                st = time.time()
                output = test_jetson(img_tensor, model, args, window_size, use_tile=use_tile)
                torch.cuda.synchronize()
                inference_time = time.time() - st
                
                print(f"[Jetson] Inference time: {inference_time:.4f}s")
                
                # Free input tensor
                del img_tensor
                clear_memory()
                
                # Postprocess
                output_np = output.float().squeeze().clamp(0, 1).cpu().numpy()
                del output
                clear_memory()
                
                if output_np.ndim == 3:
                    output_np = np.transpose(output_np, (1, 2, 0))
                    output_np = output_np[:, :, [2, 1, 0]]
                
                output_np = (output_np * 255.0).round().astype(np.uint8)
                
                # Resize to target
                final_output = cv2.resize(
                    output_np,
                    (w_old * args.scale, h_old * args.scale),
                    interpolation=cv2.INTER_CUBIC
                )
                
                # Save
                save_path = f"{save_dir}/{imgname}_SwinIR_Jetson.png"
                cv2.imwrite(save_path, final_output)
                print(f"[Jetson] Saved: {save_path}")
                
                del output_np, final_output
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "nvmap" in str(e).lower():
                    print(f"[ERROR] Memory error on {path}: {e}")
                    print("[Jetson] Attempting recovery...")
                    clear_memory()
                    continue
                else:
                    raise
    
    # Cleanup model
    del model
    clear_memory()
    print_memory_status("After cleanup:")
    
    return "success"


def main():
    parser = argparse.ArgumentParser(description="Jetson-optimized SwinIR inference")
    parser.add_argument('--scale', type=int, default=2, help='Scale factor: 1, 2, 3, 4')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model weights path')
    parser.add_argument('--folder_lq', type=str, default=None, 
                        help='Input image folder')
    parser.add_argument('--tile', type=int, default=None, 
                        help='Tile size for large images (128 recommended for Jetson)')
    parser.add_argument('--tile_overlap', type=int, default=16, 
                        help='Tile overlap (smaller = faster but may show seams)')
    parser.add_argument('--mech', type=str, default='pnp_nystra', 
                        choices=['original', 'pnp_nystra'])
    parser.add_argument('--sample_k', type=int, default=None, 
                        help='Process only K random images')
    parser.add_argument('--use_tile', action='store_true',
                        help='Enable tile-based inference for memory savings')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    if args.model_path is None:
        args.model_path = get_absolute_path("model_weights/x2.pth")
    elif not os.path.isabs(args.model_path):
        args.model_path = get_absolute_path(args.model_path)
    
    if args.folder_lq is None:
        args.folder_lq = get_absolute_path("input_images")
    elif not os.path.isabs(args.folder_lq):
        args.folder_lq = get_absolute_path(args.folder_lq)
    
    print(f"[Jetson] Model path: {args.model_path}")
    print(f"[Jetson] Input folder: {args.folder_lq}")
    
    device = get_device()
    print(f"[Jetson] Using device: {device}")
    
    use_fp16 = False
    window_size = 32
    
    if args.mech == "original":
        print(process_jetson(
            args=args,
            window_size=window_size,
            mech="original",
            device=device,
            num_landmarks=None,
            iters=None,
            sample_k=args.sample_k,
            use_fp16=use_fp16,
            use_tile=args.use_tile
        ))
    else:
        print(process_jetson(
            args=args,
            window_size=window_size,
            mech="pnp_nystra",
            device=device,
            num_landmarks=16,
            iters=2,
            sample_k=args.sample_k,
            use_fp16=use_fp16,
            use_tile=args.use_tile
        ))


if __name__ == '__main__':
    main()
