"""
Convert PyTorch Model to ONNX
==============================
Exports the SwinIR PyTorch model to ONNX format for TensorRT conversion.

Usage:
    python pth_to_onnx.py [--model_path PATH] [--output PATH]
"""

import os
import sys
import torch
import argparse
import random

random.seed(42)

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # PnP_Nystra/

# Add parent to path for model imports
sys.path.insert(0, PROJECT_ROOT)

from models.swinir import SwinIR as net


def get_absolute_path(relative_path):
    """Convert relative path to absolute from project root."""
    # Handle both forward and back slashes
    relative_path = relative_path.replace('\\', os.sep).replace('/', os.sep)
    # Remove leading dots and separators
    relative_path = relative_path.lstrip('.').lstrip(os.sep).lstrip('.').lstrip(os.sep)
    return os.path.join(PROJECT_ROOT, relative_path)


def export_to_onnx(model_path, output_path, mech='pnp_nystra', num_landmarks=16, 
                   iters=2, upscale=2, img_size=224, window_size=32):
    """Export SwinIR model to ONNX format.
    
    Args:
        model_path: Path to input .pth file
        output_path: Path to output .onnx file
        mech: Mechanism type ('original' or 'pnp_nystra')
        num_landmarks: Number of landmarks for pnp_nystra
        iters: Iterations for pnp_nystra
        upscale: Upscaling factor
        img_size: Input image size
        window_size: Window size for SwinIR
    """
    print(f"[ONNX] Creating SwinIR model...")
    print(f"[ONNX]   mech={mech}, num_landmarks={num_landmarks}, iters={iters}")
    print(f"[ONNX]   upscale={upscale}, img_size={img_size}, window_size={window_size}")
    
    # Create model
    model = net(
        mech=mech,
        num_landmarks=num_landmarks,
        iters=iters,
        upscale=upscale,
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
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ONNX] Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Load weights
    print(f'[ONNX] Loading weights from: {model_path}')
    
    if not os.path.exists(model_path):
        print(f"[ONNX] ERROR: Model file not found: {model_path}")
        return False
    
    # Load to CPU first to avoid memory issues
    pretrained_model = torch.load(model_path, map_location='cpu')
    
    param_key_g = 'params'
    state_dict = pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model
    
    model_state_dict = model.state_dict()
    print(f'[ONNX] Model state_dict keys: {len(model_state_dict)}')
    
    # Filter state dict to match
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state_dict:
            if value.size() == model_state_dict[key].size():
                filtered_state_dict[key] = value
    
    print(f'[ONNX] Loaded {len(filtered_state_dict)} matching keys')
    
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    print('[ONNX] Model weights loaded successfully')
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Export to ONNX
    print(f'[ONNX] Exporting to: {output_path}')
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=16,
        do_constant_folding=True,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  # Fixed size for TensorRT
    )
    
    print(f"[ONNX] Export successful!")
    print(f"[ONNX] Output size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    # Check for external data file
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        print(f"[ONNX] External data file: {os.path.getsize(data_file) / (1024*1024):.1f} MB")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument('--model_path', type=str, default=None,
                        help='Input .pth model path (default: model_weights/x2.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX path (default: swinir_pnp2.onnx)')
    parser.add_argument('--mech', type=str, default='pnp_nystra',
                        choices=['original', 'pnp_nystra'])
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--scale', type=int, default=2,
                        help='Upscale factor (default: 2)')
    
    args = parser.parse_args()
    
    # Set default paths using absolute paths
    if args.model_path is None:
        args.model_path = get_absolute_path("model_weights/x2.pth")
    elif not os.path.isabs(args.model_path):
        args.model_path = get_absolute_path(args.model_path)
    
    if args.output is None:
        args.output = get_absolute_path("swinir_pnp2.onnx")
    elif not os.path.isabs(args.output):
        args.output = get_absolute_path(args.output)
    
    print(f"[ONNX] Model path: {args.model_path}")
    print(f"[ONNX] Output path: {args.output}")
    
    success = export_to_onnx(
        model_path=args.model_path,
        output_path=args.output,
        mech=args.mech,
        upscale=args.scale,
        img_size=args.img_size
    )
    
    sys.exit(0 if success else 1)
