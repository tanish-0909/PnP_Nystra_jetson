"""
Test ONNX Model
================
Verify ONNX model output matches PyTorch model output.
"""

import onnxruntime as ort
import numpy as np
import torch
import os
import sys

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # PnP_Nystra/
sys.path.insert(0, PROJECT_ROOT)


def get_absolute_path(relative_path):
    """Convert relative path to absolute from project root."""
    relative_path = relative_path.replace('\\', os.sep).replace('/', os.sep)
    relative_path = relative_path.lstrip('.').lstrip(os.sep).lstrip('.').lstrip(os.sep)
    return os.path.join(PROJECT_ROOT, relative_path)


def test_onnx_vs_pytorch():
    """Compare ONNX and PyTorch model outputs."""
    from models.swinir import SwinIR as net
    
    # Load ONNX model
    onnx_path = get_absolute_path("swinir_pnp2.onnx")
    print(f"Loading ONNX model: {onnx_path}")
    ort_sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"Using providers: {ort_sess.get_providers()}")
    
    # Create PyTorch model
    model = net(
        mech='pnp_nystra',
        num_landmarks=16,
        iters=2,
        upscale=2,
        in_chans=3,
        img_size=224,
        window_size=32,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    
    # Load weights
    model_path = get_absolute_path("model_weights/x2.pth")
    print(f"Loading PyTorch weights: {model_path}")
    
    pretrained_model = torch.load(model_path, map_location='cpu')
    param_key_g = 'params'
    state_dict = pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model
    
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if k in model_state_dict and v.size() == model_state_dict[k].size()}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    
    model = model.cuda()
    model.eval()
    
    # Create random input
    inp = torch.randn(1, 3, 224, 224)
    inp_np = inp.numpy()
    
    # ONNX inference
    out_onnx = ort_sess.run(None, {"input": inp_np})[0]
    
    # PyTorch inference
    with torch.no_grad():
        out_torch = model(inp.cuda()).cpu().numpy()
    
    # Compare
    diff = np.abs(out_onnx - out_torch)
    print(f"\nComparison Results:")
    print(f"  Mean difference: {diff.mean():.6f}")
    print(f"  Max difference:  {diff.max():.6f}")
    print(f"  Output shape:    {out_onnx.shape}")
    
    if diff.mean() < 1e-4:
        print("\n✓ ONNX model matches PyTorch model!")
    else:
        print("\n⚠ Warning: Significant difference between ONNX and PyTorch outputs")


if __name__ == "__main__":
    test_onnx_vs_pytorch()
