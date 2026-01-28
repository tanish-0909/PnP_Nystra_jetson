"""
Jetson Orin Nano Super - Optimized Benchmark Script
=====================================================
Benchmarks SwinIR across PyTorch, TensorRT, and ONNX runtimes
with memory-safe operations for Jetson's 8GB unified memory.

Key Features:
- Runs ONE runtime at a time (prevents OOM from multiple models)
- FP16 inference by default
- Memory monitoring between benchmarks
- Graceful error recovery
"""

import os
import sys
import cv2
import numpy as np
import torch
import time
import gc
import argparse

# Configure Jetson environment FIRST (before any CUDA ops)
from jetson_utils import (
    configure_jetson_environment,
    clear_memory,
    print_memory_status,
    get_device,
    JetsonInferenceContext
)

# Then import models
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.swinir import SwinIR as net


def benchmark_pytorch_jetson(input_image_dir, save_dir, iterations=10, scale=2, use_fp16=False):
    """Benchmark PyTorch model on Jetson.
    
    Args:
        input_image_dir: Directory with input images
        save_dir: Output directory
        iterations: Max number of images to process
        scale: Upscaling factor
        use_fp16: Use FP16 for inference
    """
    save_dir_full = f"{save_dir}_Pytorch_Jetson"
    os.makedirs(save_dir_full, exist_ok=True)
    
    clear_memory()
    print_memory_status("Start PyTorch benchmark:")
    
    # Load model ONCE
    model = net(
        mech="pnp_nystra",
        num_landmarks=16,
        iters=2,
        upscale=scale,
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
    
    model_path = os.path.join(parent_dir, "model_weights", "x2.pth")
    print(f"[PyTorch] Loading: {model_path}")
    
    # Load to CPU first
    pretrained_model = torch.load(model_path, map_location='cpu')
    param_key_g = 'params'
    state_dict = pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model
    
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if k in model_state_dict and v.size() == model_state_dict[k].size()}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    
    del pretrained_model, state_dict, filtered_state_dict, model_state_dict
    gc.collect()
    
    # Move to GPU
    model.eval()
    if use_fp16:
        model = model.half()
    model = model.cuda()
    
    print_memory_status("After model load:")
    
    timings = []
    image_files = [f for f in os.listdir(input_image_dir) 
                   if os.path.isfile(os.path.join(input_image_dir, f))]
    
    for i, image_file in enumerate(image_files[:iterations]):
        with JetsonInferenceContext():
            try:
                image_path = os.path.join(input_image_dir, image_file)
                img_name = os.path.splitext(image_file)[0]
                
                img_bgr = cv2.imread(image_path)
                if img_bgr is None:
                    continue
                
                h_old, w_old = img_bgr.shape[:2]
                
                # Preprocess
                img = img_bgr.astype(np.float32) / 255.0
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                img = img[:, :, [2, 1, 0]].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                
                dtype = torch.float16 if use_fp16 else torch.float32
                img_tensor = torch.from_numpy(img).unsqueeze(0).to(dtype=dtype, device='cuda')
                
                del img_bgr, img
                
                # Inference
                st = time.time()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_fp16, dtype=torch.float16):
                        outputs = model(img_tensor)
                torch.cuda.synchronize()
                inference_time = time.time() - st
                
                timings.append(inference_time)
                print(f"[PyTorch] {img_name}: {inference_time:.4f}s")
                
                # Postprocess and save
                output_np = outputs.float().squeeze().clamp(0, 1).cpu().numpy()
                output_np = np.transpose(output_np, (1, 2, 0))
                output_np = output_np[:, :, [2, 1, 0]]
                output_np = (output_np * 255.0).round().astype(np.uint8)
                
                final_output = cv2.resize(output_np, (w_old * scale, h_old * scale), 
                                         interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f"{save_dir_full}/{img_name}_Pytorch.png", final_output)
                
                del img_tensor, outputs, output_np, final_output
                
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                clear_memory()
    
    # Cleanup
    del model
    clear_memory()
    
    if timings:
        avg = sum(timings) / len(timings)
        print(f"\n[PyTorch] Average: {avg:.4f}s | FPS: {1/avg:.2f}")
    
    return timings


def benchmark_trt_jetson(input_image_dir, save_dir, iterations=10, scale=2, use_fp16=False):
    """Benchmark TensorRT model on Jetson."""
    from trt_jetson import TRTWrapperJetson
    
    save_dir_full = f"{save_dir}_TRT_Jetson"
    os.makedirs(save_dir_full, exist_ok=True)
    
    clear_memory()
    print_memory_status("Start TRT benchmark:")
    
    # Determine engine path
    engine_name = "swinir_pnp_fp16.engine" if use_fp16 else "swinir_pnp_fp32.engine"
    engine_path = os.path.join(parent_dir, engine_name)
    
    # Fallback to fp32 if fp16 not available
    if not os.path.exists(engine_path):
        engine_path = os.path.join(parent_dir, "swinir_pnp_fp32.engine")
        print(f"[TRT] FP16 engine not found, using FP32")
    
    if not os.path.exists(engine_path):
        print(f"[TRT] ERROR: No engine found at {engine_path}")
        return []
    
    print(f"[TRT] Loading: {engine_path}")
    trt_model = TRTWrapperJetson(engine_path, use_fp16=use_fp16)
    
    print_memory_status("After TRT load:")
    
    timings = []
    image_files = [f for f in os.listdir(input_image_dir) 
                   if os.path.isfile(os.path.join(input_image_dir, f))]
    
    for i, image_file in enumerate(image_files[:iterations]):
        with JetsonInferenceContext(clear_before=False):  # Don't clear model
            try:
                image_path = os.path.join(input_image_dir, image_file)
                img_name = os.path.splitext(image_file)[0]
                
                img_bgr = cv2.imread(image_path)
                if img_bgr is None:
                    continue
                
                h_old, w_old = img_bgr.shape[:2]
                
                # Preprocess
                img = img_bgr.astype(np.float32) / 255.0
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                img = img[:, :, [2, 1, 0]].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                
                dtype = torch.float16 if use_fp16 else torch.float32
                img_tensor = torch.from_numpy(img).unsqueeze(0).to(dtype=dtype, device='cuda')
                
                del img_bgr, img
                
                # Inference
                st = time.time()
                outputs = trt_model.infer(img_tensor)
                torch.cuda.synchronize()
                inference_time = time.time() - st
                
                timings.append(inference_time)
                print(f"[TRT] {img_name}: {inference_time:.4f}s")
                
                # Postprocess and save
                output_np = outputs[0].float().squeeze().clamp(0, 1).cpu().numpy()
                output_np = np.transpose(output_np, (1, 2, 0))
                output_np = output_np[:, :, [2, 1, 0]]
                output_np = (output_np * 255.0).round().astype(np.uint8)
                
                final_output = cv2.resize(output_np, (w_old * scale, h_old * scale),
                                         interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f"{save_dir_full}/{img_name}_TRT.png", final_output)
                
                del img_tensor, outputs, output_np, final_output
                
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                clear_memory()
    
    # Cleanup
    trt_model.cleanup()
    del trt_model
    clear_memory()
    
    if timings:
        avg = sum(timings) / len(timings)
        print(f"\n[TRT] Average: {avg:.4f}s | FPS: {1/avg:.2f}")
    
    return timings


def benchmark_onnx_jetson(input_image_dir, save_dir, iterations=10, scale=2):
    """Benchmark ONNX Runtime on Jetson.
    
    Note: ONNX Runtime on Jetson can use TensorRT EP or CUDA EP.
    """
    import onnxruntime as ort
    
    save_dir_full = f"{save_dir}_ONNX_Jetson"
    os.makedirs(save_dir_full, exist_ok=True)
    
    clear_memory()
    print_memory_status("Start ONNX benchmark:")
    
    model_path = os.path.join(parent_dir, "swinir_pnp2.onnx")
    if not os.path.exists(model_path):
        print(f"[ONNX] ERROR: No model found at {model_path}")
        return []
    
    print(f"[ONNX] Loading: {model_path}")
    
    # Prefer TensorRT EP on Jetson, fallback to CUDA
    providers = []
    
    # Check for TensorRT EP (best performance)
    if 'TensorrtExecutionProvider' in ort.get_available_providers():
        providers.append(('TensorrtExecutionProvider', {
            'trt_max_workspace_size': 256 * 1024 * 1024,  # 256MB
            'trt_fp16_enable': True
        }))
    
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    
    providers.append('CPUExecutionProvider')
    
    ort_session = ort.InferenceSession(model_path, providers=providers)
    print(f"[ONNX] Using providers: {ort_session.get_providers()}")
    
    inp_name = ort_session.get_inputs()[0].name
    out_name = ort_session.get_outputs()[0].name
    
    print_memory_status("After ONNX load:")
    
    timings = []
    image_files = [f for f in os.listdir(input_image_dir) 
                   if os.path.isfile(os.path.join(input_image_dir, f))]
    
    for i, image_file in enumerate(image_files[:iterations]):
        try:
            image_path = os.path.join(input_image_dir, image_file)
            img_name = os.path.splitext(image_file)[0]
            
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                continue
            
            h_old, w_old = img_bgr.shape[:2]
            
            # Preprocess
            img = img_bgr.astype(np.float32) / 255.0
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = img[:, :, [2, 1, 0]].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            
            inp_np = img[np.newaxis, ...]  # Add batch dim
            
            del img_bgr, img
            
            # Inference
            st = time.time()
            out_onnx = ort_session.run([out_name], {inp_name: inp_np})[0]
            inference_time = time.time() - st
            
            timings.append(inference_time)
            print(f"[ONNX] {img_name}: {inference_time:.4f}s")
            
            # Postprocess and save
            output_np = np.clip(out_onnx.squeeze(), 0, 1)
            output_np = np.transpose(output_np, (1, 2, 0))
            output_np = output_np[:, :, [2, 1, 0]]
            output_np = (output_np * 255.0).round().astype(np.uint8)
            
            final_output = cv2.resize(output_np, (w_old * scale, h_old * scale),
                                     interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{save_dir_full}/{img_name}_ONNX.png", final_output)
            
            del inp_np, out_onnx, output_np, final_output
            gc.collect()
            
        except Exception as e:
            print(f"[ERROR] {e}")
    
    # Cleanup
    del ort_session
    clear_memory()
    
    if timings:
        avg = sum(timings) / len(timings)
        print(f"\n[ONNX] Average: {avg:.4f}s | FPS: {1/avg:.2f}")
    
    return timings


def main():
    parser = argparse.ArgumentParser(description="Jetson SwinIR Benchmark")
    parser.add_argument('--input_dir', type=str, default=r"../input_images",
                        help='Input image directory')
    parser.add_argument('--output_dir', type=str, default=r"../results/swinIR",
                        help='Output directory base')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Max images to process per runtime')
    parser.add_argument('--scale', type=int, default=2, help='Upscale factor')
    parser.add_argument('--fp32', action='store_true', 
                        help='Use FP32 instead of FP16')
    parser.add_argument('--runtime', type=str, default='all',
                        choices=['pytorch', 'trt', 'onnx', 'all'],
                        help='Which runtime to benchmark')
    
    args = parser.parse_args()
    
    use_fp16 = False
    print(f"\n{'='*60}")
    print(f"Jetson SwinIR Benchmark - FP{'32' if args.fp32 else '16'}")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Run benchmarks ONE AT A TIME to prevent OOM
    if args.runtime in ['pytorch', 'all']:
        print("\n" + "="*40)
        print("BENCHMARKING: PyTorch")
        print("="*40)
        results['pytorch'] = benchmark_pytorch_jetson(
            args.input_dir, args.output_dir, 
            args.iterations, args.scale, use_fp16
        )
        
        # Force cleanup between runtimes
        clear_memory()
        time.sleep(1)  # Let system stabilize
    
    if args.runtime in ['trt', 'all']:
        print("\n" + "="*40)
        print("BENCHMARKING: TensorRT")
        print("="*40)
        results['trt'] = benchmark_trt_jetson(
            args.input_dir, args.output_dir,
            args.iterations, args.scale, use_fp16
        )
        
        clear_memory()
        time.sleep(1)
    
    if args.runtime in ['onnx', 'all']:
        print("\n" + "="*40)
        print("BENCHMARKING: ONNX Runtime")
        print("="*40)
        results['onnx'] = benchmark_onnx_jetson(
            args.input_dir, args.output_dir,
            args.iterations, args.scale
        )
        
        clear_memory()
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for runtime, timings in results.items():
        if timings:
            avg = sum(timings) / len(timings)
            print(f"{runtime.upper():10s}: Avg {avg:.4f}s | FPS {1/avg:.2f}")
        else:
            print(f"{runtime.upper():10s}: No results")
    
    print("="*60)


if __name__ == "__main__":
    main()
