"""
Jetson Orin Nano Super - Optimized TensorRT Wrapper
=====================================================
Memory-efficient TensorRT inference for Jetson devices with limited memory.

Key Optimizations:
- FP32 inference (stable)
- Aggressive memory cleanup
- Stream synchronization
- Proper tensor lifecycle management
"""

import cv2
import numpy as np
import torch
import tensorrt as trt
import os
import time
import gc

# Import Jetson utilities (auto-configures environment)
from jetson_utils import (
    clear_memory, 
    print_memory_status, 
    preprocess_image_jetson,
    postprocess_output_jetson,
    JetsonInferenceContext,
    get_absolute_path
)


class TRTWrapperJetson:
    """TensorRT wrapper optimized for Jetson Orin Nano Super (8GB unified memory).
    
    Features:
    - Minimal memory footprint
    - Explicit resource management
    - FP16 support by default
    - Proper cleanup on destruction
    """
    
    def __init__(self, engine_path, use_fp16=False):
        """Initialize TensorRT engine.
        
        Args:
            engine_path: Path to the .engine file
            use_fp16: Force FP16 inference (recommended for Jetson)
        """
        clear_memory()
        print_memory_status("Before TRT load:")
        
        self.use_fp16 = use_fp16
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine
        print(f"[TRT] Loading engine: {engine_path}")
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        del engine_data  # Free immediately
        gc.collect()
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        # IO tensors
        self.input_info = None
        self.output_info = None
        self._setup_io()
        
        print_memory_status("After TRT load:")
    
    def _setup_io(self):
        """Setup input/output tensor info."""
        # Map TRT types to Torch types
        dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8
        }
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(abs(x) for x in self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = dtype_map.get(trt_dtype, torch.float32)
            
            info = {
                'name': name,
                'shape': shape,
                'dtype': torch_dtype,
                'tensor': None  # Allocated lazily
            }
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_info = info
            else:
                self.output_info = info
    
    def _allocate_tensors(self):
        """Allocate IO tensors just before inference."""
        # Allocate input tensor
        dtype = torch.float16 if self.use_fp16 else self.input_info['dtype']
        self.input_info['tensor'] = torch.zeros(
            self.input_info['shape'], 
            dtype=dtype, 
            device='cuda'
        )
        
        # Allocate output tensor
        self.output_info['tensor'] = torch.zeros(
            self.output_info['shape'],
            dtype=dtype,
            device='cuda'
        )
        
        # Set tensor addresses
        self.context.set_tensor_address(
            self.input_info['name'], 
            self.input_info['tensor'].data_ptr()
        )
        self.context.set_tensor_address(
            self.output_info['name'], 
            self.output_info['tensor'].data_ptr()
        )
    
    def infer(self, input_tensor):
        """Run inference on input tensor.
        
        Args:
            input_tensor: Input tensor (1, 3, H, W)
            
        Returns:
            list: Output tensors
        """
        # Allocate on first use
        if self.input_info['tensor'] is None:
            self._allocate_tensors()
        
        # Convert dtype if needed
        if self.use_fp16 and input_tensor.dtype != torch.float16:
            input_tensor = input_tensor.half()
        elif not self.use_fp16 and input_tensor.dtype != torch.float32:
            input_tensor = input_tensor.float()
        
        # Copy input
        self.input_info['tensor'].copy_(input_tensor)
        
        # Run inference
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        
        return [self.output_info['tensor'].clone()]
    
    def cleanup(self):
        """Explicitly cleanup resources."""
        if hasattr(self, 'input_info') and self.input_info and self.input_info['tensor'] is not None:
            del self.input_info['tensor']
            self.input_info['tensor'] = None
        
        if hasattr(self, 'output_info') and self.output_info and self.output_info['tensor'] is not None:
            del self.output_info['tensor']
            self.output_info['tensor'] = None
        
        clear_memory()
    
    def __del__(self):
        """Destructor for cleanup."""
        self.cleanup()


def run_pipeline_jetson(image_path, engine_path, save_dir, scale=2, use_fp16=False):
    """Run TensorRT inference pipeline optimized for Jetson.
    
    Args:
        image_path: Path to input image
        engine_path: Path to TensorRT engine
        save_dir: Output directory
        scale: Upscaling factor
        use_fp16: Use FP16 inference
    """
    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Clear memory before starting
    clear_memory()
    print_memory_status("Start of pipeline:")
    
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return
    
    # Preprocess
    img_tensor, original_size = preprocess_image_jetson(
        img_bgr, 
        target_size=224, 
        use_fp16=use_fp16
    )
    
    # Free original image
    del img_bgr
    gc.collect()
    
    # Load model and run inference
    trt_model = None
    try:
        trt_model = TRTWrapperJetson(engine_path, use_fp16=use_fp16)
        
        print_memory_status("Before inference:")
        
        st = time.time()
        outputs = trt_model.infer(img_tensor)
        inference_time = time.time() - st
        
        print(f"[TRT] Inference time: {inference_time:.4f}s")
        print_memory_status("After inference:")
        
        # Free input tensor
        del img_tensor
        clear_memory()
        
        # Postprocess
        final_output = postprocess_output_jetson(
            outputs[0], 
            original_size, 
            scale=scale
        )
        
        # Save
        save_path = f"{save_dir}/{img_name}_TRT_Jetson.png"
        cv2.imwrite(save_path, final_output)
        print(f"[TRT] Saved: {save_path}")
        
        # Cleanup output
        del outputs, final_output
        
    finally:
        # Ensure cleanup
        if trt_model is not None:
            trt_model.cleanup()
            del trt_model
        clear_memory()
        print_memory_status("End of pipeline:")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jetson-optimized TRT inference")
    parser.add_argument('--image', type=str, default=None, help='Input image path')
    parser.add_argument('--engine', type=str, default=None, help='TensorRT engine path')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--scale', type=int, default=2, help='Upscaling factor')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    if args.image is None:
        # Find first image in input_images
        input_dir = get_absolute_path("input_images")
        images = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        if images:
            args.image = os.path.join(input_dir, images[0])
        else:
            print("No images found in input_images/")
            import sys
            sys.exit(1)
    elif not os.path.isabs(args.image):
        args.image = get_absolute_path(args.image)
    
    if args.engine is None:
        args.engine = get_absolute_path("swinir_pnp_fp32.engine")
    elif not os.path.isabs(args.engine):
        args.engine = get_absolute_path(args.engine)
    
    if args.output is None:
        args.output = get_absolute_path("results/swinir_jetson")
    elif not os.path.isabs(args.output):
        args.output = get_absolute_path(args.output)
    
    print(f"Image: {args.image}")
    print(f"Engine: {args.engine}")
    print(f"Output: {args.output}")
    
    run_pipeline_jetson(
        image_path=args.image,
        engine_path=args.engine,
        save_dir=args.output,
        scale=args.scale,
        use_fp16=False  # FP32 mode
    )
