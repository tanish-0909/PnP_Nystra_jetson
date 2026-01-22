import cv2
import numpy as np
import torch
import tensorrt as trt
import os
import time

class TRTWrapperTorch:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        # SAFETY CHECK: Ensure engine loaded
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}. Check file path or TRT version compatibility.")

        self.context = self.engine.create_execution_context()
        
        self.bindings = [None] * self.engine.num_io_tensors
        self.inputs = []
        self.outputs = []
        
        self._setup_io()

    def _setup_io(self):
        # Map TRT types to Torch types
        dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32:   torch.int32,
            trt.int8:    torch.int8
        }

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            trt_dtype = self.engine.get_tensor_dtype(name)
            
            # FIX 1: Dynamic Dtype Handling
            dtype = dtype_map.get(trt_dtype, torch.float32)

            # Handle dynamic shapes (-1) -> default to 1 for allocation
            size = tuple(abs(x) for x in shape)
            
            # Create tensor on GPU
            tensor = torch.zeros(size, dtype=dtype, device='cuda')
            
            self.bindings[i] = tensor.data_ptr()
            self.context.set_tensor_address(name, tensor.data_ptr())
            
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'tensor': tensor})
            else:
                self.outputs.append({'name': name, 'tensor': tensor})

    def infer(self, input_tensor):
        # Copy input
        # Note: Ensure input_tensor dtype matches engine input dtype
        if input_tensor.dtype != self.inputs[0]['tensor'].dtype:
            input_tensor = input_tensor.to(self.inputs[0]['tensor'].dtype)
            
        self.inputs[0]['tensor'].copy_(input_tensor)
        
        # Run
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        
        # Return list of tensors
        return [out['tensor'] for out in self.outputs]

def run_pipeline(image_path, engine_path, save_dir, scale=2):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Load Image ONCE
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read image {image_path}")
        return

    # FIX 2: Cache original shape immediately
    h_old, w_old, _ = img_bgr.shape

    # Preprocess
    img = img_bgr.astype(np.float32) / 255.0
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, [2, 1, 0]].transpose(2, 0, 1) # HWC -> CHW
    img = np.ascontiguousarray(img)
    
    img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()
    
    # 2. Inference
    print(f"Loading engine: {engine_path}")
    trt_model = TRTWrapperTorch(engine_path)
    
    st = time.time()
    outputs = trt_model.infer(img_tensor)
    torch.cuda.synchronize() # Wait for GPU
    print(f"Inference Time: {time.time() - st:.4f}s")
    
    # 3. Process Output
    # FIX 3: Safe clamping (not in-place)
    # We cast to float32 immediately to avoid FP16 issues in post-processing
    output_tensor = outputs[0].float().squeeze().clamp(0, 1)
    
    output_np = output_tensor.cpu().numpy()
    
    # Convert CHW -> HWC
    output_np = np.transpose(output_np, (1, 2, 0)) 
    # RGB -> BGR
    output_np = output_np[:, :, [2, 1, 0]]         
    
    output_np = (output_np * 255.0).round().astype(np.uint8)
    
    # 4. Final Resize (Using cached shape)
    final_output = cv2.resize(output_np, (w_old * scale, h_old * scale), interpolation=cv2.INTER_CUBIC)
    
    save_path = f"{save_dir}/{img_name}_TRT_Torch.png"
    cv2.imwrite(save_path, final_output)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Update paths here
    IMG_PATH = r"D:\TensorRT_ISR2.0\pythonProject\PnP_Nystra\input_images\Screenshot 2026-01-08 215234.png"
    ENGINE_PATH = r"D:\TensorRT_ISR2.0\pythonProject\PnP_Nystra\swinir_pnp_fp32.engine"
    SAVE_DIR = r"D:\TensorRT_ISR2.0\pythonProject\PnP_Nystra\results\swinir_x2"
    
    run_pipeline(IMG_PATH, ENGINE_PATH, SAVE_DIR, scale=2)