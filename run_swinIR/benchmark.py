from test_trt_model import TRTWrapperTorch
import os
import sys
import cv2
import numpy as np
import torch
import time
import onnxruntime as ort
import gc
from run_onnx_swinir import define_model

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory's path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.swinir import SwinIR as net


def benchmark(model_type="TRT", input_image_dir=r"..\input_images", save_dir=r"..\results\swinIR", iterations=100, scale=2):
    
    save_dir += f"_{model_type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize variables to ensure scope visibility
    trt_model = None
    onnx_session = None
    model = None
    inp_name = None
    out_name = None

    if model_type == "TRT":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory's path
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, r"swinir_pnp_fp32.engine")
        trt_model = TRTWrapperTorch(model_path)
        print("Loaded TRT model.")
    elif model_type == "ONNX":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory's path
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, r"swinir_pnp2.onnx")
        onnx_session = define_model(model_path)
        inp_name = onnx_session.get_inputs()[0].name
        out_name = onnx_session.get_outputs()[0].name
        print("Loaded ONNX model.")

    else:
        model = net(
            mech = "pnp_nystra",
            num_landmarks = 16,
            iters = 2,
            upscale = 2,
            in_chans = 3,
            img_size = 224,
            window_size = 32,
            img_range = 1.,
            depths = [6, 6, 6, 6, 6, 6],
            embed_dim = 180,
            num_heads = [6, 6, 6, 6, 6, 6],
            mlp_ratio = 2,
            upsampler = 'pixelshuffle',
            resi_connection = '1conv'
        )
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory's path
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, r"model_weights\x2.pth")
        
        # Load state dict
        pretrained_model = torch.load(model_path)
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
        model = model.cuda()
        model.eval() # Set to eval mode
        print("Loaded Pytorch model.")
    
    timings = []
    
    # Get list of images
    image_files = [f for f in os.listdir(input_image_dir) if os.path.isfile(os.path.join(input_image_dir, f))]
    
    # Process images
    for i, image_file in enumerate(image_files):
        # Optional: Limit iterations if needed
        if iterations and i >= iterations:
            break
            
        image_path = os.path.join(input_image_dir, image_file)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Error: Could not read image {image_path}")
            continue
        
        h_old, w_old, _ = img_bgr.shape

        img = img_bgr.astype(np.float32) / 255.0
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = img[:, :, [2, 1, 0]].transpose(2, 0, 1) # HWC -> CHW
        img = np.ascontiguousarray(img)
        
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        
        if model_type != "ONNX":
            img_tensor = img_tensor.cuda()
        
        inference_time = 0
        outputs = None
        
        if model_type == "TRT":
            st = time.time()
            outputs = trt_model.infer(img_tensor)
            torch.cuda.synchronize()
            inference_time = time.time() - st
            print(f"Inference Time: {inference_time:.4f}s")
            
        elif model_type == "Pytorch": # Or default
            st = time.time()
            with torch.no_grad():
                outputs = model(img_tensor)
            torch.cuda.synchronize()
            inference_time = time.time() - st
            print(f"Inference Time: {inference_time:.4f}s")
            
        elif model_type == "ONNX":
            inp_np = img_tensor.cpu().numpy()
            st = time.time()
            outputs = onnx_session.run([out_name], {inp_name: inp_np})
            inference_time = time.time() - st
            print(f"ONNX Inference time: {inference_time:.4f} seconds")
            # Unpack for post-processing
            outputs = [torch.from_numpy(outputs[0])]
            del inp_np

        timings.append(inference_time)

        output_tensor = outputs[0].float().squeeze().clamp(0, 1)
             
        output_np = output_tensor.cpu().numpy()

        output_np = np.transpose(output_np, (1, 2, 0))
        output_np = output_np[:, :, [2, 1, 0]] #RGB -> BGR
        output_np = (output_np * 255.0).round().astype(np.uint8)
        
        final_output = cv2.resize(output_np, (w_old * scale, h_old * scale), interpolation=cv2.INTER_CUBIC)

        save_path = f"{save_dir}/{img_name}_{model_type}.png"
        cv2.imwrite(save_path, final_output)
        print(f"Saved: {save_path}")

        # Cleanup specific variables from the loop
        del img_bgr, img, img_tensor, outputs, output_tensor, output_np, final_output
        if model_type != "ONNX":
            torch.cuda.empty_cache()
        gc.collect()

    if timings:
        avg_time = sum(timings) / len(timings)
        print(f"\nAverage Inference Time ({model_type}): {avg_time:.4f}s")
        print(f"FPS: {1/avg_time:.2f}")
    
    # Cleanup model variables
    del trt_model, onnx_session, model
    if model_type != "ONNX":
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    # Example usage
    # Ensure all necessary files are in the expected relative paths.
    
    print("------------------------------------------------")
    print("Benchmarking Pytorch...")
    benchmark(model_type="Pytorch")
    
    print("\n------------------------------------------------")
    print("Benchmarking TRT...")
    benchmark(model_type="TRT")
    
    print("\n------------------------------------------------")
    print("Benchmarking ONNX...")
    benchmark(model_type="ONNX")