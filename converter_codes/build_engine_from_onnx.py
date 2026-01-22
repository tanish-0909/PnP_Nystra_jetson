import tensorrt as trt
import os

def build_engine_python(onnx_file_path, engine_file_path):
    # 1. Setup Logger and Builder
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    
    # 2. Create Network (Explicit Batch is required for ONNX)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 3. Config
    config = builder.create_builder_config()
    # Memory pool limit (2GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    
    # Enable FP16 if supported
    # if builder.platform_has_fast_fp16:
    #     print("Enabling FP16 Support...")
    #     config.set_flag(trt.BuilderFlag.FP16)

    # 4. Parse ONNX
    print(f"Parsing ONNX: {onnx_file_path}")
    if not os.path.exists(onnx_file_path):
        print(f"Error: File {onnx_file_path} not found.")
        return

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 5. Build and Serialize
    print("Building engine... (This might take a minute)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Success! Compatible engine saved to: {engine_file_path}")
    else:
        print("Build failed.")

if __name__ == "__main__":
    # UPDATE THESE PATHS
    ONNX_PATH = "swinir_pnp2.onnx"   # Your input ONNX
    ENGINE_PATH = "swinir_pnp_fp32.engine" # Your output Engine
    
    build_engine_python(ONNX_PATH, ENGINE_PATH)