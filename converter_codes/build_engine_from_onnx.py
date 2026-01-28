"""
Build TensorRT Engine from ONNX
================================
Converts an ONNX model to TensorRT engine optimized for Jetson Orin Nano.

Usage:
    python build_engine_from_onnx.py [--fp16]
"""

import tensorrt as trt
import os
import sys
import argparse

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # PnP_Nystra/


def get_absolute_path(relative_path):
    """Convert relative path to absolute from project root."""
    return os.path.join(PROJECT_ROOT, relative_path)


def build_engine_python(onnx_file_path, engine_file_path, use_fp16=False, workspace_mb=256):
    """Build TensorRT engine from ONNX model.
    
    Args:
        onnx_file_path: Path to input ONNX model
        engine_file_path: Path to output TensorRT engine
        use_fp16: Enable FP16 precision (not recommended if having memory issues)
        workspace_mb: Workspace size in MB (256MB default for Jetson)
    """
    # 1. Setup Logger and Builder
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    
    # 2. Create Network (Explicit Batch is required for ONNX)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 3. Config
    config = builder.create_builder_config()
    
    # Memory pool limit (workspace_mb in bytes)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    
    # Enable FP16 if requested and supported
    if use_fp16:
        if builder.platform_has_fast_fp16:
            print("[TRT] Enabling FP16 Support...")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("[TRT] WARNING: FP16 not supported on this platform, using FP32")

    # 4. Parse ONNX
    print(f"[TRT] Parsing ONNX: {onnx_file_path}")
    if not os.path.exists(onnx_file_path):
        print(f"[TRT] ERROR: File {onnx_file_path} not found.")
        return False

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('[TRT] ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # 5. Build and Serialize
    print("[TRT] Building engine... (This might take several minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"[TRT] Success! Engine saved to: {engine_file_path}")
        print(f"[TRT] Engine size: {os.path.getsize(engine_file_path) / (1024*1024):.1f} MB")
        return True
    else:
        print("[TRT] Build failed.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument('--onnx', type=str, default=None,
                        help='Input ONNX model path (default: swinir_pnp2.onnx)')
    parser.add_argument('--engine', type=str, default=None,
                        help='Output engine path (default: swinir_pnp_fp32.engine)')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 precision')
    parser.add_argument('--workspace', type=int, default=256,
                        help='Workspace size in MB (default: 256 for Jetson)')
    
    args = parser.parse_args()
    
    # Set default paths using absolute paths
    if args.onnx is None:
        args.onnx = get_absolute_path("swinir_pnp2.onnx")
    elif not os.path.isabs(args.onnx):
        args.onnx = get_absolute_path(args.onnx)
    
    if args.engine is None:
        suffix = "_fp16" if args.fp16 else "_fp32"
        args.engine = get_absolute_path(f"swinir_pnp{suffix}.engine")
    elif not os.path.isabs(args.engine):
        args.engine = get_absolute_path(args.engine)
    
    print(f"[TRT] ONNX path: {args.onnx}")
    print(f"[TRT] Engine path: {args.engine}")
    print(f"[TRT] Precision: {'FP16' if args.fp16 else 'FP32'}")
    print(f"[TRT] Workspace: {args.workspace} MB")
    
    success = build_engine_python(
        args.onnx, 
        args.engine, 
        use_fp16=args.fp16,
        workspace_mb=args.workspace
    )
    
    sys.exit(0 if success else 1)