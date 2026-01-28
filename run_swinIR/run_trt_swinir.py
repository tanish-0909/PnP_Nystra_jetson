import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import sys
import torch
import random
import tensorrt as trt
from test_trt_model import run_pipeline
import time
random.seed(42)

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # PnP_Nystra/
sys.path.insert(0, PROJECT_ROOT)

from models.swinir import SwinIR as net
from utils import swinir_utils as util
from torchinfo import summary


def get_absolute_path(relative_path):
    """Convert relative path to absolute from project root."""
    relative_path = relative_path.replace('\\', os.sep).replace('/', os.sep)
    relative_path = relative_path.lstrip('.').lstrip(os.sep).lstrip('.').lstrip(os.sep)
    return os.path.join(PROJECT_ROOT, relative_path)

def process(
    args,
    window_size,
    mech,
    device,
    num_landmarks=None,
    iters=None,
    sample_k=None,
    ENGINE_PATH="../swinir_pnp2.engine"
):
    """
    Common routine for both 'original' and 'pnp_nystra' mechanisms.
    - mech: 'original' or 'pnp_nystra'
    - For mech == 'original', num_landmarks/iters are ignored (we hardcode 16,1).
    - For mech == 'pnp_nystra', you must pass (num_landmarks, iters).
    - sample_k: if not None, randomly sample `sample_k` image paths from the folder.
    """

    # 1) Setup folder, save_dir, border, and make sure the directory exists
    folder, save_dir, border, window_size = setup(args, window_size)
    os.makedirs(save_dir, exist_ok=True)


    # 3) Collect all image paths
    all_paths = sorted(glob.glob(os.path.join(folder, "*")))
    print(f"found {len(all_paths)} images in {folder}")

    if sample_k is not None:
        # randomly pick `sample_k` images (without replacement)
        sampled = random.sample(all_paths, sample_k)
        paths_to_process = sorted(sampled)
    else:
        paths_to_process = all_paths

    # 4) Iterate over each selected image
    for path in paths_to_process:
        run_pipeline(path, ENGINE_PATH, save_dir, scale=args.scale)
    return "success"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model weights path')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--mech', type=str, default='pnp_nystra')
    parser.add_argument('--sample_k', type=int, default=None, help='If not None, randomly sample `sample_k` images from the folder')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on, cuda or cpu')
    parser.add_argument('--engine_path', type=str, default=None, help='Path to the TensorRT engine file')

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
    
    if args.engine_path is None:
        args.engine_path = get_absolute_path("swinir_pnp_fp32.engine")
    elif not os.path.isabs(args.engine_path):
        args.engine_path = get_absolute_path(args.engine_path)

    device = args.device
    print(f"Using device: {device}")
    print(f"Engine path: {args.engine_path}")
    print(f"Input folder: {args.folder_lq}")
    if args.mech == "original":
        for window_size in [32]:
            print(process(
                args=args,
                window_size=window_size,
                mech="original",
                device=device,
                # num_landmarks & iters are ignored when mech=='original'
                num_landmarks=None,
                iters=None,
                sample_k=args.sample_k
            ))

    elif args.mech == "pnp_nystra":
        for window_size in [32]:
            for num_landmarks in [16]:
                for iters in [2]:
                    print(process(
                        args=args,
                        window_size=window_size,
                        mech="pnp_nystra",
                        device=device,
                        num_landmarks=num_landmarks,
                        iters=iters,
                        sample_k= args.sample_k,
                        ENGINE_PATH=args.engine_path
                    ))



def setup(args, window_size):
    save_dir = get_absolute_path(f'results/swinir_x{args.scale}')
    folder = args.folder_lq
    border = args.scale
    window_size = window_size

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

    return imgname, img_lq, img_gt


if __name__ == '__main__':
    main()
