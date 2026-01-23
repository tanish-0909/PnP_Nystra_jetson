import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import sys
import torch
import random
import onnxruntime as ort
import time
random.seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory's path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.swinir import SwinIR as net
from utils import swinir_utils as util
from torchinfo import summary

def process(
    args,
    window_size,
    mech,
    device,
    num_landmarks=None,
    iters=None,
    sample_k=None,
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
        # 4.1) Read LQ + GT
        # imgname, img_lq, img_gt = get_image_pair(args, path)  # HWC‐BGR, float32
        (imgname, imgext) = os.path.splitext(os.path.basename(path))

        # 1) Load image and normalize to [0, 1]
        img_lq = cv2.imread(f'{path}', cv2.IMREAD_COLOR).astype(np.float32) / 255.
        h_old, w_old, _ = img_lq.shape
        # 2) RESIZE INPUT TO 224x224
        # We use INTER_CUBIC for better quality resizing
        img_lq = cv2.resize(img_lq, (224, 224), interpolation=cv2.INTER_CUBIC)

        # 3) Convert LQ to CHW‐RGB tensor
        if img_lq.ndim == 3 and img_lq.shape[2] == 3:
            # BGR → RGB
            img_lq = img_lq[:, :, [2, 1, 0]]
        
        img_lq = np.transpose(img_lq, (2, 0, 1))                # HWC → CHW
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # 1×C×H×W

        # NOTE: Padding logic removed because input is fixed to 224x224.
        # 224 is divisible by 8 (common window_size), so padding is generally not required.

        # 4) Build/Get model
        # Note: img_size is hardcoded to 224 now
        print(f"Building model for {imgname} with mech={mech}, window_size={window_size}, img_size=224")

        model_ort_sess = define_model(r"..\swinir_pnp2.onnx")
        inp_name = model_ort_sess.get_inputs()[0].name
        out_name = model_ort_sess.get_outputs()[0].name

        # 5) Inference
        st = time.time()
        inp_np = img_lq.cpu().numpy()
        out_onnx = model_ort_sess.run([out_name], {inp_name: inp_np})[0]
        print(f"ONNX Inference time: {time.time() - st:.4f} seconds")

        output_np = np.clip(out_onnx.squeeze(), 0, 1)
    
        if output_np.ndim == 3:
            # CHW‐RGB → HWC‐BGR
            # Transpose (C, H, W) -> (H, W, C)
            output_np = np.transpose(output_np, (1, 2, 0))
            # RGB -> BGR
            output_np = output_np[:, :, [2, 1, 0]]
        
        output_np = (output_np * 255.0).round().astype(np.uint8)

        # 4.8) Resize Output back to original aspect ratio (h_old * scale, w_old * scale)
        target_h = int(h_old * args.scale)
        target_w = int(w_old * args.scale)
        output_np = cv2.resize(output_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{save_dir}/{imgname}_SwinIR_Onnx_224.png", output_np)

    return "success"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--model_path', type=str,
                        default=r".\model_weights\x2.pth")
    parser.add_argument('--folder_lq', type=str, default=r".\input_images", help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=r".\output_images", help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--mech', type = str, default = 'pnp_nystra')
    parser.add_argument('--sample_k', type=int, default=None, help='If not None, randomly sample `sample_k` images from the folder')
    parser.add_argument('--device', type = str, default = 'cuda', help='Device to run the model on, cuda or cpu')

    args = parser.parse_args()

    device = args.device
    print("Using device: ", device)
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
                        sample_k= args.sample_k
                    ))


def define_model(model_path):
    ort_sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    print(ort_sess.get_providers())
    inp_name = ort_sess.get_inputs()[0].name
    out_name = ort_sess.get_outputs()[0].name

    inp_shape = ort_sess.get_inputs()[0].shape
    out_shape = ort_sess.get_outputs()[0].shape

    print(f"Input name:  {inp_name}")
    print(f"Input shape: {inp_shape}")
    print(f"Output name: {out_name}")
    print(f"Output shape:{out_shape}")


    return ort_sess


def setup(args, window_size):
    save_dir = f'results/swinir_x{args.scale}'
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


def test(img_lq, model, args, window_size):
    # print(model.img_size)
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
