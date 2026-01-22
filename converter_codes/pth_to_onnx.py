import os
import torch
import random
random.seed(42)

import torch.onnx
from models.swinir import SwinIR as net

mech = 'pnp_nystra'
num_landmarks = 16
iters = 2
upscale = 2
img_size = 224
window_size = 32
model_path = r".\model_weights\x2.pth"

model = net(
    mech = mech, 
    num_landmarks= num_landmarks, 
    iters = iters, 
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

param_key_g = 'params'
print('Loading model from %s' % model_path)

pretrained_model = torch.load(model_path)
state_dict = pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model

model_state_dict = model.state_dict()
print('Model state_dict keys length:', len(model_state_dict))
filtered_state_dict = {}

for key, value in state_dict.items():
    if key in model_state_dict:
        if value.size() == model_state_dict[key].size():
            filtered_state_dict[key] = value

model_state_dict.update(filtered_state_dict)
model.load_state_dict(model_state_dict)
print('Model loaded.')

dummy_input = torch.randn(1, 3, img_size, img_size)

torch.onnx.export(
    model,
    dummy_input,
    "swinir_pnp2.onnx",
    opset_version=16,
    do_constant_folding=True,
    verbose=True
)
print("ONNX model exported successfully.")
