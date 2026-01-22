import onnxruntime as ort
import numpy as np
import torch

ort_sess = ort.InferenceSession("swinir_pnp.onnx")

inp = torch.randn(1, 3, 224, 224).numpy()
out_onnx = ort_sess.run(None, {"input": inp})[0]

out_torch = model(torch.tensor(inp).cuda()).cpu().detach().numpy()

print(np.abs(out_onnx - out_torch).mean())
