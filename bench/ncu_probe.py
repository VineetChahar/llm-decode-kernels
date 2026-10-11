"""Minimal workload for the single ncu attempt: one autotuned-style GEMV launch."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from runtime import ext
from scripts.autotune import make

K_ = ext.load()
packed, sz = make(11264, 2048, 128)
x = torch.randn(2048, device="cuda").half(); y = torch.empty(11264, dtype=torch.half, device="cuda")
K_.gemv_w4a16(2, 4, 4, 1, packed, sz, x, y, 128)
torch.cuda.synchronize()
print("probe done")
