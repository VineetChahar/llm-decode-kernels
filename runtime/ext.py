"""JIT-build and load the CUDA extension (sm_75 only, so the build stays ~1 min on Kaggle's 4 vCPUs)."""
import os
from pathlib import Path

_EXT = None
KDIR = Path(__file__).resolve().parents[1] / "kernels"


def load(verbose=False):
    global _EXT
    if _EXT is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
        from torch.utils.cpp_extension import load as _load
        srcs = [str(KDIR / f) for f in ("bindings.cpp", "gemv_w4a16.cu", "paged_attention.cu", "fused_norm.cu",
                                         "fused_swiglu.cu", "rope_kv.cu")]
        _EXT = _load(name="decode_kernels", sources=srcs, verbose=verbose,
                     extra_cflags=["-O3"],
                     extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_75,code=sm_75",
                                        "--ptxas-options=-v" if verbose else "-Xptxas=-O3"])
    return _EXT
