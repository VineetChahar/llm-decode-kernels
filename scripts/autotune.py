#!/usr/bin/env python3
"""GEMV autotuner: for each (N, K, group) shape, sweep rows-per-warp R x warps-per-block x prefetch on the
stage-2 kernel (tile = R*warps rows per block), time with L2 flush, persist the best config per shape.

Writes configs/autotune.json  {"best": {"NxK_G": [R, warps, pf]}, "grid": [...all measurements...]}
(also copied into RESULTS_DIR). The runtime loads "best" at startup.
"""
import json, os, sys, pathlib, shutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from runtime import ext
from runtime.quant import QuantLinear
from bench.timing import time_fn

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHAPES = {"qkv": (2560, 2048), "o": (2048, 2048), "gate_up": (11264, 2048), "down": (2048, 5632),
          "lm_head": (32000, 2048)}
RS, WARPS, PFS = (1, 2, 4, 8), (1, 2, 4, 8), (0, 1)


def make(N, K, G, seed=0):
    g = torch.Generator().manual_seed(seed)
    ql = QuantLinear((torch.randn(N, K, generator=g) * 0.02).half(), G, keep_ref=False)
    return ql.packed.cuda(), ql.sz.cuda()


def tune(groups=(64, 128, 256), shapes=SHAPES, iters=40):
    K_ = ext.load()
    best, grid = {}, []
    for name, (N, K) in shapes.items():
        x = torch.randn(K, device="cuda").half()
        y = torch.empty(N, dtype=torch.half, device="cuda")
        for G in groups:
            packed, sz = make(N, K, G)
            nbytes = packed.numel() * 4 + sz.numel() * 2 + K * 2 + N * 2
            cand = []
            for R in RS:
                for W in WARPS:
                    for pf in PFS:
                        t = time_fn(lambda: K_.gemv_w4a16(2, R, W, pf, packed, sz, x, y, G), iters=iters)
                        rec = {"shape": name, "N": N, "K": K, "G": G, "R": R, "warps": W, "pf": pf,
                               "median_us": t["median_us"], "iqr_us": t["iqr_us"],
                               "gbps": nbytes / t["median_us"] / 1e3, "sm_mhz": t["clocks_after"].get("sm_mhz")}
                        grid.append(rec); cand.append(rec)
            b = min(cand, key=lambda r: r["median_us"])
            best[f"{N}x{K}_G{G}"] = [b["R"], b["warps"], b["pf"]]
            print(f"{name:8s} G{G:3d} best R={b['R']} warps={b['warps']} pf={b['pf']} "
                  f"{b['median_us']:.1f}us {b['gbps']:.0f} GB/s", flush=True)
    out = {"best": best, "grid": grid}
    (ROOT / "configs").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "configs" / "autotune.json", "w"), indent=1)
    if os.environ.get("RESULTS_DIR"):
        shutil.copy(ROOT / "configs" / "autotune.json", os.path.join(os.environ["RESULTS_DIR"], "autotune.json"))
    return out


if __name__ == "__main__":
    tune()
