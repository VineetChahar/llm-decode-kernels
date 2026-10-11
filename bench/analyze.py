"""Derived analysis over results.json: roofline points and the predicted-vs-measured time-per-token models.

Nothing here is fitted to the end-to-end numbers it is compared against. Inputs are independent
microbenchmarks:
  BW, t0     linear fit  time(S) = t0 + S / BW  of the read-bandwidth-vs-transfer-size sweep
             (t0 = fixed cost of one memory-bound kernel: launch-to-first-byte latency + tail)
  gap        GPU idle between consecutive kernels inside the CUDA graph (torch.profiler, ctx 512)
  n_kernels  kernels per decode step (profiler)

  Model A  t = bytes(ctx) / BW_peak                          pure roofline: "time = bytes moved / bandwidth"
  Model B  t = bytes(ctx) / BW + n_kernels * (t0 + gap)      + the per-kernel fixed costs
"""
import json, math

T4_FP16_TFLOPS = 65.0        # tensor-core fp16 (WMMA) peak
T4_HALF2_TFLOPS = 16.3       # non-tensor fp16 (__half2 FMA) peak: 2560 cores x 2 x 2 x 1.59 GHz
T4_FP32_TFLOPS = 8.1
L, H, I, V, NKV, D, NQ = 22, 2048, 5632, 32000, 4, 64, 32


def load(path):
    return json.load(open(path))


def bw_fit(r):
    """Least-squares t = t0 + bytes/BW over the size sweep."""
    pts = [(s["kb"] * 1024, s["median_us"]) for s in r["bandwidth"]["size_sweep"]]
    n = len(pts); sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    t0 = (sy - slope * sx) / n
    return {"t0_us": t0, "bw_gbps": 1e-3 / slope}  # slope in us/byte -> GB/s


def bytes_per_token(r, cfg, ctx):
    b = r["speed"]["bytes"][cfg]
    w = b["weights"]
    kv = b["kv_per_token"].get(str(ctx), b["kv_per_token"].get(ctx))
    act = 2 * (L * (H * 6 + I * 3 + (NQ + 2 * NKV) * D) + V)  # activation vectors in/out, fp16 (tiny)
    return w["body"] + w["lm_head"] + w["embedding_row"] + kv + act


def overhead_terms(r, cfg="rt_int4_g128"):
    p = r.get("profile", {})
    g = p.get(f"{cfg}|ctx512|graph") or p.get(f"rt_int4_g128|ctx512|graph")
    if not g or not g.get("kernels_per_token"):
        return None
    return {"n_kernels": g["kernels_per_token"], "gap_us": g["idle_us"] / g["kernels_per_token"]}


def predictions(r, cfg):
    bw_peak = r["bandwidth"]["achievable_read_gbps"]
    fit = bw_fit(r)
    ov = overhead_terms(r, cfg)
    rows = []
    for m in r["speed"]["runtime"].get(cfg, []):
        ctx = m["ctx"]
        by = bytes_per_token(r, cfg, ctx)
        a = by / bw_peak / 1e3
        b = by / fit["bw_gbps"] / 1e3 + (ov["n_kernels"] * (fit["t0_us"] + ov["gap_us"]) if ov else 0)
        rows.append({"ctx": ctx, "bytes": by, "measured_us": m["median_us"], "pred_A_us": a, "pred_B_us": b,
                     "explained_A": a / m["median_us"], "explained_B": b / m["median_us"],
                     "measured_tok_s": 1e6 / m["median_us"], "pred_A_tok_s": 1e6 / a, "pred_B_tok_s": 1e6 / b,
                     "sm_mhz": m["clocks_after"].get("sm_mhz"), "throttled": m.get("throttled")})
    return {"bw_peak_gbps": bw_peak, "fit": fit, "overhead": ov, "rows": rows}


def roofline_points(r):
    """(name, arithmetic intensity FLOP/byte, achieved GFLOP/s, achieved GB/s) per kernel at its measured time."""
    pts = []
    for g in r["gemv_stages"]:
        if g["impl"] not in ("3 autotuned", "cuBLAS fp16"):
            continue
        fl = 2 * g["N"] * g["K"]
        pts.append({"name": f"{g['shape']} {'W4A16' if 'auto' in g['impl'] else 'fp16 cuBLAS'}", "kind": "gemv",
                    "ai": fl / g["bytes"], "gflops": fl / g["median_us"] / 1e3, "gbps": g["bytes"] / g["median_us"] / 1e3})
    pol = {p["max_ctx"]: p["part_size"] for p in r.get("attention_policy", [])}
    for a in r["attention_sweep"]:
        if a.get("version", 2) != 2 or pol.get(a["ctx"]) != a["part_size"]:
            continue
        kvb = 2 * a["ctx"] * NKV * D * 2
        fl = 2 * 2 * NQ * a["ctx"] * D
        pts.append({"name": f"attn ctx{a['ctx']}", "kind": "attn", "ai": fl / kvb,
                    "gflops": fl / a["median_us"] / 1e3, "gbps": kvb / a["median_us"] / 1e3})
    return pts
