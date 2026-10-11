#!/usr/bin/env python3
"""Kernel-level benchmarks -> RESULTS_DIR/results.json.

sections:
  bw        achievable DRAM read bandwidth (own microbenchmark, swept launch configs) + D2D copy
  autotune  scripts/autotune.py sweep (R x warps x prefetch, per shape, per group size)
  gemv      stage ladder naive -> vectorized -> reg-blocked -> autotuned vs cuBLAS fp16, every model shape
  groups    GEMV time per group size (64/128/256) at autotuned configs, summed per token
  attn      paged attention: ctx 512..16K x partition size; best-partition policy; HF-style SDPA baseline
  ncu       one attempt at hardware counters (expected to be blocked on Kaggle)
"""
import argparse, json, math, os, subprocess, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from runtime import ext
from bench.timing import time_fn, Results, env_info
from scripts.autotune import SHAPES, make, tune

ROOT = pathlib.Path(__file__).resolve().parents[1]
res = Results()
K_ = ext.load()
LAYERS = 22


def bench_bw():
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    out = {"read": [], "copy": []}
    for mb in (256, 1024):
        buf = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
        for blocks_per_sm in (1, 2, 4, 8, 16):
            for threads in (256, 512, 1024):
                for unroll in (1, 4):
                    if blocks_per_sm * threads > 2048: continue
                    t = time_fn(lambda: K_.bw_read(buf, sink, sms * blocks_per_sm, threads, unroll), iters=30, flush=False)
                    out["read"].append({"mb": mb, "blocks": sms * blocks_per_sm, "threads": threads, "unroll": unroll,
                                        "median_us": t["median_us"], "gbps": buf.numel() / t["median_us"] / 1e3,
                                        "sm_mhz": t["clocks_after"].get("sm_mhz"), "mem_mhz": t["clocks_after"].get("mem_mhz")})
        dst = torch.empty_like(buf)
        t = time_fn(lambda: dst.copy_(buf), iters=30, flush=False)
        out["copy"].append({"mb": mb, "median_us": t["median_us"], "gbps_rw": 2 * buf.numel() / t["median_us"] / 1e3})
        del buf, dst
    big = max(out["read"], key=lambda r: r["gbps"])
    out["size_sweep"] = []
    for kb in (256, 512, 1024, 2048, 2816, 4096, 6144, 8192, 12288, 16384, 32768, 65536, 262144):
        buf = torch.empty(kb << 10, dtype=torch.uint8, device="cuda")
        cands = []
        for blocks_per_sm in (2, 4, 8):
            t = time_fn(lambda: K_.bw_read(buf, sink, sms * blocks_per_sm, 256, 1), iters=40, flush=True)
            cands.append((t["median_us"], blocks_per_sm, t))
        us, bps, t = min(cands, key=lambda c: c[0])
        out["size_sweep"].append({"kb": kb, "median_us": us, "gbps": buf.numel() / us / 1e3, "blocks_per_sm": bps, **_t(t)})
        print(f"  read {kb:7d} KB: {us:8.1f} us  {buf.numel() / us / 1e3:6.1f} GB/s")
        del buf
    best = big
    out["achievable_read_gbps"] = best["gbps"]
    out["best_cfg"] = best
    out["spec_gbps"] = 320.0
    print(f"achievable read BW {best['gbps']:.1f} GB/s ({best['gbps'] / 320 * 100:.0f}% of 320 spec) cfg={best}")
    print(f"D2D copy (read+write) {out['copy'][-1]['gbps_rw']:.1f} GB/s")
    res.put("bandwidth", out)
    return out


def bench_gemv():
    best = json.load(open(ROOT / "configs" / "autotune.json"))["best"]
    rows = []
    for name, (N, K) in SHAPES.items():
        G = 128
        packed, sz = make(N, K, G)
        x = torch.randn(K, device="cuda").half(); y = torch.empty(N, dtype=torch.half, device="cuda")
        w16 = (torch.randn(N, K, device="cuda") * 0.02).half()
        q_bytes = packed.numel() * 4 + sz.numel() * 2 + K * 2 + N * 2
        f_bytes = N * K * 2 + K * 2 + N * 2
        R, W, pf = best[f"{N}x{K}_G{G}"]
        stages = [("0 naive", (0, 1, 4, 0)), ("1 vectorized", (1, 1, 4, 0)), ("2 reg-blocked", (2, 4, 4, 1)),
                  ("3 autotuned", (2, R, W, pf))]
        for label, cfg in stages:
            t = time_fn(lambda: K_.gemv_w4a16(*cfg, packed, sz, x, y, G))
            rows.append({"shape": name, "N": N, "K": K, "impl": label, "cfg": cfg, "bytes": q_bytes, **_t(t),
                         "gbps": q_bytes / t["median_us"] / 1e3})
        xo = x.view(1, -1); yo = y.view(1, -1)
        t = time_fn(lambda: torch.mm(xo, w16.t(), out=yo))
        rows.append({"shape": name, "N": N, "K": K, "impl": "cuBLAS fp16", "cfg": None, "bytes": f_bytes, **_t(t),
                     "gbps": f_bytes / t["median_us"] / 1e3})
        for r in rows[-5:]:
            print(f"{name:8s} {r['impl']:15s} {r['median_us']:8.1f}us  iqr {r['iqr_us']:5.1f}  {r['gbps']:6.1f} GB/s")
    res.put("gemv_stages", rows)


def _t(t):
    ca = t["clocks_after"]
    return {"median_us": t["median_us"], "iqr_us": t["iqr_us"], "p25_us": t["p25_us"], "p75_us": t["p75_us"],
            "sm_mhz": ca.get("sm_mhz"), "temp_c": ca.get("temp_c"), "power_w": ca.get("power_w"),
            "reasons": ca.get("reasons"), "throttled": t.get("throttled"), "attempt": t.get("attempt")}


def bench_groups():
    best = json.load(open(ROOT / "configs" / "autotune.json"))["best"]
    rows = []
    per_token = {"qkv": LAYERS, "o": LAYERS, "gate_up": LAYERS, "down": LAYERS, "lm_head": 1}
    for G in (64, 128, 256):
        tot, tot_bytes = 0.0, 0
        for name, (N, K) in SHAPES.items():
            packed, sz = make(N, K, G)
            x = torch.randn(K, device="cuda").half(); y = torch.empty(N, dtype=torch.half, device="cuda")
            R, W, pf = best[f"{N}x{K}_G{G}"]
            t = time_fn(lambda: K_.gemv_w4a16(2, R, W, pf, packed, sz, x, y, G))
            b = packed.numel() * 4 + sz.numel() * 2
            rows.append({"G": G, "shape": name, "weight_bytes": b, **_t(t)})
            tot += t["median_us"] * per_token[name]; tot_bytes += b * per_token[name]
        rows.append({"G": G, "shape": "per_token_sum", "median_us": tot, "weight_bytes": tot_bytes})
        print(f"G{G}: GEMV time per token (sum of kernels) {tot / 1e3:.3f} ms, {tot_bytes / 2**20:.1f} MiB")
    res.put("group_gemv", rows)


def bench_attn():
    n_q, n_kv, D, P = 32, 4, 64, 16
    rows, base = [], []
    for ctx in (512, 1024, 2048, 4096, 8192, 16384):
        n_pages = ctx // P + 2
        kc = torch.randn(n_pages, n_kv, P, D, device="cuda").half(); vc = torch.randn_like(kc)
        bt = torch.randperm(n_pages, device="cuda").int()
        q = torch.randn(n_q * D, device="cuda").half()
        pos = torch.tensor([ctx - 1], dtype=torch.int32, device="cuda")
        out = torch.empty(n_q, D, dtype=torch.half, device="cuda")
        kv_bytes = 2 * ctx * n_kv * D * 2
        for ver in (1, 2, 3):
            for PS in sorted({ctx, 8192, 4096, 2048, 1024, 512, 256, 128, 64}):
                if PS > ctx: continue
                ms = math.ceil(ctx / PS)
                o_part = torch.empty(n_q, ms, D, device="cuda"); m_part = torch.empty(n_q, ms, device="cuda"); l_part = torch.empty_like(m_part)
                t = time_fn(lambda: K_.paged_attention(q, kc, vc, bt, pos, PS, o_part, m_part, l_part, out, ver), iters=50)
                rows.append({"version": ver, "ctx": ctx, "part_size": PS, "splits": ms, "blocks": ms * n_kv, **_t(t),
                             "gbps": kv_bytes / t["median_us"] / 1e3})
        # HF-style baseline: contiguous KV, repeat_kv + SDPA (what LlamaAttention does at decode)
        k = torch.randn(1, n_kv, ctx, D, device="cuda").half(); v = torch.randn_like(k)
        qh = q.view(1, n_q, 1, D)
        def sdpa():
            kr = k.repeat_interleave(n_q // n_kv, 1); vr = v.repeat_interleave(n_q // n_kv, 1)
            return torch.nn.functional.scaled_dot_product_attention(qh, kr, vr)
        t = time_fn(sdpa, iters=50)
        base.append({"ctx": ctx, "impl": "HF repeat_kv+SDPA (contiguous)", **_t(t), "gbps": kv_bytes / t["median_us"] / 1e3})
        t = time_fn(lambda: torch.nn.functional.scaled_dot_product_attention(qh, k, v, enable_gqa=True), iters=50)
        base.append({"ctx": ctx, "impl": "SDPA enable_gqa (contiguous)", **_t(t), "gbps": kv_bytes / t["median_us"] / 1e3})
        for ver in (1, 2, 3):
            cr = [r for r in rows if r["ctx"] == ctx and r["version"] == ver]
            b = min(cr, key=lambda r: r["median_us"]); ns = [r for r in cr if r["splits"] == 1][0]
            print(f"v{ver} ctx {ctx:5d}: no-split {ns['median_us']:7.1f}us | best PS={b['part_size']:5d} ({b['splits']} splits) "
                  f"{b['median_us']:7.1f}us {b['gbps']:5.0f} GB/s | SDPA {base[-1]['median_us']:7.1f}us | repeat+SDPA {base[-2]['median_us']:7.1f}us")
        del kc, vc, k, v
    # the runtime uses one kernel version: lowest geometric-mean best-split time over contexts (a plain sum would be
    # decided by 16K alone). Throttled rows are excluded from every choice when an unthrottled row exists.
    ctxs = sorted({r["ctx"] for r in rows})
    def cands(v, c):
        cr = [r for r in rows if r["ctx"] == c and r["version"] == v]
        ok = [r for r in cr if not r.get("throttled")]
        return ok or cr
    best_t = lambda v, c: min(r["median_us"] for r in cands(v, c))
    vbest = min({r["version"] for r in rows}, key=lambda v: sum(math.log(best_t(v, c)) for c in ctxs))
    policy = []
    for ctx in ctxs:
        b = min(cands(vbest, ctx), key=lambda r: r["median_us"])
        policy.append({"max_ctx": ctx, "part_size": b["part_size"]})
    policy.append({"max_ctx": 1 << 30, "part_size": policy[-1]["part_size"]})
    print(f"attention policy: version {vbest}, {policy}")
    doc = {"version": vbest, "policy": policy}
    json.dump(doc, open(ROOT / "configs" / "attn_policy.json", "w"), indent=1)
    if os.environ.get("RESULTS_DIR"):
        json.dump(doc, open(os.path.join(os.environ["RESULTS_DIR"], "attn_policy.json"), "w"), indent=1)
    res.put("attention_sweep", rows); res.put("attention_baselines", base); res.put("attention_policy", policy)
    res.put("attention_version", vbest)


def clock_probe(seconds=60):
    """Sustained memory-bound load (back-to-back 256 MB reads) while sampling NVML once a second:
    characterises the T4's clock throttling. Also attempts to lock clocks (needs admin; recorded either way)."""
    import time
    from bench.timing import clocks
    lock = subprocess.run(["nvidia-smi", "-i", "0", "-lgc", "1590,1590"], capture_output=True, text=True)
    out = {"lock_attempt": (lock.stdout + lock.stderr).strip()[-400:], "samples": []}
    print("clock lock attempt:", out["lock_attempt"])
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    buf = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    t0 = time.time(); nxt = 0
    while time.time() - t0 < seconds:
        for _ in range(20):
            K_.bw_read(buf, sink, sms * 8, 256, 1)
        torch.cuda.synchronize()
        if time.time() - t0 >= nxt:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); K_.bw_read(buf, sink, sms * 8, 256, 1); e1.record(); torch.cuda.synchronize()
            c = clocks(); c["t_s"] = time.time() - t0; c["gbps"] = buf.numel() / (e0.elapsed_time(e1) * 1e3) / 1e3
            out["samples"].append(c); nxt += 2
            print(f"  t={c['t_s']:5.1f}s sm={c.get('sm_mhz')} mem={c.get('mem_mhz')} {c.get('temp_c')}C {c.get('power_w'):.1f}W "
                  f"{c['gbps']:.0f} GB/s {c.get('reasons')}", flush=True)
    time.sleep(30)
    out["after_30s_idle"] = clocks()
    res.put("clock_probe", out)


def try_ncu():
    probe = ROOT / "bench" / "ncu_probe.py"
    cmd = ["ncu", "--metrics", "dram__bytes_read.sum,gpu__time_duration.sum", "-k", "regex:gemv_rb", "-c", "1",
           sys.executable, str(probe)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        txt = (p.stdout + p.stderr)[-3000:]
    except Exception as e:
        txt = repr(e)
    blocked = "ERR_NVGPUCTRPERM" in txt or "permission" in txt.lower()
    print("ncu attempt:", "BLOCKED (ERR_NVGPUCTRPERM)" if blocked else "ran", "\n", txt[-1500:])
    res.put("ncu_attempt", {"cmd": " ".join(cmd), "blocked": blocked, "tail": txt[-1500:]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["clocks", "bw", "autotune", "gemv", "groups", "attn", "ncu"])
    a = ap.parse_args()
    res.put("env", env_info())
    for s in a.sections:
        print(f"\n===== {s} =====", flush=True)
        if s == "autotune":
            res.put("autotune", tune())
        else:
            {"bw": bench_bw, "gemv": bench_gemv, "groups": bench_groups, "attn": bench_attn, "ncu": try_ncu,
             "clocks": clock_probe}[s]()
