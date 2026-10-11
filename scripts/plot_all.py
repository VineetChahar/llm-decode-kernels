#!/usr/bin/env python3
"""Regenerate every plot (and plots/tables.md) from one results.json.

usage: python scripts/plot_all.py runs/<stamp>/results.json [--llamacpp runs/<stamp>/llamacpp.json]
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bench import analyze as A

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "plots"
# validated categorical palette, fixed slot order (dataviz reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 6, "figure.dpi": 130})


def ctx_label(c):
    return f"{c // 1024}K" if c >= 1024 else str(c)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------ headline: predicted vs measured
def plot_pred_vs_measured(r, llama):
    pr = A.predictions(r, "rt_int4_g128")
    rows = pr["rows"]
    ctx = [x["ctx"] for x in rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax.plot(ctx, [x["pred_A_tok_s"] for x in rows], "--", color=C[0], label="Model A: bytes / peak BW")
    ax.plot(ctx, [x["pred_B_tok_s"] for x in rows], ":", color=C[0], label="Model B: + per-kernel fixed cost")
    ax.plot(ctx, [x["measured_tok_s"] for x in rows], "o-", color=C[0], label="measured W4A16 (CUDA graph)")
    if "rt_fp16" in r["speed"]["runtime"]:
        pf = A.predictions(r, "rt_fp16")["rows"]
        ax.plot(ctx, [x["pred_A_tok_s"] for x in pf], "--", color=C[1], label="fp16 Model A")
        ax.plot(ctx, [x["measured_tok_s"] for x in pf], "s-", color=C[1], label="measured fp16 cuBLAS (CUDA graph)")
    hf = [h for h in r["speed"].get("hf", []) if "tok_s" in h]
    if hf:
        ax.plot([h["ctx"] for h in hf], [h["tok_s"] for h in hf], "^-", color=C[2], label="HF transformers fp16")
    if llama:
        ll = sorted([x for x in llama if x.get("flash_attn") in (True, 1, "1", "on", "enabled")] or llama, key=lambda x: x["n_depth"])
        ax.plot([x["n_depth"] for x in ll], [x["avg_ts"] for x in ll], "D-", color=C[6], label="llama.cpp Q4_K_M (fa on)")
    ax.set_xscale("log", base=2); ax.set_xticks(ctx); ax.set_xticklabels([ctx_label(c) for c in ctx])
    ax.set_xlabel("context length (tokens)"); ax.set_ylabel("decode tokens / s (batch 1)")
    ax.set_title("Predicted vs measured decode throughput, TinyLlama-1.1B on T4")
    ax.set_ylim(0, None); ax.legend(fontsize=8, loc="upper right")
    ax2.plot(ctx, [100 * x["explained_A"] for x in rows], "o--", color=C[0], label="Model A")
    ax2.plot(ctx, [100 * x["explained_B"] for x in rows], "o:", color=C[2], label="Model B")
    ax2.axhline(100, color=INK2, lw=1)
    for x in rows:
        ax2.annotate(f"{100 * x['explained_A']:.0f}%", (x["ctx"], 100 * x["explained_A"]), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=INK2)
    ax2.set_xscale("log", base=2); ax2.set_xticks(ctx); ax2.set_xticklabels([ctx_label(c) for c in ctx])
    ax2.set_ylim(0, 120); ax2.set_xlabel("context length"); ax2.set_ylabel("predicted time / measured time (%)")
    ax2.set_title("Share of measured time explained (W4A16)"); ax2.legend(fontsize=8, loc="lower right")
    save(fig, "tokens_per_s_vs_context.png")
    return pr


# ------------------------------------------------------------------ roofline
def plot_roofline(r):
    bw = r["bandwidth"]["achievable_read_gbps"]
    pts = A.roofline_points(r)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ai = np.logspace(-1, 3, 200)
    for peak, lab, col in ((A.T4_HALF2_TFLOPS, "half2 FMA peak 16.3 TFLOP/s", INK2), (A.T4_FP16_TFLOPS, "tensor-core fp16 peak 65 TFLOP/s", GRID)):
        ax.plot(ai, np.minimum(ai * bw, peak * 1e3), color=col, lw=1.5)
        ax.annotate(lab, (300, peak * 1e3), textcoords="offset points", xytext=(0, 4), fontsize=8, color=INK2, ha="center")
    ax.annotate(f"measured BW {bw:.0f} GB/s", (0.2, 0.2 * bw), textcoords="offset points", xytext=(4, 8), rotation=33, fontsize=8, color=INK2)
    ridge = A.T4_HALF2_TFLOPS * 1e3 / bw
    ax.axvline(ridge, color=GRID, lw=1, ls=":")
    ax.annotate(f"ridge {ridge:.0f} FLOP/B", (ridge, 30), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2)
    groups = [("gemv", "W4A16", C[0], "o"), ("gemv", "fp16", C[1], "s"), ("attn", "attn", C[2], "^")]
    for kind, key, col, mk in groups:
        sel = [p for p in pts if p["kind"] == kind and key in p["name"] or (kind == "attn" and p["kind"] == "attn" and key == "attn")]
        sel = [p for p in sel if p["kind"] == kind]
        lab = {"W4A16": "W4A16 GEMV (autotuned)", "fp16": "fp16 GEMV (cuBLAS)", "attn": "paged attention v2"}[key]
        ax.scatter([p["ai"] for p in sel], [p["gflops"] for p in sel], color=col, marker=mk, s=40, label=lab,
                   edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.1, 1000); ax.set_ylim(10, 1e5)
    ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte)"); ax.set_ylabel("achieved GFLOP/s")
    ax.set_title("Roofline (T4): every decode kernel sits far left of the ridge")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "roofline.png")
    return pts


# ------------------------------------------------------------------ GEMV stages
def plot_gemv_stages(r):
    rows = r["gemv_stages"]
    shapes = list(dict.fromkeys(x["shape"] for x in rows))
    impls = ["0 naive", "1 vectorized", "2 reg-blocked", "3 autotuned", "cuBLAS fp16"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.16
    for i, im in enumerate(impls):
        vals = [next(x["median_us"] for x in rows if x["shape"] == s and x["impl"] == im) for s in shapes]
        xs = np.arange(len(shapes)) + (i - 2) * w
        ax.bar(xs, vals, w * 0.92, color=C[[6, 3, 2, 0, 1][i]], label=im)
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels([f"{s}\n{next(x['N'] for x in rows if x['shape'] == s)}x{next(x['K'] for x in rows if x['shape'] == s)}" for s in shapes])
    ax.set_yscale("log"); ax.set_ylabel("kernel time, µs (log)"); ax.set_title("GEMV stage ladder vs cuBLAS fp16 (G=128, L2 flushed)")
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    save(fig, "gemv_stages.png")


# ------------------------------------------------------------------ autotune heatmap
def plot_autotune(r):
    grid = r["autotune"]["grid"]
    shapes = list(dict.fromkeys(x["shape"] for x in grid))
    Rs, Ws = [1, 2, 4, 8], [1, 2, 4, 8]
    fig, axs = plt.subplots(1, len(shapes), figsize=(3.1 * len(shapes), 3.3))
    for ax, s in zip(axs, shapes):
        M = np.zeros((len(Rs), len(Ws)))
        for i, R in enumerate(Rs):
            for j, W in enumerate(Ws):
                M[i, j] = min(x["median_us"] for x in grid if x["shape"] == s and x["G"] == 128 and x["R"] == R and x["warps"] == W)
        rel = M / M.min()
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("seq", SEQ[::-1])
        ax.imshow(rel, cmap=cmap, vmin=1, vmax=max(2.0, rel.max()), origin="lower")
        for i in range(len(Rs)):
            for j in range(len(Ws)):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color=SURF if rel[i, j] < 1.25 else INK, fontweight="bold" if rel[i, j] == 1 else None)
        ax.set_xticks(range(len(Ws))); ax.set_xticklabels(Ws); ax.set_yticks(range(len(Rs))); ax.set_yticklabels(Rs)
        ax.set_xlabel("warps / block"); ax.set_title(s, fontsize=10); ax.grid(False)
    axs[0].set_ylabel("rows / warp (R)")
    fig.suptitle("Autotune: GEMV µs by rows-per-warp x warps-per-block (G=128, best of prefetch on/off; darker = faster)", fontsize=10)
    save(fig, "autotune_heatmap.png")


# ------------------------------------------------------------------ group size tradeoff
def plot_groups(r):
    ppl = r["perplexity"]["results"]
    G = [64, 128, 256]
    sp = r["speed"]["runtime"]
    ms = [next(x["median_us"] for x in sp[f"rt_int4_g{g}"] if x["ctx"] == 512) / 1e3 for g in G]
    pp = [ppl[f"rt_int4_g{g}"]["ppl"] for g in G]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.bar([str(g) for g in G], ms, 0.5, color=C[0])
    for i, v in enumerate(ms):
        a1.annotate(f"{v:.3f} ms", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    a1.set_xlabel("group size"); a1.set_ylabel("ms / token (ctx 512)"); a1.set_title("Speed")
    a2.bar([str(g) for g in G], pp, 0.5, color=C[1])
    a2.axhline(ppl["hf_fp16"]["ppl"], color=INK2, lw=1, ls="--")
    a2.annotate(f"HF fp16 {ppl['hf_fp16']['ppl']:.2f}", (2.3, ppl["hf_fp16"]["ppl"]), xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    for i, v in enumerate(pp):
        a2.annotate(f"{v:.2f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    lo = min(pp + [ppl["hf_fp16"]["ppl"]])
    a2.set_ylim(lo * 0.9, max(pp) * 1.05)
    a2.set_xlabel("group size"); a2.set_ylabel("WikiText-2 perplexity"); a2.set_title("Quality (y-axis does not start at 0)")
    fig.suptitle("Quantization group size: speed vs perplexity", fontsize=11)
    save(fig, "group_size_tradeoff.png")


# ------------------------------------------------------------------ fusion + launch ablation
def plot_fusion(r):
    sp = r["speed"]["runtime"]
    ctxs = [x["ctx"] for x in sp["rt_int4_g128"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.38
    fu = [x["median_us"] / 1e3 for x in sp["rt_int4_g128"]]
    un = [next(y["median_us"] for y in sp["rt_int4_g128_unfused"] if y["ctx"] == c) / 1e3 for c in ctxs]
    xs = np.arange(len(ctxs))
    a1.bar(xs - w / 2, un, w * 0.95, color=C[1], label="unfused (HF-style torch ops)")
    a1.bar(xs + w / 2, fu, w * 0.95, color=C[0], label="fused add+RMSNorm, SwiGLU")
    for i in range(len(ctxs)):
        a1.annotate(f"-{100 * (1 - fu[i] / un[i]):.1f}%", (xs[i] + w / 2, fu[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    a1.set_xticks(xs); a1.set_xticklabels([ctx_label(c) for c in ctxs]); a1.set_xlabel("context")
    a1.set_ylabel("ms / token (CUDA graph)"); a1.set_title("Fusion ablation (W4A16, same prompts/model)"); a1.legend(fontsize=8, loc="upper left")
    labels, vals, cols = [], [], []
    for name, lab in (("rt_int4_g128_unfused", "unfused"), ("rt_int4_g128", "fused")):
        for x in sp[name]:
            if x["ctx"] == 512 and "eager" in x:
                labels += [f"{lab}\neager (wall)", f"{lab}\nCUDA graph"]
                vals += [x["eager"]["wall_us"] / 1e3, x["median_us"] / 1e3]
                cols += [C[1], C[0]]
    a2.bar(range(len(vals)), vals, 0.6, color=cols)
    for i, v in enumerate(vals):
        a2.annotate(f"{v:.2f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    a2.set_xticks(range(len(vals))); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel("ms / token (ctx 512)"); a2.set_title("Launch overhead: eager Python vs CUDA graph")
    save(fig, "fusion_ablation.png")


# ------------------------------------------------------------------ attention sweep
def plot_attention(r):
    rows = r["attention_sweep"]
    ctxs = sorted({x["ctx"] for x in rows})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for i, c in enumerate(ctxs):
        sel = sorted([x for x in rows if x["ctx"] == c and x.get("version", 2) == 2], key=lambda x: x["part_size"])
        a1.plot([x["part_size"] for x in sel], [x["median_us"] for x in sel], "o-", color=SEQ[min(i + 1, len(SEQ) - 1)], label=ctx_label(c))
    a1.set_xscale("log", base=2); a1.set_yscale("log"); a1.set_xlabel("partition size (tokens per split); rightmost = no split")
    a1.set_ylabel("µs per layer"); a1.set_title("Split-K partition sweep (v2)"); a1.legend(title="context", fontsize=8, ncol=2)
    base = r.get("attention_baselines", [])
    series = [("v1 best split", lambda c: min(x["median_us"] for x in rows if x["ctx"] == c and x.get("version") == 1), C[3]),
              ("v2 best split", lambda c: min(x["median_us"] for x in rows if x["ctx"] == c and x.get("version", 2) == 2), C[0]),
              ("v2 no split", lambda c: next(x["median_us"] for x in rows if x["ctx"] == c and x.get("version", 2) == 2 and x["splits"] == 1), C[6]),
              ("HF repeat_kv+SDPA", lambda c: next(x["median_us"] for x in base if x["ctx"] == c and x["impl"].startswith("HF")), C[1])]
    for lab, f, col in series:
        a2.plot(ctxs, [f(c) for c in ctxs], "o-", color=col, label=lab)
    kv = [2 * c * 4 * 64 * 2 for c in ctxs]
    bw = r["bandwidth"]["achievable_read_gbps"]
    a2.plot(ctxs, [b / bw / 1e3 for b in kv], "--", color=INK2, lw=1, label="KV bytes / peak BW")
    a2.set_xscale("log", base=2); a2.set_yscale("log"); a2.set_xticks(ctxs); a2.set_xticklabels([ctx_label(c) for c in ctxs])
    a2.set_xlabel("context"); a2.set_ylabel("µs per layer"); a2.set_title("Paged decode attention vs context"); a2.legend(fontsize=8)
    save(fig, "attention_sweep.png")


# ------------------------------------------------------------------ bandwidth vs size + clock probe
def plot_bw_clock(r):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 3.8))
    ss = r["bandwidth"]["size_sweep"]
    a1.plot([s["kb"] / 1024 for s in ss], [s["gbps"] for s in ss], "o-", color=C[0])
    a1.axhline(r["bandwidth"]["achievable_read_gbps"], color=INK2, lw=1, ls="--")
    a1.axhline(320, color=GRID, lw=1)
    a1.annotate("spec 320 GB/s", (0.3, 320), xytext=(0, 3), textcoords="offset points", fontsize=8, color=INK2)
    a1.set_xscale("log", base=2); a1.set_xlabel("bytes read by one kernel (MiB)"); a1.set_ylabel("GB/s")
    a1.set_title("Achievable read BW vs transfer size")
    cp = r.get("clock_probe", {}).get("samples", [])
    if cp:
        a2.plot([c["t_s"] for c in cp], [c["sm_mhz"] for c in cp], "-", color=C[1])
        a2.set_xlabel("seconds of sustained load"); a2.set_ylabel("SM clock (MHz)"); a2.set_title("T4 clock under sustained memory-bound load")
        a3.plot([c["t_s"] for c in cp], [c["gbps"] for c in cp], "-", color=C[0])
        a3.set_xlabel("seconds of sustained load"); a3.set_ylabel("read GB/s (256 MB sweep)"); a3.set_title("Bandwidth during the same run")
        a3.set_ylim(0, None)
    save(fig, "bandwidth_and_clocks.png")


# ------------------------------------------------------------------ profile breakdown
def plot_profile(r):
    p = r.get("profile", {})
    keys = [k for k in ("rt_int4_g128|ctx512|graph", "rt_int4_g128|ctx16384|graph", "rt_fp16|ctx512|graph", "rt_int4_g128_unfused|ctx512|graph") if k in p]
    if not keys:
        return
    cats = sorted({c for k in keys for c in p[k]["categories_us"]}, key=lambda c: -max(p[k]["categories_us"].get(c, 0) for k in keys))
    cats = cats[:7]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, k in enumerate(keys):
        left = 0
        segs = [(c, p[k]["categories_us"].get(c, 0)) for c in cats]
        other = sum(v for c, v in p[k]["categories_us"].items() if c not in cats)
        segs += [("other kernels", other), ("GPU idle between kernels", p[k]["idle_us"])]
        for j, (c, v) in enumerate(segs):
            col = (C + ["#9a9893", "#d6d4ce"])[j] if j < len(cats) else ("#9a9893" if c == "other kernels" else "#d6d4ce")
            ax.barh(i, v / 1e3, left=left / 1e3, color=col, edgecolor=SURF, linewidth=1.5, label=c if i == 0 else None)
            left += v
        ax.annotate(f"{left / 1e3:.2f} ms, {p[k]['kernels_per_token']:.0f} kernels", (left / 1e3, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([k.replace("|", "  ") for k in keys], fontsize=8)
    ax.set_xlabel("ms per decode step"); ax.set_title("Where one decode step's time goes (torch.profiler, CUDA graph replay)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", visible=False)
    save(fig, "profile_breakdown.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="results.json files; later files override earlier keys")
    ap.add_argument("--llamacpp", default=None)
    ap.add_argument("--merged", default=str(ROOT / "results.json"), help="write the merged results here")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    r = {}
    for p in a.results:
        r.update(json.load(open(p)))
    llama = json.load(open(a.llamacpp)) if a.llamacpp else None
    if llama is not None:
        r["llamacpp"] = llama
    json.dump(r, open(a.merged, "w"), indent=1)
    plots = [("pred_vs_measured", lambda: plot_pred_vs_measured(r, llama)), ("roofline", lambda: plot_roofline(r)),
             ("gemv_stages", lambda: plot_gemv_stages(r)), ("autotune", lambda: plot_autotune(r)),
             ("groups", lambda: plot_groups(r)), ("fusion", lambda: plot_fusion(r)),
             ("attention", lambda: plot_attention(r)), ("bw_clock", lambda: plot_bw_clock(r)),
             ("profile", lambda: plot_profile(r))]
    for name, fn in plots:
        try:
            out = fn()
            if name == "pred_vs_measured":
                json.dump(out, open(OUT / "predictions.json", "w"), indent=1)
        except KeyError as e:
            print(f"skip {name}: missing {e}")


if __name__ == "__main__":
    main()
