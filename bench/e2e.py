#!/usr/bin/env python3
"""End-to-end accuracy + speed on TinyLlama-1.1B -> RESULTS_DIR/results.json.

sections (in order; accuracy always runs before speed):
  ppl        WikiText-2 test perplexity: HF fp16, runtime fp16 (cuBLAS), runtime INT4 at G=64/128/256,
             INT4 G128 + INT4 LM head. Same 16 windows x 512 tokens for every config.
  agree      greedy decode vs HF fp16 on the fixed prompt set (64 new tokens): free-running exact-match,
             first-divergence position, teacher-forced top-1 agreement on HF's continuation.
  speed      tokens/s vs context (512..16K) for every runtime config + HF fp16; group-size speed;
             fusion ablation (fused vs unfused) and launch ablation (CUDA graph vs eager)
  profile    torch.profiler per-kernel-category breakdown of one decode step (eager and graph replay),
             kernel count, GPU idle (inter-kernel gap) inside the graph
  quant      better quantizers in the same kernel format (G128): RTN, RTN + MSE range clipping, GPTQ (calibrated on
             WikiText-2 train), GPTQ + clipping, GPTQ + GPTQ INT4 LM head -> perplexity, agreement, speed check
"""
import argparse, gc, json, math, os, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.model import Model, DecodeConfig
from bench.timing import Results, env_info, clocks, cool
import bench.timing as T
from bench.prompts import PROMPTS

MID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CTXS = [512, 1024, 2048, 4096, 8192, 16384]
MAX_CTX = 16384 + 128
res = Results()
dev = "cuda"

print("loading HF model...", flush=True)
tok = AutoTokenizer.from_pretrained(MID)
hf = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).to(dev).eval()
SD = {k: v for k, v in hf.state_dict().items()}

CONFIGS = {
    "rt_int4_g128":          dict(weights="int4", group=128, lm_head="fp16"),
    "rt_int4_g64":           dict(weights="int4", group=64, lm_head="fp16"),
    "rt_int4_g256":          dict(weights="int4", group=256, lm_head="fp16"),
    "rt_int4_g128_int4head": dict(weights="int4", group=128, lm_head="int4"),
    "rt_int4_g128_unfused":  dict(weights="int4", group=128, lm_head="fp16", fused=False),
    "rt_fp16":               dict(weights="fp16"),
    "rt_fp16_unfused":       dict(weights="fp16", fused=False),
    "rt_int4_stage1":        dict(weights="int4", group=128, gemv_stage=(1, 1, 4, 0), lm_head="fp16"),
    "rt_int4_stage2":        dict(weights="int4", group=128, gemv_stage=(2, 4, 4, 1), lm_head="fp16"),
    "rt_int4_stage0":        dict(weights="int4", group=128, gemv_stage=(0, 1, 4, 0), lm_head="fp16"),
    "rt_int4_g128_clip":     dict(weights="int4", group=128, lm_head="fp16", quant="clip"),
    "rt_int4_g128_gptq":     dict(weights="int4", group=128, lm_head="fp16", quant="pre", qp="gptq"),
    "rt_int4_g128_gptq_clip": dict(weights="int4", group=128, lm_head="fp16", quant="pre", qp="gptq_clip"),
    "rt_int4_g128_gptq_int4head": dict(weights="int4", group=128, lm_head="int4", quant="pre", qp="gptq"),
    "rt_int4_g128_gptq_clip_int4head": dict(weights="int4", group=128, lm_head="int4", quant="pre", qp="gptq_clip"),
}
QP, QP_INFO = {}, {}


def qparams(kind):
    """GPTQ weights, calibrated once per kind and cached for the process (CPU tensors, ~0.6 GB)."""
    if kind not in QP:
        from runtime.gptq import calibrate
        t0 = time.time()
        QP[kind] = calibrate(hf, tok, group=128, clip=kind.endswith("clip"), log=lambda m: print(m, flush=True))
        QP_INFO[kind] = {"calib_s": time.time() - t0, "calib_windows": 128, "calib_len": 512,
                         "calib_data": "WikiText-2 train (seed 0)", "clip": kind.endswith("clip")}
    return QP[kind]


def build(name, max_ctx=MAX_CTX):
    gc.collect(); torch.cuda.empty_cache()
    cfg = dict(CONFIGS[name])
    kind = cfg.pop("qp", None)
    if kind:
        cfg["qparams"] = qparams(kind)
    return Model(SD, hf.config, DecodeConfig(max_ctx=max_ctx, **cfg))


# ------------------------------------------------------------------ accuracy
def wikitext_windows(n=16, L=512):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")  # bare "wikitext" id rejected by new hub
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    starts = torch.linspace(0, len(ids) - L, n).long().tolist()
    bos = torch.tensor([tok.bos_token_id])
    return [torch.cat([bos, ids[s:s + L - 1]]) for s in starts], len(ids)


def nll_from_logits(logits, w):
    lp = torch.log_softmax(logits.float(), -1)
    return -lp[:-1].gather(1, w[1:, None].to(lp.device)).sum().item(), len(w) - 1


def ppl_of(name, wins):
    m = build(name, max_ctx=640); m.prepare(640)
    tot, n = 0.0, 0
    for w in wins:
        m.reset(0)
        a, b = nll_from_logits(m.teacher_force(w, collect_logits=True), w); tot += a; n += b
    r = {"ppl": math.exp(tot / n), "tokens": n, "weight_bytes": m.weight_bytes()}
    print(f"ppl {name} = {r['ppl']:.4f}", flush=True)
    del m
    return r


def bench_ppl():
    wins, ntok = wikitext_windows()
    out = {"windows": len(wins), "window_len": len(wins[0]), "dataset_tokens": ntok, "results": {}}
    tot, n = 0.0, 0
    with torch.no_grad():
        for w in wins:
            a, b = nll_from_logits(hf(w[None].to(dev)).logits[0], w); tot += a; n += b
    out["results"]["hf_fp16"] = {"ppl": math.exp(tot / n), "tokens": n}
    print(f"ppl hf_fp16 = {math.exp(tot / n):.4f}", flush=True)
    for name in ("rt_fp16", "rt_int4_g64", "rt_int4_g128", "rt_int4_g256", "rt_int4_g128_int4head"):
        out["results"][name] = ppl_of(name, wins)
    base = out["results"]["hf_fp16"]["ppl"]
    for k, v in out["results"].items():
        v["delta_vs_hf"] = v["ppl"] - base
        v["rel_delta_pct"] = 100 * (v["ppl"] / base - 1)
    res.put("perplexity", out)


@torch.no_grad()
def hf_greedy(ids, n_new):
    """Pure argmax decode with HF's KV cache (no logits processors, no EOS stop: same semantics as the runtime)."""
    o = hf(ids[None].to(dev), use_cache=True, logits_to_keep=1); pkv = o.past_key_values
    nxt = o.logits[:, -1:].argmax(-1); out = [int(nxt)]
    for _ in range(n_new - 1):
        o = hf(nxt, past_key_values=pkv, use_cache=True); pkv = o.past_key_values
        nxt = o.logits[:, -1:].argmax(-1); out.append(int(nxt))
    return torch.tensor(out)


def bench_agree(n_new=64, names=("rt_fp16", "rt_int4_g128", "rt_int4_g64", "rt_int4_g256", "rt_int4_g128_int4head"),
                key="agreement"):
    enc = [tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
    ref = [hf_greedy(ids, n_new) for ids in enc]
    out = {"n_prompts": len(PROMPTS), "n_new": n_new, "results": {}}
    for name in names:
        m = build(name, max_ctx=256); m.prepare(256)
        exact, first_div, tf_agree, tf_n, samples = 0, [], 0, 0, []
        for i, ids in enumerate(enc):
            gen = torch.tensor(m.generate(ids, n_new))
            r = ref[i]
            eq = (gen == r)
            fd = int((~eq).nonzero()[0]) if (~eq).any() else n_new
            first_div.append(fd); exact += int(fd == n_new)
            # teacher-forced: feed prompt + HF continuation, compare our argmax with HF's next token
            m.reset(0)
            lg = m.teacher_force(torch.cat([ids, r[:-1]]), collect_logits=True)
            pred = lg[len(ids) - 1:].argmax(-1).cpu()
            tf_agree += int((pred == r).sum()); tf_n += len(r)
            if i < 4:
                samples.append({"prompt": PROMPTS[i], "hf": tok.decode(r), "ours": tok.decode(gen)})
        out["results"][name] = {"exact_match_rate": exact / len(enc), "mean_first_divergence": sum(first_div) / len(first_div),
                                "first_divergence": first_div, "teacher_forced_top1_agreement": tf_agree / tf_n,
                                "samples": samples}
        print(f"agree {name}: exact {exact}/{len(enc)}, mean first-div {sum(first_div) / len(first_div):.1f}, "
              f"teacher-forced top1 {tf_agree / tf_n:.4f}", flush=True)
        del m
    res.put(key, out)


# ------------------------------------------------------------------ speed
def time_decode(m, ctx, n_tok=64, warmup=5, use_graph=True):
    """Clock-gated (see bench/timing.py): retried after a cool-down if the SM clock fell below MIN_SM_MHZ."""
    from bench.timing import MIN_SM_MHZ, COOLDOWN_S, RETRIES
    best = None
    for attempt in range(RETRIES):
        r = _time_decode_once(m, ctx, n_tok, warmup, use_graph); r["attempt"] = attempt
        if best is None or r["clocks_after"].get("sm_mhz", 0) > best["clocks_after"].get("sm_mhz", 0):
            best = r
        if (r["clocks_after"].get("sm_mhz", 0) >= MIN_SM_MHZ or attempt == RETRIES - 1
                or T._cooled >= T.COOLDOWN_BUDGET_S):
            break
        T._cooled += COOLDOWN_S
        time.sleep(COOLDOWN_S)
    best["throttled"] = best["clocks_after"].get("sm_mhz", 0) < MIN_SM_MHZ
    return best


def _time_decode_once(m, ctx, n_tok, warmup, use_graph):
    """Per-token CUDA-event time of the decode step starting at context ctx (cache pre-filled with random values)."""
    m.reset(ctx)
    for _ in range(warmup):
        m.run_step(use_graph)
    torch.cuda.synchronize()
    m.reset(ctx)
    cb = clocks()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_tok)]
    for s, e in ev:
        s.record(); m.run_step(use_graph); e.record()
    torch.cuda._sleep(20_000_000)  # keep the GPU busy so NVML reports the loaded clock, not the idle one
    ca = clocks()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000 for s, e in ev)
    # wall clock too (includes CPU-side launch cost for eager runs)
    m.reset(ctx); torch.cuda.synchronize(); w0 = time.perf_counter()
    for _ in range(n_tok):
        m.run_step(use_graph)
    torch.cuda.synchronize(); wall = (time.perf_counter() - w0) / n_tok * 1e6
    q1, q3 = t[len(t) // 4], t[(3 * len(t)) // 4]
    med = t[len(t) // 2]
    return {"ctx": ctx, "median_us": med, "iqr_us": q3 - q1, "p25_us": q1, "p75_us": q3, "wall_us": wall,
            "tok_s": 1e6 / med, "tok_s_wall": 1e6 / wall, "clocks_before": cb, "clocks_after": ca}


def bench_speed():
    out = {"runtime": {}, "hf": [], "bytes": {}}
    plan = {  # config -> contexts
        "rt_int4_g128": CTXS, "rt_fp16": CTXS, "rt_int4_g128_unfused": CTXS, "rt_int4_g128_int4head": CTXS,
        "rt_int4_g64": CTXS, "rt_int4_g256": CTXS, "rt_fp16_unfused": [512, 4096],
        "rt_int4_stage0": [512], "rt_int4_stage1": [512], "rt_int4_stage2": [512],
    }
    for name, ctxs in plan.items():
        m = build(name)
        m.kv.normal_()  # random KV: timing is data-independent; quality past 2K is not meaningful for TinyLlama
        rows = []
        for ctx in ctxs:
            m.prepare(ctx + 80)
            waited = cool()
            r = time_decode(m, ctx); r["part_size"] = m.part_size; r["max_splits"] = m.max_splits
            r["cooled_s"] = waited; r["attn_version"] = m.attn_version
            rows.append(r)
            line = (f"{name:24s} ctx {ctx:5d}: graph {r['median_us'] / 1e3:6.3f} ms ({r['tok_s']:6.1f} tok/s, iqr {r['iqr_us']:5.1f}us)"
                    f" sm={r['clocks_after'].get('sm_mhz')}MHz{' THROTTLED' if r['throttled'] else ''}")
            if name in ("rt_int4_g128", "rt_int4_g128_unfused", "rt_fp16") and ctx in (512, 4096, 16384):
                e = time_decode(m, ctx, use_graph=False); e["eager"] = True
                r["eager"] = e
                line += f" | eager {e['median_us'] / 1e3:6.3f} ms gpu-timed, {e['wall_us'] / 1e3:6.3f} ms wall"
            print(line, flush=True)
        out["runtime"][name] = rows
        out["bytes"][name] = {"weights": m.weight_bytes(), "kv_per_token": {c: m.kv_bytes_per_token(c) for c in CTXS}}
        del m
        res.put("speed", out)
    # HF fp16 baseline: prefill random ids of length ctx, then 32 single-token forwards with the KV cache
    for ctx in CTXS:
        try:
            ids = torch.randint(100, 30000, (1, ctx), device=dev)
            with torch.no_grad():
                o = hf(ids, use_cache=True, logits_to_keep=1); pkv = o.past_key_values
                nxt = o.logits[:, -1:].argmax(-1)
                for _ in range(3):
                    o = hf(nxt, past_key_values=pkv, use_cache=True); pkv = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize(); t0 = time.perf_counter(); n = 32
                for _ in range(n):
                    o = hf(nxt, past_key_values=pkv, use_cache=True); pkv = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / n
            out["hf"].append({"ctx": ctx, "median_us": dt * 1e6, "tok_s": 1 / dt})
            print(f"hf_fp16 ctx {ctx:5d}: {dt * 1e3:6.2f} ms/token ({1 / dt:6.1f} tok/s)", flush=True)
            del o, pkv
        except torch.OutOfMemoryError as e:
            out["hf"].append({"ctx": ctx, "error": "OOM"})
        gc.collect(); torch.cuda.empty_cache()
    res.put("speed", out)


# ------------------------------------------------------------------ profile
CATS = [("gemv_rb", "W4A16 GEMV"), ("gemv_naive", "W4A16 GEMV"), ("paged_attn_partial", "attention partial"),
        ("paged_attn_combine", "attention combine"), ("add_rmsnorm", "add+RMSNorm (fused)"),
        ("swiglu", "SwiGLU (fused)"), ("rope_append", "RoPE+KV append"), ("gemm", "cuBLAS GEMM/GEMV"),
        ("gemv", "cuBLAS GEMM/GEMV"), ("argmax", "argmax"), ("reduce", "torch reduce/argmax"),
        ("index", "embedding gather"), ("elementwise", "torch elementwise (unfused ops)")]


def categorize(name):
    n = name.lower()
    for key, cat in CATS:
        if key in n:
            return cat
    return "other"


def bench_profile():
    from torch.profiler import profile, ProfilerActivity
    out = {}
    for name in ("rt_int4_g128", "rt_int4_g128_unfused", "rt_fp16"):
        m = build(name); m.kv.normal_()
        for ctx in (512, 16384):
            for mode in ("graph", "eager"):
                m.prepare(ctx + 80, use_graph=(mode == "graph"))
                m.reset(ctx)
                for _ in range(3): m.run_step(mode == "graph")
                torch.cuda.synchronize(); m.reset(ctx)
                steps = 4
                with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                    for _ in range(steps): m.run_step(mode == "graph")
                    torch.cuda.synchronize()
                evs = [e for e in prof.events() if e.device_type.name == "CUDA"]
                cats, count = {}, 0
                spans = []
                for e in evs:
                    d = e.time_range.end - e.time_range.start
                    if d <= 0: continue
                    if "memcpy" in e.name.lower() or "memset" in e.name.lower(): continue
                    count += 1
                    c = categorize(e.name)
                    cats[c] = cats.get(c, 0) + d / steps
                    spans.append((e.time_range.start, e.time_range.end))
                spans.sort()
                busy = sum(b - a for a, b in spans) / steps
                window = (spans[-1][1] - spans[0][0]) / steps if spans else 0
                key = f"{name}|ctx{ctx}|{mode}"
                out[key] = {"categories_us": cats, "kernels_per_token": count / steps, "kernel_busy_us": busy,
                            "first_to_last_us": window, "idle_us": window - busy}
                print(f"{key:40s} kernels/token {count / steps:6.1f}  busy {busy:8.1f}us  span {window:8.1f}us  "
                      f"idle {window - busy:7.1f}us  " + ", ".join(f"{k}={v:.0f}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1])), flush=True)
        del m
        res.put("profile", out)
    res.put("profile", out)



QUANT_CONFIGS = ("rt_int4_g128", "rt_int4_g128_clip", "rt_int4_g128_gptq", "rt_int4_g128_gptq_clip",
                 "rt_int4_g128_gptq_int4head", "rt_int4_g128_gptq_clip_int4head", "rt_int4_g128_int4head")


def bench_quant():
    """Same packed format and kernels, different ways of choosing the INT4 values: accuracy first, then a speed check
    (the GEMV reads identical bytes, so speed must not change — measured, not assumed)."""
    wins, ntok = wikitext_windows()
    out = {"windows": len(wins), "window_len": len(wins[0]), "results": {}}
    with torch.no_grad():
        tot, n = 0.0, 0
        for w in wins:
            a, b = nll_from_logits(hf(w[None].to(dev)).logits[0], w); tot += a; n += b
    base = math.exp(tot / n)
    out["results"]["hf_fp16"] = {"ppl": base, "tokens": n}
    for name in QUANT_CONFIGS:
        r = ppl_of(name, wins)
        r["rel_delta_pct"] = 100 * (r["ppl"] / base - 1)
        out["results"][name] = r
        out["calibration"] = QP_INFO
        res.put("quant_ppl", out)
    bench_agree(names=QUANT_CONFIGS, key="quant_agreement")
    sp = {}
    for name in ("rt_int4_g128", "rt_int4_g128_gptq_clip", "rt_int4_g128", "rt_int4_g128_gptq_clip"):  # ABAB: clock drift
        m = build(name)
        m.kv.normal_()
        sp.setdefault(name, [])
        for ctx in (512, 4096):
            m.prepare(ctx + 80)
            cool()
            r = time_decode(m, ctx)
            sp[name].append({"ctx": ctx, "median_us": r["median_us"], "wall_us": r["wall_us"], "tok_s": r["tok_s"],
                             "sm_mhz": r["clocks_after"].get("sm_mhz")})
            print(f"speed {name} ctx {ctx}: {r['tok_s']:.1f} tok/s @ {r['clocks_after'].get('sm_mhz')} MHz", flush=True)
        del m
    res.put("quant_speed", sp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["ppl", "agree", "quant", "speed", "profile"])
    a = ap.parse_args()
    res.put("env_e2e", env_info())
    failed = []
    for s in a.sections:  # sections are independent: one failing must not cost the others
        print(f"\n===== {s} =====", flush=True)
        try:
            {"ppl": bench_ppl, "agree": bench_agree, "speed": bench_speed, "profile": bench_profile,
             "quant": bench_quant}[s]()
        except Exception:
            import traceback; traceback.print_exc(); failed.append(s)
            gc.collect(); torch.cuda.empty_cache()
    if failed:
        sys.exit(f"e2e sections failed: {failed}")
