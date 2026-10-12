#!/usr/bin/env python3
"""Same-node, clock-recorded head-to-head: our W4A16 G128 runtime vs llama.cpp Q4_K_M (flash attention on).

The headline comparison in RESULTS.md v1 took llama.cpp from a different Kaggle node, and llama-bench records no
clocks. Here both run on the same GPU in the same session, alternating per context and swapping the order every
round (A,B then B,A) so clock drift hits both sides equally. Each measurement starts after the same cool-down
(< 60 C), and an NVML sampler thread (every 100 ms) records the SM clock throughout, so each number has its own clock.

Like-for-like protocol: 128 generated tokens starting at depth ctx (llama-bench -n 128 -d ctx, KV pre-filled by a
prefill it does not time), against our 128-token decode loop at ctx (CUDA graph, KV pre-filled with random values).
Ours is reported both as the per-token CUDA-event median and as the wall-clock mean over the 128 tokens;
llama-bench's avg_ts is a wall-clock mean, so `ratio_wall` is the like-for-like ratio.

usage: python bench/h2h.py --llama-bench PATH --gguf PATH [--rounds 2]  -> RESULTS_DIR/results.json key "h2h"
"""
import argparse, json, statistics, subprocess, sys, threading, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM
from runtime.model import Model, DecodeConfig
from bench.timing import Results, env_info, clocks, cool

MID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CTXS = [512, 1024, 2048, 4096, 8192, 16384]
N_GEN = 128


class ClockSampler:
    """Background NVML sampling; samples with SM clock at the 585 MHz idle floor and ~0 W draw count as idle."""

    def __init__(self, dt=0.1):
        self.dt, self.s, self._stop = dt, [], threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self

    def _run(self):
        while not self._stop.is_set():
            c = clocks()
            if "sm_mhz" in c:
                self.s.append((time.time(), c["sm_mhz"], c["power_w"], c["temp_c"], tuple(c["reasons"])))
            time.sleep(self.dt)

    def __exit__(self, *a):
        self._stop.set(); self._t.join()

    def summary(self, t0=None, t1=None):
        s = [x for x in self.s if (t0 is None or x[0] >= t0) and (t1 is None or x[0] <= t1)]
        busy = [x for x in s if "gpu_idle" not in x[4]] or s
        if not busy:
            return {}
        return {"sm_mhz_median": statistics.median(x[1] for x in busy), "sm_mhz_min": min(x[1] for x in busy),
                "sm_mhz_max": max(x[1] for x in busy), "temp_c_max": max(x[3] for x in busy), "n_samples": len(busy),
                "hw_slowdown_frac": sum("hw_slowdown" in x[4] for x in busy) / len(busy)}


def ours(m, ctx):
    m.reset(ctx)
    for _ in range(5):
        m.run_step()
    torch.cuda.synchronize()
    m.reset(ctx)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N_GEN)]
    with ClockSampler() as cs:
        for s, e in ev:
            s.record(); m.run_step(); e.record()
        torch.cuda.synchronize()
        t = sorted(s.elapsed_time(e) * 1000 for s, e in ev)
        m.reset(ctx); torch.cuda.synchronize(); w0 = time.perf_counter()
        for _ in range(N_GEN):
            m.run_step()
        torch.cuda.synchronize(); wall = (time.perf_counter() - w0) / N_GEN * 1e6
    med = t[len(t) // 2]
    return {"tok_s": 1e6 / med, "tok_s_wall": 1e6 / wall, "median_us": med, "wall_us": wall, "clock": cs.summary()}


def llama(bench, gguf, ctx, reps):
    cmd = [bench, "-m", gguf, "-ngl", "99", "-p", "0", "-n", str(N_GEN), "-d", str(ctx), "-fa", "1", "-r", str(reps),
           "-o", "json"]
    with ClockSampler() as cs:
        p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    r = json.loads(p.stdout)[0]
    # llama-bench runs prefill(ctx) + generate(128) per rep and times only the generate part. The sampler covers both;
    # also report the clock over the last N_GEN/avg_ts seconds of the run (the final rep's generate phase).
    gen_s = r["avg_ns"] / 1e9
    tail = cs.summary(t0=cs.s[-1][0] - gen_s if cs.s else None)
    return {"tok_s": r["avg_ts"], "stddev_ts": r["stddev_ts"], "samples_ts": r.get("samples_ts"),
            "build_commit": r.get("build_commit"), "clock": cs.summary(), "clock_last_gen": tail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-bench", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    res = Results()
    res.put("env_h2h", env_info())

    hf = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).eval()  # CPU: only the weights are needed
    sd = {k: v.cuda() for k, v in hf.state_dict().items()}
    m = Model(sd, hf.config, DecodeConfig(max_ctx=CTXS[-1] + 160, weights="int4", group=128, lm_head="fp16"))
    del hf
    m.kv.normal_()
    torch.cuda.empty_cache()

    out = {"protocol": __doc__.split("\n\n")[2].replace("\n", " "), "n_gen": N_GEN, "reps_llama": a.reps,
           "rounds": a.rounds, "ours_config": "rt_int4_g128 (RTN weights; GPTQ weights time the same, quant_speed)",
           "llama_config": "Q4_K_M, -fa 1, -ngl 99", "points": []}
    for rnd in range(a.rounds):
        for ctx in CTXS:
            m.prepare(ctx + 150)
            order = ("ours", "llama") if (rnd + CTXS.index(ctx)) % 2 == 0 else ("llama", "ours")
            pt = {"round": rnd, "ctx": ctx, "order": list(order)}
            for who in order:
                pt[f"cooled_s_{who}"] = cool()
                pt[who] = ours(m, ctx) if who == "ours" else llama(a.llama_bench, a.gguf, ctx, a.reps)
            o, l = pt["ours"], pt["llama"]
            pt["ratio_wall"] = o["tok_s_wall"] / l["tok_s"]
            pt["ratio_event"] = o["tok_s"] / l["tok_s"]
            out["points"].append(pt)
            print(f"r{rnd} ctx {ctx:5d} [{order[0]} first]: ours {o['tok_s_wall']:6.1f} wall / {o['tok_s']:6.1f} event tok/s"
                  f" @ {o['clock'].get('sm_mhz_median')} MHz | llama.cpp {l['tok_s']:6.1f} tok/s"
                  f" @ {l['clock'].get('sm_mhz_median')} MHz (last gen {l['clock_last_gen'].get('sm_mhz_median')})"
                  f" | ratio {pt['ratio_wall']:.3f}", flush=True)
            res.put("h2h", out)

    summ = {}
    for ctx in CTXS:
        ps = [p for p in out["points"] if p["ctx"] == ctx]
        summ[ctx] = {"ours_tok_s_wall": statistics.median(p["ours"]["tok_s_wall"] for p in ps),
                     "ours_tok_s_event": statistics.median(p["ours"]["tok_s"] for p in ps),
                     "llama_tok_s": statistics.median(p["llama"]["tok_s"] for p in ps),
                     "ratio_wall": statistics.median(p["ratio_wall"] for p in ps),
                     "ratio_wall_min": min(p["ratio_wall"] for p in ps), "ratio_wall_max": max(p["ratio_wall"] for p in ps),
                     "ours_mhz": statistics.median(p["ours"]["clock"].get("sm_mhz_median", 0) for p in ps),
                     "llama_mhz": statistics.median(p["llama"]["clock"].get("sm_mhz_median", 0) for p in ps)}
        s = summ[ctx]
        print(f"ctx {ctx:5d}: ours {s['ours_tok_s_wall']:6.1f} vs llama.cpp {s['llama_tok_s']:6.1f} tok/s -> "
              f"{100 * (s['ratio_wall'] - 1):+5.1f}% (range {100 * (s['ratio_wall_min'] - 1):+.1f}..{100 * (s['ratio_wall_max'] - 1):+.1f}%)"
              f" at {s['ours_mhz']:.0f} / {s['llama_mhz']:.0f} MHz", flush=True)
    out["summary"] = summ
    res.put("h2h", out)


if __name__ == "__main__":
    main()
