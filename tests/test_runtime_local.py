"""CPU validation of runtime plumbing (fused qkv/gate-up, RoPE, paged cache w/ shuffled pages, residual flow)
against HF transformers, using the torch reference backend. Run locally: no GPU needed."""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.model import Model, DecodeConfig

MID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MID)
hf = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).eval()
ids = tok("The capital of France is Paris. The capital of Germany is", return_tensors="pt").input_ids[0]
with torch.no_grad():
    ref = hf(ids[None]).logits[0].float()
sd = {k: v.half() for k, v in hf.state_dict().items()}
for wfmt in ("fp16", "int4"):
    m = Model(sd, hf.config, DecodeConfig(weights=wfmt, lm_head="fp16", backend="ref", device="cpu", max_ctx=64))
    m.reset(0)
    out = m.teacher_force(ids, use_graph=False, collect_logits=True).float()
    agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    diff = (out - ref).abs().max().item()
    print(f"{wfmt}: top1 agreement vs HF fp32 = {agree:.3f}, max|dlogit| = {diff:.3f}, next tok = {tok.decode(out[-1].argmax())!r}")
    top2 = ref.topk(2, -1).values; margin = top2[:, 0] - top2[:, 1]
    flips = (out.argmax(-1) != ref.argmax(-1))
    print(f"   flipped positions: {flips.nonzero().flatten().tolist()}  HF top1-top2 margin there: {margin[flips].tolist()}")
    if wfmt == "fp16":  # fp16 runtime vs fp32 HF: only near-ties (margin < 0.25) may flip
        assert diff < 0.5 and (margin[flips] < 0.25).all(), (agree, diff)
print("greedy:", tok.decode(m.generate(ids, 6, use_graph=False)))
