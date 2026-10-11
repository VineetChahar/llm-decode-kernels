"""Batch-1 decode runtime for Llama-family models (TinyLlama-1.1B), built on the custom kernels.

One decode step (per layer):
    add_rmsnorm -> qkv GEMV -> rope+paged-KV append -> paged attention (+combine) -> o GEMV
    -> add_rmsnorm -> gate|up GEMV -> SwiGLU -> down GEMV
then final add_rmsnorm -> LM head GEMV -> argmax (written back to the device token slot) -> pos += 1.

State lives in device tensors (token id, position), so `step()` is capturable as one CUDA graph and replayed
for every token. Prefill is done by teacher-forcing prompt tokens through the same decode step (by design:
prefill speed is out of scope; it stays simple and exercises exactly the kernels under test).

Knobs (DecodeConfig):
    weights   "int4" (W4A16 GEMV) | "fp16" (cuBLAS via torch.mm, i.e. the HF GEMM call)
    group     quantization group size (64/128/256)
    lm_head   "fp16" | "int4"
    fused     fused add+RMSNorm and SwiGLU kernels vs the HF-style torch op sequences
    backend   "cuda" (custom kernels) | "ref" (torch reference ops; runs on CPU for local validation)
    gemv_cfg  {"NxK_G": [R, warps, pf]} from configs/autotune.json, or a fixed (stage, R, warps, pf)
"""
import json, math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from reference import ref_ops as R
from runtime.quant import QuantLinear

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DecodeConfig:
    weights: str = "int4"
    group: int = 128
    lm_head: str = "fp16"
    fused: bool = True
    backend: str = "cuda"
    max_ctx: int = 2048
    page: int = 16
    part_size: int = 0          # attention partition (split) size; 0 -> policy from configs/attn_policy.json
    gemv_stage: tuple = None    # (stage, R, warps, pf) to force one config everywhere; None -> autotune table
    autotune_path: str = str(ROOT / "configs" / "autotune.json")
    attn_policy_path: str = str(ROOT / "configs" / "attn_policy.json")
    shuffle_pages: bool = True
    device: str = "cuda"


DEFAULT_GEMV = (2, 4, 4, 1)  # hand-picked stage-2 config used when no autotune entry exists


class Linear:
    """One projection in whichever format the config asks for."""

    def __init__(self, w, cfg: DecodeConfig, quant: bool, keep_ref: bool):
        self.N, self.K = w.shape
        self.quant = quant
        dev = cfg.device
        if quant:
            ql = QuantLinear(w.cpu(), cfg.group, keep_ref=keep_ref)
            self.packed, self.sz = ql.packed.to(dev), ql.sz.to(dev)
            self.w_ref = ql.w_ref.to(dev) if keep_ref else None
            self.nbytes = ql.nbytes()
        else:
            self.w = w.to(dev).half().contiguous()
            self.nbytes = self.w.numel() * 2


class Model:
    def __init__(self, hf_state: dict, hf_config, cfg: DecodeConfig):
        self.cfg = cfg
        c = hf_config
        self.H, self.L = c.hidden_size, c.num_hidden_layers
        self.n_q, self.n_kv = c.num_attention_heads, c.num_key_value_heads
        self.D = self.H // self.n_q
        self.I, self.V = c.intermediate_size, c.vocab_size
        self.eps = c.rms_norm_eps
        assert self.D == 64, "kernels are specialised for head_dim 64"
        dev = cfg.device
        self.ref = cfg.backend == "ref"
        if not self.ref:
            from runtime import ext
            self.K = ext.load()
        quant = cfg.weights == "int4"
        keep_ref = self.ref  # reference backend computes with dequantized fp16 weights
        sd = hf_state
        g = lambda k: sd[k].half()
        self.emb = g("model.embed_tokens.weight").to(dev)
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            wqkv = torch.cat([g(p + "self_attn.q_proj.weight"), g(p + "self_attn.k_proj.weight"),
                              g(p + "self_attn.v_proj.weight")], 0)
            wgu = torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")], 0)
            self.layers.append({
                "ln1": g(p + "input_layernorm.weight").to(dev), "ln2": g(p + "post_attention_layernorm.weight").to(dev),
                "qkv": Linear(wqkv, cfg, quant, keep_ref), "o": Linear(g(p + "self_attn.o_proj.weight"), cfg, quant, keep_ref),
                "gu": Linear(wgu, cfg, quant, keep_ref), "down": Linear(g(p + "mlp.down_proj.weight"), cfg, quant, keep_ref)})
        self.norm_f = g("model.norm.weight").to(dev)
        lm = sd.get("lm_head.weight", sd["model.embed_tokens.weight"]).half()
        self.lm_head = Linear(lm, cfg, cfg.lm_head == "int4", keep_ref)

        # gemv config table
        self.gemv_table = {}
        if cfg.gemv_stage is None and Path(cfg.autotune_path).exists():
            self.gemv_table = json.load(open(cfg.autotune_path)).get("best", {})

        # paged KV cache: [L, 2, pages, n_kv, P, D]; block table shared by all layers (one sequence)
        self.n_pages = (cfg.max_ctx + cfg.page - 1) // cfg.page + 1
        self.kv = torch.zeros(self.L, 2, self.n_pages, self.n_kv, cfg.page, self.D, dtype=torch.half, device=dev)
        self.bt = (torch.randperm(self.n_pages) if cfg.shuffle_pages else torch.arange(self.n_pages)).int().to(dev)
        self.cos, self.sin = (t.to(dev) for t in R.rope_tables(cfg.max_ctx + 1, self.D, c.rope_theta if hasattr(c, "rope_theta") else 10000.0))

        ps = cfg.part_size or self._policy_part_size(cfg.max_ctx)
        self.part_size = ps
        self.max_splits = max(1, math.ceil(cfg.max_ctx / ps))
        f32 = dict(dtype=torch.float32, device=dev)
        self.o_part = torch.empty(self.n_q, self.max_splits, self.D, **f32)
        self.m_part = torch.empty(self.n_q, self.max_splits, **f32)
        self.l_part = torch.empty(self.n_q, self.max_splits, **f32)

        # device-resident decode state + static activation buffers
        h16 = dict(dtype=torch.half, device=dev)
        self.tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.int32, device=dev)
        self.x = torch.empty(self.H, **h16); self.normed = torch.empty(self.H, **h16)
        self.qkv = torch.empty((self.n_q + 2 * self.n_kv) * self.D, **h16)
        self.attn = torch.empty(self.n_q, self.D, **h16)
        self.o = torch.empty(self.H, **h16); self.gu = torch.empty(2 * self.I, **h16)
        self.act = torch.empty(self.I, **h16); self.down = torch.empty(self.H, **h16)
        self.logits = torch.empty(self.V, **h16)
        self.graph = None

    # ------------------------------------------------------------------ helpers
    def _policy_part_size(self, max_ctx):
        p = Path(self.cfg.attn_policy_path)
        if p.exists():
            for ent in json.load(open(p))["policy"]:  # sorted by max_ctx ascending
                if max_ctx <= ent["max_ctx"]:
                    return ent["part_size"]
        return 512

    def gemv_cfg(self, lin):
        if self.cfg.gemv_stage is not None:
            return self.cfg.gemv_stage
        e = self.gemv_table.get(f"{lin.N}x{lin.K}_G{self.cfg.group}")
        return (2, *e) if e else DEFAULT_GEMV

    def linear(self, lin: Linear, x, y):
        if self.ref:
            y.copy_(R.gemv_ref(lin.w_ref if lin.quant else lin.w, x))
        elif lin.quant:
            s, r, w, pf = self.gemv_cfg(lin)
            self.K.gemv_w4a16(s, r, w, pf, lin.packed, lin.sz, x, y, self.cfg.group)
        else:
            torch.mm(x.view(1, -1), lin.w.t(), out=y.view(1, -1))  # cuBLAS, same call HF makes

    def add_norm(self, delta, w):
        if self.ref or not self.cfg.fused:
            xn, out = R.add_rmsnorm_ref(self.x, delta, w, self.eps)
            if delta is not None:
                self.x.copy_(xn)
            self.normed.copy_(out)
        else:
            self.K.add_rmsnorm(self.x, delta, w, self.normed, self.eps)

    def swiglu(self):
        if self.ref or not self.cfg.fused:
            self.act.copy_(R.swiglu_ref(self.gu))
        else:
            self.K.swiglu(self.gu, self.act)

    def attention(self, li):
        kc, vc = self.kv[li, 0], self.kv[li, 1]
        if self.ref:
            pos = int(self.pos)
            q = R.rope_append_ref(self.qkv, self.cos, self.sin, pos, kc, vc, self.bt, self.n_q, self.n_kv, self.D, self.cfg.page)
            self.attn.copy_(R.paged_attention_ref(q, kc, vc, self.bt, pos + 1, self.cfg.page))
        else:
            self.K.rope_append(self.qkv, self.cos, self.sin, self.pos, self.bt, kc, vc, self.n_q, self.n_kv)
            self.K.paged_attention(self.qkv[: self.n_q * self.D], kc, vc, self.bt, self.pos, self.part_size,
                                   self.o_part, self.m_part, self.l_part, self.attn)

    # ------------------------------------------------------------------ decode step
    def step(self):
        """One token: reads self.tok/self.pos, writes self.logits, self.tok <- argmax, self.pos += 1."""
        torch.index_select(self.emb, 0, self.tok, out=self.x.view(1, -1))
        delta = None
        for li, lw in enumerate(self.layers):
            self.add_norm(delta, lw["ln1"])
            self.linear(lw["qkv"], self.normed, self.qkv)
            self.attention(li)
            self.linear(lw["o"], self.attn.view(-1), self.o)
            self.add_norm(self.o, lw["ln2"])
            self.linear(lw["gu"], self.normed, self.gu)
            self.swiglu()
            self.linear(lw["down"], self.act, self.down)
            delta = self.down
        self.add_norm(delta, self.norm_f)
        self.linear(self.lm_head, self.normed, self.logits)
        torch.argmax(self.logits, out=self.tok.view(())) if not self.ref else self.tok.fill_(int(self.logits.float().argmax()))
        self.pos += 1

    def capture(self):
        """Capture step() into a CUDA graph (warm up on a side stream first, as torch requires)."""
        assert not self.ref
        saved_tok, saved_pos = self.tok.clone(), self.pos.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()
        self.tok.copy_(saved_tok); self.pos.copy_(saved_pos)
        torch.cuda.synchronize()

    def run_step(self, use_graph=True):
        if use_graph and self.graph is not None:
            self.graph.replay()
        else:
            self.step()

    # ------------------------------------------------------------------ high-level
    def reset(self, pos=0):
        self.pos.fill_(pos)

    def teacher_force(self, tokens, use_graph=True, collect_logits=False):
        """Feed tokens[i] at positions pos..; returns logits [len, V] (fp16) if requested."""
        tokens = torch.as_tensor(tokens, device=self.tok.device).long()
        out = torch.empty(len(tokens), self.V, dtype=torch.half, device=self.tok.device) if collect_logits else None
        for i in range(len(tokens)):
            self.tok.copy_(tokens[i:i + 1])
            self.run_step(use_graph)
            if collect_logits:
                out[i].copy_(self.logits)
        return out

    def generate(self, prompt, n_new, use_graph=True):
        """Greedy: teacher-force the prompt, then free-run n_new tokens. Returns list of new token ids."""
        self.reset(0)
        self.teacher_force(prompt, use_graph)
        new = [int(self.tok)]
        for _ in range(n_new - 1):
            self.run_step(use_graph)
            new.append(int(self.tok))
        return new

    # ------------------------------------------------------------------ byte accounting (for the roofline)
    def weight_bytes(self):
        b = sum(l[k].nbytes for l in self.layers for k in ("qkv", "o", "gu", "down"))
        b += sum(l["ln1"].numel() * 2 + l["ln2"].numel() * 2 for l in self.layers)
        return {"body": b, "lm_head": self.lm_head.nbytes, "embedding_row": self.H * 2}

    def kv_bytes_per_token(self, ctx):
        """KV bytes read by attention at context ctx, plus the new K,V row written, all layers."""
        row = self.n_kv * self.D * 2  # one token, one of K/V, fp16
        return self.L * (2 * row * ctx + 2 * row)
