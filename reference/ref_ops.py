"""fp16 PyTorch reference for every custom kernel. Same signatures/semantics as runtime/ext.py ops.

These are the numerical ground truth for tests/ and the CPU backend for local runtime validation.
Accumulation is fp32 everywhere, rounding to fp16 at the same points HF Llama rounds.
"""
import math
import torch


def gemv_ref(w_ref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y = W x with fp16 (dequantized) weights, fp32 accumulate, fp16 out."""
    return (w_ref.float() @ x.float()).half()


def add_rmsnorm_ref(x: torch.Tensor, delta, w: torch.Tensor, eps: float):
    """x <- x + delta (fp16 add, as HF residual); returns (x_new, rmsnorm(x_new) * w) matching HF LlamaRMSNorm."""
    xn = x if delta is None else (x + delta)
    xf = xn.float()
    h = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).half()
    return xn, w * h


def swiglu_ref(gu: torch.Tensor) -> torch.Tensor:
    """gu = [gate | up] -> silu(gate) * up, rounded like HF (silu output fp16, then fp16 multiply)."""
    I = gu.shape[-1] // 2
    g, u = gu[..., :I], gu[..., I:]
    return torch.nn.functional.silu(g.float()).half().float().mul(u.float()).half()


def rope_tables(max_pos: int, head_dim: int = 64, theta: float = 10000.0, device="cpu"):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64, device=device) / head_dim))
    f = torch.arange(max_pos, dtype=torch.float64, device=device)[:, None] * inv[None]
    return f.cos().float().contiguous(), f.sin().float().contiguous()  # [max_pos, head_dim/2]


def rope_ref(x: torch.Tensor, cos, sin, pos: int) -> torch.Tensor:
    """x [n_heads, D] (rotate_half convention: pairs (i, i+D/2)). fp32 math, fp16 out."""
    D = x.shape[-1]
    x1, x2 = x[..., : D // 2].float(), x[..., D // 2:].float()
    c, s = cos[pos], sin[pos]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).half()


def rope_append_ref(qkv, cos, sin, pos, k_cache, v_cache, block_table, n_q, n_kv, D, page):
    """Rotate q,k at position pos; write k,v into the paged cache. Returns rotated q [n_q, D]."""
    q = qkv[: n_q * D].view(n_q, D)
    k = qkv[n_q * D: (n_q + n_kv) * D].view(n_kv, D)
    v = qkv[(n_q + n_kv) * D:].view(n_kv, D)
    qr, kr = rope_ref(q, cos, sin, pos), rope_ref(k, cos, sin, pos)
    pg, off = int(block_table[pos // page]), pos % page
    k_cache[pg, :, off] = kr
    v_cache[pg, :, off] = v
    return qr


def gather_kv(k_cache, v_cache, block_table, ctx, page):
    """Paged cache [num_pages, n_kv, P, D] -> contiguous [ctx, n_kv, D]."""
    n_pages = (ctx + page - 1) // page
    idx = block_table[:n_pages].long()
    k = k_cache[idx].permute(0, 2, 1, 3).reshape(-1, k_cache.shape[1], k_cache.shape[3])[:ctx]
    v = v_cache[idx].permute(0, 2, 1, 3).reshape(-1, v_cache.shape[1], v_cache.shape[3])[:ctx]
    return k, v


def attention_contiguous_ref(q, k, v):
    """q [n_q, D], k/v [ctx, n_kv, D] contiguous. GQA by head repetition. fp32 softmax, fp16 out."""
    n_q, D = q.shape
    rep = n_q // k.shape[1]
    kf = k.float().repeat_interleave(rep, dim=1)  # [ctx, n_q, D]
    vf = v.float().repeat_interleave(rep, dim=1)
    s = torch.einsum("hd,thd->ht", q.float(), kf) / math.sqrt(D)
    p = torch.softmax(s, -1)
    return torch.einsum("ht,thd->hd", p, vf).half()


def paged_attention_ref(q, k_cache, v_cache, block_table, ctx, page):
    k, v = gather_kv(k_cache, v_cache, block_table, ctx, page)
    return attention_contiguous_ref(q, k, v)


def split_softmax_combine_ref(q, k, v, part):
    """Pure-torch model of the split-K kernel: per-partition (unnormalized o, max m, sum l) + rescale-combine.
    Exists so the combine maths is unit-tested independently of CUDA."""
    n_q, D = q.shape
    rep = n_q // k.shape[1]
    kf = k.float().repeat_interleave(rep, 1); vf = v.float().repeat_interleave(rep, 1)
    os, ms, ls = [], [], []
    for t0 in range(0, k.shape[0], part):
        s = torch.einsum("hd,thd->ht", q.float(), kf[t0:t0 + part]) / math.sqrt(D)
        m = s.amax(-1); p = torch.exp(s - m[:, None])
        os.append(torch.einsum("ht,thd->hd", p, vf[t0:t0 + part])); ms.append(m); ls.append(p.sum(-1))
    M = torch.stack(ms).amax(0)
    a = torch.stack([torch.exp(m - M) for m in ms])          # [S, n_q]
    L = (a * torch.stack(ls)).sum(0)
    O = (a[..., None] * torch.stack(os)).sum(0) / L[:, None]
    return O.half()
