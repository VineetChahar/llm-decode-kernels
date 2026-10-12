"""Group-wise asymmetric INT4 quantizer + nibble packing for the W4A16 GEMV.

Weight W [N, K] (out x in). Per row, per group of G consecutive K elements:
    scale = (max - min) / 15        (fp16)
    zero  = round(-min / scale)     (integer-valued fp16 in [0, 15])
    q     = clamp(round(w / scale) + zero, 0, 15)
    w_hat = ((q - zero) * scale).half()

Packed layout (int32 [N, K/8]); word j holds elements 8j..8j+7:
    element 8j+2i   -> bits [4i,    4i+4)      i = 0..3
    element 8j+2i+1 -> bits [16+4i, 16+4i+4)
so (word >> 4i) & 0x000F000F | 0x64006400 reinterpreted as half2 is (1024+q[8j+2i], 1024+q[8j+2i+1]).
"""
import torch


CLIP_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)


def group_params(wf, clip=False):
    """wf [..., G] fp32 -> (scale, zero) [..., 1] in the kernel's convention (fp16-representable scale, integer zero).
    clip=True: shrink the [min, max] range by the factor in CLIP_GRID that minimises the group's squared rounding
    error (outliers otherwise stretch the 16-level grid for the whole group)."""
    mn0 = wf.amin(-1, keepdim=True)
    mx0 = wf.amax(-1, keepdim=True)
    best = None
    for a in (CLIP_GRID if clip else (1.0,)):
        mn, mx = mn0 * a, mx0 * a
        scale = ((mx - mn) / 15).clamp(min=1e-6).half().float()  # round scale to fp16 first, quantize against it
        zero = torch.round(-mn / scale).clamp(0, 15)
        if not clip:
            return scale, zero
        err = ((torch.clamp(torch.round(wf / scale) + zero, 0, 15) - zero) * scale - wf).pow(2).sum(-1, keepdim=True)
        if best is None:
            best = [err, scale, zero]
        else:
            m = err < best[0]
            best = [torch.where(m, err, best[0]), torch.where(m, scale, best[1]), torch.where(m, zero, best[2])]
    return best[1], best[2]


def quantize(w: torch.Tensor, group: int, clip: bool = False):
    N, K = w.shape
    assert K % group == 0, (K, group)
    wf = w.float().reshape(N, K // group, group)
    scale, zero = group_params(wf, clip)
    q = torch.clamp(torch.round(wf / scale) + zero, 0, 15)
    return q.reshape(N, K).to(torch.uint8), scale.reshape(N, -1).half(), zero.reshape(N, -1).half()


@torch.no_grad()
def quantize_gptq(w: torch.Tensor, H: torch.Tensor, group: int, clip: bool = False, damp: float = 0.01, block: int = 128):
    """GPTQ (Frantar et al. 2022) into the same format as quantize(): returns (q uint8 [N,K], scale, zero [N,K/G] fp16).

    Columns are quantized left to right; each column's rounding error is spread over the not-yet-quantized columns
    through the inverse Hessian of the layer input (H = 2 X^T X over calibration tokens), so the layer *output* error is
    minimised rather than the weight error. Group params are computed when a group starts, from the error-updated
    weights (as in the reference implementation); no act-order, so groups stay contiguous and the kernel is unchanged."""
    N, K = w.shape
    assert K % group == 0 and block % group == 0
    W = w.float().clone()
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(K, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros(N, K, dtype=torch.uint8, device=W.device)
    S = torch.zeros(N, K // group, device=W.device)
    Z = torch.zeros(N, K // group, device=W.device)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % group == 0:
                s, z = group_params(W1[:, i:i + group], clip)  # block % group == 0: the group lies in this block
                S[:, c // group], Z[:, c // group] = s[:, 0], z[:, 0]
                scale, zero = s[:, 0], z[:, 0]
            col = W1[:, i]
            q = torch.clamp(torch.round(col / scale) + zero, 0, 15)
            Q[:, c] = q.to(torch.uint8)
            deq = (q - zero) * scale
            e = (col - deq) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            Err[:, i] = e
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q, S.half(), Z.half()


def dequantize(q, scale, zero, group: int):
    N, K = q.shape
    qf = q.float().reshape(N, K // group, group)
    w = (qf - zero.float()[..., None]) * scale.float()[..., None]
    return w.reshape(N, K).half()  # single rounding, identical to the kernel's half arithmetic


def pack(q: torch.Tensor) -> torch.Tensor:
    N, K = q.shape
    assert K % 8 == 0
    q = q.to(torch.int64).reshape(N, K // 8, 8)
    word = torch.zeros(N, K // 8, dtype=torch.int64, device=q.device)
    for i in range(4):
        word |= q[..., 2 * i] << (4 * i)
        word |= q[..., 2 * i + 1] << (16 + 4 * i)
    word = torch.where(word >= 2**31, word - 2**32, word)  # store as signed int32
    return word.to(torch.int32).contiguous()


def unpack(packed: torch.Tensor) -> torch.Tensor:
    N, W = packed.shape
    word = packed.to(torch.int64) & 0xFFFFFFFF
    out = torch.empty(N, W, 8, dtype=torch.uint8, device=packed.device)
    for i in range(4):
        out[..., 2 * i] = ((word >> (4 * i)) & 0xF).to(torch.uint8)
        out[..., 2 * i + 1] = ((word >> (16 + 4 * i)) & 0xF).to(torch.uint8)
    return out.reshape(N, W * 8)


def pack_scale_zero(scale, zero):
    """[N, K/G] scale + zero -> [N, K/G, 2] fp16, read by the kernel as half2 (scale, zero)."""
    return torch.stack([scale, zero], -1).contiguous()


class QuantLinear:
    """Holds one quantized weight in kernel format plus the fp16 dequantized reference."""

    def __init__(self, w: torch.Tensor, group: int, keep_ref: bool = True, clip: bool = False, qsz=None):
        """qsz: precomputed (q, scale, zero), e.g. from quantize_gptq; otherwise RTN (optionally range-clipped)."""
        self.N, self.K = w.shape
        self.group = group
        q, s, z = (t.to(w.device) for t in qsz) if qsz is not None else quantize(w, group, clip)
        self.packed = pack(q)
        self.sz = pack_scale_zero(s, z)
        self.w_ref = dequantize(q, s, z, group) if keep_ref else None

    def nbytes(self):
        return self.packed.numel() * 4 + self.sz.numel() * 2
