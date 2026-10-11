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


def quantize(w: torch.Tensor, group: int):
    N, K = w.shape
    assert K % group == 0, (K, group)
    wf = w.float().reshape(N, K // group, group)
    mn = wf.amin(-1, keepdim=True)
    mx = wf.amax(-1, keepdim=True)
    scale = ((mx - mn) / 15).clamp(min=1e-6).half().float()  # round scale to fp16 first, quantize against it
    zero = torch.round(-mn / scale).clamp(0, 15)
    q = torch.clamp(torch.round(wf / scale) + zero, 0, 15)
    return q.reshape(N, K).to(torch.uint8), scale.reshape(N, -1).half(), zero.reshape(N, -1).half()


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

    def __init__(self, w: torch.Tensor, group: int, keep_ref: bool = True):
        self.N, self.K = w.shape
        self.group = group
        q, s, z = quantize(w, group)
        self.packed = pack(q)
        self.sz = pack_scale_zero(s, z)
        self.w_ref = dequantize(q, s, z, group) if keep_ref else None

    def nbytes(self):
        return self.packed.numel() * 4 + self.sz.numel() * 2
