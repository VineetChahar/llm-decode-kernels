"""CPU-only tests for the quantizer and nibble packing (run locally before any GPU time).

The hand-checkable case: q row = [0,1,2,3,4,5,6,7] must pack to 0x75316420:
  e0=0 @bits0, e2=2 @bits4, e4=4 @bits8, e6=6 @bits12, e1=1 @bits16, e3=3 @bits20, e5=5 @bits24, e7=7 @bits28.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from runtime.quant import quantize, dequantize, pack, unpack


def test_hand_packed_word():
    q = torch.arange(8, dtype=torch.uint8)[None]
    w = pack(q)
    assert (int(w[0, 0]) & 0xFFFFFFFF) == 0x75316420, hex(int(w[0, 0]) & 0xFFFFFFFF)
    # magic-number extraction as the kernel does it: (w >> 4i) & 0x000F000F gives (q[2i], q[2i+1])
    word = int(w[0, 0]) & 0xFFFFFFFF
    for i in range(4):
        t = (word >> (4 * i)) & 0x000F000F
        assert (t & 0xFFFF, t >> 16) == (2 * i, 2 * i + 1)


def test_roundtrip():
    torch.manual_seed(0)
    q = torch.randint(0, 16, (64, 2048), dtype=torch.uint8)
    assert torch.equal(unpack(pack(q)), q)


def test_quant_error_bound():
    torch.manual_seed(0)
    w = (torch.randn(256, 2048) * 0.02).half()
    for g in (64, 128, 256):
        q, s, z = quantize(w, g)
        wh = dequantize(q, s, z, g)
        err = (wh.float() - w.float()).abs().reshape(256, -1, g)
        # RTN error <= scale/2 (+ fp16 rounding, + clamp at the zero-point edge)
        bound = s.float()[..., None] * 0.5 + 2e-3 * w.float().abs().max()
        assert (err <= bound + 1e-6).all(), g


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
