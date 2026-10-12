"""CPU-only tests for the quantizer and nibble packing (run locally before any GPU time).

The hand-checkable case: q row = [0,1,2,3,4,5,6,7] must pack to 0x75316420:
  e0=0 @bits0, e2=2 @bits4, e4=4 @bits8, e6=6 @bits12, e1=1 @bits16, e3=3 @bits20, e5=5 @bits24, e7=7 @bits28.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from runtime.quant import quantize, quantize_gptq, dequantize, pack, unpack


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


def test_clip_and_gptq_beat_rtn():
    """With correlated, outlier-heavy inputs: range clipping lowers the weight error, GPTQ lowers the *output* error
    (what it optimises), and both stay in the kernel's format (integer zero in [0,15], fp16 scale)."""
    torch.manual_seed(0)
    N, K, G = 128, 512, 128
    w = (torch.randn(N, K) * 0.02); w[torch.rand(N, K) < 0.002] *= 8  # a few outlier weights
    w = w.half()
    X = torch.randn(4096, 64) @ torch.randn(64, K)  # low-rank -> strongly correlated input features
    X[:, :8] *= 10                                   # a few large-activation channels
    H = 2 * X.t() @ X / X.shape[0]
    out_err = lambda q, s, z: (X @ (dequantize(q, s, z, G).float() - w.float()).t()).pow(2).mean().item()
    w_err = lambda q, s, z: (dequantize(q, s, z, G).float() - w.float()).pow(2).mean().item()
    rtn, clip = quantize(w, G), quantize(w, G, clip=True)
    gptq = quantize_gptq(w, H, G)
    assert w_err(*clip) < w_err(*rtn)
    assert out_err(*gptq) < 0.7 * out_err(*rtn), (out_err(*gptq), out_err(*rtn))
    for q, s, z in (clip, gptq):
        assert q.dtype == torch.uint8 and int(q.max()) <= 15
        assert torch.equal(z.float(), z.float().round()) and float(z.min()) >= 0 and float(z.max()) <= 15
    print(f"  output MSE: rtn {out_err(*rtn):.3e}  clip {out_err(*clip):.3e}  gptq {out_err(*gptq):.3e}")


def test_gptq_calibrate_tiny_llama():
    """End-to-end mechanics of runtime.gptq.calibrate on a random 2-layer Llama: every matrix produced, shapes match
    the runtime's keys, and the HF model's fp16 weights are restored afterwards."""
    from transformers import LlamaConfig, LlamaForCausalLM
    import runtime.gptq as GQ
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    m = LlamaForCausalLM(cfg).eval()
    before = {k: v.clone() for k, v in m.state_dict().items()}

    class Tok:
        bos_token_id = 1
    GQ.calibration_windows = lambda tok, n, L, seed=0: [torch.randint(2, 256, (L,)) for _ in range(n)]
    out = GQ.calibrate(m, Tok(), group=64, n=8, L=32, log=lambda *a: None)
    assert set(out) == {f"layers.{i}.{k}" for i in range(2) for k in ("qkv", "o", "gu", "down")} | {"lm_head"}
    assert out["layers.0.qkv"][0].shape == (128 + 2 * 64, 128) and out["layers.1.gu"][0].shape == (512, 128)
    assert out["lm_head"][1].shape == (256, 2)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
