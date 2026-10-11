"""CPU check of the split-K combine maths (the online-softmax rescale) against plain softmax attention."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from reference import ref_ops as R

def test_combine():
    torch.manual_seed(0)
    for ctx in (1, 17, 128, 1000):
        for scale in (1.0, 8.0):
            q = torch.randn(32, 64).half(); k = (torch.randn(ctx, 4, 64) * scale).half(); v = torch.randn(ctx, 4, 64).half()
            ref = R.attention_contiguous_ref(q, k, v)
            for part in (1, 16, 64, 100, 4096):
                out = R.split_softmax_combine_ref(q, k, v, part)
                err = (out.float() - ref.float()).abs().max().item()
                assert err < 2e-3, (ctx, scale, part, err)

def test_paged_gather():
    torch.manual_seed(0)
    kc = torch.randn(12, 4, 16, 64).half(); vc = torch.randn(12, 4, 16, 64).half()
    bt = torch.randperm(12).int()
    k, v = R.gather_kv(kc, vc, bt, 70, 16)
    t = 37  # token 37 lives in logical page 2, offset 5
    assert torch.equal(k[t], kc[bt[2], :, 5]) and torch.equal(v[t], vc[bt[2], :, 5])

if __name__ == "__main__":
    test_combine(); print("PASS combine"); test_paged_gather(); print("PASS gather")
