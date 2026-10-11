"""Numerical gates for every kernel. Run on GPU before any speed number is produced.

Order is deliberate — tiny hand-checkable cases first:
  1. 4-row GEMV, K=128, closed-form answer, every stage/config, exact match required
  2. one-hot x sweep over all 128 columns: catches any nibble mis-packing position by position
  3. real shapes (all model GEMVs x group sizes) vs fp16 dequantized reference
  4. paged attention at ctx=128 vs contiguous-KV reference, several partition sizes (incl. ragged)
  5. paged attention at real contexts 512..16K, flat and peaky softmax
  6. fused norm / swiglu / rope+append vs reference
Errors: max_abs = max|kernel - ref|, max_rel = max_abs / max|ref| (normwise; elementwise rel is
meaningless at values near zero). Writes RESULTS_DIR/results.json["accuracy_kernels"]; exits 1 on failure.
"""
import math, os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from runtime import ext
from runtime.quant import QuantLinear, pack, pack_scale_zero
from reference import ref_ops as R
from bench.timing import Results

torch.manual_seed(0)
dev = "cuda"
K_ = ext.load()
rows, failures = [], []

# (stage, R, warps, pf) configurations exercised by every GEMV test
GEMV_CFGS = [(0, 1, 4, 0), (1, 1, 4, 0)] + [(2, r, w, pf) for r in (1, 2, 4, 8) for w in (1, 4, 8) for pf in (0, 1)]


def record(name, out, ref, tol):
    out, ref = out.float(), ref.float()
    max_abs = (out - ref).abs().max().item()
    max_rel = max_abs / max(ref.abs().max().item(), 1e-12)
    ok = math.isfinite(max_abs) and max_rel <= tol
    rows.append({"test": name, "max_abs": max_abs, "max_rel": max_rel, "tol_rel": tol, "pass": ok})
    if not ok:
        failures.append(name)
    print(f"{'PASS' if ok else 'FAIL'} {name:58s} max_abs={max_abs:.3e} max_rel={max_rel:.3e}", flush=True)


def gemv(cfg, packed, sz, x, N, group):
    y = torch.empty(N, dtype=torch.half, device=dev)
    K_.gemv_w4a16(cfg[0], cfg[1], cfg[2], cfg[3], packed, sz, x, y, group)
    return y


# ---- 1. hand-checkable 4-row GEMV: q[r,k]=(k+r)%16, zero=8, scale=(r+1)/2, x=1 -> y[r] = -64*(r+1)/2
def test_shape1():
    N, K, G = 4, 128, 128
    q = torch.tensor([[(k + r) % 16 for k in range(K)] for r in range(N)], dtype=torch.uint8)
    s = torch.tensor([[(r + 1) * 0.5] for r in range(N)], dtype=torch.half)
    z = torch.full((N, 1), 8.0, dtype=torch.half)
    packed, sz = pack(q).to(dev), pack_scale_zero(s, z).to(dev)
    expect = torch.tensor([-32.0, -64.0, -96.0, -128.0], device=dev)
    for cfg in GEMV_CFGS:
        y = gemv(cfg, packed, sz, torch.ones(K, dtype=torch.half, device=dev), N, G)
        record(f"gemv_shape1_x=1 stage{cfg}", y, expect, 0.0)
        # one-hot sweep: y[r] must equal (q[r,k]-8)*s[r] for every column k
        bad = 0
        for k in range(K):
            x = torch.zeros(K, dtype=torch.half, device=dev); x[k] = 1
            y = gemv(cfg, packed, sz, x, N, G).float().cpu()
            want = (q[:, k].float() - 8) * s[:, 0].float()
            bad += int((y != want).any())
        rows.append({"test": f"gemv_onehot_nibbles stage{cfg}", "max_abs": float(bad), "max_rel": float(bad),
                     "tol_rel": 0.0, "pass": bad == 0})
        print(f"{'PASS' if bad == 0 else 'FAIL'} gemv_onehot_nibbles stage{cfg} wrong_columns={bad}", flush=True)
        if bad: failures.append(f"onehot {cfg}")


# ---- 3. real shapes
SHAPES = {"qkv": (2560, 2048), "o": (2048, 2048), "gate_up": (11264, 2048), "down": (2048, 5632),
          "lm_head": (32000, 2048)}


def test_real_gemv():
    for name, (N, K) in SHAPES.items():
        w = (torch.randn(N, K) * 0.02).half()
        x = torch.randn(K, device=dev).half()
        for G in (64, 128, 256):
            ql = QuantLinear(w, G)
            packed, sz, wref = ql.packed.to(dev), ql.sz.to(dev), ql.w_ref.to(dev)
            ref = R.gemv_ref(wref, x)
            cfgs = GEMV_CFGS if G == 128 else [(0, 1, 4, 0), (1, 1, 4, 0), (2, 4, 4, 1), (2, 8, 2, 1)]
            for cfg in cfgs:
                record(f"gemv {name} {N}x{K} G{G} stage{cfg}", gemv(cfg, packed, sz, x, N, G), ref, 5e-3)


# ---- 4/5. paged attention
def make_cache(ctx, n_kv, page, D=64, peaky=1.0):
    n_pages = (ctx + page - 1) // page + 3
    kc = (torch.randn(n_pages, n_kv, page, D, device=dev) * peaky).half()
    vc = torch.randn(n_pages, n_kv, page, D, device=dev).half()
    bt = torch.randperm(n_pages, device=dev).int()  # non-contiguous physical pages
    return kc, vc, bt


def run_attn(q, kc, vc, bt, ctx, PS, version=2, spare=0):
    """spare > 0 sizes the split buffers for more splits than ctx needs, as the runtime does (buffers are sized for
    the prepared cap): surplus split blocks must exit and combine must ignore the unused slots."""
    n_q = q.numel() // 64
    max_splits = max(1, (ctx + PS - 1) // PS) + spare
    o_part = torch.empty(n_q, max_splits, 64, device=dev)
    m_part = torch.empty(n_q, max_splits, device=dev); l_part = torch.empty_like(m_part)
    out = torch.empty(n_q, 64, dtype=torch.half, device=dev)
    pos = torch.tensor([ctx - 1], dtype=torch.int32, device=dev)
    K_.paged_attention(q, kc, vc, bt, pos, PS, o_part, m_part, l_part, out, version)
    return out


def test_attention():
    # ctx=128 vs contiguous reference first
    for n_q, n_kv in ((32, 4), (8, 8), (16, 4), (8, 4)):  # HPG 8, 1, 4, 2
        for page in (16, 32):
            kc, vc, bt = make_cache(128, n_kv, page)
            q = torch.randn(n_q, 64, device=dev).half()
            k, v = R.gather_kv(kc, vc, bt, 128, page)
            ref = R.attention_contiguous_ref(q, k, v)
            for PS in (4096, 64, 32, 48, 16):
                for ver in (1, 2, 3):
                    record(f"attn ctx128 hq{n_q} hkv{n_kv} P{page} PS{PS} v{ver}", run_attn(q, kc, vc, bt, 128, PS, ver), ref, 2e-3)
    # odd context lengths (partial last page and partial last chunk)
    for ctx in (1, 2, 17, 63, 65, 129, 1000):
        kc, vc, bt = make_cache(ctx, 4, 16)
        q = torch.randn(32, 64, device=dev).half()
        ref = R.paged_attention_ref(q, kc, vc, bt, ctx, 16)
        for PS in (8192, 64, 256):
            for ver in (1, 2, 3):
                record(f"attn ctx{ctx} PS{PS} v{ver}", run_attn(q, kc, vc, bt, ctx, PS, ver), ref, 2e-3)
    # real contexts, flat and peaky (sharp softmax stresses the max/rescale path)
    for ctx in (512, 1024, 2048, 4096, 8192, 16384):
        for peaky in (1.0, 6.0):
            kc, vc, bt = make_cache(ctx, 4, 16, peaky=peaky)
            q = torch.randn(32, 64, device=dev).half()
            ref = R.paged_attention_ref(q, kc, vc, bt, ctx, 16)
            for PS in (ctx, 1024, 256, 128, 64):
                for ver in (1, 2, 3):
                    record(f"attn ctx{ctx} peaky{peaky} PS{PS} v{ver}", run_attn(q, kc, vc, bt, ctx, PS, ver), ref, 2e-3)
    # runtime conditions: buffers sized for a larger cap (surplus splits exit), incl. the single-live-split path
    # where the partial kernel writes `out` itself and combine must exit; PS values from configs/attn_policy.json
    for ctx in (1, 40, 64, 100, 700, 2048, 5000):
        kc, vc, bt = make_cache(ctx, 4, 16, peaky=6.0)
        q = torch.randn(32, 64, device=dev).half()
        ref = R.paged_attention_ref(q, kc, vc, bt, ctx, 16)
        for PS in (64, 128, 256):
            for ver in (1, 2, 3):
                record(f"attn runtime ctx{ctx} PS{PS} spare8 v{ver}", run_attn(q, kc, vc, bt, ctx, PS, ver, spare=8), ref, 2e-3)


def test_fused():
    for H in (2048,):
        w = (1 + 0.1 * torch.randn(H, device=dev)).half()
        x = (torch.randn(H, device=dev) * 3).half(); d = torch.randn(H, device=dev).half()
        xr, outr = R.add_rmsnorm_ref(x.clone(), d, w, 1e-5)
        xk, out = x.clone(), torch.empty_like(x)
        K_.add_rmsnorm(xk, d, w, out, 1e-5)
        record("add_rmsnorm residual", xk, xr, 0.0)
        record("add_rmsnorm out", out, outr, 2e-3)
        xk = x.clone(); K_.add_rmsnorm(xk, None, w, out, 1e-5)
        record("rmsnorm (no residual)", out, R.add_rmsnorm_ref(x, None, w, 1e-5)[1], 2e-3)
    gu = (torch.randn(2 * 5632, device=dev) * 2).half(); out = torch.empty(5632, dtype=torch.half, device=dev)
    K_.swiglu(gu, out)
    record("swiglu", out, R.swiglu_ref(gu), 2e-3)
    # rope + append
    n_q, n_kv, page = 32, 4, 16
    cos, sin = (t.to(dev) for t in R.rope_tables(4096))
    for pos in (0, 1, 37, 2047, 4095):
        kc, vc, bt = make_cache(4096, n_kv, page)
        kr_c, vr_c = kc.clone(), vc.clone()
        qkv = torch.randn((n_q + 2 * n_kv) * 64, device=dev).half()
        qref = R.rope_append_ref(qkv.clone(), cos, sin, pos, kr_c, vr_c, bt, n_q, n_kv, 64, page)
        qk = qkv.clone()
        K_.rope_append(qk, cos, sin, torch.tensor([pos], dtype=torch.int32, device=dev), bt, kc, vc, n_q, n_kv)
        record(f"rope q pos{pos}", qk[: n_q * 64].view(n_q, 64), qref, 2e-3)
        record(f"rope k-cache pos{pos}", kc, kr_c, 2e-3)
        record(f"rope v-cache pos{pos}", vc, vr_c, 0.0)


if __name__ == "__main__":
    only = sys.argv[1:] or ["shape1", "real_gemv", "attention", "fused"]
    for t in only:
        globals()[f"test_{t}"]()
    print(f"\n{len(rows) - len(failures)}/{len(rows)} passed")
    Results().put("accuracy_kernels", rows)
    sys.exit(1 if failures else 0)
