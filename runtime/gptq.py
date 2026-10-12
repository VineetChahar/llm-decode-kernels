"""Sequential GPTQ calibration for a HF Llama model -> precomputed INT4 weights for the decode runtime.

Layer by layer: capture the inputs of decoder layer i (produced by the already-quantized layers 0..i-1), accumulate
the input Hessian H = sum x^T x of each projection, quantize with runtime.quant.quantize_gptq, write the dequantized
weights back into the HF layer and recompute its outputs as the next layer's inputs. q/k/v share one input (one H, one
GPTQ call on the concatenated [q;k;v] matrix — rows are independent, so this equals three calls) and so do gate/up.
The LM head is calibrated last on the final-norm outputs. The HF model's weights are restored afterwards.

Output keys match runtime.model: "layers.{i}.qkv|o|gu|down", "lm_head" -> (q uint8 [N,K], scale, zero [N,K/G] fp16).
Calibration text is WikiText-2 *train*; perplexity is measured on *test*.
"""
import time
import torch
from runtime.quant import quantize_gptq, dequantize


def calibration_windows(tok, n=128, L=512, seed=0):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - L, (n,), generator=g).tolist()
    bos = torch.tensor([tok.bos_token_id])
    return [torch.cat([bos, ids[s:s + L - 1]]) for s in starts]


class _Catch(Exception):
    pass


@torch.no_grad()
def calibrate(hf, tok, group=128, clip=False, n=128, L=512, lm_head=True, log=print):
    dev = next(hf.parameters()).device
    layers = hf.model.layers
    wins = calibration_windows(tok, n, L)

    # 1. inputs of layer 0 (+ the kwargs the model passes to every layer: rotary embeddings, mask, positions)
    inps, kw = [], {}
    orig0 = layers[0]

    class Catcher(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, hidden_states, *args, **kwargs):
            inps.append(hidden_states); kw.update(kwargs); raise _Catch

    layers[0] = Catcher(orig0)
    for w in wins:
        try:
            hf.model(w[None].to(dev), use_cache=False)
        except _Catch:
            pass
    layers[0] = orig0
    kw = {k: v for k, v in kw.items() if k not in ("past_key_values", "past_key_value", "use_cache")}

    def run(layer, x):
        o = layer(x, **kw)
        return o[0] if isinstance(o, tuple) else o

    saved, out = {}, {}
    t0 = time.time()
    for i, layer in enumerate(layers):
        a, m = layer.self_attn, layer.mlp
        targets = {"qkv": (a.q_proj, [a.q_proj, a.k_proj, a.v_proj]), "o": (a.o_proj, [a.o_proj]),
                   "gu": (m.gate_proj, [m.gate_proj, m.up_proj]), "down": (m.down_proj, [m.down_proj])}
        H = {k: None for k in targets}
        cnt = {k: 0 for k in targets}

        def hook(key):
            def f(mod, args, _out):
                x = args[0].reshape(-1, args[0].shape[-1]).float()
                H[key] = x.t() @ x if H[key] is None else H[key] + x.t() @ x
                cnt[key] += x.shape[0]
            return f

        hs = [mod.register_forward_hook(hook(k)) for k, (mod, _) in targets.items()]
        for x in inps:
            run(layer, x)
        for h in hs:
            h.remove()
        for key, (_, mods) in targets.items():
            W = torch.cat([mm.weight.data for mm in mods], 0)
            q, s, z = quantize_gptq(W, 2 * H[key] / cnt[key], group, clip=clip)
            out[f"layers.{i}.{key}"] = (q.cpu(), s.cpu(), z.cpu())
            deq = dequantize(q, s, z, group)
            r = 0
            for mm in mods:
                saved[id(mm)] = (mm, mm.weight.data.cpu().clone())  # .cpu() is a no-op on CPU
                mm.weight.data.copy_(deq[r:r + mm.weight.shape[0]]); r += mm.weight.shape[0]
        del H
        inps = [run(layer, x) for x in inps]  # next layer sees this layer's *quantized* outputs
        log(f"gptq layer {i}: {time.time() - t0:.0f}s")

    if lm_head:
        Hh, c = None, 0
        for x in inps:
            y = hf.model.norm(x).reshape(-1, x.shape[-1]).float()
            Hh = y.t() @ y if Hh is None else Hh + y.t() @ y
            c += y.shape[0]
        q, s, z = quantize_gptq(hf.lm_head.weight.data, 2 * Hh / c, group, clip=clip)
        out["lm_head"] = (q.cpu(), s.cpu(), z.cpu())

    for mm, w in saved.values():  # restore the fp16 reference model
        mm.weight.data.copy_(w.to(dev))
    log(f"gptq done: {time.time() - t0:.0f}s, {len(out)} matrices")
    return out
