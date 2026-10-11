"""CUDA-event timing: 5 warmup + >= 30 timed iterations, per-iteration event pairs, median + IQR.

flush=True memsets a 16 MB buffer (4x the T4's 4 MB L2) between iterations, outside the timed region,
so a weight matrix is never L2-resident from the previous iteration (the real model never reuses a
layer's weights within a token).
"""
import json, os, statistics, time
import torch

_FLUSH = None


def _flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    _FLUSH.zero_()


def clocks():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0))
        return {"sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                "mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def time_fn(fn, warmup=5, iters=50, flush=True):
    """Returns dict(median_us, p25_us, p75_us, iqr_us, n, clocks_before, clocks_after)."""
    for _ in range(warmup):
        if flush: _flush()
        fn()
    torch.cuda.synchronize()
    cb = clocks()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush: _flush()
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    ca = clocks()
    t = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    q = statistics.quantiles(t, n=4)
    return {"median_us": statistics.median(t), "p25_us": q[0], "p75_us": q[2], "iqr_us": q[2] - q[0],
            "min_us": t[0], "n": iters, "clocks_before": cb, "clocks_after": ca}


class Results:
    """Merge-on-write JSON store: each bench section writes its own key into RESULTS_DIR/results.json."""

    def __init__(self, path=None):
        d = os.environ.get("RESULTS_DIR", ".")
        self.path = path or os.path.join(d, "results.json")
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def put(self, key, value):
        self.data[key] = value
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1, default=float)
        os.replace(tmp, self.path)


def env_info():
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "sm": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
            "mem_gb": p.total_memory / 2**30, "l2_mb": getattr(p, "L2_cache_size", 0) / 2**20,
            "torch": torch.__version__, "cuda": torch.version.cuda, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "clocks": clocks()}
