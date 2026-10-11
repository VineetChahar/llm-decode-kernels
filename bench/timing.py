"""CUDA-event timing: 5 warmup + >= 30 timed iterations, per-iteration event pairs, median + IQR.

flush=True READS a 16 MB buffer (4x the T4's 4 MB L2) between iterations, outside the timed region,
so a weight matrix is never L2-resident from the previous iteration (the real model never reuses a
layer's weights within a token). It must be a read: a memset flush leaves L2 full of dirty lines whose
write-back then competes with the kernel under test (unlike the real decode step, where L2 holds clean
weight lines) — and it burns power, which makes the T4's clock throttling worse.

Clock gating: the T4 throttles hard under sustained load (observed 1590 -> 585 MHz SM clock). Every
measurement records NVML clocks/temperature/power/throttle reasons; if the SM clock sampled right after
the timed loop is below MIN_SM_MHZ, the GPU is left idle to cool and the measurement is retried. The
result is flagged `throttled` if it never recovered.
"""
import json, os, statistics, time
import torch

_FLUSH = None


MIN_SM_MHZ = 1400
COOLDOWN_S = 10
RETRIES = 2
COOLDOWN_BUDGET_S = 900  # total cool-down per process; past it, measure once and flag `throttled`
_cooled = 0.0


def _flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.ones(4 << 20, dtype=torch.float32, device="cuda")  # 16 MB
    _FLUSH.sum()  # read-only sweep (writes one scalar)


def clocks():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0))
        reasons = pynvml.nvmlDeviceGetCurrentClocksEventReasons(h)
        names = {0x2: "sw_power_cap", 0x4: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal",
                 0x80: "hw_power_brake", 0x1: "gpu_idle", 0x8: "sync_boost"}
        return {"sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                "mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                "reasons": [n for b, n in names.items() if reasons & b]}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def time_fn(fn, warmup=5, iters=50, flush=True, gate=True):
    """Clock-gated measurement: retried after a cool-down while the in-loop SM clock < MIN_SM_MHZ."""
    global _cooled
    best = None
    for attempt in range(RETRIES if gate else 1):
        r = _time_once(fn, warmup, iters, flush)
        r["attempt"] = attempt
        mhz = r["clocks_after"].get("sm_mhz", 0)
        if best is None or mhz > best["clocks_after"].get("sm_mhz", 0):
            best = r
        if not gate or mhz >= MIN_SM_MHZ or attempt == RETRIES - 1 or _cooled >= COOLDOWN_BUDGET_S:
            break
        _cooled += COOLDOWN_S
        time.sleep(COOLDOWN_S)
    best["throttled"] = best["clocks_after"].get("sm_mhz", 0) < MIN_SM_MHZ
    return best


def _time_once(fn, warmup, iters, flush):
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
    # Sample the clock while the GPU is still busy: a ~20 ms spin queued behind the timed loop. Reading after
    # synchronize() sees the idle clock (585 MHz on T4) and made every gated measurement "throttled" (full1 bug).
    torch.cuda._sleep(20_000_000)
    ca = clocks()
    torch.cuda.synchronize()
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
