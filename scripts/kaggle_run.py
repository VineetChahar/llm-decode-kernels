#!/usr/bin/env python3
"""Pack the repo, push it as a private Kaggle GPU script kernel, poll, pull outputs into runs/<UTC>/.

usage: kaggle_run.py --cmd "bash scripts/run_all.sh" [--accel NvidiaTeslaT4] [--slug llm-decode-kernels]

* polls `kaggle kernels status` every 30 s
* pulls everything the run wrote to /kaggle/working (RESULTS_DIR) into runs/<UTC timestamp>/
* exits non-zero if the kernel failed, printing the last 100 log lines
Nothing is pushed anywhere except the private Kaggle kernel itself.
"""
import argparse, base64, datetime, io, json, re, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = str(ROOT / ".venv/bin/kaggle") if (ROOT / ".venv/bin/kaggle").exists() else "kaggle"
INCLUDE = ["kernels", "runtime", "reference", "bench", "scripts", "tests", "configs"]

DRIVER = r'''
import base64, io, os, subprocess, sys, tarfile, shutil
PAYLOAD = "__PAYLOAD__"
CMD = __CMD__
W = "/kaggle/working"
os.makedirs(W, exist_ok=True)
repo = "/kaggle/repo"
shutil.rmtree(repo, ignore_errors=True); os.makedirs(repo)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz").extractall(repo)
env = dict(os.environ, RESULTS_DIR=W, PYTHONUNBUFFERED="1", TORCH_EXTENSIONS_DIR="/kaggle/torch_ext",
           CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
log = open(os.path.join(W, "run.log"), "w")
def run(c):
    print("$ " + c, flush=True); log.write("$ " + c + "\n"); log.flush()
    p = subprocess.Popen(c, shell=True, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write(line); log.write(line); log.flush()
    p.wait(); return p.returncode
run("nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv; nvcc --version | tail -2")
rc = run(CMD)
log.write("EXIT_CODE=%d\n" % rc); log.close()
print("EXIT_CODE=%d" % rc)
sys.exit(rc)
'''


def pack():
    buf = io.BytesIO()
    skip = lambda ti: None if ("__pycache__" in ti.name or ti.name.endswith(".pyc")) else ti
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in INCLUDE:
            if (ROOT / name).exists():
                tf.add(ROOT / name, arcname=name, filter=skip)
    return base64.b64encode(buf.getvalue()).decode()


def sh(args):
    return subprocess.run([KAGGLE] + args, capture_output=True, text=True)


def push(ref, a):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "main.py").write_text(DRIVER.replace("__PAYLOAD__", pack()).replace("__CMD__", repr(a.cmd)))
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": ref, "title": a.slug, "code_file": "main.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}))
        r = sh(["kernels", "push", "-p", str(td), "-t", str(a.timeout), "--accelerator", a.accel])
        print(r.stdout.strip(), r.stderr.strip())
        if r.returncode != 0 or "error" in (r.stdout + r.stderr).lower():
            sys.exit("push failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="")
    ap.add_argument("--attach", action="store_true", help="don't push; poll + pull the kernel's current run")
    ap.add_argument("--accel", default="NvidiaTeslaT4")
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--slug", default="llm-decode-kernels")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    user = re.search(r"username: (\S+)", sh(["config", "view"]).stdout).group(1)
    ref = f"{user}/{a.slug}"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "runs" / (stamp + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    if a.attach:
        print("attaching to running kernel", ref)
    elif not a.cmd:
        sys.exit("--cmd required unless --attach")
    else:
        push(ref, a)
    (out / "meta.json").write_text(json.dumps({"ref": ref, "cmd": a.cmd, "accel": a.accel, "utc": stamp}))

    t0 = time.time(); status = ""
    while True:
        time.sleep(a.poll)
        s = sh(["kernels", "status", ref]); txt = (s.stdout + s.stderr).strip()
        m = re.search(r'status "?([\w.]+)"?', txt); status = m.group(1) if m else txt
        print(f"[{int(time.time() - t0):5d}s] {status}", flush=True)
        if any(k in status.upper() for k in ("COMPLETE", "ERROR", "CANCEL")):
            break

    r = sh(["kernels", "output", ref, "-p", str(out), "-o"])
    print(r.stdout[-300:], r.stderr[-300:])
    log = out / "run.log"
    text = log.read_text(errors="ignore") if log.exists() else ""
    if not text:  # fall back to the kernel's own log file
        for p in out.glob("*.log"):
            text = p.read_text(errors="ignore")
    ok = "COMPLETE" in status.upper() and "EXIT_CODE=0" in text[-3000:]
    if not ok:
        print(f"KERNEL FAILED ({status}); last 100 log lines:")
        print("".join(text.splitlines(True)[-100:]))
        sys.exit(1)
    print("".join(text.splitlines(True)[-15:]))
    print("OK ->", out)


if __name__ == "__main__":
    main()
