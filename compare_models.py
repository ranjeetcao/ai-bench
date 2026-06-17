#!/usr/bin/env python3
"""Cross-model comparison built from each model's results/ subfolder."""
import glob, json, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))

# display order + (label, arch note)
META = {
    "phi-4-awq":           ("Phi-4 (dense ~14B)",      "dense ~14B"),
    "gemma-4-26b-a4b-awq": ("Gemma-4-26B-A4B (MoE)",   "MoE ~4B act"),
    "qwen3.6-27b-awq":     ("Qwen3.6-27B (dense)",     "dense 27B"),
    "qwen3-32b-awq":       ("Qwen3-32B (dense)",       "dense 32B"),
    "qwen3.6-35b-a3b-awq": ("Qwen3.6-35B-A3B (MoE)",   "MoE ~3B act"),
}

def g(d, *ks):
    for k in ks:
        v = d.get(k)
        if v is not None:
            return v
    return None

def load(sub):
    res = {}
    for p in glob.glob(os.path.join(ROOT, sub, "results", "in*_out*_c*.json")):
        m = re.search(r"in(\d+)_out(\d+)_c(\d+)", os.path.basename(p))
        if not m:
            continue
        i, o, c = (int(x) for x in m.groups())
        res[(i, o, c)] = json.load(open(p))
    return res

models = [s for s in META if load(s)]
shapes = [(256, 256), (1024, 512), (4096, 512)]

print("\n===== SINGLE-STREAM (c=1):  decode tok/s  /  TTFT p50 ms =====")
hdr = f"{'model':<26}" + "".join(f"| {f'{i}/{o}':>15} " for i, o in shapes)
print(hdr); print("-" * len(hdr))
for s in models:
    r = load(s); cells = ""
    for (i, o) in shapes:
        d = r.get((i, o, 1))
        if d:
            tpot = g(d, "mean_tpot_ms"); dec = 1000 / tpot if tpot else 0
            ttft = g(d, "median_ttft_ms", "p50_ttft_ms")
            cells += f"| {dec:5.0f} / {ttft:6.0f} "
        else:
            cells += f"| {'-':>15} "
    print(f"{META[s][0]:<26}{cells}")

print("\n===== PEAK AGGREGATE output tok/s (concurrency=64) =====")
hdr = f"{'model':<26}" + "".join(f"| {f'{i}/{o}':>12} " for i, o in shapes)
print(hdr); print("-" * len(hdr))
for s in models:
    r = load(s); cells = ""
    for (i, o) in shapes:
        d = r.get((i, o, 64))
        v = g(d, "output_throughput") if d else None
        cells += f"| {v:12.0f} " if v else f"| {'-':>12} "
    print(f"{META[s][0]:<26}{cells}")
print()
