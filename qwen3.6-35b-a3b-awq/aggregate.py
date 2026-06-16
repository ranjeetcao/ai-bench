#!/usr/bin/env python3
"""Aggregate vllm bench serve result JSONs into latency + throughput summaries."""
import glob, json, os, re

RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def g(d, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None

rows = []
for path in glob.glob(os.path.join(RESDIR, "in*_out*_c*.json")):
    m = re.search(r"in(\d+)_out(\d+)_c(\d+)", os.path.basename(path))
    if not m:
        continue
    inlen, outlen, conc = (int(x) for x in m.groups())
    with open(path) as f:
        d = json.load(f)
    mean_tpot = g(d, "mean_tpot_ms")
    rows.append({
        "in": inlen, "out": outlen, "c": conc,
        "completed": g(d, "completed"),
        "dur": g(d, "duration"),
        "req_tput": g(d, "request_throughput"),
        "out_tput": g(d, "output_throughput"),
        "tot_tput": g(d, "total_token_throughput"),
        "ttft_med": g(d, "median_ttft_ms", "p50_ttft_ms"),
        "ttft_p99": g(d, "p99_ttft_ms"),
        "tpot_mean": mean_tpot,
        "itl_mean": g(d, "mean_itl_ms"),
        "e2el_med": g(d, "median_e2el_ms", "p50_e2el_ms"),
        "decode_toks": (1000.0 / mean_tpot) if mean_tpot else None,
    })

rows.sort(key=lambda r: (r["in"], r["out"], r["c"]))

def fmt(v, nd=1):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

shapes = sorted({(r["in"], r["out"]) for r in rows})

print("\n================ SINGLE-STREAM LATENCY (concurrency = 1) ================")
print(f"{'shape in/out':>14} | {'TTFT p50':>9} | {'TTFT p99':>9} | {'TPOT':>8} | {'decode':>9} | {'E2E p50':>9}")
print(f"{'(tokens)':>14} | {'(ms)':>9} | {'(ms)':>9} | {'(ms/tok)':>8} | {'(tok/s)':>9} | {'(ms)':>9}")
print("-" * 78)
for (i, o) in shapes:
    for r in rows:
        if r["in"] == i and r["out"] == o and r["c"] == 1:
            print(f"{str(i)+'/'+str(o):>14} | {fmt(r['ttft_med']):>9} | {fmt(r['ttft_p99']):>9} | "
                  f"{fmt(r['tpot_mean'],2):>8} | {fmt(r['decode_toks']):>9} | {fmt(r['e2el_med'],0):>9}")

print("\n================ THROUGHPUT SCALING (per request shape) ================")
for (i, o) in shapes:
    print(f"\n  shape {i}/{o} tokens:")
    print(f"    {'conc':>4} | {'out tok/s':>10} | {'total tok/s':>11} | {'req/s':>6} | {'TTFT p50':>9} | {'per-stream':>10}")
    print(f"    {'':>4} | {'(decode)':>10} | {'(in+out)':>11} | {'':>6} | {'(ms)':>9} | {'tok/s':>10}")
    print("    " + "-" * 64)
    for r in rows:
        if r["in"] == i and r["out"] == o:
            print(f"    {r['c']:>4} | {fmt(r['out_tput']):>10} | {fmt(r['tot_tput']):>11} | "
                  f"{fmt(r['req_tput'],2):>6} | {fmt(r['ttft_med'],0):>9} | {fmt(r['decode_toks']):>10}")

print("\n(decode tok/s = 1000/TPOT = per-stream generation speed; out tok/s = aggregate across all concurrent streams)")
