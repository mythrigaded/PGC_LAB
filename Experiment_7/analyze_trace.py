import collections
import json
import sys

GAP_US = 50

COMPUTE_HINTS = (
    "gemm", "sgemm", "cutlass",
    "cublas", "conv", "wmma"
)


def merge(iv):
    iv = sorted(iv)
    out = []

    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])

    return out


def total(iv):
    return sum(e - s for s, e in iv)


def overlap(a, b):
    i = j = t = 0

    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])

        if e > s:
            t += e - s

        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

    return t


def label(name):
    low = name.lower()

    if any(h in low for h in COMPUTE_HINTS):
        return "compute-bound"

    return "memory-bound"


def analyze(path):
    with open(path) as f:
        data = json.load(f)

    events = data["traceEvents"] if isinstance(
        data, dict
    ) else data

    kern, copy = [], []

    knames = collections.defaultdict(
        lambda: [0, 0.0]
    )

    cnames = collections.defaultdict(
        lambda: [0, 0.0]
    )

    for e in events:
        if e.get("ph") != "X":
            continue

        cat = e.get("cat", "")
        s = e["ts"]
        d = e.get("dur", 0)

        if cat == "kernel":
            kern.append([s, s + d])
            knames[e["name"][:60]][0] += 1
            knames[e["name"][:60]][1] += d

        elif cat in ("gpu_memcpy", "gpu_memset"):
            copy.append([s, s + d])
            cnames[e["name"][:60]][0] += 1
            cnames[e["name"][:60]][1] += d

    if not kern:
        print(f"{path}: No GPU kernels found")
        return None

    k = merge(kern)
    c = merge(copy)

    start = min(
        k[0][0], c[0][0] if c else k[0][0]
    )

    end = max(
        k[-1][1], c[-1][1] if c else k[-1][1]
    )

    span = end - start
    busy = total(k)

    gaps = [
        k[i + 1][0] - k[i][1]
        for i in range(len(k) - 1)
    ]

    big = [g for g in gaps if g > GAP_US]

    print("=" * 72)
    print(path)
    print("=" * 72)

    print(f"Profiled window       : {span / 1000:.2f} ms")
    print(f"Kernel busy time      : {busy / 1000:.2f} ms")
    print(f"GPU utilization       : {100 * busy / span:.1f} %")
    print(f"Copy busy time        : {total(c) / 1000:.2f} ms")
    print(f"Copy-compute overlap  : {overlap(k, c) / 1000:.2f} ms")
    print(f"Idle gaps > 50 us     : {len(big)}")
    print(f"Total idle gap time   : {sum(big) / 1000:.2f} ms")

    print("\nKERNEL CLASSIFICATION (NAME-BASED)")
    print("-" * 72)

    print(f"{'Calls':>6} {'Total ms':>10} {'Label':<15} Name")

    for n, (cnt, dur) in sorted(
        knames.items(),
        key=lambda x: -x[1][1]
    ):
        print(
            f"{cnt:6d} {dur / 1000:10.2f} "
            f"{label(n):<15} {n}"
        )

    print("\nMEMORY COPIES")
    print("-" * 72)

    for n, (cnt, dur) in sorted(
        cnames.items(),
        key=lambda x: -x[1][1]
    ):
        print(f"{cnt:6d} {dur / 1000:10.2f} {n}")

    print()

    return span, busy, total(c), overlap(k, c), len(big)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python analyze_trace.py trace_*.json"
        )

    rows = []

    for p in sys.argv[1:]:
        r = analyze(p)

        if r:
            rows.append((p, r))

    if len(rows) > 1:
        print("=" * 72)
        print("BASELINE VS OPTIMIZED SUMMARY")
        print("=" * 72)

        print(
            f"{'File':<24}{'Window ms':>10}"
            f"{'Util %':>10}{'Overlap ms':>12}"
            f"{'Gaps':>8}"
        )

        for p, (span, busy, _, ov, g) in rows:
            print(
                f"{p:<24}{span / 1000:10.1f}"
                f"{100 * busy / span:10.1f}"
                f"{ov / 1000:12.1f}{g:8d}"
            )


if __name__ == "__main__":
    main()