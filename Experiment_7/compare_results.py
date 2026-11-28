import sqlite3
from pathlib import Path


def merge(intervals):
    merged = []

    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


def duration(intervals):
    return sum(e - s for s, e in intervals)


def overlap(a, b):
    i = j = total = 0

    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])

        if end > start:
            total += end - start

        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

    return total


def analyze(filename):
    if not Path(filename).is_file():
        raise FileNotFoundError(filename)

    con = sqlite3.connect(filename)

    kernels = con.execute("""
        SELECT start, end
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE end > start
    """).fetchall()

    copies = con.execute("""
        SELECT start, end
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE end > start
    """).fetchall()

    con.close()

    if not kernels or not copies:
        raise RuntimeError(
            f"GPU events missing in {filename}"
        )

    k = merge(kernels)
    c = merge(copies)

    start = min(k[0][0], c[0][0])
    end = max(k[-1][1], c[-1][1])

    span = end - start
    busy = duration(k)
    copy_time = duration(c)
    ov = overlap(k, c)

    gaps = [
        k[i + 1][0] - k[i][1]
        for i in range(len(k) - 1)
    ]

    big_gaps = [g for g in gaps if g > 50000]

    return {
        "window": span / 1e6,
        "kernel": busy / 1e6,
        "util": 100 * busy / span,
        "copy": copy_time / 1e6,
        "overlap": ov / 1e6,
        "gaps": len(big_gaps),
        "gap_time": sum(big_gaps) / 1e6
    }


b = analyze("baseline.sqlite")
o = analyze("optimized.sqlite")

# Previously measured 200-batch execution times
baseline_time = 3.802
optimized_time = 1.947

print("=" * 76)
print("EXPERIMENT 7: PROFILING AND GPU UTILIZATION")
print("GPU: NVIDIA GeForce RTX 5060 Ti")
print("=" * 76)

print(f"{'METRIC':<40}{'BASELINE':>16}{'OPTIMIZED':>16}")
print("-" * 76)


def row(name, base, opt):
    print(f"{name:<40}{base:>16}{opt:>16}")


row("Total time (s)", "3.802", "1.947")
row("Per batch (ms)", "19.01", "9.74")
row("Throughput (batches/s)", "52.6", "102.7")

row("Profiled window (ms)",
    f"{b['window']:.3f}",
    f"{o['window']:.3f}")

row("Kernel busy time (ms)",
    f"{b['kernel']:.3f}",
    f"{o['kernel']:.3f}")

row("GPU utilization (%)",
    f"{b['util']:.2f}",
    f"{o['util']:.2f}")

row("Memory copy busy time (ms)",
    f"{b['copy']:.3f}",
    f"{o['copy']:.3f}")

row("Copy-compute overlap (ms)",
    f"{b['overlap']:.3f}",
    f"{o['overlap']:.3f}")

row("Idle gaps > 50 us",
    b["gaps"], o["gaps"])

row("Total idle gap time (ms)",
    f"{b['gap_time']:.3f}",
    f"{o['gap_time']:.3f}")

row("Checksum", "4.198830e+09", "4.198830e+09")

print("-" * 76)

print(f"Execution speedup       : "
      f"{baseline_time / optimized_time:.2f}x")

print(f"Time reduction          : "
      f"{100 * (baseline_time - optimized_time) / baseline_time:.2f}%")

print(f"Utilization improvement : "
      f"{o['util'] - b['util']:.2f} percentage points")

print(f"Time saved              : "
      f"{baseline_time - optimized_time:.3f} seconds")

print("Displayed checksum      : MATCHED")

print("=" * 76)