"""
Experiment 5 - Neural Network Layers in CUDA
Graph generator.

Reads  : ../results/timing_results.csv
         ../results/first_10_outputs.csv
Writes : ../graphs/*.png
         ../results/summary_metrics.csv

Usage  : python scripts/generate_graphs.py   (run from the Experiment_5 folder)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "graphs")
os.makedirs(OUT, exist_ok=True)

t = pd.read_csv(os.path.join(RES, "timing_results.csv"))
outs = pd.read_csv(os.path.join(RES, "first_10_outputs.csv"))
me = t.iloc[0]          # measured (RTX 5060 Ti)
ref = t.iloc[1]         # lab manual reference (RTX 4500 Ada)

C_CPU, C_KER, C_E2E = "#DC2626", "#16A34A", "#2563EB"
C_COPY, C_REF = "#F59E0B", "#9CA3AF"
DARK = "#1F2937"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- summary --
cpu, ker, e2e = me.CPU_ms, me.GPU_kernel_only_ms, me.GPU_end_to_end_ms
overhead = e2e - ker
summary = pd.DataFrame([
    ["CPU average compute time", cpu, "ms"],
    ["GPU kernel-only time", ker, "ms"],
    ["GPU end-to-end time", e2e, "ms"],
    ["Kernel-only speedup (CPU / kernel-only, by hand)", round(cpu / ker, 4), "x"],
    ["End-to-end speedup (CPU / end-to-end, by hand)", round(cpu / e2e, 4), "x"],
    ["End-to-end minus kernel-only (copies + launch/sync overhead)", round(overhead, 6), "ms"],
    ["Share of end-to-end time that is not kernel work", round(overhead / e2e * 100, 1), "%"],
    ["Maximum absolute CPU/GPU difference", me.Max_abs_difference, ""],
], columns=["Metric", "Value", "Unit"])
summary.to_csv(os.path.join(RES, "summary_metrics.csv"), index=False)

# ------------------------------------------- 01 network and kernel order --
fig, ax = plt.subplots(figsize=(13, 3.0))
ax.set_xlim(0, 13)
ax.set_ylim(0.8, 3.6)
ax.axis("off")
boxes = [
    ("X\ninput", "256 x 128", "#E5E7EB"),
    ("Matrix multiply\n+ bias  (W1, b1)", "256 x 64", "#DCFCE7"),
    ("ReLU", "256 x 64", "#DCFCE7"),
    ("Matrix multiply\n+ bias  (W2, b2)", "256 x 10", "#DBEAFE"),
    ("Sigmoid", "256 x 10", "#DBEAFE"),
    ("Y\noutput", "256 x 10", "#E5E7EB"),
]
w, h, gap, x0, y0 = 1.75, 1.5, 0.42, 0.25, 1.35
for i, (name, shape, col) in enumerate(boxes):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=col, ec=DARK, lw=1.2))
    ax.text(x + w / 2, y0 + h * 0.62, name, ha="center", va="center", fontsize=11, weight="bold", color=DARK)
    ax.text(x + w / 2, y0 + h * 0.2, shape, ha="center", va="center", fontsize=10, color=DARK)
    if 1 <= i <= 4:
        ax.text(x + w / 2, y0 - 0.3, f"kernel {i}", ha="center", fontsize=10, color="#6B7280")
    if i < len(boxes) - 1:
        ax.annotate("", xy=(x + w + gap - 0.03, y0 + h / 2), xytext=(x + w + 0.03, y0 + h / 2),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color=DARK))
ax.text(x0 + 1 * (w + gap) + w + gap / 2, 3.25, "Layer 1", ha="center", fontsize=12, weight="bold", color="#15803D")
ax.text(x0 + 3 * (w + gap) + w + gap / 2, 3.25, "Layer 2", ha="center", fontsize=12, weight="bold", color="#1D4ED8")
ax.set_title("Fig. 1 - The two-layer forward pass and the order of the four CUDA kernels", loc="left", fontsize=12)
save(fig, "01_network_forward_pass.png")

# ---------------------------------------------------- 02 execution time --
fig, ax = plt.subplots(figsize=(8, 4.8))
labels = ["CPU", "GPU kernel-only", "GPU end-to-end"]
vals = [cpu, ker, e2e]
bars = ax.bar(labels, vals, color=[C_CPU, C_KER, C_E2E], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.6f} ms", ha="center", fontsize=11)
ax.set_ylabel("Average time per forward pass (ms)")
ax.set_ylim(0, cpu * 1.18)
ax.set_title("Fig. 2 - Execution time (average of 20 repeats, RTX 5060 Ti)", loc="left", fontsize=12)
save(fig, "02_execution_time_comparison.png")

# ----------------------------------------------------------- 03 speedup --
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sp = [me.Kernel_only_speedup_x, me.End_to_end_speedup_x]
bars = ax.bar(["Kernel-only", "End-to-end"], sp, color=[C_KER, C_E2E], width=0.5)
for b, v in zip(bars, sp):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}x", ha="center", fontsize=12, weight="bold")
ax.axhline(1, color=DARK, ls="--", lw=1)
ax.text(1.32, 1.3, "1x = same speed as CPU", ha="right", fontsize=9, color=DARK)
ax.set_ylabel("Speedup over CPU (x)")
ax.set_ylim(0, max(sp) * 1.2)
ax.set_title("Fig. 3 - GPU speedup over the CPU", loc="left", fontsize=12)
save(fig, "03_speedup_kernel_vs_end_to_end.png")

# ------------------------------------------- 04 end-to-end time breakdown --
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.barh([0], [ker], color=C_KER, height=0.5, label=f"Kernels: {ker:.4f} ms ({ker / e2e * 100:.1f} %)")
ax.barh([0], [overhead], left=[ker], color=C_COPY, height=0.5,
        label=f"Copies + launch/sync overhead: {overhead:.4f} ms ({overhead / e2e * 100:.1f} %)")
ax.set_yticks([0])
ax.set_yticklabels(["GPU end-to-end\n(0.264380 ms)"])
ax.set_xlabel("Time (ms)")
ax.set_xlim(0, e2e * 1.05)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1, frameon=False)
ax.set_title("Fig. 4 - Where the GPU end-to-end time goes", loc="left", fontsize=12)
save(fig, "04_gpu_end_to_end_breakdown.png")

# ------------------------------------------------------- 05 correctness --
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(outs.Index, outs.Lab_manual_reference, "o", ms=13, mfc="none", mec=C_REF, mew=2,
        label="Lab manual reference (RTX 4500 Ada)")
ax.plot(outs.Index, outs.Measured_GPU, "o", ms=6, color=C_E2E, label="Measured GPU output (RTX 5060 Ti)")
ax.set_xticks(outs.Index)
ax.set_xlabel("Output element index")
ax.set_ylabel("Sigmoid output value")
ax.set_ylim(0.49, 0.525)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.text(9, 0.4915, f"Max |CPU - GPU| = {me.Max_abs_difference:.6f}  ->  MATCH", ha="right",
        fontsize=10, color="#15803D", weight="bold")
ax.set_title("Fig. 5 - First 10 output values: identical to the manual's reference", loc="left", fontsize=12)
save(fig, "05_output_validation.png")

# ------------------------------------------- 06 measured vs the reference --
fig, ax = plt.subplots(figsize=(9, 4.8))
cats = ["CPU", "GPU kernel-only", "GPU end-to-end"]
mine = [cpu, ker, e2e]
theirs = [ref.CPU_ms, ref.GPU_kernel_only_ms, ref.GPU_end_to_end_ms]
x = range(len(cats))
b1 = ax.bar([i - 0.2 for i in x], mine, 0.4, color=C_E2E, label="This experiment (RTX 5060 Ti)")
b2 = ax.bar([i + 0.2 for i in x], theirs, 0.4, color=C_REF, label="Lab manual (RTX 4500 Ada)")
for bars_, vals_ in ((b1, mine), (b2, theirs)):
    for b, v in zip(bars_, vals_):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(cats)
ax.set_ylabel("Average time (ms)")
ax.set_ylim(0, cpu * 1.18)
ax.legend(frameon=False)
ax.set_title(f"Fig. 6 - Measured times vs the manual's reference run\n"
             f"(speedups: kernel-only {me.Kernel_only_speedup_x:.2f}x vs {ref.Kernel_only_speedup_x:.2f}x, "
             f"end-to-end {me.End_to_end_speedup_x:.2f}x vs {ref.End_to_end_speedup_x:.2f}x)",
             loc="left", fontsize=12)
save(fig, "06_measured_vs_manual_reference.png")

print("Wrote 6 graphs to", OUT)
print("Wrote", os.path.join(RES, "summary_metrics.csv"))
