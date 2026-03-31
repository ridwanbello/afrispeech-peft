"""
Plot audio duration histogram for AfriSpeech-200.

Usage:
    source ~/projects/afrispeech-project/analysis-env/bin/activate
    cd ~/projects/afrispeech-project
    python plot_duration_histogram.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_from_disk

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
DATA_PATH  = "./afrispeech_arrow"
OUTPUT_DIR = "./figures"
CLIP_AT    = 60   # clip x-axis at 60s for readability (outliers beyond shown separately)

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
ds = load_from_disk(DATA_PATH)

dfs = []
for split in ["train", "validation", "test"]:
    df = ds[split].remove_columns(["audio"]).to_pandas()
    df["split"] = split
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

durations = df["duration"]
mean_dur   = durations.mean()
median_dur = durations.median()
std_dur    = durations.std()
pct_over30 = (durations > 30).mean() * 100

print(f"Total clips : {len(durations):,}")
print(f"Mean        : {mean_dur:.2f}s")
print(f"Median      : {median_dur:.2f}s")
print(f"Std         : {std_dur:.2f}s")
print(f"Min / Max   : {durations.min():.2f}s / {durations.max():.2f}s")
print(f"IQR         : {durations.quantile(0.25):.2f}s – {durations.quantile(0.75):.2f}s")
print(f"Clips > 30s : {pct_over30:.1f}%")

# ─────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

clipped   = durations[durations <= CLIP_AT]
bins      = np.arange(0, CLIP_AT + 2, 2)

n, edges, patches = ax.hist(
    clipped, bins=bins,
    color="#378ADD", edgecolor="#185FA5", linewidth=0.4,
    label="Clip count",
)

# Color bins beyond 30s differently
for patch, left in zip(patches, edges[:-1]):
    if left >= 30:
        patch.set_facecolor("#F0997B")
        patch.set_edgecolor("#D85A30")

# Mean and median lines
ax.axvline(mean_dur,   color="#E24B4A", linewidth=1.8, linestyle="-",  label=f"Mean ({mean_dur:.2f}s)")
ax.axvline(median_dur, color="#1D9E75", linewidth=1.8, linestyle="--", label=f"Median ({median_dur:.2f}s)")

# Annotation for clipped outliers
n_outliers = (durations > CLIP_AT).sum()
if n_outliers > 0:
    ax.text(
        CLIP_AT - 0.5, ax.get_ylim()[1] * 0.95,
        f"{n_outliers:,} clips\n>{CLIP_AT}s\n(not shown)",
        ha="right", va="top", fontsize=9,
        color="#D85A30",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#F0997B", linewidth=0.8),
    )

# Formatting
ax.set_xlabel("Duration (seconds)", fontsize=12)
ax.set_ylabel("Number of clips", fontsize=12)
ax.set_title("AfriSpeech-200 — Audio Duration Distribution", fontsize=13, fontweight="normal", pad=12)
ax.set_xlim(0, CLIP_AT)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.tick_params(axis="both", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)

outlier_patch = mpatches.Patch(facecolor="#F0997B", edgecolor="#D85A30", linewidth=0.5, label="Clips 30–60s (truncated in preprocessing)")
ax.legend(handles=[
    mpatches.Patch(facecolor="#378ADD", edgecolor="#185FA5", linewidth=0.5, label="Clips 0–30s"),
    outlier_patch,
    plt.Line2D([0], [0], color="#E24B4A", linewidth=1.8, label=f"Mean ({mean_dur:.2f}s)"),
    plt.Line2D([0], [0], color="#1D9E75", linewidth=1.8, linestyle="--", label=f"Median ({median_dur:.2f}s)"),
], fontsize=10, framealpha=0.8, edgecolor="#cccccc")

# Stats box
stats_text = (
    f"N = {len(durations):,}\n"
    f"Mean = {mean_dur:.2f}s\n"
    f"Median = {median_dur:.2f}s\n"
    f"Std = {std_dur:.2f}s\n"
    f"IQR = {durations.quantile(0.25):.1f}–{durations.quantile(0.75):.1f}s"
)
ax.text(
    0.62, 0.97, stats_text,
    transform=ax.transAxes, fontsize=9,
    va="top", ha="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", linewidth=0.8),
)

plt.tight_layout()
out_path = f"{OUTPUT_DIR}/duration_histogram.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nSaved to {out_path}")
plt.show()
