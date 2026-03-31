"""
Cleaning pipeline impact analysis for AfriSpeech-200 paper.

Shows:
- How many clips removed by each filter step
- Breakdown by domain (general, clinical, all)
- Before vs after clip counts per split

Usage:
    source ~/projects/afrispeech-project/analysis-env/bin/activate
    cd ~/projects/afrispeech-project
    python 04_cleaning_pipeline_analysis.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_from_disk

DATA_PATH  = "./afrispeech_arrow"
OUTPUT_DIR = "./figures"
TABLE_DIR  = "./tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Cleaning pipeline
# ─────────────────────────────────────────────────────────────────
_INAUDIBLE = re.compile(
    r"\b(inaudible|inaudiable|inauidble|inauidible|inauible|inaudibe"
    r"|inaudibel|inaudilbe|inudible|inaudiible|inaudiblee|inuadible"
    r"|inaudbile|inauidble|inaudoible|inaudicble|inaudilbe|inaudible"
    r"|inaudivle|inaudinle|inaduible|inaudibl|inaudbible|inuadibale"
    r"|inauidbe|inaudibble|inaduible|inauddible|inauudible|inauidble"
    r"|inaudibale|inauidible)\b",
    re.IGNORECASE,
)
_FILLERS  = re.compile(r"\b(uh+|um+|hmm+|mmhmm|mhm|hm+|ugh+|ah+)\b", re.IGNORECASE)
_SPECIAL  = re.compile(r"\[(UNK|PAD)\]", re.IGNORECASE)
_DISALLOW = re.compile(r"[^a-zA-Z0-9\s?()\:\-\+<>\.\/\'\[\]]")

def apply_pipeline(texts):
    """Return per-filter removal counts."""
    n = len(texts)
    counts = {"original": n}

    mask_null  = texts.isna() | (texts.str.strip() == "")
    counts["null_empty"] = int(mask_null.sum())

    mask_inaud = texts.str.contains(_INAUDIBLE, regex=True, na=False)
    counts["inaudible"] = int(mask_inaud.sum())

    cleaned = texts.copy()
    cleaned = cleaned.apply(lambda t: _SPECIAL.sub("", str(t)) if pd.notna(t) else "")
    cleaned = cleaned.apply(lambda t: _INAUDIBLE.sub("", t))
    cleaned = cleaned.apply(lambda t: _FILLERS.sub("", t))
    cleaned = cleaned.apply(lambda t: _DISALLOW.sub(" ", t))
    cleaned = cleaned.apply(lambda t: re.sub(r"\s+", " ", t).strip().lower())

    mask_short = cleaned.str.len() < 5
    mask_long  = cleaned.str.len() > 300
    counts["too_short"] = int((mask_short & ~mask_null).sum())
    counts["too_long"]  = int((mask_long  & ~mask_null).sum())

    kept = ~(mask_null | mask_short | mask_long)
    counts["kept"] = int(kept.sum())
    counts["total_removed"] = n - counts["kept"]
    counts["pct_removed"] = round(counts["total_removed"] / n * 100, 1)
    return counts


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
df_all = pd.concat(dfs, ignore_index=True)
print(f"Total clips: {len(df_all):,}")


# ─────────────────────────────────────────────────────────────────
# Run pipeline on each domain × split
# ─────────────────────────────────────────────────────────────────
rows = []
for split in ["train", "validation", "test", "all"]:
    for domain in ["general", "clinical", "all"]:
        subset = df_all.copy()
        if split != "all":
            subset = subset[subset["split"] == split]
        if domain != "all":
            subset = subset[subset["domain"] == domain]
        res = apply_pipeline(subset["transcript"])
        res["split"]  = split
        res["domain"] = domain
        rows.append(res)
        if split == "all":
            print(f"  {domain:10s} | {res['original']:>6,} clips → "
                  f"{res['kept']:>6,} kept | "
                  f"removed: null={res['null_empty']}, "
                  f"inaudible={res['inaudible']}, "
                  f"short={res['too_short']}, "
                  f"long={res['too_long']} "
                  f"({res['pct_removed']}%)")

results_df = pd.DataFrame(rows)
results_df.to_csv(f"{TABLE_DIR}/cleaning_pipeline_stats.csv", index=False)
print(f"\nSaved: {TABLE_DIR}/cleaning_pipeline_stats.csv")


# ─────────────────────────────────────────────────────────────────
# Figure 1 — Stacked bar: removed per filter per domain
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

all_results = results_df[results_df["split"] == "all"]
domains     = ["general", "clinical", "all"]
labels      = ["General", "Clinical", "All"]
x           = np.arange(len(domains))
width       = 0.5

filters = [
    ("null_empty",  "Null / empty",      "#888780"),
    ("inaudible",   "Inaudible marker",  "#E24B4A"),
    ("too_short",   "Too short (<5 chars)","#378ADD"),
    ("too_long",    "Too long (>300 chars)","#1D9E75"),
]

bottoms = np.zeros(len(domains))
for key, label, color in filters:
    vals = [int(all_results[all_results["domain"] == d][key].values[0]) for d in domains]
    bars = ax.bar(x, vals, width, bottom=bottoms, label=label,
                  color=color, edgecolor="white", linewidth=0.5)
    # Add value labels inside bars if large enough
    for bar, val, bot in zip(bars, vals, bottoms):
        if val > 100:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    f"{val:,}", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="500")
    bottoms += np.array(vals)

# Total removed label on top of each bar
for i, d in enumerate(domains):
    row = all_results[all_results["domain"] == d].iloc[0]
    ax.text(i, bottoms[i] + 50, f"{row['total_removed']:,}\n({row['pct_removed']}%)",
            ha="center", va="bottom", fontsize=9, color="black")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Clips removed", fontsize=11)
ax.set_title("Cleaning Pipeline — Clips Removed by Filter and Domain", fontsize=12, fontweight="normal")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.2, linewidth=0.5)
ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc", loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cleaning_pipeline_stacked.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/cleaning_pipeline_stacked.png")
plt.close()


# ─────────────────────────────────────────────────────────────────
# Figure 2 — Before vs after per split × domain
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

splits  = ["train", "validation", "test"]
domains = ["general", "clinical", "all"]
x       = np.arange(len(domains))
width   = 0.35
colors  = {"before": "#B5D4F4", "after": "#378ADD"}

for ax, split in zip(axes, splits):
    subset = results_df[results_df["split"] == split]
    before = [int(subset[subset["domain"] == d]["original"].values[0]) for d in domains]
    after  = [int(subset[subset["domain"] == d]["kept"].values[0])    for d in domains]

    ax.bar(x - width/2, before, width, label="Before cleaning",
           color=colors["before"], edgecolor="#185FA5", linewidth=0.5)
    ax.bar(x + width/2, after,  width, label="After cleaning",
           color=colors["after"],  edgecolor="#185FA5", linewidth=0.5)

    # Value labels
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - width/2, b + b*0.01, f"{b:,}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, a + a*0.01, f"{a:,}", ha="center", va="bottom", fontsize=8)

    ax.set_title(split.capitalize(), fontsize=12, fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(["General", "Clinical", "All"], fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    if split == "train":
        ax.set_ylabel("Number of clips", fontsize=11)
    if split == "validation":
        ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")

fig.suptitle("Clip Counts Before and After Cleaning — by Split and Domain",
             fontsize=12, fontweight="normal", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cleaning_before_after.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/cleaning_before_after.png")
plt.close()

print("\nAll done!")
