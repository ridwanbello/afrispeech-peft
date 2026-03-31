"""
Quick script to get all dataset stats needed for paper writeup.
Usage:
    source ~/projects/afrispeech-project/analysis-env/bin/activate
    cd ~/projects/afrispeech-project
    python get_dataset_stats.py
"""
import pandas as pd
from datasets import load_from_disk

ds = load_from_disk("./afrispeech_arrow")

dfs = []
for split in ["train", "validation", "test"]:
    df = ds[split].remove_columns(["audio"]).to_pandas()
    df["split"] = split
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

print("=== OVERALL ===")
print(f"Total clips     : {len(df):,}")
print(f"Total hours     : {df['duration'].sum()/3600:.1f}")
print(f"Unique speakers : {df['speaker_id'].nunique():,}")
print(f"Unique accents  : {df['accent'].nunique():,}")
print(f"Countries       : {df['country'].nunique():,}")

print("\n=== SPLITS ===")
for split in ["train","validation","test"]:
    s = df[df["split"]==split]
    print(f"{split:12s}: {len(s):>6,} clips | {s['duration'].sum()/3600:.1f}h | {s['speaker_id'].nunique()} speakers")

print("\n=== DOMAINS ===")
for domain in ["general","clinical"]:
    s = df[df["domain"]==domain]
    print(f"{domain:12s}: {len(s):>6,} clips | {s['duration'].sum()/3600:.1f}h")

print("\n=== DURATION ===")
print(f"Mean   : {df['duration'].mean():.2f}s")
print(f"Median : {df['duration'].median():.2f}s")
print(f"Std    : {df['duration'].std():.2f}s")
print(f"Min    : {df['duration'].min():.2f}s")
print(f"Max    : {df['duration'].max():.2f}s")

print("\n=== DEMOGRAPHICS ===")
print("Gender:")
print(df["gender"].value_counts().to_string())
print("\nAge group:")
print(df["age_group"].value_counts().to_string())

print("\n=== TOP 10 ACCENTS (by clip count) ===")
print(df["accent"].value_counts().head(10).to_string())

print("\n=== TOP 10 COUNTRIES ===")
print(df["country"].value_counts().head(10).to_string())
