"""
Preprocess AfriSpeech-200 for Whisper-medium fine-tuning.
Runs locally — no GPU needed, CPU only.

Processes all 3 domains: general, clinical, all
Saves preprocessed datasets to ./preprocessed/

Usage:
    source ~/projects/afrispeech-project/afrispeech-env/bin/activate
    cd ~/projects/afrispeech-project
    python preprocess_whisper_local.py
"""

import os
import re
import types
import numpy as np
from datasets import load_from_disk, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "openai/whisper-medium"
DATA_PATH     = "./afrispeech_arrow"
OUTPUT_BASE   = "./preprocessed"
SAMPLING_RATE = 16_000
MAX_AUDIO_SEC = 30
MAX_INPUT_LEN = SAMPLING_RATE * MAX_AUDIO_SEC
MAX_LABEL_LEN = 448
TEXT_COLUMN   = "transcript"
NUM_PROC      = 4

DOMAINS = [
    ("general",  "whisper-medium-general"),
    ("clinical", "whisper-medium-clinical"),
    (None,       "whisper-medium-all"),
]

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
_NUM_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
    "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
    "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
    "nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,
    "sixty":60,"seventy":70,"eighty":80,"ninety":90,
    "hundred":100,"thousand":1000,"million":1000000,
}
_ORDINALS = {
    "first":"1st","second":"2nd","third":"3rd","fourth":"4th","fifth":"5th",
    "sixth":"6th","seventh":"7th","eighth":"8th","ninth":"9th","tenth":"10th",
    "eleventh":"11th","twelfth":"12th","thirteenth":"13th","fourteenth":"14th",
    "fifteenth":"15th","sixteenth":"16th","seventeenth":"17th","eighteenth":"18th",
    "nineteenth":"19th","twentieth":"20th","thirtieth":"30th",
    "fortieth":"40th","fiftieth":"50th",
}
_MONTHS = {
    "january":"01","jan":"01","february":"02","feb":"02","march":"03","mar":"03",
    "april":"04","apr":"04","may":"05","june":"06","jun":"06","july":"07",
    "jul":"07","august":"08","aug":"08","september":"09","sep":"09","sept":"09",
    "october":"10","oct":"10","november":"11","nov":"11","december":"12","dec":"12",
}
_DOW     = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)"
_ORD_PAT = r"(\d{1,2})(?:st|nd|rd|th)?"
_MON_PAT = r"(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")"
_YEAR    = r"(\d{4})"
_DAY     = r"(\d{1,2})"
_DATE_P1 = re.compile(rf"{_DOW}\s+{_ORD_PAT}\s+{_MON_PAT},?\s+{_YEAR}", re.IGNORECASE)
_DATE_P2 = re.compile(rf"{_ORD_PAT}\s+{_MON_PAT},?\s+{_YEAR}",           re.IGNORECASE)
_DATE_P3 = re.compile(rf"{_DAY}[-/]{_DAY}[-/]{_YEAR}")
_SPECIAL  = re.compile(r"\[(UNK|PAD)\]", re.IGNORECASE)
_DISALLOW = re.compile(r"[^a-zA-Z0-9\s?()\:\-\+<>\.\/\'\[\]]")

def _m2n(m): return _MONTHS.get(m.lower(), m)

def _normalise_dates(text):
    text = _DATE_P1.sub(lambda m: f"{int(m.group(1)):02d}/{_m2n(m.group(2))}/{m.group(3)}", text)
    text = _DATE_P2.sub(lambda m: f"{int(m.group(1)):02d}/{_m2n(m.group(2))}/{m.group(3)}", text)
    text = _DATE_P3.sub(lambda m: f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}", text)
    return text

def _normalise_times(text):
    def _sub(m): return f"{m.group(1)}:{m.group(2) or '00'} {m.group(3).lower()}"
    return re.sub(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]m)\b", _sub, text, flags=re.IGNORECASE)

def _words_to_digits(text):
    return " ".join(
        _ORDINALS[w] if w in _ORDINALS else str(_NUM_WORDS[w]) if w in _NUM_WORDS else w
        for w in text.split()
    )

def clean_transcript(text):
    if not isinstance(text, str): return None
    text = text.strip().replace("\r", " ")
    text = _normalise_dates(text)
    text = _normalise_times(text)
    text = _SPECIAL.sub("", text)
    text = _INAUDIBLE.sub("", text)
    text = _FILLERS.sub("", text)
    text = _words_to_digits(text)
    text = _DISALLOW.sub(" ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 5 or len(text) > 300: return None
    return text

# ─────────────────────────────────────────────────────────────────
# Preprocess one domain
# ─────────────────────────────────────────────────────────────────
def preprocess_domain(dataset, domain, out_dir, feature_extractor, tokenizer):
    print(f"\n{'='*60}")
    print(f"  Domain: {domain or 'all'} → {out_dir}")
    print(f"{'='*60}")

    if os.path.exists(out_dir):
        print(f"  Already exists — skipping. Delete to reprocess.")
        return

    # Filter by domain
    if domain is not None:
        ds = dataset.filter(
            lambda x: x["domain"] == domain,
            num_proc=NUM_PROC,
            desc=f"Filtering {domain}",
        )
    else:
        ds = dataset

    # Filter invalid transcripts
    ds = ds.filter(
        lambda x: clean_transcript(x[TEXT_COLUMN]) is not None,
        num_proc=NUM_PROC,
        desc="Filtering transcripts",
    )

    print(f"  Train: {len(ds['train']):,} | Val: {len(ds['validation']):,} | Test: {len(ds['test']):,}")

    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    def prepare(batch):
        array = np.asarray(batch["audio"]["array"], dtype=np.float32)
        if len(array) > MAX_INPUT_LEN:
            array = array[:MAX_INPUT_LEN]
        batch["input_features"] = feature_extractor(
            array, sampling_rate=SAMPLING_RATE,
            return_tensors="np", return_attention_mask=False,
        ).input_features[0]
        batch["labels"] = tokenizer(
            clean_transcript(batch[TEXT_COLUMN]),
            max_length=MAX_LABEL_LEN,
            truncation=True,
        ).input_ids
        return batch

    ds = ds.map(
        prepare,
        remove_columns=[c for c in ds["train"].column_names
                        if c not in {"input_features", "labels"}],
        num_proc=NUM_PROC,
        writer_batch_size=200,
        desc=f"Feature extraction ({domain or 'all'})",
    )

    os.makedirs(out_dir, exist_ok=True)
    ds.save_to_disk(out_dir)
    print(f"  Saved to {out_dir}")
    print(f"  Train: {len(ds['train']):,} | Val: {len(ds['validation']):,} | Test: {len(ds['test']):,}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    print(f"Loading model components from {MODEL_NAME}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    print("  Done")

    print(f"\nLoading dataset from {DATA_PATH}...")
    dataset = load_from_disk(DATA_PATH)
    print(f"  Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,} | Test: {len(dataset['test']):,}")

    for domain, folder in DOMAINS:
        out_dir = os.path.join(OUTPUT_BASE, folder)
        preprocess_domain(dataset, domain, out_dir, feature_extractor, tokenizer)

    print("\n\nAll done! Preprocessed data saved to:")
    for _, folder in DOMAINS:
        out_dir = os.path.join(OUTPUT_BASE, folder)
        if os.path.exists(out_dir):
            os.system(f"du -sh {out_dir}")


if __name__ == "__main__":
    main()
