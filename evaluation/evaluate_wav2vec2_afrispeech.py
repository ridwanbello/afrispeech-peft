"""
Evaluate fine-tuned wav2vec2-xlsr on AfriSpeech-200 test/validation sets.

Usage:
    source ~/projects/afrispeech-project/wav2vec2-env/bin/activate
    cd ~/projects/afrispeech-project
    python evaluate_wav2vec2_local.py
"""

import os
import re
import json
import numpy as np
import torch
from datasets import load_from_disk, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
import evaluate as evaluate_lib
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "robello2/wav2vec2-xlsr-afrispeech-all"
MODEL_TAG    = "wav2vec2-xlsr-all-10ep"

DATA_PATH    = "./afrispeech_arrow"
OUTPUT_DIR   = f"./eval_results/{MODEL_TAG}"
SAMPLING_RATE = 16_000
MAX_AUDIO_SEC = 30
MAX_INPUT_LEN = SAMPLING_RATE * MAX_AUDIO_SEC
BATCH_SIZE    = 8
SPLITS        = ["validation", "test"]
DOMAINS       = ["general", "clinical", None]   # None = all domains

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
# Evaluate one split+domain combination
# ─────────────────────────────────────────────────────────────────
def evaluate_split(model, processor, tokenizer, dataset, split_name, domain, device, wer_metric):
    domain_label = domain or "all"
    print(f"\n{'='*60}")
    print(f"  Split: {split_name.upper()}  |  Domain: {domain_label.upper()}")

    ds = dataset[split_name]
    if domain:
        ds = ds.filter(lambda x: x["domain"] == domain)
    # filter out samples with no valid transcript
    ds = ds.filter(lambda x: clean_transcript(x["transcript"]) is not None)

    if len(ds) == 0:
        print("  No samples — skipping.")
        return None

    print(f"  Samples: {len(ds):,}")
    print(f"{'='*60}")

    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    def prepare(batch):
        array = np.asarray(batch["audio"]["array"], dtype=np.float32)
        if len(array) > MAX_INPUT_LEN:
            array = array[:MAX_INPUT_LEN]
        batch["input_values"] = processor(
            array, sampling_rate=SAMPLING_RATE
        ).input_values[0]
        batch["reference"] = clean_transcript(batch["transcript"]) or ""
        return batch

    ds = ds.map(
        prepare,
        remove_columns    = [c for c in ds.column_names
                             if c not in {"input_values", "reference"}],
        num_proc          = 1,
        load_from_cache_file = False,
        desc              = f"Preprocessing {split_name}/{domain_label}",
    )

    all_preds = []
    all_refs  = []
    n_batches = (len(ds) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(ds), BATCH_SIZE):
        batch_samples = ds[i : i + BATCH_SIZE]
        input_values  = [np.array(v) for v in batch_samples["input_values"]]
        refs          = batch_samples["reference"]

        # pad to same length
        max_len = max(len(v) for v in input_values)
        padded  = np.zeros((len(input_values), max_len), dtype=np.float32)
        for j, v in enumerate(input_values):
            padded[j, :len(v)] = v
        attention_mask = np.zeros_like(padded, dtype=np.int64)
        for j, v in enumerate(input_values):
            attention_mask[j, :len(v)] = 1

        inputs = {
            "input_values":   torch.tensor(padded).to(device),
            "attention_mask": torch.tensor(attention_mask).to(device),
        }

        with torch.no_grad():
            logits  = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        preds    = tokenizer.batch_decode(pred_ids)
        preds    = [clean_transcript(p) or "" for p in preds]

        all_preds.extend(preds)
        all_refs.extend(refs)

        batch_num = i // BATCH_SIZE + 1
        if batch_num % 20 == 0 or batch_num == n_batches:
            running_wer = wer_metric.compute(
                predictions=all_preds, references=all_refs
            )
            print(f"  Batch {batch_num}/{n_batches} | Running WER: {running_wer:.3f}")

    wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    wer = round(wer, 3)
    print(f"\n  ✓ Final WER [{split_name}/{domain_label}]: {wer}")
    return wer

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── load model ────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer         = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_NAME)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    processor         = Wav2Vec2Processor(
        feature_extractor = feature_extractor,
        tokenizer         = tokenizer,
    )
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print("  Model loaded and set to eval mode")

    # ── load dataset ──────────────────────────────────────────────
    print(f"\nLoading dataset from {DATA_PATH}...")
    dataset = load_from_disk(DATA_PATH)
    for split in SPLITS:
        print(f"  {split}: {len(dataset[split]):,} samples")

    wer_metric = evaluate_lib.load("wer")
    all_results = {}

    # ── evaluate all combinations ─────────────────────────────────
    for split in SPLITS:
        all_results[split] = {}
        for domain in DOMAINS:
            domain_label = domain or "all"
            wer = evaluate_split(
                model, processor, tokenizer,
                dataset, split, domain, device, wer_metric
            )
            if wer is not None:
                all_results[split][domain_label] = wer

    # ── print summary ─────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  RESULTS SUMMARY — {MODEL_TAG}")
    print("="*60)
    print(f"  {'Split':<12} {'Domain':<12} {'WER':>6}")
    print(f"  {'-'*32}")
    for split in SPLITS:
        for domain_label, wer in all_results[split].items():
            print(f"  {split:<12} {domain_label:<12} {wer:>6.3f}")

    # ── save results ──────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model":   MODEL_NAME,
            "tag":     MODEL_TAG,
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
