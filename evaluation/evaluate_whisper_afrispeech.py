"""
Evaluate robello2/whisper-medium-afrispeech-all on AfriSpeech-200.
Evaluates all 6 combinations: validation+test × general+clinical+all

Usage:
    source ~/projects/afrispeech-project/afrispeech-env/bin/activate
    cd ~/projects/afrispeech-project
    python evaluate_whisper_all_local.py
"""

import os
import re
import json
import numpy as np
import torch
from datasets import load_from_disk, Audio
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
import evaluate as evaluate_lib

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "robello2/whisper-medium-afrispeech-all"
DATA_PATH     = "./afrispeech_arrow"
OUTPUT_DIR    = "./eval_results/whisper-medium-all"
SAMPLING_RATE = 16_000
BATCH_SIZE    = 8
TEXT_COLUMN   = "transcript"
MAX_AUDIO_SEC = 30

SPLITS  = ["validation", "test"]
DOMAINS = [("general", "general"), ("clinical", "clinical"), (None, "all")]

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

def clean_transcript(text):
    if not isinstance(text, str): return None
    text = text.strip().replace("\r", " ")
    text = _DATE_P1.sub(lambda m: f"{int(m.group(1)):02d}/{_m2n(m.group(2))}/{m.group(3)}", text)
    text = _DATE_P2.sub(lambda m: f"{int(m.group(1)):02d}/{_m2n(m.group(2))}/{m.group(3)}", text)
    text = _DATE_P3.sub(lambda m: f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}", text)
    def _sub(m): return f"{m.group(1)}:{m.group(2) or '00'} {m.group(3).lower()}"
    text = re.sub(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]m)\b", _sub, text, flags=re.IGNORECASE)
    text = _SPECIAL.sub("", text)
    text = _INAUDIBLE.sub("", text)
    text = _FILLERS.sub("", text)
    text = " ".join(str(_ORDINALS.get(w, _NUM_WORDS.get(w, w))) for w in text.split())
    text = _DISALLOW.sub(" ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 5 or len(text) > 300: return None
    return text

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    processor         = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model             = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    model.generation_config.forced_decoder_ids    = None
    model.generation_config.suppress_tokens       = None
    model.generation_config.begin_suppress_tokens = None
    model.generation_config.language              = "english"
    model.generation_config.task                  = "transcribe"
    print("  Model loaded | generation_config patched")

    english_normalizer = EnglishTextNormalizer(
        getattr(tokenizer, "english_spelling_normalizer", None)
    )
    def normalise(text):
        cleaned = clean_transcript(text)
        if not cleaned: return ""
        try: return english_normalizer(cleaned).strip()
        except: return cleaned

    # Load dataset
    print(f"\nLoading dataset from {DATA_PATH}...")
    dataset    = load_from_disk(DATA_PATH)
    wer_metric = evaluate_lib.load("wer")
    all_results = {}

    for split in SPLITS:
        all_results[split] = {}
        for domain, domain_label in DOMAINS:
            print(f"\n{'='*50}")
            print(f"  Split: {split} | Domain: {domain_label}")
            print(f"{'='*50}")

            ds = dataset[split]
            if domain:
                ds = ds.filter(lambda x: x["domain"] == domain)
            ds = ds.filter(lambda x: clean_transcript(x[TEXT_COLUMN]) is not None)
            ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
            print(f"  Samples: {len(ds):,}")

            all_preds, all_refs = [], []
            n_batches = (len(ds) + BATCH_SIZE - 1) // BATCH_SIZE

            for i in range(0, len(ds), BATCH_SIZE):
                batch   = ds[i : i + BATCH_SIZE]
                arrays  = [np.asarray(a["array"], dtype=np.float32)[:SAMPLING_RATE*MAX_AUDIO_SEC]
                           for a in batch["audio"]]
                refs    = [normalise(t) for t in batch[TEXT_COLUMN]]

                inputs = processor(
                    arrays, sampling_rate=SAMPLING_RATE,
                    return_tensors="pt", padding=True,
                ).to(device)

                with torch.no_grad():
                    pred_ids = model.generate(
                        inputs["input_features"],
                        language="english", task="transcribe",
                        suppress_tokens=None, begin_suppress_tokens=None,
                    )

                preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                preds = [normalise(p) for p in preds]
                all_preds.extend(preds)
                all_refs.extend(refs)

                batch_num = i // BATCH_SIZE + 1
                if batch_num % 50 == 0 or batch_num == n_batches:
                    running_wer = wer_metric.compute(predictions=all_preds, references=all_refs)
                    print(f"  Batch {batch_num}/{n_batches} | WER: {running_wer:.3f}")

            wer = round(wer_metric.compute(predictions=all_preds, references=all_refs), 3)
            print(f"\n  ✓ Final WER [{split}/{domain_label}]: {wer}")
            all_results[split][domain_label] = {"wer": wer, "samples": len(all_preds)}

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"model": MODEL_NAME, "results": all_results}, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: {MODEL_NAME}")
    print(f"{'='*60}")
    print(f"  {'Split':<12} {'General':>10} {'Clinical':>10} {'All':>10}")
    print(f"  {'-'*44}")
    for split in SPLITS:
        g = all_results[split].get("general",  {}).get("wer", "-")
        c = all_results[split].get("clinical", {}).get("wer", "-")
        a = all_results[split].get("all",      {}).get("wer", "-")
        print(f"  {split:<12} {str(g):>10} {str(c):>10} {str(a):>10}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
