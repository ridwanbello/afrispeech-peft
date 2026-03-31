"""
Evaluate DoRA fine-tuned models on LibriSpeech test-clean.
Measures catastrophic forgetting of standard English after AfriSpeech adaptation.

Models:
  - robello2/whisper-medium-dora-afrispeech-general
  - robello2/wav2vec2-xlsr-dora-afrispeech-general

Usage:
    source ~/projects/afrispeech-project/afrispeech-env/bin/activate
    cd ~/projects/afrispeech-project
    python evaluate_dora_librispeech.py
"""

import os
import json
import numpy as np
import torch
from datasets import load_from_disk, load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from peft import PeftModel
import evaluate as evaluate_lib

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
WHISPER_BASE  = "openai/whisper-medium"
WHISPER_DORA  = "robello2/whisper-medium-dora-afrispeech-general"

WAV2VEC2_BASE = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
WAV2VEC2_DORA = "robello2/wav2vec2-xlsr-dora-afrispeech-general"

LOCAL_PATH    = "./librispeech_test_clean"  # set to None to download
OUTPUT_DIR    = "./eval_results/dora_librispeech"
SAMPLING_RATE = 16_000
BATCH_SIZE    = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Load LibriSpeech
# ─────────────────────────────────────────────────────────────────
def load_librispeech():
    if LOCAL_PATH and os.path.exists(LOCAL_PATH):
        print(f"Loading LibriSpeech from {LOCAL_PATH}...")
        ds = load_from_disk(LOCAL_PATH)
        if hasattr(ds, "keys"):
            split = "test" if "test" in ds else list(ds.keys())[0]
            ds = ds[split]
    else:
        print("Downloading LibriSpeech test-clean from HuggingFace Hub...")
        ds = load_dataset(
            "openslr/librispeech_asr", "clean",
            split="test", trust_remote_code=True,
        )
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    print(f"  Samples: {len(ds):,}")
    return ds

# ─────────────────────────────────────────────────────────────────
# Evaluate Whisper DoRA on LibriSpeech
# ─────────────────────────────────────────────────────────────────
def eval_whisper_dora(ds, wer_metric, device):
    print(f"\n{'='*55}")
    print(f"  Whisper-medium DoRA — LibriSpeech test-clean")
    print(f"{'='*55}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_BASE)
    tokenizer         = WhisperTokenizer.from_pretrained(WHISPER_BASE, use_fast=False)
    processor         = WhisperProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    english_normalizer = EnglishTextNormalizer(
        getattr(tokenizer, "english_spelling_normalizer", None)
    )
    def normalise(text):
        try: return english_normalizer(text).strip()
        except: return text.lower().strip()

    # Load base + DoRA adapter
    base_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE)
    model      = PeftModel.from_pretrained(base_model, WHISPER_DORA)
    model      = model.to(device)
    model.eval()

    for m in [model, model.base_model]:
        if hasattr(m, "generation_config"):
            m.generation_config.forced_decoder_ids    = None
            m.generation_config.suppress_tokens       = None
            m.generation_config.begin_suppress_tokens = None
            m.generation_config.language              = "english"
            m.generation_config.task                  = "transcribe"

    ref_col   = "text" if "text" in ds.column_names else "transcription"
    all_preds, all_refs = [], []
    n_batches = (len(ds) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(ds), BATCH_SIZE):
        batch  = ds[i : i + BATCH_SIZE]
        arrays = [np.asarray(a["array"], dtype=np.float32) for a in batch["audio"]]
        refs   = [normalise(r) for r in batch[ref_col]]

        inputs = processor(
            arrays, sampling_rate=SAMPLING_RATE,
            return_tensors="pt", padding=True,
            return_attention_mask=True,
        ).to(device)

        with torch.no_grad():
            pred_ids = model.generate(
                input_features        = inputs["input_features"],
                attention_mask        = inputs.get("attention_mask"),
                language              = "english",
                task                  = "transcribe",
                suppress_tokens       = None,
                begin_suppress_tokens = None,
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
    print(f"\n  ✓ Whisper DoRA LibriSpeech WER: {wer}")

    # Print sample predictions
    print("\n  Sample predictions:")
    for i in range(min(3, len(all_preds))):
        print(f"    REF : {all_refs[i]}")
        print(f"    PRED: {all_preds[i]}")
        print()

    del model
    torch.cuda.empty_cache()
    return wer

# ─────────────────────────────────────────────────────────────────
# Evaluate wav2vec2 DoRA on LibriSpeech
# ─────────────────────────────────────────────────────────────────
def eval_wav2vec2_dora(ds, wer_metric, device):
    print(f"\n{'='*55}")
    print("  wav2vec2-xlsr-53 DoRA — LibriSpeech test-clean")
    print(f"{'='*55}")

    processor  = Wav2Vec2Processor.from_pretrained(WAV2VEC2_BASE)
    base_model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_BASE)
    model      = PeftModel.from_pretrained(base_model, WAV2VEC2_DORA)
    model      = model.to(device)
    model.eval()

    ref_col   = "text" if "text" in ds.column_names else "transcription"
    all_preds, all_refs = [], []
    n_batches = (len(ds) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(ds), BATCH_SIZE):
        batch  = ds[i : i + BATCH_SIZE]
        arrays = [np.asarray(a["array"], dtype=np.float32) for a in batch["audio"]]
        refs   = [r.lower().strip() for r in batch[ref_col]]

        inputs = processor(
            arrays, sampling_rate=SAMPLING_RATE,
            return_tensors="pt", padding=True,
        )
        # wav2vec2 uses input_values not input_ids
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(
                input_values=input_values,
                attention_mask=attention_mask,
            ).logits
        pred_ids = torch.argmax(logits, dim=-1)
        preds    = processor.batch_decode(pred_ids)
        preds    = [p.lower().strip() for p in preds]

        all_preds.extend(preds)
        all_refs.extend(refs)
        torch.cuda.empty_cache()

        batch_num = i // BATCH_SIZE + 1
        if batch_num % 50 == 0 or batch_num == n_batches:
            running_wer = wer_metric.compute(predictions=all_preds, references=all_refs)
            print(f"  Batch {batch_num}/{n_batches} | WER: {running_wer:.3f}")

    wer = round(wer_metric.compute(predictions=all_preds, references=all_refs), 3)
    print(f"\n  ✓ wav2vec2 DoRA LibriSpeech WER: {wer}")

    print("\n  Sample predictions:")
    for i in range(min(3, len(all_preds))):
        print(f"    REF : {all_refs[i]}")
        print(f"    PRED: {all_preds[i]}")
        print()

    del model
    torch.cuda.empty_cache()
    return wer

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    ds         = load_librispeech()
    wer_metric = evaluate_lib.load("wer")

    results = {}

    # Whisper DoRA
    wer_whisper = eval_whisper_dora(ds, wer_metric, device)
    results["whisper_medium_dora_general"] = {
        "model":    WHISPER_DORA,
        "dataset":  "librispeech_test_clean",
        "wer":      wer_whisper,
        "baseline": 0.027,   # zero-shot whisper-medium on librispeech
        "fft":      0.057,   # full fine-tuning whisper-medium-general on librispeech
        "forgetting_vs_baseline": round(wer_whisper - 0.027, 3),
        "forgetting_vs_fft":      round(wer_whisper - 0.057, 3),
    }

    # wav2vec2 DoRA
    wer_wav2vec2 = eval_wav2vec2_dora(ds, wer_metric, device)
    results["wav2vec2_xlsr_dora_general"] = {
        "model":    WAV2VEC2_DORA,
        "dataset":  "librispeech_test_clean",
        "wer":      wer_wav2vec2,
        "baseline": 0.074,   # zero-shot wav2vec2-xlsr on librispeech
        "fft":      0.138,   # full fine-tuning wav2vec2-xlsr-general on librispeech
        "forgetting_vs_baseline": round(wer_wav2vec2 - 0.074, 3),
        "forgetting_vs_fft":      round(wer_wav2vec2 - 0.138, 3),
    }

    # Save
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*55}")
    print("  CATASTROPHIC FORGETTING SUMMARY — LibriSpeech test-clean")
    print(f"{'='*55}")
    print(f"  {'Model':<30} {'Zero-shot':>10} {'FFT':>8} {'DoRA':>8} {'Δ vs zero-shot':>16}")
    print(f"  {'-'*72}")
    for key, r in results.items():
        name = key.replace("_", " ")
        print(f"  {name:<30} {r['baseline']:>10.3f} {r['fft']:>8.3f} {r['wer']:>8.3f} {r['forgetting_vs_baseline']:>+16.3f}")

    print(f"\n  Results saved to {out_path}")
    print("\n  Interpretation:")
    print("  Positive Δ = forgetting (WER rose vs zero-shot baseline)")
    print("  Negative Δ = improvement (DoRA also improved LibriSpeech WER)")


if __name__ == "__main__":
    main()
