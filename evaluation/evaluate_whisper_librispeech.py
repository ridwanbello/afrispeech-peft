"""
Evaluate openai/whisper-small on LibriSpeech test-clean.
No fine-tuning — pure baseline eval.

Usage:
    source ~/projects/afrispeech-project/afrispeech-env/bin/activate
    cd ~/projects/afrispeech-project
    python evaluate_whisper_small_librispeech.py
"""

import os
import json
import numpy as np
import torch
from datasets import load_from_disk, load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
import evaluate as evaluate_lib

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "openai/whisper-small"
SAMPLING_RATE = 16_000
BATCH_SIZE    = 8
OUTPUT_DIR    = "./eval_results/whisper-small-librispeech"

# Path to local librispeech if already downloaded, else load from HF Hub
LOCAL_PATH    = "./librispeech_test_clean"   # set to None to download from Hub

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
    # Load components separately to avoid processor_config duplicate key bug
    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    processor         = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model             = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Patch generation_config — same pattern as training scripts
    model.generation_config.forced_decoder_ids    = None
    model.generation_config.suppress_tokens       = None
    model.generation_config.begin_suppress_tokens = None
    model.generation_config.language              = "english"
    model.generation_config.task                  = "transcribe"
    print("  Model loaded | generation_config patched")

    # ── normaliser ────────────────────────────────────────────────
    english_normalizer = EnglishTextNormalizer(
        getattr(processor.tokenizer, "english_spelling_normalizer", None)
    )

    def normalise(text: str) -> str:
        try:
            return english_normalizer(text).strip()
        except Exception:
            return text.lower().strip()

    # ── load dataset ──────────────────────────────────────────────
    if LOCAL_PATH and os.path.exists(LOCAL_PATH):
        print(f"\nLoading LibriSpeech from {LOCAL_PATH}...")
        ds = load_from_disk(LOCAL_PATH)
        # handle both DatasetDict and Dataset
        if hasattr(ds, 'keys'):
            # DatasetDict — take test split or first available
            split = "test" if "test" in ds else list(ds.keys())[0]
            ds = ds[split]
    else:
        print("\nDownloading LibriSpeech test-clean from HuggingFace Hub...")
        ds = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split      = "test",
            trust_remote_code = True,
        )

    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    print(f"  Samples: {len(ds):,}")

    # ── eval loop ─────────────────────────────────────────────────
    wer_metric = evaluate_lib.load("wer")
    all_preds  = []
    all_refs   = []
    n_batches  = (len(ds) + BATCH_SIZE - 1) // BATCH_SIZE

    # detect reference text column
    ref_col = "text" if "text" in ds.column_names else "transcription"
    print(f"  Reference column: '{ref_col}'")

    for i in range(0, len(ds), BATCH_SIZE):
        batch   = ds[i : i + BATCH_SIZE]
        arrays  = [np.asarray(a["array"], dtype=np.float32)
                   for a in batch["audio"]]
        refs    = [normalise(r) for r in batch[ref_col]]

        inputs  = processor(
            arrays,
            sampling_rate   = SAMPLING_RATE,
            return_tensors  = "pt",
            padding         = True,
        ).to(device)

        with torch.no_grad():
            pred_ids = model.generate(
                inputs["input_features"],
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
        if batch_num % 20 == 0 or batch_num == n_batches:
            running_wer = wer_metric.compute(predictions=all_preds, references=all_refs)
            print(f"  Batch {batch_num}/{n_batches} | Running WER: {running_wer:.3f}")

    wer = round(wer_metric.compute(predictions=all_preds, references=all_refs), 3)

    print(f"\n{'='*50}")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Dataset: LibriSpeech test-clean")
    print(f"  WER    : {wer}")
    print(f"{'='*50}")

    # print a few examples
    print("\nSample predictions:")
    for i in range(min(5, len(all_preds))):
        print(f"  REF : {all_refs[i]}")
        print(f"  PRED: {all_preds[i]}")
        print()

    results = {
        "model":   MODEL_NAME,
        "dataset": "librispeech_test_clean",
        "wer":     wer,
        "samples": len(all_preds),
    }
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
