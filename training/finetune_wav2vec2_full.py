"""
Fine-tune jonatasgrosman/wav2vec2-large-xlsr-53-english on AfriSpeech-200.
Runs locally on RTX 5080 (or any CUDA GPU).

Usage:
    source ~/projects/afrispeech-project/wav2vec2-env/bin/activate
    cd ~/projects/afrispeech-project
    python finetune_wav2vec2_local.py

Prerequisites:
    - afrispeech_arrow/ dataset in same directory (or update DATA_PATH)
    - HF_TOKEN set as environment variable, or huggingface-cli login
    - WANDB_API_KEY set as environment variable, or wandb login
"""

import os
import re
import json
import time
import numpy as np
import torch
import wandb
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_from_disk, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import evaluate as evaluate_lib

# ─────────────────────────────────────────────────────────────────
# Configuration — edit these
# ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
DOMAIN        = None           # None = all domains | "general" | "clinical"
TEXT_COLUMN   = "transcript"
SAMPLING_RATE = 16_000
MAX_AUDIO_SEC = 30
MAX_INPUT_LEN = SAMPLING_RATE * MAX_AUDIO_SEC

DATA_PATH        = "./afrispeech_arrow"
PREPROCESSED_DIR = "./preprocessed/wav2vec2-xlsr-all"
OUTPUT_DIR       = "./checkpoints/wav2vec2-xlsr-all"
HF_HUB_REPO      = "robello2/wav2vec2-xlsr-afrispeech-all"
WANDB_PROJECT    = "wav2vec2-xlsr-afrispeech-local"

NUM_EPOCHS       = 10
BATCH_SIZE       = 16      # reduce to 8 if OOM
GRAD_ACCUM_STEPS = 1
LEARNING_RATE    = 1e-4
WARMUP_RATIO     = 0.10
WEIGHT_DECAY     = 0.01
FP16             = True
SAVE_TOTAL_LIMIT = 3

# AfriSpeech 50-character vocabulary
AFRISPEECH_VOCAB = [
    "-", "w", "a", "7", ",", "0", "d", "i", ":", "p",
    "g", "u", "(", "5", "1", "e", "9", "j", "b", "3",
    "s", "'", "h", "o", "+", "l", "v", "y", "q", "n",
    "2", "r", "f", "m", "%", "t", "/", "6", "z", "?",
    "8", ")", "x", ".", "4", "c", "k", "|", "[UNK]", "[PAD]",
]
VOCAB_DICT = {v: i for i, v in enumerate(AFRISPEECH_VOCAB)}

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
# Data collator
# ─────────────────────────────────────────────────────────────────
@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding:   bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_values = [{"input_values": f["input_values"]} for f in features]
        label_feats  = [{"input_ids":    f["labels"]}       for f in features]
        batch = self.processor.pad(
            input_values, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_feats, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

# ─────────────────────────────────────────────────────────────────
# Timing callback
# ─────────────────────────────────────────────────────────────────
class EpochTimingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start = None
        self.train_start = None
        self.epoch_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        t = time.time() - self.epoch_start
        self.epoch_times.append(t)
        print(f"  Epoch {int(state.epoch)} time: {t/60:.2f} min")
        if wandb.run:
            wandb.log({"epoch_time_min": round(t/60, 2), "epoch": int(state.epoch)})

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - self.train_start
        avg   = sum(self.epoch_times) / max(len(self.epoch_times), 1)
        print(f"  Total: {total/3600:.2f} hrs | Avg/epoch: {avg/60:.2f} min")

# ─────────────────────────────────────────────────────────────────
# Preprocessing (run once, cached to PREPROCESSED_DIR)
# ─────────────────────────────────────────────────────────────────
def preprocess():
    print(f"Preprocessing dataset → {PREPROCESSED_DIR}")

    # build tokenizer from vocab
    tok_dir = os.path.join(PREPROCESSED_DIR, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    vocab_path = os.path.join(tok_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(VOCAB_DICT, f, ensure_ascii=False, indent=2)
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token            = "[UNK]",
        pad_token            = "[PAD]",
        word_delimiter_token = "|",
    )
    tokenizer.save_pretrained(tok_dir)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    processor = Wav2Vec2Processor(
        feature_extractor = feature_extractor,
        tokenizer         = tokenizer,
    )

    dataset = load_from_disk(DATA_PATH)
    if DOMAIN:
        dataset = dataset.filter(lambda x: x["domain"] == DOMAIN, num_proc=4)
    dataset = dataset.filter(lambda x: clean_transcript(x[TEXT_COLUMN]) is not None, num_proc=4)
    print(f"  Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,}")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    def prepare(batch):
        array = np.asarray(batch["audio"]["array"], dtype=np.float32)
        if len(array) > MAX_INPUT_LEN:
            array = array[:MAX_INPUT_LEN]
        batch["input_values"] = processor(
            array, sampling_rate=SAMPLING_RATE
        ).input_values[0]
        cleaned = clean_transcript(batch[TEXT_COLUMN])
        batch["labels"] = tokenizer(text=cleaned, padding=False).input_ids
        batch["input_length"] = len(batch["input_values"])
        return batch

    dataset = dataset.map(
        prepare,
        remove_columns    = [c for c in dataset["train"].column_names
                             if c not in {"input_values", "labels", "input_length"}],
        num_proc          = 4,
        writer_batch_size = 200,
        desc              = "Feature extraction",
    )

    data_dir = os.path.join(PREPROCESSED_DIR, "dataset")
    os.makedirs(data_dir, exist_ok=True)
    dataset.save_to_disk(data_dir)
    print(f"  Saved to {data_dir}")
    return dataset, tok_dir

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── load or preprocess dataset ────────────────────────────────
    data_dir = os.path.join(PREPROCESSED_DIR, "dataset")
    tok_dir  = os.path.join(PREPROCESSED_DIR, "tokenizer")
    if os.path.exists(data_dir) and os.path.exists(tok_dir):
        print(f"Loading preprocessed dataset from {data_dir}...")
        dataset = load_from_disk(data_dir)
    else:
        dataset, tok_dir = preprocess()
    print(f"  Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,}")

    # ── processor ─────────────────────────────────────────────────
    print(f"Loading tokenizer from {tok_dir}...")
    tokenizer         = Wav2Vec2CTCTokenizer.from_pretrained(tok_dir)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    processor         = Wav2Vec2Processor(
        feature_extractor = feature_extractor,
        tokenizer         = tokenizer,
    )
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── model ─────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        ctc_loss_reduction      = "mean",
        pad_token_id            = tokenizer.pad_token_id,
        vocab_size              = len(VOCAB_DICT),
        ignore_mismatched_sizes = True,
    ).to(device)
    model.freeze_feature_encoder()
    print("  Feature encoder frozen")

    # ── compute metrics ───────────────────────────────────────────
    wer_metric = evaluate_lib.load("wer")

    def compute_metrics(pred):
        pred_ids  = np.argmax(pred.predictions, axis=-1)
        label_ids = np.where(pred.label_ids == -100, tokenizer.pad_token_id, pred.label_ids)
        pred_str  = tokenizer.batch_decode(pred_ids)
        label_str = tokenizer.batch_decode(label_ids, group_tokens=False)
        pred_clean  = [clean_transcript(p) or "" for p in pred_str]
        label_clean = [clean_transcript(l) or "" for l in label_str]
        wer = wer_metric.compute(predictions=pred_clean, references=label_clean)
        return {"wer": round(wer, 3)}

    # ── warmup steps ──────────────────────────────────────────────
    steps_per_epoch = len(dataset["train"]) // (BATCH_SIZE * GRAD_ACCUM_STEPS)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = max(10, int(total_steps * WARMUP_RATIO))
    print(f"\n  Steps/epoch: {steps_per_epoch:,} | Total: {total_steps:,} | Warmup: {warmup_steps:,}")

    # ── wandb ─────────────────────────────────────────────────────
    wandb.init(
        project = WANDB_PROJECT,
        name    = f"wav2vec2-xlsr-{DOMAIN or 'all'}-local",
        tags    = ["finetune", "wav2vec2", "xlsr", DOMAIN or "all",
                   "afrispeech", "lr-1e-4", f"{NUM_EPOCHS}-epochs", "rtx5080"],
        config  = {
            "model": MODEL_NAME, "domain": DOMAIN,
            "vocab_size": len(VOCAB_DICT),
            "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "warmup_steps": warmup_steps,
            "weight_decay": WEIGHT_DECAY, "fp16": FP16,
            "frozen_encoder": True,
        },
    )

    # ── training args ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        run_name                    = f"wav2vec2-xlsr-{DOMAIN or 'all'}-local",
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = 8,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        optim                       = "adamw_torch",
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        lr_scheduler_type           = "linear",
        warmup_steps                = warmup_steps,
        fp16                        = FP16,
        gradient_checkpointing      = True,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        logging_steps               = 25,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        remove_unused_columns       = False,
        report_to                   = ["wandb"],
        push_to_hub                 = True,
        hub_model_id                = HF_HUB_REPO,
        hub_token                   = os.environ.get("HF_TOKEN"),
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
    )

    data_collator   = DataCollatorCTCWithPadding(processor=processor, padding=True)
    timing_callback = EpochTimingCallback()

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = dataset["train"],
        eval_dataset     = dataset["validation"],
        data_collator    = data_collator,
        compute_metrics  = compute_metrics,
        tokenizer        = processor.feature_extractor,  # processing_class in newer transformers
        callbacks        = [timing_callback],
    )

    # ── baseline eval ─────────────────────────────────────────────
    print("\nRunning baseline eval...")
    baseline = trainer.evaluate(metric_key_prefix="baseline")
    trainer.log_metrics("baseline", baseline)
    print(f"  Baseline WER: {baseline.get('baseline_wer', 'N/A')}")

    # ── train ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  wav2vec2 XLSR | domain={DOMAIN or 'all'} | LR={LEARNING_RATE}")
    print("="*60 + "\n")
    trainer.train()

    # ── final eval ────────────────────────────────────────────────
    print("\nFinal evaluation...")
    final_metrics = trainer.evaluate(metric_key_prefix="final")
    print(f"  Final WER: {final_metrics.get('final_wer', 'N/A')}")

    if wandb.run:
        wandb.run.summary["baseline_wer"] = baseline.get("baseline_wer", -1)
        wandb.run.summary["final_wer"]    = final_metrics.get("final_wer", -1)

    # ── save ──────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    summary = {
        "baseline": baseline, "final": final_metrics,
        "timing": {
            "total_hrs":       round((time.time() - timing_callback.train_start) / 3600, 3),
            "avg_epoch_min":   round(sum(timing_callback.epoch_times) / max(len(timing_callback.epoch_times), 1) / 60, 2),
            "epoch_times_min": [round(t/60, 2) for t in timing_callback.epoch_times],
        },
        "config": {
            "model": MODEL_NAME, "domain": DOMAIN,
            "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "warmup_steps": warmup_steps,
        },
    }
    with open(f"{OUTPUT_DIR}/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── push to Hub ───────────────────────────────────────────────
    print(f"\nPushing to Hub: {HF_HUB_REPO}...")
    trainer.push_to_hub(commit_message="End of training")
    print("  Done.")

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
