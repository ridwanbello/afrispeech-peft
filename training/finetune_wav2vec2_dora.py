"""
Fine-tune wav2vec2-xlsr-53 on AfriSpeech-200 using DoRA via HuggingFace PEFT.

Applied to both feature extractor and transformer layers, rank=32.
Runs locally on RTX 5080.

Usage:
    source ~/projects/afrispeech-project/wav2vec2-env/bin/activate
    cd ~/projects/afrispeech-project
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune_wav2vec2_dora_local.py --domain general
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune_wav2vec2_dora_local.py --domain clinical
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune_wav2vec2_dora_local.py --domain all
"""

import os
import re
import json
import time
import argparse
import numpy as np
import torch
import wandb
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from datasets import load_from_disk
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate as evaluate_lib

# ─────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, default="general",
                    choices=["general", "clinical", "all"])
args = parser.parse_args()
DOMAIN = args.domain

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
SAMPLING_RATE = 16_000

DOMAIN_CONFIG = {
    "general":  {
        "preprocessed": "./preprocessed/wav2vec2-general",
        "output_dir":   "./checkpoints/wav2vec2-dora-general",
        "hub_repo":     "robello2/wav2vec2-xlsr-dora-afrispeech-general",
        "wandb_proj":   "local-wav2vec2-dora-general",
    },
    "clinical": {
        "preprocessed": "./preprocessed/wav2vec2-clinical",
        "output_dir":   "./checkpoints/wav2vec2-dora-clinical",
        "hub_repo":     "robello2/wav2vec2-xlsr-dora-afrispeech-clinical",
        "wandb_proj":   "local-wav2vec2-dora-clinical",
    },
    "all": {
        "preprocessed": "./preprocessed/wav2vec2-all",
        "output_dir":   "./checkpoints/wav2vec2-dora-all",
        "hub_repo":     "robello2/wav2vec2-xlsr-dora-afrispeech-all",
        "wandb_proj":   "local-wav2vec2-dora-all",
    },
}

cfg              = DOMAIN_CONFIG[DOMAIN]
PREPROCESSED_DIR = cfg["preprocessed"]
OUTPUT_DIR       = cfg["output_dir"]
HF_HUB_REPO      = cfg["hub_repo"]
WANDB_PROJECT    = cfg["wandb_proj"]

# DoRA config
LORA_RANK      = 32
LORA_ALPHA     = 64
LORA_DROPOUT   = 0.05
USE_DORA       = True
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]  # attention layers only

# Training config
NUM_EPOCHS              = 10
BATCH_SIZE              = 8
GRAD_ACCUM_STEPS        = 2
LEARNING_RATE           = 1e-4
WARMUP_RATIO            = 0.10
WEIGHT_DECAY            = 0.01
SAVE_TOTAL_LIMIT        = 10
EARLY_STOPPING_PATIENCE = 3
FP16                    = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    processor:  Any
    padding:    Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nDomain : {DOMAIN}")
    print(f"Data   : {PREPROCESSED_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Hub    : {HF_HUB_REPO}")

    # ── wandb ─────────────────────────────────────────────────────
    wandb.login()
    wandb.init(
        project = WANDB_PROJECT,
        name    = f"wav2vec2-dora-r{LORA_RANK}-{DOMAIN}",
        tags    = ["dora", "peft", "wav2vec2-xlsr-53", DOMAIN,
                   "afrispeech", f"r{LORA_RANK}", "10-epochs"],
        config  = {
            "model": MODEL_NAME, "domain": DOMAIN,
            "peft_method": "DoRA", "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        }
    )

    # ── processor & model ─────────────────────────────────────────
    print(f"\nLoading processor: {MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        ctc_loss_reduction = "mean",
        pad_token_id       = processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_encoder()  # freeze CNN feature extractor

    # ── apply DoRA ────────────────────────────────────────────────
    print(f"\nApplying DoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        use_dora       = USE_DORA,
        target_modules = TARGET_MODULES,
        task_type      = TaskType.TOKEN_CLS,
        bias           = "none",
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "trainable_pct":    round(trainable_params / total_params * 100, 3),
    })
    print(f"  Total     : {total_params:,}")
    print(f"  Trainable : {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # ── metrics ───────────────────────────────────────────────────
    wer_metric = evaluate_lib.load("wer")

    def compute_metrics(pred):
        pred_ids  = np.argmax(pred.predictions, axis=-1)
        pred_str  = processor.batch_decode(pred_ids)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        pred_clean  = [clean_transcript(p) or "" for p in pred_str]
        label_clean = [clean_transcript(l) or "" for l in label_str]
        wer = wer_metric.compute(predictions=pred_clean, references=label_clean)
        return {"wer": round(wer, 3)}

    # ── load dataset ──────────────────────────────────────────────
    print(f"\nLoading preprocessed dataset from {PREPROCESSED_DIR}...")
    dataset = load_from_disk(PREPROCESSED_DIR)
    print(f"  Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,}")

    steps_per_epoch = len(dataset["train"]) // (BATCH_SIZE * GRAD_ACCUM_STEPS)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = max(10, int(total_steps * WARMUP_RATIO))

    # ── timing callback ───────────────────────────────────────────
    class EpochTimingCallback(TrainerCallback):
        def __init__(self):
            self.epoch_start = None
            self.train_start = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.train_start = time.time()

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_start = time.time()

        def on_epoch_end(self, args, state, control, **kwargs):
            t = time.time() - self.epoch_start
            print(f"  Epoch {int(state.epoch)} time: {t/60:.2f} min")
            if wandb.run:
                wandb.log({"epoch_time_min": round(t/60, 2), "epoch": int(state.epoch)})

        def on_train_end(self, args, state, control, **kwargs):
            total = time.time() - self.train_start
            print(f"  Total training time: {total/3600:.2f} hrs")

    # ── training args ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
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
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_steps               = 25,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        remove_unused_columns       = False,
        push_to_hub                 = True,
        hub_model_id                = HF_HUB_REPO,
        report_to                   = "wandb",
        run_name                    = f"wav2vec2-dora-r{LORA_RANK}-{DOMAIN}",
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    trainer = Trainer(
        args             = training_args,
        model            = model,
        train_dataset    = dataset["train"],
        eval_dataset     = dataset["validation"],
        data_collator    = data_collator,
        compute_metrics  = compute_metrics,
        processing_class = processor.feature_extractor,
        callbacks        = [
            EpochTimingCallback(),
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
        ],
    )

    # ── train ─────────────────────────────────────────────────────
    print("\nStarting DoRA training...")
    trainer.train()

    # ── final eval ────────────────────────────────────────────────
    print("\nFinal evaluation...")
    final_metrics = trainer.evaluate(metric_key_prefix="final")
    print(f"  Final WER: {final_metrics.get('final_wer', 'N/A')}")

    # ── save & push ───────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)

    summary = {
        "model": MODEL_NAME, "domain": DOMAIN,
        "peft_method": "DoRA",
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(trainable_params / total_params * 100, 3),
        "final_wer": final_metrics.get("final_wer"),
    }
    with open(f"{OUTPUT_DIR}/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    trainer.push_to_hub(commit_message="DoRA fine-tuning complete")
    print("  Done!")

    wandb.finish()
    print(f"\nSummary:")
    print(f"  Final WER : {final_metrics.get('final_wer')}")
    print(f"  Trainable : {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

if __name__ == "__main__":
    main()
