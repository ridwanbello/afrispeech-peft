"""
Fine-tune openai/whisper-medium on AfriSpeech-200 using DoRA (Weight-Decomposed
Low-Rank Adaptation) via HuggingFace PEFT.

Applied to both encoder and decoder, rank=32, general domain.
Runs on Modal H100.

Preprocessed data already on Modal volume at:
  /data/preprocessed/whisper-medium-general

Run:
    modal run finetune_whisper_dora_modal.py
    modal run finetune_whisper_dora_modal.py --resume
"""

import modal

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "openai/whisper-medium"
LANGUAGE      = "english"
TASK          = "transcribe"
DOMAIN        = "general"
MAX_LABEL_LEN = 448

PREPROCESSED_DIR = "/data/preprocessed/whisper-medium-general"
OUTPUT_DIR       = "/vol/whisper-medium-dora-general"
HF_HUB_REPO      = "robello2/whisper-medium-dora-afrispeech-general"
WANDB_PROJECT    = "modal-whisper-dora-general"

# DoRA / LoRA config
LORA_RANK       = 32
LORA_ALPHA      = 64        # typically 2x rank
LORA_DROPOUT    = 0.05
USE_DORA        = True      # DoRA = weight-decomposed LoRA
# Target modules — both encoder and decoder attention + feed-forward
TARGET_MODULES  = [
    "q_proj", "v_proj", "k_proj", "out_proj",   # attention
    "fc1", "fc2",                                 # feed-forward
]

# Training config
NUM_EPOCHS       = 10
BATCH_SIZE       = 16
GRAD_ACCUM_STEPS = 1
LEARNING_RATE    = 1e-4      # higher LR is typical for PEFT vs full fine-tuning
WARMUP_RATIO     = 0.10
WEIGHT_DECAY     = 0.01
SAVE_TOTAL_LIMIT = 10
FP16             = True

MINUTES = 60
HOURS   = 60 * MINUTES

# ─────────────────────────────────────────────────────────────────
# Modal setup
# ─────────────────────────────────────────────────────────────────
base_pkgs = [
    "torch==2.6.0", "torchaudio==2.6.0",
    "transformers>=4.40.0", "datasets==2.19.0",
    "accelerate>=0.30.0", "evaluate>=0.4.1", "jiwer>=3.0.3",
    "peft>=0.10.0",
    "soundfile", "librosa", "wandb", "huggingface_hub",
    "numpy<2.0",
]

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*base_pkgs, extra_index_url="https://download.pytorch.org/whl/cu124")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
)

app       = modal.App("whisper-medium-dora-general")
data_vol  = modal.Volume.from_name("afrispeech-data")
model_vol = modal.Volume.from_name("whisper-afrispeech-model", create_if_missing=True)

# ─────────────────────────────────────────────────────────────────
# Cleaning pipeline
# ─────────────────────────────────────────────────────────────────
CLEANER_CODE = r'''
import re

_INAUDIBLE = re.compile(
    r"\b(inaudible|inaudiable|inauidble|inauidible|inauible|inaudibe"
    r"|inaudibel|inaudilbe|inudible|inaudiible|inaudiblee|inuadible"
    r"|inaudbile|inauidble|inaudoible|inaudicble|inaudilbe|inaudible"
    r"|inaudivle|inaudinle|inaduible|inaudibl|inaudbible|inuadibale"
    r"|inauidbe|inaudibble|inaduible|inauddible|inauudible|inauidble"
    r"|inaudibale|inauidible)\b",
    re.IGNORECASE,
)
_FILLERS = re.compile(r"\b(uh+|um+|hmm+|mmhmm|mhm|hm+|ugh+|ah+)\b", re.IGNORECASE)
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
_SPECIAL = re.compile(r"\[(UNK|PAD)\]", re.IGNORECASE)
_DISALLOW= re.compile(r"[^a-zA-Z0-9\s?()\:\-\+<>\.\/\'\[\]]")

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
'''

# ─────────────────────────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────────────────────────
@app.function(
    image   = gpu_image,
    gpu     = "H100",
    timeout = 24 * HOURS,
    volumes = {"/data": data_vol, "/vol": model_vol},
    secrets = [
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train(resume: bool = False):
    import os, types, json, time
    import numpy as np
    import torch
    import wandb
    from dataclasses import dataclass
    from typing import Any, Dict, List

    from datasets import load_from_disk
    from transformers import (
        WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        TrainerCallback, EarlyStoppingCallback,
    )
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    import evaluate as evaluate_lib

    device = torch.device("cuda")
    print(f"Device : {device} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    _mod = types.ModuleType("cleaner")
    exec(CLEANER_CODE, _mod.__dict__)
    clean_transcript = _mod.clean_transcript

    # ── wandb ─────────────────────────────────────────────────────
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project = WANDB_PROJECT,
        name    = f"whisper-medium-dora-r{LORA_RANK}-{DOMAIN}",
        tags    = ["dora", "peft", "whisper-medium", DOMAIN, "afrispeech",
                   f"r{LORA_RANK}", "10-epochs"],
        config  = {
            "model": MODEL_NAME, "domain": DOMAIN,
            "peft_method": "DoRA", "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
        }
    )

    # ── processor ─────────────────────────────────────────────────
    print(f"\nLoading processor from {MODEL_NAME}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    processor         = WhisperProcessor(
        feature_extractor = feature_extractor,
        tokenizer         = tokenizer,
    )

    # ── base model ────────────────────────────────────────────────
    print(f"Loading base model: {MODEL_NAME}")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.generation_config.forced_decoder_ids    = None
    model.generation_config.suppress_tokens       = None
    model.generation_config.begin_suppress_tokens = None
    model.generation_config.language              = LANGUAGE
    model.generation_config.task                  = TASK

    # ── apply DoRA ────────────────────────────────────────────────
    print(f"\nApplying DoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
    print(f"Target modules: {TARGET_MODULES}")

    lora_config = LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        use_dora       = USE_DORA,          # DoRA vs LoRA
        target_modules = TARGET_MODULES,
        task_type      = TaskType.SEQ_2_SEQ_LM,
        bias           = "none",
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # Log parameter counts to wandb
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "trainable_pct":    round(trainable_params / total_params * 100, 3),
    })
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # ── normaliser ────────────────────────────────────────────────
    _SENTINEL = "abcxyz"
    english_normalizer = EnglishTextNormalizer(
        getattr(tokenizer, "english_spelling_normalizer", None)
    )

    def normalise(text: str) -> str:
        cleaned = clean_transcript(text)
        if cleaned is None: return _SENTINEL
        try: return english_normalizer(cleaned).strip() or _SENTINEL
        except: return cleaned or _SENTINEL

    # ── data collator ─────────────────────────────────────────────
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor:        Any
        decoder_start_id: int
        max_label_length: int = 448

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_feats  = [{"input_features": f["input_features"]} for f in features]
            batch        = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
            batch["attention_mask"] = torch.ones(
                batch["input_features"].shape[:2], dtype=torch.long
            )
            label_feats  = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (labels[:, 0] == self.decoder_start_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    # ── timing + checkpoint callback ──────────────────────────────
    class EpochTimingCallback(TrainerCallback):
        def __init__(self, vol):
            self.epoch_start = None
            self.train_start = None
            self.epoch_times = []
            self.vol         = vol

        def on_train_begin(self, args, state, control, **kwargs):
            self.train_start = time.time()

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_start = time.time()

        def on_epoch_end(self, args, state, control, **kwargs):
            t = time.time() - self.epoch_start
            self.epoch_times.append(t)
            print(f"  Epoch {int(state.epoch)} time: {t/60:.2f} min")
            self.vol.commit()
            print(f"  Checkpoint committed to Modal volume.")
            if wandb.run:
                wandb.log({"epoch_time_min": round(t/60, 2), "epoch": int(state.epoch)})

        def on_train_end(self, args, state, control, **kwargs):
            total = time.time() - self.train_start
            avg   = sum(self.epoch_times) / max(len(self.epoch_times), 1)
            print(f"  Total: {total/3600:.2f} hrs | Avg/epoch: {avg/60:.2f} min")

    # ── metrics ───────────────────────────────────────────────────
    wer_metric = evaluate_lib.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_norm  = [normalise(p) for p in pred_str]
        label_norm = [normalise(l) for l in label_str]
        wer = wer_metric.compute(predictions=pred_norm, references=label_norm)
        return {"wer": round(wer, 3)}

    # ── load preprocessed dataset ─────────────────────────────────
    print(f"\nLoading preprocessed dataset from {PREPROCESSED_DIR}...")
    dataset = load_from_disk(PREPROCESSED_DIR)
    print(f"  Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,}")

    steps_per_epoch = len(dataset["train"]) // (BATCH_SIZE * GRAD_ACCUM_STEPS)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = max(10, int(total_steps * WARMUP_RATIO))
    print(f"  Steps/epoch: {steps_per_epoch:,} | Total: {total_steps:,} | Warmup: {warmup_steps:,}")

    # ── training args ─────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = 16,
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
        predict_with_generate       = True,
        generation_max_length       = MAX_LABEL_LEN,
        remove_unused_columns       = False,
        label_names                 = ["labels"],
        push_to_hub                 = True,
        hub_model_id                = HF_HUB_REPO,
        hub_token                   = os.environ["HF_TOKEN"],
        report_to                   = "wandb",
        run_name                    = f"whisper-medium-dora-r{LORA_RANK}-{DOMAIN}",
        dataloader_num_workers      = 0,
        dataloader_pin_memory       = True,
    )

    data_collator   = DataCollatorSpeechSeq2SeqWithPadding(
        processor        = processor,
        decoder_start_id = model.config.decoder_start_token_id,
        max_label_length = MAX_LABEL_LEN,
    )
    timing_callback = EpochTimingCallback(vol=model_vol)

    trainer = Seq2SeqTrainer(
        args             = training_args,
        model            = model,
        train_dataset    = dataset["train"],
        eval_dataset     = dataset["validation"],
        data_collator    = data_collator,
        compute_metrics  = compute_metrics,
        processing_class = processor.feature_extractor,
        callbacks        = [timing_callback, EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── baseline eval ─────────────────────────────────────────────
    print("\nRunning baseline eval...")
    baseline = trainer.evaluate(metric_key_prefix="baseline")
    trainer.log_metrics("baseline", baseline)
    print(f"  Baseline WER: {baseline.get('baseline_wer', 'N/A')}")

    # ── train ─────────────────────────────────────────────────────
    print("\nStarting DoRA training...")
    if resume:
        checkpoints = [
            os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR)
            if d.startswith("checkpoint-")
        ] if os.path.exists(OUTPUT_DIR) else []
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            print(f"  Resuming from checkpoint: {latest}")
            trainer.train(resume_from_checkpoint=latest)
        else:
            print("  No checkpoint found — starting from scratch")
            trainer.train()
    else:
        trainer.train()

    # ── final eval ────────────────────────────────────────────────
    print("\nFinal evaluation...")
    final_metrics = trainer.evaluate(metric_key_prefix="final")
    print(f"  Final WER: {final_metrics.get('final_wer', 'N/A')}")

    # ── save ──────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)

    summary = {
        "model": MODEL_NAME, "domain": DOMAIN,
        "peft_method": "DoRA",
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "target_modules": TARGET_MODULES,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(trainable_params / total_params * 100, 3),
        "baseline_wer": baseline.get("baseline_wer"),
        "final_wer":    final_metrics.get("final_wer"),
        "config": {
            "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "warmup_steps": warmup_steps,
        },
    }
    with open(f"{OUTPUT_DIR}/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    model_vol.commit()

    # ── push to Hub ───────────────────────────────────────────────
    print(f"\nPushing to Hub: {HF_HUB_REPO}...")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=HF_HUB_REPO, exist_ok=True)

    # Save full model (merged) + adapter separately
    save_dir = f"{OUTPUT_DIR}/final"
    os.makedirs(save_dir, exist_ok=True)

    # Save adapter weights only (lightweight — just DoRA deltas)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    feature_extractor.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Upload vocab files from base model
    for fname in ["vocab.json", "merges.txt", "normalizer.json",
                  "added_tokens.json", "special_tokens_map.json"]:
        try:
            fpath = hf_hub_download(MODEL_NAME, fname)
            api.upload_file(path_or_fileobj=fpath, path_in_repo=fname,
                            repo_id=HF_HUB_REPO, commit_message=f"Add {fname}")
        except Exception as e:
            print(f"  Skipped {fname}: {e}")

    trainer.push_to_hub(commit_message="DoRA fine-tuning complete")
    print("  Done!")

    wandb.finish()
    print("\nAll done!")
    print(f"\nSummary:")
    print(f"  Baseline WER : {baseline.get('baseline_wer')}")
    print(f"  Final WER    : {final_metrics.get('final_wer')}")
    print(f"  Trainable    : {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")


@app.local_entrypoint()
def main(resume: bool = False):
    if resume:
        print("Resuming DoRA training from latest checkpoint...")
    else:
        print("Starting DoRA fine-tuning on whisper-medium (general domain)...")
    train.remote(resume=resume)
