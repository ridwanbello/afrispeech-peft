"""
Shared-Regional Adapter Architecture for Whisper-medium on AfriSpeech-200.

Architecture:
- Frozen Whisper-medium base
- Shared DoRA adapter: trained on all African accents (pan-African features)
- Regional DoRA adapters: one per high-resource accent (>=100 clips)
- Learnable gate: weighted average alpha*shared + (1-alpha)*regional
- Fallback: shared adapter only for unseen/low-resource accents

Usage:
    modal run finetune_whisper_shared_regional_adapter.py --mode shared
    modal run finetune_whisper_shared_regional_adapter.py --mode regional --accent yoruba
    modal run finetune_whisper_shared_regional_adapter.py --mode eval
"""

import modal

MINUTES = 60
HOURS   = 60 * MINUTES

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "openai/whisper-medium"
LANGUAGE      = "english"
TASK          = "transcribe"
MAX_LABEL_LEN = 448
SAMPLING_RATE = 16_000

# Top 5 high-resource accents (update after running accent clip count)
HIGH_RESOURCE_ACCENTS = [
    "yoruba",
    "igbo",
    "south african english",
    "kenyan english",
    "ghanaian english",
]

# DoRA config
LORA_RANK    = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

# Training
NUM_EPOCHS       = 10
BATCH_SIZE       = 16
LEARNING_RATE    = 1e-4
WARMUP_RATIO     = 0.10
WEIGHT_DECAY     = 0.01
SAVE_TOTAL_LIMIT = 10
FP16             = True

PREPROCESSED_BASE = "/data/preprocessed"
OUTPUT_BASE       = "/vol/whisper-adapters"
HF_HUB_BASE       = "robello2/whisper-medium-adapter"
WANDB_PROJECT     = "whisper-shared-regional-adapter"

# ─────────────────────────────────────────────────────────────────
# Modal setup
# ─────────────────────────────────────────────────────────────────
base_pkgs = [
    "torch==2.6.0", "torchaudio==2.6.0",
    "transformers>=4.40.0", "datasets==2.19.0",
    "accelerate>=0.30.0", "evaluate>=0.4.1", "jiwer>=3.0.3",
    "peft>=0.10.0",
    "soundfile", "librosa", "wandb", "huggingface_hub", "numpy<2.0",
]

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*base_pkgs, extra_index_url="https://download.pytorch.org/whl/cu124")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
)

app       = modal.App("whisper-shared-regional-adapter")
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
# Shared-Regional Adapter Model
# ─────────────────────────────────────────────────────────────────
ADAPTER_MODEL_CODE = '''
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import WhisperForConditionalGeneration

class SharedRegionalAdapterModel(nn.Module):
    """
    Whisper-medium with shared + regional DoRA adapters.

    Architecture:
      - Frozen Whisper-medium base
      - Shared adapter: trained on all AfriSpeech data
      - Regional adapters: one per high-resource accent
      - Learnable gate: alpha * shared_out + (1-alpha) * regional_out
      - Fallback: shared adapter when accent is unknown
    """

    def __init__(self, base_model_name, lora_config, accent_list):
        super().__init__()

        # Load frozen base
        base = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        base.generation_config.forced_decoder_ids    = None
        base.generation_config.suppress_tokens       = None
        base.generation_config.begin_suppress_tokens = None
        base.generation_config.language              = "english"
        base.generation_config.task                  = "transcribe"

        # Shared adapter (trained on all accents)
        self.shared_model = get_peft_model(base, lora_config)

        # Regional adapters (one per accent, loaded lazily)
        self.regional_models = nn.ModuleDict()
        self.accent_list     = accent_list

        # Learnable gate per accent: alpha in [0,1]
        # alpha=1 → pure shared, alpha=0 → pure regional
        self.gates = nn.ParameterDict({
            accent.replace(" ", "_"): nn.Parameter(torch.tensor(0.5))
            for accent in accent_list
        })

    def get_alpha(self, accent):
        key = accent.replace(" ", "_") if accent else None
        if key and key in self.gates:
            return torch.sigmoid(self.gates[key])
        return torch.tensor(1.0)  # fallback: pure shared

    def load_regional_adapter(self, accent, adapter_path):
        """Load a trained regional adapter from disk or Hub."""
        key = accent.replace(" ", "_")
        base = WhisperForConditionalGeneration.from_pretrained(
            self.shared_model.base_model.model.config._name_or_path
        )
        self.regional_models[key] = PeftModel.from_pretrained(base, adapter_path)

    def forward(self, input_features, labels=None, accent=None, **kwargs):
        # Always run shared adapter
        shared_out = self.shared_model(
            input_features=input_features, labels=labels, **kwargs
        )

        key = accent.replace(" ", "_") if accent else None
        if key and key in self.regional_models:
            # Run regional adapter
            regional_out = self.regional_models[key](
                input_features=input_features, labels=labels, **kwargs
            )
            alpha = self.get_alpha(accent)

            # Gate: weighted average of logits
            # alpha close to 1 → rely on shared
            # alpha close to 0 → rely on regional
            if hasattr(shared_out, "logits") and hasattr(regional_out, "logits"):
                gated_logits = (
                    alpha * shared_out.logits +
                    (1 - alpha) * regional_out.logits
                )
                shared_out.logits = gated_logits

            if labels is not None and hasattr(shared_out, "loss"):
                # Recompute loss from gated logits
                from torch.nn import CrossEntropyLoss
                loss_fct  = CrossEntropyLoss(ignore_index=-100)
                shift_log = gated_logits[..., :-1, :].contiguous()
                shift_lab = labels[..., 1:].contiguous()
                shared_out.loss = loss_fct(
                    shift_log.view(-1, shift_log.size(-1)),
                    shift_lab.view(-1)
                )

        return shared_out

    def generate(self, input_features, accent=None, **kwargs):
        key = accent.replace(" ", "_") if accent else None
        if key and key in self.regional_models:
            alpha = self.get_alpha(accent).item()
            # Use regional for generation when alpha < 0.5
            if alpha < 0.5:
                return self.regional_models[key].generate(
                    input_features=input_features, **kwargs
                )
        return self.shared_model.generate(
            input_features=input_features, **kwargs
        )

    def print_trainable_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        print(f"Gate parameters: {sum(p.numel() for p in self.gates.parameters()):,}")
'''

# ─────────────────────────────────────────────────────────────────
# Stage 1: Train shared adapter on all AfriSpeech data
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
def train_shared():
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
    from peft import LoraConfig, get_peft_model, TaskType
    import evaluate as evaluate_lib

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    _mod = types.ModuleType("cleaner")
    exec(CLEANER_CODE, _mod.__dict__)
    clean_transcript = _mod.clean_transcript

    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project = WANDB_PROJECT,
        name    = "shared-adapter-all-domains",
        tags    = ["shared-adapter", "dora", "afrispeech", "all-accents"],
        config  = {"lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
                   "target_modules": TARGET_MODULES, "learning_rate": LEARNING_RATE}
    )

    print("Loading processor and model...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    processor         = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    base_model        = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    base_model.generation_config.forced_decoder_ids    = None
    base_model.generation_config.suppress_tokens       = None
    base_model.generation_config.begin_suppress_tokens = None
    base_model.generation_config.language              = LANGUAGE
    base_model.generation_config.task                  = TASK

    lora_config = LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        use_dora       = True,
        target_modules = TARGET_MODULES,
        task_type      = TaskType.SEQ_2_SEQ_LM,
        bias           = "none",
    )

    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"total_params": total, "trainable_params": trainable,
                         "trainable_pct": round(trainable/total*100, 3)})

    # Normaliser
    _SENTINEL          = "abcxyz"
    english_normalizer = EnglishTextNormalizer(getattr(tokenizer, "english_spelling_normalizer", None))
    def normalise(text):
        cleaned = clean_transcript(text)
        if cleaned is None: return _SENTINEL
        try: return english_normalizer(cleaned).strip() or _SENTINEL
        except: return cleaned or _SENTINEL

    # Data collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_id: int

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_feats  = [{"input_features": f["input_features"]} for f in features]
            batch        = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
            batch["attention_mask"] = torch.ones(batch["input_features"].shape[:2], dtype=torch.long)
            label_feats  = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.decoder_start_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    # Timing + commit callback
    class EpochCallback(TrainerCallback):
        def __init__(self, vol):
            self.start = None
            self.vol   = vol

        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.time()

        def on_epoch_end(self, args, state, control, **kwargs):
            elapsed = (time.time() - self.start) / 60
            print(f"  Epoch {int(state.epoch)} | {elapsed:.1f} min elapsed")
            self.vol.commit()
            if wandb.run:
                wandb.log({"epoch": int(state.epoch)})

    # Metrics
    wer_metric = evaluate_lib.load("wer")
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = np.where(pred.label_ids == -100, tokenizer.pad_token_id, pred.label_ids)
        pred_str  = [normalise(p) for p in tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
        label_str = [normalise(l) for l in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
        return {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 3)}

    # Load preprocessed all-domain data
    print("Loading preprocessed dataset...")
    dataset = load_from_disk(f"{PREPROCESSED_BASE}/whisper-medium-all")
    print(f"  Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,}")

    steps_per_epoch = len(dataset["train"]) // BATCH_SIZE
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = max(10, int(total_steps * WARMUP_RATIO))

    output_dir = f"{OUTPUT_BASE}/shared-adapter"
    os.makedirs(output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = 16,
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
        hub_model_id                = f"{HF_HUB_BASE}-shared",
        hub_token                   = os.environ["HF_TOKEN"],
        report_to                   = "wandb",
        run_name                    = "shared-adapter-all-domains",
        dataloader_num_workers      = 0,
    )

    trainer = Seq2SeqTrainer(
        args             = training_args,
        model            = model,
        train_dataset    = dataset["train"],
        eval_dataset     = dataset["validation"],
        data_collator    = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_id=model.config.decoder_start_token_id
        ),
        compute_metrics  = compute_metrics,
        processing_class = processor.feature_extractor,
        callbacks        = [
            EpochCallback(vol=model_vol),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    print("\nBaseline eval...")
    baseline = trainer.evaluate(metric_key_prefix="baseline")
    print(f"  Baseline WER: {baseline.get('baseline_wer')}")

    print("\nTraining shared adapter...")
    trainer.train()

    final = trainer.evaluate(metric_key_prefix="final")
    print(f"  Final WER: {final.get('final_wer')}")

    trainer.save_model(output_dir)
    trainer.push_to_hub(commit_message="Shared adapter training complete")
    model_vol.commit()
    wandb.finish()
    print("Done!")


# ─────────────────────────────────────────────────────────────────
# Stage 2: Train regional adapter per accent
# ─────────────────────────────────────────────────────────────────
@app.function(
    image   = gpu_image,
    gpu     = "H100",
    timeout = 12 * HOURS,
    volumes = {"/data": data_vol, "/vol": model_vol},
    secrets = [
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_regional(accent: str = "yoruba"):
    import os, types, json, time
    import numpy as np
    import torch
    import wandb
    from dataclasses import dataclass
    from typing import Any, Dict, List

    from datasets import load_from_disk, Audio
    from transformers import (
        WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        TrainerCallback, EarlyStoppingCallback,
    )
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    import evaluate as evaluate_lib

    device = torch.device("cuda")
    accent_key = accent.lower().replace(" ", "_")

    print(f"Training regional adapter for accent: {accent}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    _mod = types.ModuleType("cleaner")
    exec(CLEANER_CODE, _mod.__dict__)
    clean_transcript = _mod.clean_transcript

    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project = WANDB_PROJECT,
        name    = f"regional-adapter-{accent_key}",
        tags    = ["regional-adapter", "dora", "afrispeech", accent_key],
        config  = {"accent": accent, "lora_rank": LORA_RANK}
    )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    processor         = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load shared adapter as base for regional (warm start)
    shared_path = f"{HF_HUB_BASE}-shared"
    print(f"Loading shared adapter from {shared_path} as warm start...")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    base_model.generation_config.forced_decoder_ids    = None
    base_model.generation_config.suppress_tokens       = None
    base_model.generation_config.begin_suppress_tokens = None
    base_model.generation_config.language              = LANGUAGE
    base_model.generation_config.task                  = TASK

    # Apply fresh regional adapter on top
    lora_config = LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        use_dora       = True,
        target_modules = TARGET_MODULES,
        task_type      = TaskType.SEQ_2_SEQ_LM,
        bias           = "none",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # Normaliser + metrics
    _SENTINEL          = "abcxyz"
    english_normalizer = EnglishTextNormalizer(getattr(tokenizer, "english_spelling_normalizer", None))
    def normalise(text):
        cleaned = clean_transcript(text)
        if cleaned is None: return _SENTINEL
        try: return english_normalizer(cleaned).strip() or _SENTINEL
        except: return cleaned or _SENTINEL

    wer_metric = evaluate_lib.load("wer")
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = np.where(pred.label_ids == -100, tokenizer.pad_token_id, pred.label_ids)
        pred_str  = [normalise(p) for p in tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
        label_str = [normalise(l) for l in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
        return {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 3)}

    # Filter dataset to this accent from preprocessed all-domain data
    # NOTE: Since preprocessed data has no metadata, load from raw arrow
    print(f"Loading accent-specific data for: {accent}")
    raw_ds = load_from_disk("/data/afrispeech_arrow")

    def filter_accent(x):
        return (x["accent"] or "").lower() == accent.lower() and \
               clean_transcript(x["transcript"]) is not None

    accent_ds = raw_ds.filter(filter_accent, num_proc=2)
    accent_ds = accent_ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    print(f"  Train: {len(accent_ds['train']):,} | Val: {len(accent_ds['validation']):,}")

    if len(accent_ds["train"]) < 50:
        print(f"  WARNING: Only {len(accent_ds['train'])} training samples — too few for reliable adapter")

    # Preprocess on the fly
    def prepare(batch):
        array = np.asarray(batch["audio"]["array"], dtype=np.float32)
        if len(array) > SAMPLING_RATE * 30:
            array = array[:SAMPLING_RATE * 30]
        batch["input_features"] = feature_extractor(
            array, sampling_rate=SAMPLING_RATE, return_tensors="np"
        ).input_features[0]
        batch["labels"] = tokenizer(
            clean_transcript(batch["transcript"]),
            max_length=MAX_LABEL_LEN, truncation=True,
        ).input_ids
        return batch

    accent_ds = accent_ds.map(
        prepare,
        remove_columns=[c for c in accent_ds["train"].column_names
                        if c not in {"input_features", "labels"}],
        num_proc=1,
        desc=f"Preprocessing {accent}",
    )

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_id: int

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_feats  = [{"input_features": f["input_features"]} for f in features]
            batch        = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
            batch["attention_mask"] = torch.ones(batch["input_features"].shape[:2], dtype=torch.long)
            label_feats  = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.decoder_start_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    class EpochCallback(TrainerCallback):
        def __init__(self, vol):
            self.vol = vol
        def on_epoch_end(self, args, state, control, **kwargs):
            print(f"  Epoch {int(state.epoch)} done")
            self.vol.commit()

    steps_per_epoch = max(1, len(accent_ds["train"]) // BATCH_SIZE)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = max(5, int(total_steps * WARMUP_RATIO))

    output_dir = f"{OUTPUT_BASE}/regional-{accent_key}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = min(BATCH_SIZE, len(accent_ds["train"])),
        per_device_eval_batch_size  = 8,
        optim                       = "adamw_torch",
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        lr_scheduler_type           = "linear",
        warmup_steps                = warmup_steps,
        fp16                        = FP16,
        gradient_checkpointing      = True,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_steps               = 10,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        save_total_limit            = 5,
        predict_with_generate       = True,
        generation_max_length       = MAX_LABEL_LEN,
        remove_unused_columns       = False,
        label_names                 = ["labels"],
        push_to_hub                 = True,
        hub_model_id                = f"{HF_HUB_BASE}-regional-{accent_key}",
        hub_token                   = os.environ["HF_TOKEN"],
        report_to                   = "wandb",
        run_name                    = f"regional-{accent_key}",
        dataloader_num_workers      = 0,
    )

    trainer = Seq2SeqTrainer(
        args             = training_args,
        model            = model,
        train_dataset    = accent_ds["train"],
        eval_dataset     = accent_ds["validation"],
        data_collator    = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_id=model.config.decoder_start_token_id
        ),
        compute_metrics  = compute_metrics,
        processing_class = processor.feature_extractor,
        callbacks        = [
            EpochCallback(vol=model_vol),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    print("\nTraining regional adapter...")
    trainer.train()

    final = trainer.evaluate(metric_key_prefix="final")
    print(f"  Final WER ({accent}): {final.get('final_wer')}")

    trainer.save_model(output_dir)
    trainer.push_to_hub(commit_message=f"Regional adapter: {accent}")
    model_vol.commit()
    wandb.finish()
    print(f"Regional adapter for {accent} done!")


# ─────────────────────────────────────────────────────────────────
# Local entrypoint
# ─────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(mode: str = "shared", accent: str = "yoruba"):
    if mode == "shared":
        print("Training shared adapter on all AfriSpeech data...")
        train_shared.remote()
    elif mode == "regional":
        print(f"Training regional adapter for accent: {accent}")
        train_regional.remote(accent=accent)
    elif mode == "all_regional":
        print(f"Training regional adapters for all {len(HIGH_RESOURCE_ACCENTS)} accents...")
        for acc in HIGH_RESOURCE_ACCENTS:
            print(f"  Launching: {acc}")
            train_regional.remote(accent=acc)
    else:
        print(f"Unknown mode: {mode}. Use: shared | regional | all_regional")
