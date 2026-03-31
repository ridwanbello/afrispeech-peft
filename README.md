# AfriSpeech-PEFT: Parameter-Efficient Fine-tuning for African-Accented English ASR

This repository contains the code for the paper:

> **Parameter-Efficient Fine-tuning of Speech Foundation Models for African-Accented English ASR**  

## Overview

We apply LoRA and DoRA (Weight-Decomposed Low-Rank Adaptation) to fine-tune
Whisper-medium and wav2vec2-xlsr-53 on AfriSpeech-200 — a 200-hour pan-African
accented English corpus covering general and clinical speech domains.

Key findings:
- DoRA fine-tuning achieves competitive WER with full fine-tuning using <5% of parameters
- DoRA generalises better cross-domain (general → clinical) than full fine-tuning
- DoRA reduces catastrophic forgetting of standard English (LibriSpeech WER: 0.045 vs 0.057 for full fine-tuning)

## Repository Structure

```
afrispeech-peft/
├── training/
│   ├── finetune_whisper_full_all_domains.py      # Whisper full fine-tuning (all domains, Modal)
│   ├── finetune_whisper_full_clinical.py         # Whisper full fine-tuning (clinical, Modal)
│   ├── finetune_whisper_dora.py                  # Whisper DoRA fine-tuning (Modal H100)
│   ├── finetune_wav2vec2_full.py                 # wav2vec2 full fine-tuning (local GPU)
│   └── finetune_wav2vec2_dora.py                 # wav2vec2 DoRA fine-tuning (local GPU)
│
├── evaluation/
│   ├── evaluate_whisper_afrispeech.py            # Whisper eval on AfriSpeech (all splits/domains)
│   ├── evaluate_wav2vec2_afrispeech.py           # wav2vec2 eval on AfriSpeech
│   ├── evaluate_whisper_dora_afrispeech.py       # Whisper DoRA eval on AfriSpeech
│   ├── evaluate_wav2vec2_dora_afrispeech.py      # wav2vec2 DoRA eval on AfriSpeech
│   ├── evaluate_dora_librispeech.py              # Catastrophic forgetting eval (LibriSpeech)
│   ├── evaluate_whisper_librispeech.py           # Whisper baseline eval on LibriSpeech
│   ├── evaluate_wav2vec2_librispeech.py          # wav2vec2 baseline eval on LibriSpeech
│   └── evaluate_mms_zeroshot.py                  # MMS zero-shot eval on AfriSpeech (Modal)
│
├── preprocessing/
│   ├── preprocess_whisper.py                     # Feature extraction for Whisper
│   └── preprocess_wav2vec2.py                    # Feature extraction for wav2vec2
│
└── analysis/
    ├── cleaning_pipeline_analysis.py             # Cleaning pipeline impact analysis
    ├── plot_duration_histogram.py                # Audio duration distribution plot
    └── get_dataset_stats.py                      # Dataset statistics for paper
```

## Dataset

[AfriSpeech-200](https://huggingface.co/datasets/tobiolatunji/afrispeech-200) —
200 hours of pan-African accented English speech from 2,463 speakers across
120 accents from 13 countries, covering general and clinical domains.

## Models

| Model | HuggingFace Hub |
|---|---|
| Whisper-medium full fine-tuned (general) | `robello2/whisper-medium-afrispeech-general-v4` |
| Whisper-medium full fine-tuned (clinical) | `robello2/whisper-medium-afrispeech-clinical` |
| Whisper-medium full fine-tuned (all) | `robello2/whisper-medium-afrispeech-all` |
| Whisper-medium DoRA (general) | `robello2/whisper-medium-dora-afrispeech-general` |
| wav2vec2-xlsr-53 full fine-tuned (all) | `robello2/wav2vec2-xlsr-afrispeech-all` |
| wav2vec2-xlsr-53 DoRA (general) | `robello2/wav2vec2-xlsr-dora-afrispeech-general` |

## Setup

```bash
# Create environments
python3 -m venv afrispeech-env
source afrispeech-env/bin/activate
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate evaluate jiwer peft soundfile librosa wandb huggingface_hub numpy

# For Modal training (Whisper)
pip install modal
modal token set --token-id YOUR_ID --token-secret YOUR_SECRET
```

## Training

### Full fine-tuning (Modal H100)
```bash
# Whisper all-domain
modal run training/finetune_whisper_full_all_domains.py

# Resume if interrupted
modal run training/finetune_whisper_full_all_domains.py --resume
```

### DoRA fine-tuning
```bash
# Whisper DoRA (Modal H100)
modal run training/finetune_whisper_dora.py

# wav2vec2 DoRA (local GPU)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python training/finetune_wav2vec2_dora.py --domain general
```

## Evaluation

```bash
# AfriSpeech evaluation (all splits and domains)
python evaluation/evaluate_whisper_dora_afrispeech.py
python evaluation/evaluate_wav2vec2_dora_afrispeech.py

# Catastrophic forgetting (LibriSpeech)
python evaluation/evaluate_dora_librispeech.py

# MMS zero-shot (Modal)
modal run evaluation/evaluate_mms_zeroshot.py
```

## Preprocessing

```bash
# Whisper (all 3 domains)
python preprocessing/preprocess_whisper.py --domain general
python preprocessing/preprocess_whisper.py --domain clinical
python preprocessing/preprocess_whisper.py --domain all

# wav2vec2
python preprocessing/preprocess_wav2vec2.py --domain general
```

## Key Results

| Model | Method | Trainable | LS-clean | Test Gen | Test Cli | Test All |
|---|---|---|---|---|---|---|
| Whisper-medium | Zero-shot | — | 0.027 | 0.278 | 0.341 | 0.310 |
| Whisper-medium | Full FT | 100% | 0.057 | 0.093 | 0.266 | 0.181 |
| Whisper-medium | DoRA r=32 | <5% | 0.045 | 0.101 | 0.246 | 0.174 |
| wav2vec2-xlsr-53 | Zero-shot | — | 0.074 | 0.502 | 0.654 | 0.578 |
| wav2vec2-xlsr-53 | Full FT | 100% | 0.138 | 0.198 | 0.424 | 0.310 |
| wav2vec2-xlsr-53 | DoRA r=32 | <2% | 0.105 | 0.321 | 0.498 | 0.409 |

## Citation

```bibtex
@article{bello2026afrispeech_peft,
  title   = {Parameter-Efficient Fine-tuning of Speech Foundation Models for African-Accented English ASR},
  author  = {Bello, Ridwan},
  journal = {Computer Speech \& Language},
  year    = {2026}
}
```

## Acknowledgements

This work uses the [AfriSpeech-200](https://arxiv.org/abs/2310.00274) dataset
from Olatunji et al. (2023). Training was conducted on Modal cloud (H100) and
locally on an NVIDIA RTX 5080.
