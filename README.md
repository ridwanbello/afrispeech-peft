# AfriSpeech-PEFT: Parameter-Efficient Fine-tuning for African-Accented English ASR

## Overview

We apply DoRA (Weight-Decomposed Low-Rank Adaptation) to fine-tune
Whisper-medium and wav2vec2-xlsr-53 on AfriSpeech-200 (https://arxiv.org/abs/2310.00274) and compare with the conventional full-finetuning approach

Key findings:
- Despite using only <5% of model parameters, Whisper finetuned with DoRA all-domain achieves 0.130 test WER vs 0.125 for full fine-tuning (100%)
- DoRA reduces catastrophic forgetting by 61.7% on LibriSpeech (0.049 vs 0.128 for full fine-tuning)
- DoRA generalises better cross-domain: general-trained DoRA achieves 0.246 on clinical test vs 0.266 for full fine-tuning
- Full fine-tuning outperforms DoRA for wav2vec2-xlsr-53 on AfriSpeech, but DoRA still reduces forgetting (0.133 vs 0.158)

## Dataset

[AfriSpeech-200](https://huggingface.co/datasets/tobiolatunji/afrispeech-200) —
200 hours of pan-African accented English speech from 2,463 speakers across
120 accents from 13 countries, covering general and clinical domains.

## Models
The finetuned models are hosted on HuggingFace

| Model | HuggingFace Hub |
|---|---|
| Whisper-medium full fine-tuned (general) | `robello2/whisper-medium-afrispeech-general` |
| Whisper-medium full fine-tuned (clinical) | `robello2/whisper-medium-afrispeech-clinical` |
| Whisper-medium full fine-tuned (all) | `robello2/whisper-medium-afrispeech-all` |
| Whisper-medium DoRA (general) | `robello2/whisper-medium-dora-afrispeech-general` |
| Whisper-medium DoRA (clinical) | `robello2/whisper-medium-dora-afrispeech-clinical` |
| Whisper-medium DoRA (all) | `robello2/whisper-medium-dora-afrispeech-all` |
| wav2vec2-xlsr-53 full fine-tuned (all) | `robello2/wav2vec2-xlsr-afrispeech-all` |
| wav2vec2-xlsr-53 DoRA (general) | `robello2/wav2vec2-xlsr-dora-afrispeech-general` |
| wav2vec2-xlsr-53 DoRA (clinical) | `robello2/wav2vec2-xlsr-dora-afrispeech-clinical` |
| wav2vec2-xlsr-53 DoRA (all) | `robello2/wav2vec2-xlsr-dora-afrispeech-all` |

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

### Full fine-tuning
```bash
# Whisper all-domain
modal run training/finetune_whisper_full_all_domains.py

# Resume if interrupted
modal run training/finetune_whisper_full_all_domains.py --resume
```

### DoRA fine-tuning
```bash
# Whisper DoRA — general / clinical / all (Modal H100)
# Set DOMAIN at top of script: "general" | "clinical" | "all"
modal run training/finetune_whisper_dora.py
```

## Evaluation

```bash
# AfriSpeech evaluation — all 3 DoRA models per architecture
python evaluation/evaluate_whisper_dora_afrispeech.py
python evaluation/evaluate_wav2vec2_dora_afrispeech.py

# Catastrophic forgetting — all 6 DoRA models on LibriSpeech
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
python preprocessing/preprocess_wav2vec2.py --domain clinical
python preprocessing/preprocess_wav2vec2.py --domain all
```

## DoRA Hyperparameters
 
| Hyperparameter | Whisper-medium | wav2vec2-xlsr-53 |
|---|---|---|
| Rank (r) | 32 | 32 |
| Alpha (α) | 64 | 64 |
| Dropout | 0.05 | 0.05 |
| Target modules | q, k, v, out, fc1, fc2 | q, k, v, out |
| Trainable params | <5% | <2% |
| Learning rate | 1e-4 | 1e-4 |
| Early stopping patience | 3 epochs | 3 epochs |
 
## Key Results
 
### Whisper-medium
 
| Training domain | Method | Trainable | LS-clean | Test Gen | Test Cli | Test All |
|---|---|---|---|---|---|---|
| Whisper-medium (DoRA) | Zero-shot | 0% | 0.027 | 0.278 | 0.341 | 0.310 |
| General | Full FT | 100% | 0.057 | 0.093 | 0.266 | 0.181 |
| General | DoRA r=32 | <5% | 0.045 | 0.101 | 0.246 | 0.174 |
| Clinical | Full FT | 100% | 0.098 | 0.190 | 0.150 | 0.170 |
| Clinical | DoRA r=32 | <5% | 0.047 | 0.160 | 0.158 | 0.159 |
| All | Full FT | 100% | 0.128 | **0.097** | **0.152** | **0.125** |
| All | DoRA r=32 | <5% | **0.049** | 0.101 | 0.158 | 0.130 |
 
### wav2vec2-xlsr-53
 
| Training domain | Method | Trainable | LS-clean | Test Gen | Test Cli | Test All |
|---|---|---|---|---|---|---|
| Wav2vec2 (DoRA) | Zero-shot | 0% | 0.074 | 0.502 | 0.654 | 0.578 |
| General | Full FT | 100% | 0.138 | 0.198 | 0.424 | 0.310 |
| General | DoRA r=32 | <2% | 0.105 | 0.321 | 0.498 | 0.409 |
| Clinical | Full FT | 100% | 0.178 | 0.307 | 0.267 | 0.287 |
| Clinical | DoRA r=32 | <2% | 0.135 | 0.364 | 0.434 | 0.399 |
| All | Full FT | 100% | 0.158 | **0.196** | **0.257** | **0.226** |
| All | DoRA r=32 | <2% | **0.133** | 0.316 | 0.424 | 0.370 |

## Acknowledgements

This work uses the [AfriSpeech-200](https://arxiv.org/abs/2310.00274) dataset
from Olatunji et al. (2023). Training was conducted on Modal cloud (H100) and
locally on an NVIDIA RTX 5080.
