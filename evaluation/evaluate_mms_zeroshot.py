"""
Zero-shot evaluation of MMS models on AfriSpeech-200 test set.
Runs on Modal A100.

Models: facebook/mms-1b-fl102, facebook/mms-1b-all
Evaluates: test × general + clinical + all

Run:
    modal run evaluate_mms_modal.py
"""

import modal

MINUTES = 60
HOURS   = 60 * MINUTES

base_pkgs = [
    "torch==2.4.1", "torchaudio==2.4.1",
    "transformers>=4.40.0", "datasets==2.19.0",
    "accelerate>=0.30.0", "evaluate>=0.4.1", "jiwer>=3.0.3",
    "soundfile", "librosa", "huggingface_hub", "numpy<2.0",
]

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*base_pkgs, extra_index_url="https://download.pytorch.org/whl/cu121")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
)

app      = modal.App("mms-afrispeech-eval")
data_vol = modal.Volume.from_name("afrispeech-data")

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
'''

MODELS = [
    "facebook/mms-1b-fl102",
    "facebook/mms-1b-all",
]

DOMAINS = [
    ("general",  "general"),
    ("clinical", "clinical"),
    (None,       "all"),
]

SAMPLING_RATE = 16_000
BATCH_SIZE    = 4
MAX_AUDIO_SEC = 30
TEXT_COLUMN   = "transcript"
OUTPUT_PATH   = "/data/eval_results/mms_zeroshot_results.json"


@app.function(
    image   = gpu_image,
    gpu     = "H100",
    timeout = 6 * HOURS,
    volumes = {"/data": data_vol},
    secrets = [modal.Secret.from_name("huggingface-secret")],
)
def evaluate():
    import os, types, json
    import numpy as np
    import torch
    from datasets import load_from_disk, Audio
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    import evaluate as evaluate_lib

    device = torch.device("cuda")
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}")

    _mod = types.ModuleType("cleaner")
    exec(CLEANER_CODE, _mod.__dict__)
    clean_transcript = _mod.clean_transcript

    print("Loading dataset from /data/afrispeech_arrow...")
    dataset    = load_from_disk("/data/afrispeech_arrow")
    wer_metric = evaluate_lib.load("wer")

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        processor = AutoProcessor.from_pretrained(model_name)
        processor.tokenizer.set_target_lang("eng")
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            target_lang             = "eng",
            ignore_mismatched_sizes = True,
        ).to(device)
        model.load_adapter("eng")
        model.eval()

        model_results = {}

        for domain, domain_label in DOMAINS:
            print(f"\n  Domain: {domain_label}")
            ds = dataset["test"]
            if domain:
                ds = ds.filter(lambda x: x["domain"] == domain)
            ds = ds.filter(lambda x: clean_transcript(x[TEXT_COLUMN]) is not None)
            ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
            print(f"  Samples: {len(ds):,}")

            all_preds, all_refs = [], []
            n_batches = (len(ds) + BATCH_SIZE - 1) // BATCH_SIZE

            for i in range(0, len(ds), BATCH_SIZE):
                batch   = ds[i : i + BATCH_SIZE]
                arrays  = [
                    np.asarray(a["array"], dtype=np.float32)[:SAMPLING_RATE * MAX_AUDIO_SEC]
                    for a in batch["audio"]
                ]
                refs = [clean_transcript(t) or "" for t in batch[TEXT_COLUMN]]

                inputs = processor(
                    arrays,
                    sampling_rate  = SAMPLING_RATE,
                    return_tensors = "pt",
                    padding        = True,
                ).to(device)

                with torch.no_grad():
                    logits   = model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=-1)
                preds    = processor.batch_decode(pred_ids)
                preds    = [clean_transcript(p) or "" for p in preds]

                all_preds.extend(preds)
                all_refs.extend(refs)
                torch.cuda.empty_cache()

                batch_num = i // BATCH_SIZE + 1
                if batch_num % 50 == 0 or batch_num == n_batches:
                    running_wer = wer_metric.compute(predictions=all_preds, references=all_refs)
                    print(f"    Batch {batch_num}/{n_batches} | WER: {running_wer:.3f}")

            wer = round(wer_metric.compute(predictions=all_preds, references=all_refs), 3)
            print(f"  ✓ WER [{domain_label}]: {wer}")
            model_results[domain_label] = {"wer": wer, "samples": len(all_preds)}

        all_results[model_name] = model_results

        # Free GPU before loading next model
        del model
        torch.cuda.empty_cache()

        # Save after each model
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        data_vol.commit()
        print(f"  Results saved to {OUTPUT_PATH}")

    # Print summary
    print(f"\n{'='*65}")
    print(f"  ZERO-SHOT MMS RESULTS — AfriSpeech Test Set")
    print(f"{'='*65}")
    print(f"  {'Model':<30} {'General':>10} {'Clinical':>10} {'All':>10}")
    print(f"  {'-'*62}")
    for model_name, res in all_results.items():
        short = model_name.replace("facebook/", "")
        g = res.get("general",  {}).get("wer", "-")
        c = res.get("clinical", {}).get("wer", "-")
        a = res.get("all",      {}).get("wer", "-")
        print(f"  {short:<30} {str(g):>10} {str(c):>10} {str(a):>10}")

    return all_results


@app.local_entrypoint()
def main():
    print("Running MMS zero-shot evaluation on Modal A100...")
    results = evaluate.remote()
    print("\nDone! Results:")
    for model_name, res in results.items():
        print(f"\n  {model_name}")
        for domain, metrics in res.items():
            print(f"    {domain}: WER={metrics['wer']}")
