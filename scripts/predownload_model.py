"""
Anima - pre-download the clinical language model into the local Hugging Face cache.

Run this ONCE while online. Afterwards the API and training scripts can run with
HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 (set in .env) - they load Bio_ClinicalBERT
straight from the cache with no network calls, giving faster, network-independent
startup. Re-running is a harmless no-op if the model is already cached.

This also seeds the cache for a Docker image build (bake this into the image so
professors get instant, offline-safe startup).

Usage (from the repo root or app/, with the .venv active):
    python scripts/predownload_model.py
"""

import os
import sys

# Force ONLINE for the download itself, even if the shell / .env set offline mode.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

MODEL_NAME = os.getenv("CLINICAL_BERT_MODEL", "emilyalsentzer/Bio_ClinicalBERT")


def main() -> int:
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        sys.exit("ERROR: transformers is not installed. Run: pip install -r requirements.txt")

    print(f"Caching '{MODEL_NAME}' into the local Hugging Face cache ...")
    try:
        AutoTokenizer.from_pretrained(MODEL_NAME)
        AutoModel.from_pretrained(MODEL_NAME)
    except Exception as exc:
        sys.exit(f"ERROR: could not download '{MODEL_NAME}' ({exc}). "
                 f"Check your internet connection and try again.")

    print("Done. The model is cached.")
    print("The API / training scripts can now run offline "
          "(HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 in .env).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
