"""
Anima - DSM-5 feature extraction (shared train/serve contract).

Turns a patient's structured details + free-text clinical narrative into a single
PyTorch tensor that the text & demographic model consumes. This module is the
SHARED contract: both DSM5_Analysis.py (inference) and the training script import
it, so features are computed identically at train and serve time (no skew).

Design
------
* Bio_ClinicalBERT is loaded ONCE per process (cached singleton) and used only as
  a FROZEN feature extractor - eval mode, no gradients, so it's light on memory.
* A clinical note is embedded into a fixed 768-d vector via masked mean-pooling
  over the last hidden state (more robust than the raw [CLS] token).
* The note embedding is concatenated with the structured features (age,
  biological_sex, is_child, inattentive_score, hyperactive_score) into one vector.

Feature layout (length = STRUCTURED_DIM + EMBED_DIM = 773)
    [0]   age
    [1]   biological_sex        (1 = Male, 0 = Female, 0.5 = unknown)
    [2]   is_child              (1 / 0)
    [3]   inattentive_score     (Conners' T-score)
    [4]   hyperactive_score     (Conners' T-score)
    [5:]  clinical-note embedding (768 dims)

Features are returned RAW; feature scaling (StandardScaler) is fitted during
training and persisted with the model artifact, so it stays part of the trained
head rather than being baked in here.

The heavy model is loaded lazily on first use, so importing this module is cheap;
call warmup() at API startup if you want to pay that cost up front.
"""

import os
import logging
from functools import lru_cache
from typing import Optional, Sequence, Union

import torch
import transformers
from transformers import AutoTokenizer, AutoModel

# Quieten the verbose weight-load report (the "UNEXPECTED keys" table is the
# expected MLM head we discard when loading Bio_ClinicalBERT as a plain encoder).
transformers.logging.set_verbosity_error()

logger = logging.getLogger("anima.dsm5.features")

# --- Configuration -----------------------------------------------------------
MODEL_NAME = os.getenv("CLINICAL_BERT_MODEL", "emilyalsentzer/Bio_ClinicalBERT")
MAX_TOKENS = int(os.getenv("CLINICAL_BERT_MAX_TOKENS", "256"))
EMBED_DIM = 768        # Bio_ClinicalBERT is BERT-base -> 768-d hidden state
STRUCTURED_FEATURE_NAMES = [
    "age", "biological_sex", "is_child", "inattentive_score", "hyperactive_score",
]
STRUCTURED_DIM = len(STRUCTURED_FEATURE_NAMES)
FEATURE_DIM = STRUCTURED_DIM + EMBED_DIM   # 773


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_model():
    """Load (and cache) the Bio_ClinicalBERT tokenizer + model exactly once.

    Cached via lru_cache so repeated calls reuse the same in-memory instance.
    The model is placed in eval mode and used only for inference (frozen).
    """
    device = _device()
    logger.info("Loading clinical language model '%s' on %s ...", MODEL_NAME, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    logger.info("Clinical language model loaded.")
    return tokenizer, model, device


def warmup() -> None:
    """Preload the model (e.g. at API startup) so the first request isn't slow."""
    _load_model()


def _to_float(value, default: float = 0.0) -> float:
    """Coerce a DB value to float, mapping None/blank/unparseable to a default."""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default)


@torch.no_grad()
def embed_notes(notes: Union[str, Sequence[Optional[str]]]) -> torch.Tensor:
    """Embed one or many clinical notes into fixed 768-d vectors.

    Accepts a single string or a sequence of strings (batched in ONE forward
    pass for efficiency - important for training). Empty/None notes map to a
    zero vector so "no narrative" is a neutral input the head can learn from.

    Returns a (EMBED_DIM,) tensor for a single note, or (N, EMBED_DIM) for a list.
    """
    single = isinstance(notes, str) or notes is None
    seq = [notes] if single else list(notes)

    # Blank out missing notes; remember which rows to zero afterwards.
    texts = [(t if (t is not None and str(t).strip()) else "") for t in seq]
    is_empty = [t == "" for t in texts]

    tokenizer, model, device = _load_model()
    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=MAX_TOKENS, return_tensors="pt",
    ).to(device)

    last_hidden = model(**enc).last_hidden_state          # (N, T, 768)
    mask = enc["attention_mask"].unsqueeze(-1).float()    # (N, T, 1)
    summed = (last_hidden * mask).sum(dim=1)              # (N, 768)
    counts = mask.sum(dim=1).clamp(min=1.0)              # (N, 1)
    embeddings = (summed / counts).cpu().float()         # (N, 768) masked mean

    for i, empty in enumerate(is_empty):
        if empty:
            embeddings[i].zero_()

    return embeddings[0] if single else embeddings


@torch.no_grad()
def embed_note_tokens(note: Optional[str]):
    """Per-word contextual vectors for a note (for word-level explanations).

    Returns ``(words, word_vectors)`` where ``words`` is a list of the note's
    words and ``word_vectors`` is a (num_words, 768) tensor - each row is the SUM
    of that word's sub-token hidden states. Since the note embedding is a mean of
    these same token states, a dot product of each word vector with the model's
    note weight gives how hard that word pushed the score. Empty note -> ([], 0x768).
    """
    if not note or not str(note).strip():
        return [], torch.zeros((0, EMBED_DIM))

    tokenizer, model, device = _load_model()
    enc = tokenizer(str(note), truncation=True, max_length=MAX_TOKENS,
                    return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()   # char spans per token
    word_ids = enc.word_ids(0)                        # which word each token belongs to
    enc = enc.to(device)
    hidden = model(**enc).last_hidden_state[0].cpu()  # (T, 768)

    # Group sub-tokens by their source word (skip special tokens where word_id is None).
    groups = {}
    order = []
    for ti, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in groups:
            groups[wid] = []
            order.append(wid)
        groups[wid].append(ti)

    words, vectors = [], []
    text = str(note)
    for wid in order:
        idxs = groups[wid]
        start, end = offsets[idxs[0]][0], offsets[idxs[-1]][1]
        word = text[start:end]
        if not word.strip():
            continue
        words.append(word)
        vectors.append(hidden[idxs].sum(dim=0))
    word_vectors = torch.stack(vectors) if vectors else torch.zeros((0, EMBED_DIM))
    return words, word_vectors


def build_structured(age, biological_sex, is_child,
                     inattentive_score, hyperactive_score) -> torch.Tensor:
    """Assemble the 5-d structured feature vector in the fixed feature order."""
    return torch.tensor([
        _to_float(age),
        _to_float(biological_sex, 0.5),   # unknown sex -> neutral midpoint
        _to_float(is_child),
        _to_float(inattentive_score),
        _to_float(hyperactive_score),
    ], dtype=torch.float32)


def build_feature_vector(age, biological_sex, is_child,
                         hyperactive_score, inattentive_score,
                         clinical_notes: Optional[str]) -> torch.Tensor:
    """Package one patient's data into a single (773,) PyTorch feature tensor.

    Combines the structured demographic/score features with the Bio_ClinicalBERT
    embedding of the clinical notes. This is the per-patient entry point used at
    inference time.
    """
    structured = build_structured(
        age, biological_sex, is_child, inattentive_score, hyperactive_score
    )                                                     # (5,)
    note_embedding = embed_notes(clinical_notes)          # (768,)
    return torch.cat([structured, note_embedding], dim=0)  # (773,)


def build_feature_matrix(records: Sequence[dict]) -> torch.Tensor:
    """Package many patient records into an (N, 773) tensor for training.

    Each record is a dict with keys: age, biological_sex, is_child,
    inattentive_score, hyperactive_score, clinical_notes. Notes are embedded in a
    single batched forward pass, which is far faster than one call per patient.
    """
    structured = torch.stack([
        build_structured(
            r.get("age"), r.get("biological_sex"), r.get("is_child"),
            r.get("inattentive_score"), r.get("hyperactive_score"),
        )
        for r in records
    ])                                                    # (N, 5)

    embeddings = embed_notes([r.get("clinical_notes") for r in records])
    if embeddings.dim() == 1:                             # single record edge case
        embeddings = embeddings.unsqueeze(0)
    return torch.cat([structured, embeddings], dim=1)     # (N, 773)