"""
Anima - DSM-5 text & demographic model (shared architecture).

Defines the trained "head" that sits on top of the frozen Bio_ClinicalBERT
features produced by dsm5_features.py. This module is the SHARED contract:

    * train_dsm5.py       builds + trains DSM5Head, saves its weights.
    * DSM5_Analysis.py    rebuilds DSM5Head and loads those weights to score.

Keeping the class in one place means train and serve use the byte-for-byte same
architecture (no skew). The artifact is a {config, state_dict} dict of tensors +
primitives, so torch.load stays safe under the weights_only default of torch >= 2.6.

Design (deliberately tiny - only ~130 labelled patients)
--------------------------------------------------------
A linear layer -> one logit -> sigmoid ADHD-risk. Standardisation is baked in as
buffers (fitted on the TRAINING fold), so the SAME raw feature vector the API
builds can be fed straight in - no separate scaler to ship.

Optional note-embedding PCA (fusion control)
--------------------------------------------
The full feature vector is 5 strong structured features + a 768-d note embedding.
On small data those 768 dims can swamp the 5, so the head optionally projects the
note block down to `note_pca_k` dimensions with a PCA fitted on the training fold
(components stored as buffers). The projection runs INSIDE forward(), so the API
still passes the raw 773-d vector and needs no change. With note_pca_k=None the
head is a plain linear model over whatever input_dim it is given (also used for
the scores-only / notes-only ablation slices).
"""

import torch
import torch.nn as nn

# Must match dsm5_features: STRUCTURED_DIM (5) + EMBED_DIM (768) = 773.
STRUCTURED_DIM = 5
NOTE_DIM = 768
FEATURE_DIM = STRUCTURED_DIM + NOTE_DIM


class DSM5Head(nn.Module):
    """Linear ADHD-risk head; standardisation (and optional note-PCA) baked in."""

    def __init__(self, input_dim: int = FEATURE_DIM, note_pca_k: int = None,
                 structured_dim: int = STRUCTURED_DIM):
        super().__init__()
        self.input_dim = int(input_dim)
        self.note_pca_k = int(note_pca_k) if note_pca_k else None
        self.structured_dim = int(structured_dim)

        if self.note_pca_k:
            note_dim = self.input_dim - self.structured_dim
            self.register_buffer("note_mean", torch.zeros(note_dim))
            self.register_buffer("note_components", torch.zeros(self.note_pca_k, note_dim))
            feat_dim = self.structured_dim + self.note_pca_k
        else:
            feat_dim = self.input_dim

        self.feat_dim = feat_dim
        self.register_buffer("feat_mean", torch.zeros(feat_dim))
        self.register_buffer("feat_std", torch.ones(feat_dim))
        self.linear = nn.Linear(feat_dim, 1)

    # -- fold-fit statistics (buffers; never touched by the optimiser) ----------
    def set_pca(self, mean, components) -> None:
        """Store the note-block PCA (fit on the training fold)."""
        with torch.no_grad():
            self.note_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).flatten())
            self.note_components.copy_(torch.as_tensor(components, dtype=torch.float32))

    def set_normalization(self, mean, std, eps: float = 1e-6) -> None:
        """Set standardisation stats on the PROCESSED features (train fold only)."""
        with torch.no_grad():
            self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).flatten())
            std_t = torch.as_tensor(std, dtype=torch.float32).flatten().clamp(min=eps)
            self.feat_std.copy_(std_t)

    # -- forward ---------------------------------------------------------------
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Raw features -> processed features (identity, or structured+note-PCA)."""
        if not self.note_pca_k:
            return x
        structured = x[:, :self.structured_dim]
        notes = x[:, self.structured_dim:]
        notes = (notes - self.note_mean) @ self.note_components.t()   # (N, k)
        return torch.cat([structured, notes], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw features (N, input_dim) -> logits (N,)."""
        x = self.project(x)
        x = (x - self.feat_mean) / self.feat_std
        return self.linear(x).squeeze(-1)

    def config(self) -> dict:
        return {"input_dim": self.input_dim, "note_pca_k": self.note_pca_k,
                "structured_dim": self.structured_dim}

    # -- explainability helpers (exact, because the head is linear) -------------
    def structured_effective_weight(self) -> torch.Tensor:
        """Per-structured-feature weight acting on the RAW (unstandardised) value.

        Because the head standardises then applies a linear layer, feature i's
        contribution to the logit is w_i * (x_i - mean_i), where w_i is this
        weight. Lets the API show exactly how much each score/demographic pushed
        the prediction.
        """
        w = self.linear.weight.detach().squeeze(0)          # (feat_dim,)
        return w[:self.structured_dim] / self.feat_std[:self.structured_dim]

    def note_effective_weight(self) -> torch.Tensor:
        """768-d weight ``w`` such that the clinical note contributes ``w . e`` to
        the logit (plus a constant), where ``e`` is the note embedding.

        Folds the note-block linear weights, the standardisation, and (if used) the
        note-PCA back onto the raw 768-d embedding space, so a single dot product
        with each word's embedding gives that word's push on the score.
        """
        w = self.linear.weight.detach().squeeze(0)          # (feat_dim,)
        g = w[self.structured_dim:] / self.feat_std[self.structured_dim:]  # note dims
        if self.note_pca_k:
            return self.note_components.detach().t() @ g     # (768,)
        return g                                             # (768,)


def save_head(model: DSM5Head, path: str) -> None:
    """Persist as {config, state_dict} - all tensors/primitives (weights_only-safe)."""
    torch.save({"config": model.config(), "state_dict": model.state_dict()}, path)


def load_head(path: str, map_location: str = "cpu") -> DSM5Head:
    """Rebuild DSM5Head from a saved artifact; returns an eval-mode model.

    Accepts the {config, state_dict} format and (for backward compatibility) a
    bare state_dict, which is assumed to be a plain full 773-d head.
    """
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        cfg = obj.get("config", {})
        model = DSM5Head(input_dim=cfg.get("input_dim", FEATURE_DIM),
                         note_pca_k=cfg.get("note_pca_k"),
                         structured_dim=cfg.get("structured_dim", STRUCTURED_DIM))
        model.load_state_dict(obj["state_dict"])
    else:
        model = DSM5Head(input_dim=FEATURE_DIM)
        model.load_state_dict(obj)
    model.eval()
    return model
