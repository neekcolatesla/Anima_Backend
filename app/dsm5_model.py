"""
Anima - DSM-5 text & demographic model (shared architecture).

Defines the trained "head" that sits on top of the frozen Bio_ClinicalBERT
features produced by dsm5_features.py. This module is the SHARED contract:

    * train_dsm5.py       builds + trains DSM5Head, saves its weights.
    * DSM5_Analysis.py    rebuilds DSM5Head and loads those weights to score.

Keeping the class in one place means train and serve use the byte-for-byte same
architecture (no skew), and the saved artifact is a plain state_dict (tensors
only) so torch.load stays safe under the weights_only default of torch >= 2.6.

Design (deliberately tiny - only ~130 labelled patients)
--------------------------------------------------------
A single linear layer over the 773-d feature vector (age, sex, is_child, the two
Conners' subscale T-scores, and the 768-d note embedding) -> one logit ->
sigmoid = ADHD-risk probability. A linear model is the least likely to just
memorise a small dataset; overfitting is further held back by L2 weight decay in
the trainer and by standardising the inputs.

Standardisation is baked into the module as (feat_mean, feat_std) buffers, fitted
on the TRAINING fold and saved with the weights. So the exact same RAW feature
vector the API builds can be fed straight in - the module normalises internally,
and there is no separate scaler object to ship or keep in sync.
"""

import torch
import torch.nn as nn

# Must match dsm5_features.FEATURE_DIM (STRUCTURED_DIM 5 + EMBED_DIM 768).
FEATURE_DIM = 773


class DSM5Head(nn.Module):
    """Linear ADHD-risk head with input standardisation baked in as buffers."""

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        # Buffers (not parameters): saved in the state_dict, moved with .to(),
        # but never updated by the optimiser. Default to a no-op transform.
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))
        self.linear = nn.Linear(input_dim, 1)

    def set_normalization(self, mean, std, eps: float = 1e-6) -> None:
        """Set the standardisation stats (fit on the training fold only).

        std is clamped away from zero so constant features (e.g. a BERT dim that
        is always 0 for empty notes) don't produce inf/NaN.
        """
        with torch.no_grad():
            mean_t = torch.as_tensor(mean, dtype=torch.float32).flatten()
            std_t = torch.as_tensor(std, dtype=torch.float32).flatten().clamp(min=eps)
            self.feat_mean.copy_(mean_t)
            self.feat_std.copy_(std_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw features (N, 773) -> logits (N,). Standardises internally."""
        x = (x - self.feat_mean) / self.feat_std
        return self.linear(x).squeeze(-1)


def load_head(path: str, input_dim: int = FEATURE_DIM,
              map_location: str = "cpu") -> DSM5Head:
    """Rebuild DSM5Head and load a saved state_dict; returns an eval-mode model.

    The artifact is a state_dict (tensors only), so this is safe under the
    torch >= 2.6 ``weights_only=True`` default.
    """
    model = DSM5Head(input_dim)
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model
