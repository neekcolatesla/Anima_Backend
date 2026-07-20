"""
Anima - shared MRI slice selection + pairing.

Used by BOTH the trainer (train_mri.py) and the serving engine (MRI_Analysis.py)
so they pick the SAME slices from a patient's anat / anat_gm folders. Identical
selection at train and inference is what prevents train/serve skew - the same
lesson dsm5_features.py enforces for the text model.

Why select slices at all?
-------------------------
A scan's full axial stack (189 slices) includes the very top (skull cap) and very
bottom (neck) of the volume, which are mostly background. Training and scoring on
those near-empty slices dilutes the signal: their per-slice predictions hover at
~0.5 and drag the patient-level mean toward chance. Restricting to the informative
central region (where brain tissue actually is) is standard, principled practice -
it is NOT tuning to the test set.

Two selectors, applied in order:
  1. central-fraction crop - keep the middle ``central_frac`` of the stack
     (deterministic, cheap, no image loading);
  2. optional foreground filter - drop any remaining slice whose fraction of
     non-background pixels is below ``min_foreground`` (data-driven; loads the
     anat image to measure). Off by default.
Then an optional even subsample to ``max_slices`` bounds cost while keeping
whole-region coverage.
"""

import os
import glob

# Pixel intensity (0-255) above which a pixel counts as brain tissue rather than
# background, for the optional foreground filter.
FG_PIXEL_THRESHOLD = 15


def list_slices(directory: str) -> list:
    """Sorted JPEG slice paths in a scan folder (numbered names preserve order)."""
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(glob.glob(os.path.join(directory, "*.jpg")))


def _central_indices(n: int, central_frac: float) -> list:
    """Indices of the middle ``central_frac`` of a length-n stack (centered)."""
    if not central_frac or central_frac >= 1.0:
        return list(range(n))
    keep = max(1, int(round(n * float(central_frac))))
    start = (n - keep) // 2
    return list(range(start, start + keep))


def _evenly_sample(items: list, k) -> list:
    """Evenly sample at most k items across the list (order preserved)."""
    if not k or len(items) <= k:
        return items
    step = len(items) / float(k)
    return [items[int(i * step)] for i in range(k)]


def _foreground_fraction(path: str) -> float:
    """Fraction of pixels above the tissue threshold (lazy PIL/numpy import)."""
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return float((arr > FG_PIXEL_THRESHOLD).mean())


def pair_slices(anat_dir: str, gm_dir: str, central_frac: float = 0.6,
                min_foreground: float = 0.0, max_slices=0) -> list:
    """Return a list of (anat_path, gm_path) pairs after selection.

    The two stacks are paired positionally (both are the same ordered axial
    series). Selection order: central crop -> optional foreground filter ->
    optional even subsample. Returns [] if either stack is empty.
    """
    anat = list_slices(anat_dir)
    gm = list_slices(gm_dir)
    n = min(len(anat), len(gm))
    if n == 0:
        return []

    idx = _central_indices(n, central_frac)
    pairs = [(anat[i], gm[i]) for i in idx]

    if min_foreground and min_foreground > 0:
        kept = [(a, g) for (a, g) in pairs
                if _foreground_fraction(a) >= min_foreground]
        # Never let the filter empty a patient out - fall back to the central set.
        if kept:
            pairs = kept

    if max_slices:
        pairs = _evenly_sample(pairs, max_slices)

    return pairs
