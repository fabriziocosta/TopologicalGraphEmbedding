"""Small utilities shared by optional stability selection."""

from __future__ import annotations

import numpy as np


def subsample_indices(n_samples: int, fraction: float, random_state: int) -> np.ndarray:
    """Return a without-replacement stability subsample."""
    size = max(3, int(np.ceil(float(fraction) * n_samples)))
    return np.random.default_rng(random_state).choice(n_samples, size=size, replace=False)


__all__ = ["subsample_indices"]
