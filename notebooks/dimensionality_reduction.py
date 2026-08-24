"""Configurable two-dimensional reducers for notebook visualizations."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA


def make_reducer(
    method: str = "umap",
    random_state: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> Any:
    """Create a fitted-model-compatible 2D reducer.

    UMAP is the default for visual structure; PCA remains available as a
    deterministic linear baseline with ``method="pca"``.
    """
    method = method.lower().strip()
    if method == "pca":
        return PCA(n_components=2, random_state=random_state)
    if method != "umap":
        raise ValueError("method must be 'umap' or 'pca'")
    try:
        from umap import UMAP
    except ImportError as error:
        raise ImportError(
            "UMAP is the default reducer. Install it with "
            "`python -m pip install umap-learn`, or select method='pca'."
        ) from error
    return UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=1,
    )


def fit_reducer(points: np.ndarray, method: str = "umap", **kwargs: Any) -> Any:
    """Construct and fit a configured reducer on ``points``."""
    reducer = make_reducer(method=method, **kwargs)
    return reducer.fit(np.asarray(points, dtype=float))


__all__ = ["fit_reducer", "make_reducer"]
