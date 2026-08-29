"""Configurable two-dimensional reducers for embedding visualizations."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


class ClassicalMDS:
    """Classical metric MDS with a deterministic out-of-sample transform.

    Classical MDS is normally defined only for the points used during fitting.
    The transform below uses the standard Gower out-of-sample extension so the
    fitted reducer can also project spline samples and station coordinates.
    """

    display_name_ = "Classical MDS"

    def __init__(
        self,
        n_components: int = 2,
        metric: str = "euclidean",
        random_state: int | None = 0,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.metric = metric
        # Kept for reducer API compatibility; classical MDS is deterministic.
        self.random_state = random_state

    @staticmethod
    def _squared_distances(
        left: np.ndarray, right: np.ndarray, metric: str,
    ) -> np.ndarray:
        distances = pairwise_distances(left, right, metric=metric)
        return np.asarray(distances, dtype=float) ** 2

    def fit(self, points: np.ndarray, y: Any = None) -> ClassicalMDS:
        del y
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or len(points) < 2:
            raise ValueError("points must be a two-dimensional array with at least two rows")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite values")

        squared_distances = self._squared_distances(points, points, self.metric)
        row_mean = np.mean(squared_distances, axis=1)
        grand_mean = float(np.mean(squared_distances))
        centered = -0.5 * (
            squared_distances
            - row_mean[:, None]
            - row_mean[None, :]
            + grand_mean
        )
        eigenvalues, eigenvectors = np.linalg.eigh(centered)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.asarray(eigenvalues[order], dtype=float)
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)

        positive = eigenvalues > max(1e-12, np.max(np.abs(eigenvalues)) * 1e-12)
        count = min(self.n_components, int(np.count_nonzero(positive)))
        selected_values = eigenvalues[:count]
        selected_vectors = eigenvectors[:, :count]
        embedding = selected_vectors * np.sqrt(selected_values)[None, :]

        # Fix the otherwise arbitrary eigenvector signs for reproducibility.
        for component in range(count):
            pivot = int(np.argmax(np.abs(embedding[:, component])))
            if embedding[pivot, component] < 0.0:
                embedding[:, component] *= -1.0
                selected_vectors[:, component] *= -1.0

        self._fit_points_ = points.copy()
        self._row_mean_ = row_mean
        self._grand_mean_ = grand_mean
        self._eigenvalues_ = selected_values
        self._eigenvectors_ = selected_vectors
        self.embedding_ = np.zeros((len(points), self.n_components), dtype=float)
        self.embedding_[:, :count] = embedding
        self.n_features_in_ = points.shape[1]
        return self

    def transform(self, points: np.ndarray) -> np.ndarray:
        if not hasattr(self, "embedding_"):
            raise RuntimeError("Call fit before transform")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.n_features_in_:
            raise ValueError("points has a different feature dimension than the fitted data")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite values")

        coordinates = np.zeros((len(points), self.n_components), dtype=float)
        if not len(self._eigenvalues_):
            return coordinates
        squared_distances = self._squared_distances(
            points, self._fit_points_, self.metric,
        )
        cross_mean = np.mean(squared_distances, axis=1)
        centered = -0.5 * (
            squared_distances
            - cross_mean[:, None]
            - self._row_mean_[None, :]
            + self._grand_mean_
        )
        coordinates[:, :len(self._eigenvalues_)] = (
            centered @ self._eigenvectors_
            / np.sqrt(self._eigenvalues_)[None, :]
        )
        return coordinates

    def fit_transform(self, points: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(points, y=y).embedding_


def make_reducer(
    method: str = "umap",
    random_state: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> Any:
    """Create a fitted-model-compatible 2D reducer.

    UMAP is the default for visual structure. PCA and classical MDS are
    available as deterministic alternatives with ``method="pca"`` and
    ``method="mds"`` respectively.
    """
    method = method.lower().strip()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        reducer.display_name_ = "PCA"
        return reducer
    if method in {"mds", "classical_mds"}:
        return ClassicalMDS(
            n_components=2, metric=metric, random_state=random_state,
        )
    if method != "umap":
        raise ValueError("method must be 'umap', 'pca', or 'mds'")
    try:
        from umap import UMAP
    except ImportError as error:
        raise ImportError(
            "UMAP is the default reducer. Install it with "
            "`python -m pip install umap-learn`, or select method='pca' or 'mds'."
        ) from error
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=1,
    )
    reducer.display_name_ = "UMAP"
    return reducer


def fit_reducer(points: np.ndarray, method: str = "umap", **kwargs: Any) -> Any:
    """Construct and fit a configured reducer on ``points``."""
    reducer = make_reducer(method=method, **kwargs)
    return reducer.fit(np.asarray(points, dtype=float))


__all__ = ["ClassicalMDS", "fit_reducer", "make_reducer"]
