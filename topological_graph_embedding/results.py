"""Typed outputs returned by the spline graph embedding estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Projection of observations onto a fitted spline route network.

    All arrays have one row per observation.  ``tangent`` is expressed in the
    standardized fitting coordinates used to construct the normal frame.
    """

    route_id: np.ndarray
    position: np.ndarray
    projected: np.ndarray
    residual: np.ndarray
    residual_norm: np.ndarray
    tangent: np.ndarray

    def __post_init__(self) -> None:
        if self.route_id.ndim != 1 or self.position.ndim != 1:
            raise ValueError("route_id and position must be one-dimensional")
        if self.projected.ndim != 2 or self.residual.ndim != 2 or self.tangent.ndim != 2:
            raise ValueError("projected, residual, and tangent must be two-dimensional")
        count = len(self.route_id)
        if any(len(array) != count for array in (
            self.position, self.projected, self.residual, self.residual_norm, self.tangent,
        )):
            raise ValueError("all result arrays must contain the same number of observations")
        if self.projected.shape != self.residual.shape or self.projected.shape != self.tangent.shape:
            raise ValueError("projected, residual, and tangent must have matching shapes")

    @property
    def n_samples(self) -> int:
        """Number of observations represented by the result."""
        return len(self.route_id)

    @property
    def n_features(self) -> int:
        """Number of original features represented by the result."""
        return self.projected.shape[1]


__all__ = ["EmbeddingResult"]
