"""Smooth tangent-orthogonal residual subspaces."""

from ._residual_pca import attach_residual_pca, fit_residual_pca


def fit_residual_subspaces(model, points, centerline_result):
    """Fit the model's smooth residual subspace fields."""
    return fit_residual_pca(model, points, centerline_result)


def attach_residual_subspaces(model, original, points, centerline_result):
    """Attach residual coordinates and reconstruction diagnostics."""
    return attach_residual_pca(model, original, points, centerline_result)

__all__ = [
    "attach_residual_pca",
    "attach_residual_subspaces",
    "fit_residual_pca",
    "fit_residual_subspaces",
]
