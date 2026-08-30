"""Reusable workflow for the scikit-learn toy-dataset notebook."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_classification,
    make_gaussian_quantiles,
    make_moons,
)

from ...estimator import SkeletalEmbedding


def make_spiral(n_samples: int = 500, noise: float = 0.045, turns: float = 1.15, random_state: int = 5):
    """Generate a noisy open spiral and a single plotting label."""
    rng = np.random.default_rng(random_state)
    theta = np.linspace(0.0, 2.0 * np.pi * turns, n_samples)
    radius = np.linspace(0.1, 1.0, n_samples)
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    labels = np.zeros(n_samples, dtype=int)
    permutation = rng.permutation(n_samples)
    points += rng.normal(scale=noise, size=points.shape)
    return points[permutation], labels[permutation]


def build_toy_datasets(n_samples: int = 500) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build the deterministic toy catalogue used by the notebook."""
    return {
        "moons": make_moons(n_samples=n_samples, noise=0.07, random_state=0),
        "circles": make_circles(n_samples=n_samples, factor=0.42, noise=0.045, random_state=1),
        "spiral": make_spiral(n_samples=n_samples),
        "blobs": make_blobs(
            n_samples=n_samples,
            centers=[(-1.2, -0.8), (0.0, 1.0), (1.2, -0.4)],
            cluster_std=[0.22, 0.28, 0.20],
            random_state=2,
        ),
        "classification": make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=1.25,
            flip_y=0.04,
            random_state=3,
        ),
        "gaussian-quantiles": make_gaussian_quantiles(
            n_samples=n_samples, n_features=2, n_classes=3, random_state=4,
        ),
    }


def fit_toy_datasets(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    random_state: int = 10,
) -> tuple[dict[str, SkeletalEmbedding], dict[str, Any], Any]:
    """Fit all toy datasets and return models, results, and a summary frame."""
    import pandas as pd

    models: dict[str, SkeletalEmbedding] = {}
    embeddings: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for index, (name, (points, labels)) in enumerate(datasets.items()):
        model = SkeletalEmbedding(
            n_centroids=45,
            persistence_threshold=4.0,
            spline_smoothing=0.1,
            max_cycles=4,
            random_state=random_state + index,
            persistence_max_points=60,
            spline_samples_per_node=12,
            topology_neighbors=6,
            detect_cycles=True,
            detect_junctions=True,
            use_local_pca=True,
            local_pca_neighbors=20,
            max_branch_angle_degrees=45.0,
            use_tangent_boundary_conditions=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = model.fit_transform(points)
        models[name] = model
        embeddings[name] = result
        summary_rows.append({
            "dataset": name,
            "cycles": model.realized_cycle_count_,
            "junctions": len(model.junctions_),
            "endpoints": len(model.endpoints_),
            "spline_chains": len(model.splines_),
            "ribs": len(model.rib_paths_),
            "median_residual": float(np.median(result.residual_norm)),
        })
    return models, embeddings, pd.DataFrame(summary_rows)


def render_toy_datasets(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    models: dict[str, SkeletalEmbedding],
    embeddings: dict[str, Any],
    summary: Any,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    """Render the toy grid and summary table."""
    import matplotlib.pyplot as plt

    from ...plots import plot_embedding_row

    figure, axes = plt.subplots(len(datasets), 4, figsize=(26, 4 * len(datasets)))
    axes = np.atleast_2d(axes)
    for row, (name, (points, labels)) in enumerate(datasets.items()):
        plot_embedding_row(
            axes=axes[row], points=points, labels=labels, model=models[name],
            result=embeddings[name], projected_title=f"{name}: data and skeletal spline network",
            graph_title=f"{name}: skeleton embedding", metro_lines_title=f"{name}: metro-map lines",
            metro_points_title=f"{name}: metro-map points", reducer=None, jitter_seed=row,
            show_metro_nodes=False, metro_residual_width=0.08,
        )
    figure.suptitle("Scikit-learn toy datasets embedded with skeleton splines", fontsize=16, y=0.995)
    figure.tight_layout()
    table_figure, table_axis = plt.subplots(figsize=(12, 3.0))
    table_axis.axis("off")
    table = table_axis.table(
        cellText=summary.values.tolist(), colLabels=summary.columns.tolist(),
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    table_figure.tight_layout()
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination / "sklearn_toy_datasets.png", dpi=160, bbox_inches="tight")
        table_figure.savefig(destination / "sklearn_toy_summary.png", dpi=160, bbox_inches="tight")
    return figure, table_figure


__all__ = ["build_toy_datasets", "fit_toy_datasets", "make_spiral", "render_toy_datasets"]
