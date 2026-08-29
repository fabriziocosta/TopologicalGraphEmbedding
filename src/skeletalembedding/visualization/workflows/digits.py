"""Reusable workflow for the lifted 3D/dataset-catalog notebook."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
    make_blobs,
    make_circles,
    make_classification,
    make_gaussian_quantiles,
    make_moons,
)

from ...datasets import generate_synthetic_datasets
from ...estimator import SkeletalEmbedding


def make_spiral(n_samples: int = 500, noise: float = 0.045, turns: float = 1.15, random_state: int = 5):
    rng = np.random.default_rng(random_state)
    theta = np.linspace(0.0, 2.0 * np.pi * turns, n_samples)
    radius = np.linspace(0.1, 1.0, n_samples)
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    points += rng.normal(scale=noise, size=points.shape)
    return points[rng.permutation(n_samples)]


def lift_planar_dataset(dataset, z_noise: float = 0.045, random_state: int = 0):
    """Add independent Z noise while retaining the planar signal."""
    points, labels = dataset
    points = np.asarray(points, dtype=float)
    rng = np.random.default_rng(random_state)
    return np.column_stack([points, rng.normal(scale=z_noise, size=len(points))]), np.asarray(labels)


def build_dataset_catalog(n_samples: int = 500, random_state: int = 0) -> tuple[dict[str, Any], set[str]]:
    """Build the deterministic mixed-dimensional dataset catalogue."""
    synthetic = {
        f"synthetic/{name}": (points, np.zeros(len(points), dtype=int))
        for name, points in generate_synthetic_datasets(
            n=n_samples, noise=0.045, random_state=random_state, binary_tree_depth=3,
        ).items()
    }
    toys = {
        "toy/moons": make_moons(n_samples=n_samples, noise=0.07, random_state=0),
        "toy/circles": make_circles(n_samples=n_samples, factor=0.42, noise=0.045, random_state=1),
        "toy/spiral": (make_spiral(n_samples=n_samples), np.zeros(n_samples, dtype=int)),
        "toy/blobs": make_blobs(n_samples=n_samples, centers=[(-1.2, -0.8), (0.0, 1.0), (1.2, -0.4)], cluster_std=[0.22, 0.28, 0.20], random_state=2),
        "toy/classification": make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=1.25, flip_y=0.04, random_state=3),
        "toy/gaussian-quantiles": make_gaussian_quantiles(n_samples=n_samples, n_features=2, n_classes=3, random_state=4),
    }
    planar = {**synthetic, **toys}
    planar = {
        name: lift_planar_dataset(dataset, random_state=index)
        for index, (name, dataset) in enumerate(planar.items())
    }
    high_dim = {
        "high-dimensional/digits": (load_digits().data, load_digits().target),
        "high-dimensional/wine": (load_wine().data, load_wine().target),
        "high-dimensional/breast-cancer": (load_breast_cancer().data, load_breast_cancer().target),
        "high-dimensional/diabetes": (load_diabetes().data, load_diabetes().target),
    }
    return {**planar, **high_dim}, set(planar)


def fit_catalog_entry(name: str, dataset: Any, *, smoothness: float = 0.02, random_state: int = 0):
    """Fit one catalogue entry with notebook-compatible defaults."""
    points, _labels = dataset
    is_planar = name.startswith(("synthetic/", "toy/"))
    is_binary_tree = name == "synthetic/binary-tree"
    model = SkeletalEmbedding(
        initialization="legacy_coarsen" if is_binary_tree else "skeletal",
        n_centroids=36,
        persistence_threshold=4.0 if is_planar else None,
        spline_smoothing=float(smoothness),
        max_cycles=0 if is_binary_tree else 4,
        random_state=random_state,
        standardize=not is_planar,
        persistence_max_points=500 if is_planar else 60,
        spline_samples_per_node=12,
        topology_neighbors=6,
        use_local_pca=True,
        local_pca_neighbors=20,
        use_tangent_boundary_conditions=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = model.fit_transform(points)
    return model, result


__all__ = ["build_dataset_catalog", "fit_catalog_entry", "lift_planar_dataset", "make_spiral"]
