"""Synthetic manifolds used by the end-to-end prototype demo."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np


def _sample_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lengths = np.asarray([np.linalg.norm(end - start) for start, end in segments], dtype=float)
    probabilities = lengths / np.sum(lengths)
    choices = rng.choice(len(segments), size=n, p=probabilities)
    values = rng.random(n)
    points = np.empty((n, 2), dtype=float)
    for index, segment_index in enumerate(choices):
        start, end = segments[segment_index]
        points[index] = start + values[index] * (end - start)
    return points


def _add_noise(points: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    return points + rng.normal(scale=noise, size=points.shape)


def noisy_line(n: int = 500, noise: float = 0.045, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    t = rng.uniform(-1.2, 1.2, size=n)
    return _add_noise(np.column_stack([t, np.zeros(n)]), noise, rng)


def noisy_y(n: int = 500, noise: float = 0.045, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    center = np.array([0.0, 0.0])
    angles = np.deg2rad([90.0, 210.0, 330.0])
    segments = [(center, center + 1.25 * np.array([np.cos(angle), np.sin(angle)])) for angle in angles]
    return _add_noise(_sample_segments(segments, n, rng), noise, rng)


def noisy_x(n: int = 500, noise: float = 0.045, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    center = np.zeros(2)
    segments = [
        (center, length * np.array([np.cos(angle), np.sin(angle)]))
        for angle in np.deg2rad([45.0, 135.0, 225.0, 315.0])
        for length in [1.25]
    ]
    return _add_noise(_sample_segments(segments, n, rng), noise, rng)


def noisy_circle(n: int = 500, noise: float = 0.045, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = 1.0 + rng.normal(scale=noise * 0.45, size=n)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]) + rng.normal(scale=noise, size=(n, 2))


def noisy_figure_eight(n: int = 500, noise: float = 0.045, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    # Both lobes meet at the origin, giving two cycles with one shared node.
    points = np.column_stack([np.sin(angle), np.sin(angle) * np.cos(angle)])
    return _add_noise(points, noise, rng)


def noisy_branching_tree(n: int = 500, noise: float = 0.045, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    center = np.array([0.0, -0.15])
    upper = np.array([0.0, 0.95])
    left = np.array([-0.85, 1.45])
    right = np.array([0.85, 1.45])
    lower_left = np.array([-0.95, -1.15])
    lower_right = np.array([0.95, -1.15])
    segments = [(center, endpoint) for endpoint in [upper, left, right, lower_left, lower_right]]
    return _add_noise(_sample_segments(segments, n, rng), noise, rng)


def noisy_loop_branch(n: int = 500, noise: float = 0.045, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    loop_count = int(round(n * 0.78))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=loop_count)
    loop = np.column_stack([np.cos(angle), np.sin(angle)])
    branch_count = n - loop_count
    branch_t = rng.uniform(0.0, 1.0, size=branch_count)
    branch = np.column_stack([np.ones(branch_count) + 0.95 * branch_t, 0.75 * branch_t])
    return _add_noise(np.vstack([loop, branch]), noise, rng)


def noisy_hypercube(
    n: int = 500,
    dim: int = 4,
    noise: float = 0.045,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample noisy points along the edges of a ``dim``-dimensional hypercube."""
    if dim < 1:
        raise ValueError("dim must be at least 1")
    rng = np.random.default_rng() if rng is None else rng
    vertices = np.asarray(list(product((-1.0, 1.0), repeat=dim)))
    vertex_lookup = {tuple(vertex): index for index, vertex in enumerate(vertices)}
    edges: list[tuple[int, int]] = []
    for index, vertex in enumerate(vertices):
        for axis in range(dim):
            if vertex[axis] < 0:
                other = vertex.copy()
                other[axis] = 1.0
                edges.append((index, vertex_lookup[tuple(other)]))
    edge_indices = rng.integers(len(edges), size=n)
    selected_edges = np.asarray(edges, dtype=int)[edge_indices]
    position = rng.random(n)
    starts = vertices[selected_edges[:, 0]]
    ends = vertices[selected_edges[:, 1]]
    points = starts + position[:, None] * (ends - starts)
    return points + rng.normal(scale=noise, size=points.shape)


DATASET_FACTORIES: dict[str, Callable[..., np.ndarray]] = {
    "line": noisy_line,
    "y": noisy_y,
    "x": noisy_x,
    "circle": noisy_circle,
    "figure-eight": noisy_figure_eight,
    "branching-tree": noisy_branching_tree,
    "loop-branch": noisy_loop_branch,
}


def generate_datasets(
    n: int = 500,
    noise: float = 0.045,
    random_state: int = 0,
) -> dict[str, np.ndarray]:
    """Generate all seven benchmark point clouds with independent seeds."""
    result = {}
    for offset, (name, factory) in enumerate(DATASET_FACTORIES.items()):
        rng = np.random.default_rng(random_state + offset)
        result[name] = factory(n=n, noise=noise, rng=rng)
    return result


__all__ = ["DATASET_FACTORIES", "generate_datasets", "noisy_hypercube"]

