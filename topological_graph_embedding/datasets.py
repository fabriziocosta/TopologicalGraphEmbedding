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


def noisy_binary_tree(
    n: int = 500,
    noise: float = 0.045,
    rng: np.random.Generator | None = None,
    *,
    depth: int = 3,
) -> np.ndarray:
    """Sample a noisy complete binary tree in two dimensions.

    ``depth`` is the number of edges from the root to every leaf.  Thus a
    depth-1 tree has one root and two leaves, while a depth-3 tree has eight
    leaves and seven internal branching nodes.
    """
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or depth < 1:
        raise ValueError("depth must be a positive integer")

    rng = np.random.default_rng() if rng is None else rng
    depth = int(depth)
    horizontal_span = 2.5
    vertical_span = 2.4
    nodes_by_level: list[list[np.ndarray]] = []
    for level in range(depth + 1):
        count = 2**level
        if level == 0:
            x_positions = np.zeros(1, dtype=float)
        else:
            x_positions = np.linspace(-horizontal_span / 2.0, horizontal_span / 2.0, count)
        y = vertical_span / 2.0 - vertical_span * level / depth
        nodes_by_level.append([
            np.array([x, y], dtype=float) for x in x_positions
        ])

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for level in range(1, depth + 1):
        parents = nodes_by_level[level - 1]
        for child_index, child in enumerate(nodes_by_level[level]):
            segments.append((parents[child_index // 2], child))
    return _add_noise(_sample_segments(segments, n, rng), noise, rng)


def noisy_loop_branch(n: int = 500, noise: float = 0.045, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    loop_count = round(n * 0.78)
    angle = rng.uniform(0.0, 2.0 * np.pi, size=loop_count)
    loop = np.column_stack([np.cos(angle), np.sin(angle)])
    branch_count = n - loop_count
    branch_t = rng.uniform(0.0, 1.0, size=branch_count)
    branch = np.column_stack([np.ones(branch_count) + 0.95 * branch_t, 0.75 * branch_t])
    return _add_noise(np.vstack([loop, branch]), noise, rng)


def noisy_polygon_rays_circles(
    n: int = 500,
    noise: float = 0.045,
    rng: np.random.Generator | None = None,
    *,
    n_sides: int = 5,
    radius: float = 1.0,
    circle_radius: float = 0.22,
) -> np.ndarray:
    """Sample a polygon-and-circles graph with isotropic observation noise.

    The latent graph contains the boundary of a regular polygon, a ray from
    the center to every vertex, and a circle attached to each vertex.  Each
    circle is placed outside the polygon so that its nearest point is the
    corresponding vertex.  Primitive curves are sampled in proportion to
    their lengths before Gaussian noise is added.
    """
    if n_sides < 3:
        raise ValueError("n_sides must be at least 3")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if circle_radius <= 0.0:
        raise ValueError("circle_radius must be positive")

    rng = np.random.default_rng() if rng is None else rng
    angles = np.linspace(
        np.pi / 2.0,
        np.pi / 2.0 + 2.0 * np.pi,
        n_sides,
        endpoint=False,
    )
    vertices = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.zeros(2)

    polygon_edges = [
        (vertices[index], vertices[(index + 1) % n_sides])
        for index in range(n_sides)
    ]
    rays = [(center, vertex) for vertex in vertices]
    segments = polygon_edges + rays

    # Shifting a circle center outward by its radius makes the polygon vertex
    # the inward tangent point of that circle.
    circle_centers = vertices + circle_radius * vertices / radius
    segment_lengths = np.asarray(
        [np.linalg.norm(end - start) for start, end in segments], dtype=float,
    )
    circle_lengths = np.full(n_sides, 2.0 * np.pi * circle_radius)
    primitive_lengths = np.concatenate([segment_lengths, circle_lengths])
    probabilities = primitive_lengths / np.sum(primitive_lengths)
    choices = rng.choice(len(primitive_lengths), size=n, p=probabilities)
    values = rng.random(n)
    points = np.empty((n, 2), dtype=float)

    for index, primitive_index in enumerate(choices):
        if primitive_index < len(segments):
            start, end = segments[primitive_index]
            points[index] = start + values[index] * (end - start)
        else:
            circle_index = primitive_index - len(segments)
            angle = 2.0 * np.pi * values[index]
            points[index] = circle_centers[circle_index] + circle_radius * np.array([
                np.cos(angle), np.sin(angle),
            ])

    return _add_noise(points, noise, rng)


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


SYNTHETIC_DATASETS: dict[str, Callable[..., np.ndarray]] = {
    "line": noisy_line,
    "y": noisy_y,
    "x": noisy_x,
    "circle": noisy_circle,
    "figure-eight": noisy_figure_eight,
    "binary-tree": noisy_binary_tree,
    "loop-branch": noisy_loop_branch,
    "polygon-rays-circles": noisy_polygon_rays_circles,
}


def generate_synthetic_datasets(
    n: int = 500,
    noise: float = 0.045,
    random_state: int = 0,
    polygon_sides: int = 5,
    binary_tree_depth: int = 3,
) -> dict[str, np.ndarray]:
    """Generate all eight benchmark point clouds with independent seeds.

    ``polygon_sides`` controls the regular polygon in the
    ``"polygon-rays-circles"`` dataset, and ``binary_tree_depth`` controls
    the root-to-leaf depth of the ``"binary-tree"`` dataset.
    """
    result = {}
    for offset, (name, factory) in enumerate(SYNTHETIC_DATASETS.items()):
        rng = np.random.default_rng(random_state + offset)
        if name == "polygon-rays-circles":
            result[name] = factory(
                n=n,
                noise=noise,
                rng=rng,
                n_sides=polygon_sides,
            )
        elif name == "binary-tree":
            result[name] = factory(
                n=n,
                noise=noise,
                rng=rng,
                depth=binary_tree_depth,
            )
        else:
            result[name] = factory(n=n, noise=noise, rng=rng)
    return result


__all__ = [
    "SYNTHETIC_DATASETS",
    "generate_synthetic_datasets",
    "noisy_binary_tree",
    "noisy_hypercube",
    "noisy_polygon_rays_circles",
]
