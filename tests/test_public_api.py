import importlib.util
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from topological_graph_embedding import EmbeddingResult, SplineGraphEmbedding
from topological_graph_embedding.datasets import generate_synthetic_datasets


def test_embedding_result_is_frozen_and_attribute_based():
    result = EmbeddingResult(
        route_id=np.array([0, 0]),
        position=np.array([0.1, 0.9]),
        projected=np.zeros((2, 2)),
        residual=np.ones((2, 2)),
        residual_norm=np.ones(2),
        tangent=np.tile([1.0, 0.0], (2, 1)),
    )
    assert [field.name for field in fields(result)] == [
        "route_id", "position", "projected", "residual", "residual_norm", "tangent",
    ]
    with pytest.raises(FrozenInstanceError):
        result.position = np.zeros(2)
    assert not hasattr(result, "__getitem__")


def test_embedding_projection_and_normal_coordinates_are_batch_independent():
    rng = np.random.default_rng(4)
    points = np.column_stack([
        np.linspace(-2.0, 2.0, 90),
        rng.normal(0.0, 0.06, 90),
        rng.normal(0.0, 0.06, 90),
        rng.normal(0.0, 0.06, 90),
    ])
    model = SplineGraphEmbedding(n_centroids=12, random_state=2).fit(points)
    full = model.transform(points)
    subset = model.transform(points[::3])
    coordinates = model.normal_coordinates(full)
    subset_coordinates = model.normal_coordinates(subset)
    assert np.all(np.isfinite(full.projected))
    assert np.all(full.route_id >= 0)
    assert np.allclose(coordinates[::3], subset_coordinates)


@pytest.mark.parametrize(
    ("name", "expected_cycles"),
    [("line", 0), ("star", 0), ("circle", 1), ("figure-eight", 2), ("loop-branch", 1)],
)
def test_synthetic_topology_diagnostics(name, expected_cycles):
    points = generate_synthetic_datasets(n=300, noise=0.03, random_state=0)[name]
    model = SplineGraphEmbedding(n_centroids=24, max_cycles=5, random_state=0).fit(points)
    assert model.realized_cycle_count_ == expected_cycles
    assert model.topology_shortfall_ == 0
    assert np.all(model.transform(points).route_id >= 0)


def test_empty_and_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        SplineGraphEmbedding().fit(np.empty((0, 2)))
    with pytest.raises(ValueError):
        SplineGraphEmbedding().fit(np.empty((3, 0)))
    with pytest.raises(ValueError):
        SplineGraphEmbedding(topology_neighbors=1)
    model = SplineGraphEmbedding(n_centroids=3).fit(np.full((5, 2), 2.0))
    assert np.isfinite(model.local_scale_)
    assert np.all(model.transform(np.full((5, 2), 2.0)).route_id >= 0)


def test_removed_root_modules_are_not_available():
    for module_name in ("topological_spline_graph", "spline_sklearn", "metro_layout", "synthetic_datasets"):
        assert importlib.util.find_spec(module_name) is None
