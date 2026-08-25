import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_circles

from topological_graph_embedding import SplineGraphEmbedding
from topological_graph_embedding._electrical import (
    _effective_resistance,
    _kron_reduction,
)
from topological_graph_embedding._topology import _weighted_symmetric_knn_graph
from topological_graph_embedding.datasets import generate_synthetic_datasets
from topological_graph_embedding.sklearn import SplineEmbeddingTransformer
from topological_graph_embedding.visualization.metro import MetroLayout


@pytest.mark.parametrize(
    ("name", "cycles", "junction_count", "branch_count"),
    [
        ("line", 0, 0, None),
        ("y", 0, 1, 3),
        ("x", 0, 1, 4),
        ("circle", 1, 0, None),
        ("figure-eight", 2, 1, 4),
        ("loop-branch", 1, 1, 3),
    ],
)
def test_topological_backbone_preserves_synthetic_structure(
    name, cycles, junction_count, branch_count,
):
    points = generate_synthetic_datasets(n=120, noise=0.03, random_state=0)[name]
    model = SplineGraphEmbedding(
        n_centroids=16,
        max_cycles=5,
        random_state=0,
        backbone_initialization="topological",
    ).fit(points)
    result = model.transform(points)

    assert model.realized_cycle_count_ == cycles
    assert model.topology_shortfall_ == 0
    assert len(model.junctions_) == junction_count
    assert np.all(np.isfinite(result.projected))
    assert np.all(result.route_id >= 0)
    assert model.backbone_graph_ is model.landmark_graph_
    assert model.backbone_paths_
    if branch_count is not None:
        assert model.junctions_[0].branch_count == branch_count
        assert model.junctions_[0].node_id in model.junction_branch_directions_
        directions = model.junction_branch_directions_[model.junctions_[0].node_id]
        assert directions.shape == (branch_count, points.shape[1])
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_topological_electrical_diagnostics_and_kron_reduction():
    points = generate_synthetic_datasets(n=80, noise=0.02, random_state=1)["y"]
    model = SplineGraphEmbedding(
        n_centroids=12,
        random_state=0,
        backbone_initialization="topological",
        use_effective_resistance=True,
        use_electrical_flow=True,
        use_kron_reduction=True,
        routing_resistance_weight=0.2,
        routing_current_weight=0.2,
    ).fit(points)

    assert model.effective_resistance_
    assert model.edge_leverage_
    assert model.electrical_traffic_
    assert model.kron_laplacian_.shape[0] == len(model.kron_vertex_ids_)
    assert np.all(np.isfinite(model.kron_laplacian_))


def test_electrical_path_resistance_and_kron_shape():
    points = np.arange(4, dtype=float)[:, None]
    graph, _ = _weighted_symmetric_knn_graph(points, neighbors=2)
    _, resistance, leverage = _effective_resistance(graph)
    assert all(value >= 0.0 for value in resistance.values())
    assert all(value >= 0.0 for value in leverage.values())
    reduced, retained = _kron_reduction(graph, [0, 3])
    assert retained.tolist() == [0, 3]
    assert reduced.shape == (2, 2)
    assert np.allclose(reduced, reduced.T)


def test_topological_parameters_propagate_through_sklearn_adapter():
    estimator = SplineEmbeddingTransformer(
        backbone_initialization="topological",
        junction_scales=[1.0, 2.0, 3.0],
        use_effective_resistance=True,
    )
    cloned = clone(estimator)
    assert cloned.get_params()["backbone_initialization"] == "topological"
    assert cloned.get_params()["junction_scales"] == [1.0, 2.0, 3.0]
    assert cloned.get_params()["use_effective_resistance"] is True


def test_topological_parameter_validation():
    with pytest.raises(ValueError):
        SplineGraphEmbedding(backbone_initialization="unknown")
    with pytest.raises(ValueError):
        SplineGraphEmbedding(junction_inner_fraction=1.0)
    with pytest.raises(ValueError):
        SplineGraphEmbedding(max_branch_angle_degrees=0.0)


def test_topological_junction_routes_are_stable_across_kmeans_seeds():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)["x"]
    model = SplineGraphEmbedding(
        n_centroids=32,
        random_state=2,
        backbone_initialization="topological",
    ).fit(points)

    assert len(model.junctions_) == 1
    assert model.junctions_[0].branch_count == 4
    assert model.junction_degree_shortfall_ == {0: 0}
    assert model.landmark_graph_.degree(model.junctions_[0].node_id) == 4
    assert all(model.landmark_graph_.degree(node) == 1 for node in model.endpoint_node_ids_)


def test_topological_branching_tree_does_not_promote_centroid_stubs():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)[
        "branching-tree"
    ]
    model = SplineGraphEmbedding(
        n_centroids=32,
        random_state=5,
        backbone_initialization="topological",
    ).fit(points)

    assert len(model.junctions_) == 1
    assert model.junctions_[0].branch_count == 5
    assert len(model.endpoints_) == 5
    assert model.landmark_graph_.degree(model.junctions_[0].node_id) == 5
    assert model.junction_degree_shortfall_ == {0: 0}
    assert model.endpoint_degree_violations_ == []
    assert all(not chain["closed"] for chain in model.route_chains_)


def test_topological_y_keeps_one_junction_across_kmeans_seeds():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)["y"]
    for random_state in range(8):
        model = SplineGraphEmbedding(
            n_centroids=32,
            random_state=random_state,
            backbone_initialization="topological",
        ).fit(points)

        assert len(model.junctions_) == 1
        assert model.junctions_[0].branch_count == 3
        assert len(model.endpoints_) == 3
        assert model.junction_degree_shortfall_ == {0: 0}
        assert model.endpoint_degree_violations_ == []


def test_topological_disconnected_cycles_keep_separate_closed_routes():
    points, _ = make_circles(
        n_samples=500,
        factor=0.42,
        noise=0.045,
        random_state=1,
    )
    model = SplineGraphEmbedding(
        n_centroids=32,
        max_cycles=4,
        spline_smoothing=0.1,
        random_state=11,
        backbone_initialization="topological",
    ).fit(points)

    assert model.component_cycle_counts_ == [1, 1]
    assert model.realized_cycle_count_ == 2
    assert len(model.route_chains_) == 2
    assert all(chain["closed"] for chain in model.route_chains_)
    result = model.transform(points)
    assert np.all(result.route_id >= 0)


def test_topological_single_loop_spline_covers_both_sides_of_cycle():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)["circle"]
    model = SplineGraphEmbedding(
        n_centroids=32,
        random_state=3,
        backbone_initialization="topological",
    ).fit(points)

    samples = model.routes_[0].samples
    assert model.routes_[0].closed
    assert samples[:, 0].min() < -0.7
    assert samples[:, 0].max() > 0.7
    assert samples[:, 1].min() < -0.7
    assert samples[:, 1].max() > 0.7
    assert np.median(model.transform(points).residual_norm) < 0.1


def test_topological_line_spline_is_linear_in_original_metric():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)["line"]
    model = SplineGraphEmbedding(
        n_centroids=32,
        random_state=0,
        backbone_initialization="topological",
    ).fit(points)

    samples = model.routes_[0].samples
    centered = samples - np.mean(samples, axis=0)
    orthogonal = np.asarray([-model.linear_direction_[1], model.linear_direction_[0]])
    assert np.max(np.abs(centered @ orthogonal)) < 1e-8
    assert model.landmark_graph_.degree(model.endpoint_node_ids_[0]) == 1
    assert model.landmark_graph_.degree(model.endpoint_node_ids_[1]) == 1


def test_topological_shared_cycles_use_opposite_metro_sides():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)["figure-eight"]
    model = SplineGraphEmbedding(
        n_centroids=32,
        random_state=0,
        backbone_initialization="topological",
    ).fit(points)
    result = model.transform(points)
    layout = MetroLayout(model).fit(result)

    junction = next(node for node in model.landmark_graph_.nodes if model.landmark_graph_.degree(node) >= 3)
    station_x = layout.station_positions_[junction][0]
    closed_paths = [
        path for route, path in enumerate(layout.route_paths_)
        if model.route_chains_[route]["closed"]
    ]
    relative_centers = [float(np.mean(path[:, 0]) - station_x) for path in closed_paths]
    assert len(relative_centers) == 2
    assert relative_centers[0] * relative_centers[1] < 0.0


def test_topological_complex_workflow_keeps_cycle_backbones_closed():
    from topological_graph_embedding.datasets import (
        noisy_hypercube,
        noisy_polygon_rays_circles,
    )

    clouds = [
        noisy_polygon_rays_circles(
            n=500, noise=0.045, rng=np.random.default_rng(0), n_sides=4,
        ),
        noisy_hypercube(
            n=500, dim=3, noise=0.055, rng=np.random.default_rng(7),
        ),
    ]
    for points in clouds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model = SplineGraphEmbedding(
                n_centroids=32,
                max_cycles=5,
                random_state=0,
                persistence_threshold=4.0,
                persistence_max_points=300,
                backbone_initialization="topological",
            ).fit(points)
        result = model.transform(points)
        assert model.realized_cycle_count_ == model.requested_cycle_count_
        assert model.endpoints_ == []
        assert np.all(np.isfinite(result.projected))
