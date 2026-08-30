import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_circles, make_moons

from skeletalembedding import SkeletalEmbedding
from skeletalembedding._electrical import (
    _effective_resistance,
    _kron_reduction,
)
from skeletalembedding._topology import _weighted_symmetric_knn_graph
from skeletalembedding.datasets import generate_synthetic_datasets
from skeletalembedding.sklearn import SkeletalEmbeddingTransformer
from skeletalembedding.visualization.metro import MetroLayout
from skeletalembedding.visualization.workflows.digits import make_spiral


@pytest.mark.parametrize(
    ("name", "cycles", "junction_count", "branch_count"),
    [
        ("line", 0, 0, None),
        ("star", 0, 1, 4),
        ("circle", 1, 0, None),
        ("figure-eight", 2, 1, 4),
        ("loop-branch", 1, 1, 3),
    ],
)
def test_topological_backbone_preserves_synthetic_structure(
    name, cycles, junction_count, branch_count,
):
    points = generate_synthetic_datasets(n=120, noise=0.03, random_state=0)[name]
    model = SkeletalEmbedding(
        n_centroids=16,
        max_cycles=5,
        random_state=0,
        initialization="skeletal",
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
    points = generate_synthetic_datasets(n=80, noise=0.02, random_state=1)["star"]
    model = SkeletalEmbedding(
        n_centroids=12,
        random_state=0,
        initialization="skeletal",
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


def test_mutual_knn_retains_only_reciprocal_natural_edges():
    points = np.array([[0.0], [1.0], [3.0], [10.0]])
    symmetric, _ = _weighted_symmetric_knn_graph(points, neighbors=1)
    mutual, _ = _weighted_symmetric_knn_graph(points, neighbors=1, mutual_knn=True)

    assert len(symmetric.original_components) == 1
    assert sorted(map(len, mutual.original_components)) == [1, 1, 2]


def test_mst_flag_augments_mutual_knn_edges_before_component_recording():
    points = np.array([[0.0], [1.0], [3.0], [10.0]])
    graph, _ = _weighted_symmetric_knn_graph(
        points, neighbors=1, mutual_knn=True, add_mst=True,
    )

    assert len(graph.original_components) == 1
    assert len(graph.edges) == len(points) - 1


def test_topological_parameters_propagate_through_sklearn_adapter():
    estimator = SkeletalEmbeddingTransformer(
        initialization="skeletal",
        junction_scales=[1.0, 2.0, 3.0],
        use_effective_resistance=True,
        mutual_knn=True,
        add_mst=True,
    )
    cloned = clone(estimator)
    assert cloned.get_params()["initialization"] == "skeletal"
    assert cloned.get_params()["junction_scales"] == [1.0, 2.0, 3.0]
    assert cloned.get_params()["use_effective_resistance"] is True
    assert cloned.get_params()["mutual_knn"] is True
    assert cloned.get_params()["add_mst"] is True


def test_topological_parameter_validation():
    with pytest.raises(ValueError):
        SkeletalEmbedding(initialization="unknown")
    with pytest.raises(ValueError):
        SkeletalEmbedding(junction_inner_fraction=1.0)
    with pytest.raises(ValueError):
        SkeletalEmbedding(max_branch_angle_degrees=0.0)


def test_topological_junction_routes_are_stable_across_kmeans_seeds():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)["star"]
    model = SkeletalEmbedding(
        n_centroids=32,
        random_state=2,
        initialization="skeletal",
    ).fit(points)

    assert len(model.junctions_) == 1
    assert model.junctions_[0].branch_count == 4
    assert model.junction_degree_shortfall_ == {0: 0}
    assert model.landmark_graph_.degree(model.junctions_[0].node_id) == 4
    assert all(model.landmark_graph_.degree(node) == 1 for node in model.endpoint_node_ids_)


def test_topological_binary_tree_produces_a_valid_embedding():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)[
        "binary-tree"
    ]
    model = SkeletalEmbedding(
        n_centroids=32,
        random_state=5,
        initialization="skeletal",
    ).fit(points)

    result = model.transform(points)
    assert np.all(np.isfinite(result.projected))
    assert np.all(result.route_id >= 0)


def test_topological_star_keeps_one_junction_across_kmeans_seeds():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)["star"]
    for random_state in range(8):
        model = SkeletalEmbedding(
            n_centroids=32,
            random_state=random_state,
            initialization="skeletal",
        ).fit(points)

        assert len(model.junctions_) == 1
        assert model.junctions_[0].branch_count == 4
        assert len(model.endpoints_) == 4
        assert model.junction_degree_shortfall_ == {0: 0}
        assert model.endpoint_degree_violations_ == []


def test_topological_disconnected_cycles_keep_separate_closed_routes():
    points, _ = make_circles(
        n_samples=500,
        factor=0.42,
        noise=0.045,
        random_state=1,
    )
    model = SkeletalEmbedding(
        n_centroids=32,
        max_cycles=4,
        spline_smoothing=0.1,
        random_state=11,
        initialization="skeletal",
    ).fit(points)

    assert model.component_cycle_counts_ == [1, 1]
    assert model.realized_cycle_count_ == 2
    assert len(model.route_chains_) == 2
    assert all(chain["closed"] for chain in model.route_chains_)
    result = model.transform(points)
    assert np.all(result.route_id >= 0)


def test_topological_disconnected_moons_use_complete_arc_endpoints():
    points, _ = make_moons(n_samples=500, noise=0.07, random_state=0)
    model = SkeletalEmbedding(
        n_centroids=45,
        persistence_threshold=4.0,
        spline_smoothing=0.1,
        max_cycles=4,
        random_state=10,
        initialization="skeletal",
    ).fit(points)

    curves = [spline.samples * model.scale_ + model.mean_ for spline in model.routes_]
    assert len(curves) == 2
    assert all(np.ptp(curve[:, 0]) > 1.7 for curve in curves)
    assert all(abs(curve[0, 1] - curve[-1, 1]) < 0.25 for curve in curves)


def test_topological_disconnected_cycles_keep_full_loop_spans():
    points, _ = make_circles(
        n_samples=500,
        factor=0.42,
        noise=0.045,
        random_state=1,
    )
    model = SkeletalEmbedding(
        n_centroids=45,
        max_cycles=4,
        persistence_threshold=4.0,
        spline_smoothing=0.1,
        random_state=11,
        initialization="skeletal",
    ).fit(points)
    result = model.transform(points)

    for route, spline in enumerate(model.routes_):
        curve = spline.samples * model.scale_ + model.mean_
        members = points[result.route_id == route]
        assert spline.closed
        assert np.all(np.ptp(curve, axis=0) >= 0.8 * np.ptp(members, axis=0))


def test_topological_single_loop_spline_covers_both_sides_of_cycle():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)["circle"]
    model = SkeletalEmbedding(
        n_centroids=32,
        random_state=3,
        initialization="skeletal",
    ).fit(points)

    samples = model.routes_[0].samples
    assert model.routes_[0].closed
    assert samples[:, 0].min() < -0.7
    assert samples[:, 0].max() > 0.7
    assert samples[:, 1].min() < -0.7
    assert samples[:, 1].max() > 0.7
    assert np.median(model.transform(points).residual_norm) < 0.1


def test_topological_dense_single_loop_is_not_split_by_coarse_tree():
    points = generate_synthetic_datasets(n=1000, noise=0.045, random_state=0)["circle"]
    model = SkeletalEmbedding(
        n_centroids=50,
        random_state=0,
        persistence_threshold=4.0,
        persistence_max_points=60,
        initialization="skeletal",
    ).fit(points)

    assert model.realized_cycle_count_ == 1
    assert len(model.junctions_) == 0
    assert len(model.route_chains_) == 1
    assert model.route_chains_[0]["closed"]
    assert np.median(model.transform(points).residual_norm) < 0.1


def test_topological_line_spline_is_linear_in_original_metric():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)["line"]
    model = SkeletalEmbedding(
        n_centroids=32,
        random_state=0,
        initialization="skeletal",
    ).fit(points)

    samples = model.routes_[0].samples
    centered = samples - np.mean(samples, axis=0)
    orthogonal = np.asarray([-model.linear_direction_[1], model.linear_direction_[0]])
    assert np.max(np.abs(centered @ orthogonal)) < 1e-8
    assert model.landmark_graph_.degree(model.endpoint_node_ids_[0]) == 1
    assert model.landmark_graph_.degree(model.endpoint_node_ids_[1]) == 1


def test_topological_shared_cycles_use_opposite_metro_sides():
    points = generate_synthetic_datasets(n=500, noise=0.045, random_state=0)["figure-eight"]
    model = SkeletalEmbedding(
        n_centroids=32,
        random_state=0,
        initialization="skeletal",
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


def test_topological_loop_branch_keeps_stem_open():
    points_2d = generate_synthetic_datasets(
        n=500, noise=0.045, random_state=0,
    )["loop-branch"]
    points = np.column_stack([
        points_2d,
        np.random.default_rng(6).normal(scale=0.045, size=len(points_2d)),
    ])
    model = SkeletalEmbedding(
        n_centroids=36,
        random_state=0,
        initialization="skeletal",
        persistence_threshold=4.0,
        persistence_max_points=60,
        spline_smoothing=0.08,
        max_cycles=4,
        standardize=False,
    ).fit(points)

    assert model.realized_cycle_count_ == 1
    assert len(model.junctions_) == 1
    assert len(model.route_chains_) == 2
    assert sum(chain["closed"] for chain in model.route_chains_) == 1
    endpoint = model.endpoint_regions_[0].center
    assert endpoint[0] > 1.5
    closed_chain = next(chain for chain in model.route_chains_ if chain["closed"])
    open_chain = next(chain for chain in model.route_chains_ if not chain["closed"])
    assert model.junction_regions_[0].node_id in open_chain["nodes"]
    assert model.endpoint_regions_[0].node_id in open_chain["nodes"]
    assert model.endpoint_regions_[0].node_id not in closed_chain["nodes"]
    closed_route = model.routes_[model.route_chains_.index(closed_chain)]
    support_distances = np.min(
        np.linalg.norm(
            closed_chain["points"][:, None, :] - closed_route.samples[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    assert np.max(support_distances) < 1e-8
    loop_center = np.mean(closed_chain["points"][:, :2], axis=0)
    loop_angles = np.unwrap(np.arctan2(
        closed_route.samples[:, 1] - loop_center[1],
        closed_route.samples[:, 0] - loop_center[0],
    ))
    # The noisy lifted observations can introduce a tiny angular jitter in
    # the display plane, but the fitted route must not materially reverse.
    assert np.all(np.diff(loop_angles) >= -0.02)


def test_topological_open_spiral_route_covers_the_whole_component():
    points_2d = make_spiral(n_samples=500, noise=0.045, random_state=5)
    points = np.column_stack([
        points_2d,
        np.random.default_rng(9).normal(scale=0.045, size=len(points_2d)),
    ])
    model = SkeletalEmbedding(
        n_centroids=36,
        n_neighbors=6,
        topology_neighbors=6,
        persistence_threshold=4.0,
        persistence_max_points=60,
        spline_smoothing=0.08,
        max_cycles=4,
        standardize=False,
        random_state=0,
        initialization="skeletal",
    ).fit(points)
    result = model.transform(points)

    assert model.realized_cycle_count_ == 0
    assert len(model.route_chains_) == 1
    assert not model.route_chains_[0]["closed"]
    curve = model.routes_[0].samples * model.scale_ + model.mean_
    assert np.all(np.ptp(curve, axis=0)[:2] >= 0.7 * np.ptp(points, axis=0)[:2])
    assert np.median(result.residual_norm) < 0.15


def test_topological_complex_workflow_keeps_cycle_backbones_closed():
    from skeletalembedding.datasets import (
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
            model = SkeletalEmbedding(
                n_centroids=32,
                max_cycles=5,
                random_state=0,
                persistence_threshold=4.0,
                persistence_max_points=300,
                initialization="skeletal",
            ).fit(points)
        result = model.transform(points)
        assert model.realized_cycle_count_ == model.requested_cycle_count_
        assert model.endpoints_ == []
        assert np.all(np.isfinite(result.projected))


def test_topological_hypercube_recovers_all_corners_and_faces():
    from skeletalembedding.datasets import noisy_hypercube

    points = noisy_hypercube(
        n=1000,
        dim=3,
        noise=0.055,
        rng=np.random.default_rng(7),
    )
    model = SkeletalEmbedding(
        n_centroids=50,
        max_cycles=5,
        random_state=1,
        persistence_threshold=4.0,
        persistence_max_points=300,
        initialization="skeletal",
    ).fit(points)

    assert model.hypercube_dimension_ == 3
    assert model.face_cycle_count_ == 6
    assert model.realized_cycle_count_ == 5
    assert len(model.junctions_) == 8
    assert all(region.branch_count == 3 for region in model.junctions_)
    assert all(
        model.landmark_graph_.degree(region.node_id) == 3
        for region in model.junctions_
    )
    assert model.landmark_graph_.cycle_rank() == 5
