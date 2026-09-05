"""Compression invariants and end-to-end multiresolution regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from skeletalembedding import SkeletalEmbedding
from skeletalembedding._multiresolution import build_hierarchy, medoid
from skeletalembedding._residual_pca import _ancestry_covariance, _route_distances
from skeletalembedding._stability import (
    match_cycles_across_levels,
    match_junctions_across_levels,
)
from skeletalembedding._topology import JunctionRegion, PersistentCycle
from skeletalembedding.datasets import generate_synthetic_datasets


def test_hierarchy_partitions_and_parent_composition():
    rng = np.random.default_rng(7)
    points = rng.normal(size=(500, 3))
    levels = build_hierarchy(points, target_size=35)
    assert len(levels) > 2
    for index, level in enumerate(levels):
        assert np.array_equal(level.points, points[level.representative_indices])
        assert np.array_equal(
            np.sort(np.concatenate(level.descendant_indices)), np.arange(len(points))
        )
        assert level.descendant_original_indices is level.descendant_indices
        for original, descendants in zip(
            level.representative_indices, level.descendant_indices
        ):
            assert original in descendants
        if index + 1 < len(levels):
            parent = levels[index + 1]
            assert len(parent.points) < len(level.points)
            for group, descendants in enumerate(parent.descendant_indices):
                children = np.flatnonzero(level.parent_indices == group)
                assert np.array_equal(
                    descendants,
                    np.sort(
                        np.concatenate([level.descendant_indices[i] for i in children])
                    ),
                )
        else:
            assert level.parent_indices is None


def test_medoid_exact_objective_and_approximate_determinism():
    rng = np.random.default_rng(8)
    points = rng.normal(size=(200, 4))
    members = np.arange(200)
    selected = medoid(points, members, members)
    assert selected == np.argmin(cdist(points, points).sum(axis=1))
    assert medoid(points, members, members, "approx_medoid", 2) == medoid(
        points, members, members, "approx_medoid", 2
    )
    # Equal-cost candidates are resolved by original row, not iteration order.
    assert medoid(np.array([[0.0], [1.0]]), [1, 0], np.array([8, 3])) == 1


def test_duplicate_and_stopping_cases():
    points = np.ones((100, 2))
    levels = build_hierarchy(points, target_size=10)
    assert len(levels[-1].points) <= 10
    assert all(np.isfinite(level.scale) for level in levels)
    assert len(build_hierarchy(points, max_levels=0)) == 1
    assert len(build_hierarchy(points, target_size=100)) == 1
    assert len(build_hierarchy(points, target_size=3, min_reduction=0.999)) == 1


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("hierarchy_max_levels", -1),
        ("hierarchy_target_size", 2),
        ("hierarchy_distance_quantile", 1),
        ("hierarchy_min_reduction", 0),
        ("hierarchy_local_neighbors", True),
        ("representative_method", "centroid"),
        ("backbone_level", -1),
        ("backbone_consensus_levels", 0),
        ("route_resolution_weight", float("nan")),
        ("rib_resolution_weight", -1),
        ("rib_seed_source", "unknown"),
    ],
)
def test_parameter_validation(parameter, value):
    with pytest.raises(ValueError):
        SkeletalEmbedding(**{parameter: value})


def test_spatial_cycle_and_degree_matching():
    t = np.linspace(0, 2 * np.pi, 30)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    left = PersistentCycle(0, 5, 5, circle)
    remote = PersistentCycle(0, 5, 5, circle + 20)
    assert not match_cycles_across_levels([left], [remote], tolerance=0.5)[0]
    assert match_cycles_across_levels([left], [left], tolerance=0.5)[0]
    junction = JunctionRegion(np.zeros(2), 3, 1)
    incompatible = JunctionRegion(np.zeros(2), 4, 1)
    compatible = JunctionRegion(np.ones(2) * 0.1, 3, 1)
    assert match_junctions_across_levels(
        [junction], [incompatible, compatible], tolerance=0.5
    )[0]


@pytest.mark.parametrize(
    "name,cycles", [("circle", 1), ("figure-eight", 2), ("star", 0), ("loop-branch", 1)]
)
def test_multilevel_synthetic_topology(name, cycles):
    points = generate_synthetic_datasets(n=300, noise=0.01, random_state=0)[name]
    model = SkeletalEmbedding(hierarchy_target_size=40, n_centroids=20).fit(points)
    assert len(model.levels_) >= 3
    assert model.realized_cycle_count_ == cycles
    assert (
        model.coarse_backbone_graph_.cycle_rank() == model.backbone_graph_.cycle_rank()
    )
    if model.backbone_input_size_ == len(points):
        assert model.hierarchy_summary_["selection_reason"] == "insufficient_consensus"
    assert np.all(np.isfinite(model.transform(points).projected))
    assert len(model.route_descendant_support_) == len(model.routes_)
    if name == "circle":
        assert model.cycle_resolution_support_[0] > 0.8
        assert len(model.route_chains_[0]["points"]) > model.backbone_input_size_
        assert np.mean(model.transform(points).residual_norm) < 0.1
    if name == "star":
        assert len(model.junctions_) == 1
        assert np.isfinite(model.junctions_[0].resolution_support)
        assert model.junctions_[0].branch_count_by_level


def test_disabled_single_level_and_refit():
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    points = np.column_stack([np.cos(t), np.sin(t)])
    default = SkeletalEmbedding().fit(points)
    disabled = SkeletalEmbedding(use_multiresolution=False).fit(points)
    assert np.allclose(
        default.transform(points).projected, disabled.transform(points).projected
    )
    assert np.all(np.isnan(default.cycle_resolution_support_))
    default.set_params(hierarchy_target_size=20).fit(points)
    assert len(default.levels_) > 1
    default.set_params(use_multiresolution=False).fit(points[:50])
    assert default.hierarchy_sizes_ == [50]
    assert default.rib_paths_ == []
    assert np.all(np.isnan(default.cycle_resolution_support_))


@pytest.mark.parametrize("closed", [False, True])
def test_ancestry_covariance_retains_gaussian_weights(closed):
    rng = np.random.default_rng(2)
    coordinates = rng.normal(size=(100, 3))
    local_t = rng.uniform(0, 1, 100)
    blocks = [
        (block, min(local_t[block]), max(local_t[block]))
        for block in np.array_split(np.arange(100), 8)
    ]
    for center in [0, 0.3, 0.99]:
        for bandwidth in [0.002, 0.1]:
            distances = _route_distances(local_t, center, closed)
            weights = np.exp(-0.5 * ((distances - min(distances)) / bandwidth) ** 2)
            expected = (coordinates * weights[:, None]).T @ coordinates / weights.sum()
            actual = _ancestry_covariance(
                coordinates, local_t, blocks, center, closed, bandwidth
            )
            assert np.allclose(actual, expected, atol=1e-12)


def test_density_robustness_and_rare_branch():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    dense = np.repeat(circle[:40], 8, axis=0)
    for points in [circle, np.vstack([circle, dense])]:
        model = SkeletalEmbedding(hierarchy_target_size=40).fit(points)
        assert model.realized_cycle_count_ == 1
    branch = np.column_stack([np.linspace(0, 1, 30), np.linspace(1, 1.8, 30)])
    points = np.vstack([circle, dense, branch])
    levels = build_hierarchy(points, target_size=50)
    assert any(
        index >= len(circle) + len(dense) for index in levels[-1].representative_indices
    )
    assert np.max(levels[-1].points[:, 1]) > 1.6


def test_sklearn_parameter_forwarding():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    from skeletalembedding.sklearn import (
        SkeletalEmbeddingClassifier,
        SkeletalEmbeddingTransformer,
    )

    for cls in [SkeletalEmbeddingClassifier, SkeletalEmbeddingTransformer]:
        model = clone(
            cls(hierarchy_target_size=30, representative_method="approx_medoid")
        )
        assert model._new_embedding().hierarchy_target_size == 30
        assert model._new_embedding().representative_method == "approx_medoid"


def test_hierarchy_plot():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from skeletalembedding.visualization import plot_hierarchy

    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    model = SkeletalEmbedding(hierarchy_target_size=20).fit(
        np.column_stack([np.cos(t), np.sin(t)])
    )
    figure, axes = plot_hierarchy(model, show_ancestry=True)
    assert len(axes) == len(model.levels_) + 2
    figure.canvas.draw()
    plt.close(figure)


def test_stable_band_fallback_and_explicit_level(monkeypatch):
    from skeletalembedding import _multiresolution as mr

    points = np.column_stack([np.linspace(0, 1, 100), np.zeros(100)])
    model = SkeletalEmbedding(hierarchy_target_size=12, backbone_max_representatives=30)
    mr.initialize_hierarchy(model, points)
    assert len(model.levels_) >= 3

    def fake_evaluate(model, level, index):
        # Coarsest two levels collapse; the fine guard retains its cycle.
        count = 0 if len(level.points) <= model.backbone_max_representatives else 1
        return {
            "status": "tested",
            "cycle_count": count,
            "junctions": [],
            "endpoints": [],
            "persistent_cycles": [],
            "scale": 0.1,
            "n_points": len(level.points),
            "level": index,
        }

    monkeypatch.setattr(mr, "evaluate_level", fake_evaluate)
    with pytest.warns(RuntimeWarning, match="exceeding"):
        mr.infer_hierarchy_topology(model)
    assert model.hierarchy_sizes_[model.selected_backbone_level_] > 30
    assert "size_budget_exceeded" in model.hierarchy_summary_["selection_reason"]
    model.backbone_level = len(model.levels_) - 2
    mr.infer_hierarchy_topology(model)
    assert model.selected_backbone_level_ == model.backbone_level
    with pytest.raises(ValueError, match="outside"):
        SkeletalEmbedding(backbone_level=99).fit(points)


def test_no_dense_distances_in_compression(monkeypatch):
    from skeletalembedding import _topology

    monkeypatch.setattr(
        _topology,
        "_pairwise_distances",
        lambda *_: pytest.fail("global pairwise distances"),
    )
    t = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
    levels = build_hierarchy(np.column_stack([np.cos(t), np.sin(t)]), target_size=1000)
    assert len(levels[-1].points) <= 1000


def test_multiresolution_torus():
    u, v = np.meshgrid(
        np.linspace(0, 2 * np.pi, 24, endpoint=False),
        np.linspace(0, 2 * np.pi, 16, endpoint=False),
    )
    u, v = u.ravel(), v.ravel()
    points = np.column_stack(
        (
            (2 + 0.8 * np.cos(v)) * np.cos(u),
            (2 + 0.8 * np.cos(v)) * np.sin(u),
            0.8 * np.sin(v),
        )
    )
    model = SkeletalEmbedding(
        hierarchy_target_size=80,
        n_centroids=12,
        standardize=False,
        persistence_max_points=180,
        persistence_threshold=1.5,
        max_cycles=2,
        max_residual_dim=1,
        residual_subspace_smoothness=0.2,
        coverage_refinement=True,
        coverage_max_iterations=1,
        coverage_max_candidates_per_iteration=3,
        coverage_error_tolerance=0.1,
    ).fit(points)
    assert model.backbone_input_size_ < len(points)
    assert (
        sum(
            entry.get("cycle_count") == 2 for entry in model.topology_by_level_.values()
        )
        >= 2
    )
    assert model.realized_cycle_count_ == 2
    assert np.all(model.cycle_resolution_support_ > 0.5)
    assert len(model.rib_paths_) > 0
    assert model.residual_dim_ == 1
    assert all(basis.shape[-1] == 1 for basis in model.residual_bases_)
    assert np.all(np.isfinite(model.transform(points).reconstructed))
    assert sum(map(sum, model.residual_ancestry_partition_sizes_)) == len(points)


def test_rib_sources_deduplication_and_resolution_utility():
    from skeletalembedding._multiresolution import RepresentativeLevel
    from skeletalembedding._ribs import (
        RibCandidate,
        prepare_rib_candidates,
        select_ribs,
    )

    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    first = RibCandidate(points, 0, 0, "transverse", 2.0, 0.8, resolution_support=0.8)
    duplicate = RibCandidate(
        points.copy(),
        1,
        0,
        "hierarchy",
        1.0,
        0.7,
        resolution_support=0.7,
        seed_sources=("hierarchy",),
    )
    level = RepresentativeLevel(
        points, np.arange(3), None, [np.array([i]) for i in range(3)], 0.1
    )
    model = SimpleNamespace(
        levels_=[level, level],
        use_multiresolution=True,
        rib_seed_source="residual",
        _structural_subsamples_=[],
        local_scale_=0.1,
        coverage_min_gain=0,
        rib_min_support=0.6,
        coverage_max_candidates_per_iteration=3,
    )
    result = prepare_rib_candidates(model, points, None, [first, duplicate])
    assert len(result) == 1
    assert result[0].seed_sources == ("hierarchy", "residual")
    model.rib_seed_source = "hierarchy"
    model.use_multiresolution = False
    assert prepare_rib_candidates(model, points, None, [first]) == []
    other = RibCandidate(points + 3, 2, 0, "transverse", 2.0, 0.8, resolution_support=0)
    chosen = select_ribs(
        [other, first],
        max_ribs=1,
        min_gain=0,
        length_penalty=0,
        rib_penalty=0,
        junction_penalty=0,
        resolution_weight=1,
    )
    assert chosen == [first]


def test_structural_subsampling_precedes_mip(monkeypatch):
    import skeletalembedding.embedding as module

    original_select = module.select_backbone_mip
    seen = []

    def record(candidates, *args, **kwargs):
        seen.append(
            any(np.isfinite(candidate.subsample_support) for candidate in candidates)
        )
        return original_select(candidates, *args, **kwargs)

    monkeypatch.setattr(module, "select_backbone_mip", record)
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    model = SkeletalEmbedding(
        hierarchy_target_size=30,
        stability_selection=True,
        stability_runs=2,
        n_centroids=12,
    ).fit(np.column_stack([np.cos(t), np.sin(t)]))
    assert seen[0]
    assert len(model._structural_subsamples_) == 2
    assert np.all(np.isfinite(model.cycle_subsample_support_))


def test_rare_y_arm_survives_density_imbalance():
    rng = np.random.default_rng(4)
    points = np.vstack(
        [
            np.linspace(0, 1, count)[:, None] * np.array([np.cos(angle), np.sin(angle)])
            for angle, count in zip([0, 2 * np.pi / 3, 4 * np.pi / 3], [300, 300, 30])
        ]
    )
    points += rng.normal(0, 0.002, points.shape)
    model = SkeletalEmbedding(
        hierarchy_target_size=35,
        n_centroids=24,
        detect_cycles=False,
        prune_short_branches=False,
    ).fit(points)
    assert len(model.junctions_) == 1
    assert model.junctions_[0].branch_count == 3
    assert model.junction_resolution_support_[0] > 0.8
    assert len(model.endpoints_) == 3


def test_branching_tree_resolution():
    points = generate_synthetic_datasets(
        n=400, noise=0.001, random_state=2, binary_tree_depth=2
    )["binary-tree"]
    model = SkeletalEmbedding(
        hierarchy_target_size=40, n_centroids=24, detect_cycles=False
    ).fit(points)
    assert len(model.levels_) > 2
    assert model.realized_cycle_count_ == 0
    assert len(model.junctions_) >= 2
    assert len(model.endpoints_) >= 4
    assert np.all(model.junction_resolution_support_ >= 2 / len(model.levels_))
