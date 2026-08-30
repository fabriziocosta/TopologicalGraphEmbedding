import numpy as np
import pytest
from sklearn.base import clone

import skeletalembedding.embedding as embedding_module
from skeletalembedding import SkeletalEmbedding
from skeletalembedding.datasets import generate_synthetic_datasets
from skeletalembedding.sklearn import SkeletalEmbeddingTransformer


def _plane(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(160, 2))
    return np.column_stack((x, np.zeros(len(x))))


def test_topology_mip_api_has_no_legacy_controls():
    estimator = SkeletalEmbedding(n_neighbors=6)
    assert "initialization" not in clone(estimator).get_params()
    assert "use_mip" not in estimator.get_params()
    with pytest.raises(TypeError):
        SkeletalEmbedding(initialization="legacy_coarsen")
    with pytest.raises(TypeError):
        SkeletalEmbedding(use_mip=False)


def test_topology_always_attempts_mip(monkeypatch):
    calls = []
    original = embedding_module.select_backbone_mip

    def wrapped(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(embedding_module, "select_backbone_mip", wrapped)
    SkeletalEmbedding(n_centroids=8, random_state=0).fit(_plane())
    assert calls


def test_infeasible_mip_uses_deterministic_topology_fallback(monkeypatch):
    monkeypatch.setattr(
        embedding_module,
        "select_backbone_mip",
        lambda *args, **kwargs: ({}, "infeasible:test"),
    )
    model = SkeletalEmbedding(n_centroids=8, random_state=0).fit(_plane())
    assert model.mip_status_ == "infeasible:test"
    assert len(model.backbone_graph_.nodes) > 0


def test_skeletal_backbone_node_target_subdivides_edges_and_preserves_topology():
    points = np.column_stack((np.linspace(-2.0, 2.0, 120), np.zeros(120)))
    model = SkeletalEmbedding(
        n_centroids=12,
        n_backbone_nodes=9,
        random_state=3,
    ).fit(points)

    assert model.backbone_node_count_ == 9
    assert len(model.backbone_graph_.nodes) == 9
    assert model.backbone_graph_.cycle_rank() == 0
    assert model.backbone_graph_.degree(model.endpoint_node_ids_[0]) == 1
    assert model.backbone_graph_.degree(model.endpoint_node_ids_[1]) == 1
    assert np.all(model.transform(points).route_id >= 0)


def test_skeletal_backbone_spacing_subdivides_long_edges():
    points = np.column_stack((np.linspace(-2.0, 2.0, 120), np.zeros(120)))
    model = SkeletalEmbedding(
        n_centroids=12,
        backbone_node_spacing=0.5,
        random_state=3,
    ).fit(points)

    assert len(model.backbone_graph_.nodes) > 2
    assert max(model.backbone_graph_.edges.values()) <= 0.5 + 1e-10


def test_skeletal_backbone_node_target_enforces_topological_lower_bound():
    points = generate_synthetic_datasets(n=180, noise=0.02, random_state=1)["circle"]
    with pytest.raises(ValueError, match="minimum|could not be realized"):
        SkeletalEmbedding(
            n_centroids=20,
            n_backbone_nodes=2,
            random_state=0,
        ).fit(points)

    relaxed = SkeletalEmbedding(
        n_centroids=20,
        n_backbone_nodes=2,
        backbone_node_policy="allow_topology_relaxation",
        random_state=0,
    ).fit(points)
    assert len(relaxed.backbone_graph_.nodes) == 2
    assert relaxed.backbone_graph_.cycle_rank() == 0


def test_skeletal_backbone_node_target_preserves_cycle_rank_when_refining_cycle():
    points = generate_synthetic_datasets(n=180, noise=0.02, random_state=1)["circle"]
    model = SkeletalEmbedding(
        n_centroids=20,
        n_backbone_nodes=8,
        random_state=0,
    ).fit(points)

    assert len(model.backbone_graph_.nodes) == 8
    assert model.backbone_graph_.cycle_rank() == 1
    assert np.all(model.transform(points).route_id >= 0)


def test_backbone_resolution_controls_are_validated_for_topology_mode():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SkeletalEmbedding(n_backbone_nodes=8, backbone_node_spacing=0.5)


def test_skeleton_metadata_and_reconstruction_diagnostics():
    points = _plane()
    model = SkeletalEmbedding(
        n_centroids=10,
        n_neighbors=6,
        max_residual_dim=1,
        random_state=2,
    ).fit(points)
    model.transform(points)
    assert model.skeleton_graph_ is not None
    assert model.backbone_graph_ is not None
    assert model.rib_graph_ is not None
    assert len(model.element_types_) == len(model.splines_)
    assert set(model.element_types_) == {"backbone"}
    assert model.reconstruction_.shape == points.shape
    assert model.residual_coordinates_.shape == (len(points), 1)
    assert model.post_pca_residual_norm_.shape == (len(points),)
    assert np.allclose(points, model.reconstruction_ + model.post_pca_residual_)
    assert model.skeleton_cycle_rank_ == model.backbone_cycle_rank_


def test_strict_coverage_adds_ribs_and_improves_error():
    points = _plane(seed=4)
    baseline = SkeletalEmbedding(
        n_centroids=10,
        n_neighbors=6,
        max_residual_dim=0,
        random_state=2,
    ).fit(points)
    refined = SkeletalEmbedding(
        n_centroids=10,
        n_neighbors=6,
        max_residual_dim=0,
        coverage_refinement=True,
        coverage_error_tolerance=0.05,
        coverage_max_iterations=2,
        coverage_max_candidates_per_iteration=3,
        random_state=2,
    ).fit(points)
    assert len(refined.rib_paths_) > 0
    assert len(refined.routes_) == len(refined.element_types_)
    assert refined.reconstruction_error_ < baseline.reconstruction_error_
    assert refined.coverage_iterations_ >= 1
    assert refined.coverage_history_


def test_sklearn_adapter_exposes_new_parameters():
    transformer = SkeletalEmbeddingTransformer(
        n_neighbors=6,
        n_backbone_nodes=8,
        coverage_refinement=True,
        coverage_error_tolerance=0.1,
        max_residual_dim=1,
    )
    cloned = clone(transformer)
    assert cloned.get_params()["coverage_refinement"] is True
    assert cloned.get_params()["n_backbone_nodes"] == 8
    transformed = transformer.fit(_plane(seed=5)).transform(_plane(seed=5))
    assert transformed.shape[0] == 160


def test_rib_candidate_type_is_validated_and_cloneable():
    estimator = SkeletalEmbedding(rib_candidate_type="parallel")
    assert clone(estimator).get_params()["rib_candidate_type"] == "parallel"
    for value in ("diagonal", "", None):
        try:
            SkeletalEmbedding(rib_candidate_type=value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid rib candidate type was accepted")


def test_stability_consensus_is_reproducible_and_refits_all_observations():
    points = _plane(seed=11)
    kwargs = {
        "n_centroids": 8,
        "n_neighbors": 5,
        "max_residual_dim": 1,
        "stability_selection": True,
        "stability_runs": 2,
        "stability_fraction": 0.7,
        "stability_residual_subspaces": True,
        "random_state": 4,
    }
    first = SkeletalEmbedding(**kwargs).fit(points)
    second = SkeletalEmbedding(**kwargs).fit(points)
    np.testing.assert_allclose(first.route_support_, second.route_support_)
    np.testing.assert_allclose(first.junction_support_, second.junction_support_)
    assert first.geometry_fit_n_samples_ == len(points)
    assert first.geometry_fit_indices_.tolist() == list(range(len(points)))
    assert first.full_data_refit_ is True
    assert first.stability_residual_subspaces_ is not None
    assert first.stability_summary_["successful_runs"] == 2
