import numpy as np
from sklearn.base import clone

from skeletalembedding import SkeletalEmbedding
from skeletalembedding.sklearn import SkeletalEmbeddingTransformer


def _plane(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(160, 2))
    return np.column_stack((x, np.zeros(len(x))))


def test_skeletal_api_and_legacy_initializer():
    estimator = SkeletalEmbedding(n_neighbors=6, initialization="skeletal")
    assert clone(estimator).get_params()["initialization"] == "skeletal"
    assert SkeletalEmbedding(initialization="legacy_coarsen").initialization == "legacy_coarsen"


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
        initialization="skeletal",
        n_neighbors=6,
        coverage_refinement=True,
        coverage_error_tolerance=0.1,
        max_residual_dim=1,
    )
    cloned = clone(transformer)
    assert cloned.get_params()["coverage_refinement"] is True
    transformed = transformer.fit(_plane(seed=5)).transform(_plane(seed=5))
    assert transformed.shape[0] == 160
