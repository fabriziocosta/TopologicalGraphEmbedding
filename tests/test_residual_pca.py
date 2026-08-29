import numpy as np
import pytest

from skeletalembedding import SkeletalEmbedding
from skeletalembedding.sklearn import (
    SkeletalEmbeddingClassifier,
    SkeletalEmbeddingTransformer,
)


def _noisy_line(n=90, seed=10):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        np.linspace(-2.0, 2.0, n),
        rng.normal(0.0, 0.20, n),
        rng.normal(0.0, 0.03, n),
    ])


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("max_residual_dim", -1), ("residual_pca_bandwidth", 0.0),
     ("residual_pca_bandwidth", np.inf), ("residual_subspace_smoothness", -1.0),
     ("residual_subspace_smoothness", np.inf)],
)
def test_residual_pca_parameters_are_validated(parameter, value):
    with pytest.raises((TypeError, ValueError)):
        SkeletalEmbedding(**{parameter: value})


def test_residual_pca_shapes_reconstruction_and_legacy_backfill():
    points = _noisy_line()
    for dimension in (0, 1, 2):
        result = SkeletalEmbedding(
            n_centroids=10, random_state=4, max_residual_dim=dimension,
        ).fit_transform(points)
        assert result.residual_coordinates.shape == (len(points), dimension)
        assert result.reconstructed.shape == points.shape
        assert result.unexplained_residual.shape == points.shape
        assert np.allclose(points, result.reconstructed + result.unexplained_residual)
    model = SkeletalEmbedding(n_centroids=10, random_state=4, max_residual_dim=5).fit(points)
    assert model.residual_dim_ == 2


def test_residual_bases_are_tangent_orthogonal_and_orthonormal():
    model = SkeletalEmbedding(
        n_centroids=10, random_state=4, max_residual_dim=2,
    ).fit(_noisy_line())
    for route, basis in enumerate(model.residual_bases_):
        grid = model.residual_parameter_grids_[route]
        tangents = model.routes_[route].tangent(grid)
        assert np.allclose(np.einsum("nd,ndr->nr", tangents, basis), 0.0, atol=1e-8)
        gram = np.einsum("ndr,nds->nrs", basis, basis)
        assert np.allclose(gram, np.eye(2), atol=1e-8)


def test_residual_pca_recovers_dominant_transverse_variance():
    model = SkeletalEmbedding(
        n_centroids=10, random_state=4, standardize=False, max_residual_dim=1,
    ).fit(_noisy_line(seed=12))
    values = np.concatenate(model.residual_eigenvalues_)
    assert np.nanmedian(values[:, 0]) > 0.005
    directions = np.concatenate([basis[:, :, 0] for basis in model.residual_bases_])
    assert np.nanmedian(np.abs(directions[:, 1])) > 0.8


def test_residual_pca_transform_is_batch_independent_and_closed_seam_is_shared():
    points = _noisy_line()
    model = SkeletalEmbedding(
        n_centroids=10, random_state=4, max_residual_dim=1,
    ).fit(points)
    full = model.transform(points)
    subset = model.transform(points[::3])
    assert np.allclose(full.residual_coordinates[::3], subset.residual_coordinates)
    assert np.allclose(full.reconstructed[::3], subset.reconstructed)

    theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta), 0.05 * np.cos(3.0 * theta)])
    closed = SkeletalEmbedding(
        n_centroids=16, random_state=2, max_residual_dim=1,
    ).fit(circle)
    for spline, basis in zip(closed.routes_, closed.residual_bases_):
        if spline.closed:
            assert np.allclose(basis[0], basis[-1], atol=1e-8)


def test_sklearn_residual_pca_features_and_classifier_dimensions():
    points = _noisy_line()
    transformer = SkeletalEmbeddingTransformer(
        n_centroids=10, random_state=4, max_residual_dim=1,
    ).fit(points)
    names = transformer.get_feature_names_out()
    features = transformer.transform(points)
    assert "residual_pca_0" in names
    assert not any(name.startswith("residual_0") for name in names)
    assert features.shape[1] == len(names)

    labels = (points[:, 0] > 0).astype(int)
    classifier = SkeletalEmbeddingClassifier(
        n_centroids=10, random_state=4, max_residual_dim=1,
    ).fit(points, labels)
    assert classifier.embedding_.max_residual_dim == 1
    assert classifier.estimator_.n_features_in_ == len(
        classifier.embedding_.routes_
    ) + 1 + 1
