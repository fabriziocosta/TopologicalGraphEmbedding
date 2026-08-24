import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from topological_graph_embedding.sklearn import (
    SplineEmbeddingClassifier,
    SplineEmbeddingTransformer,
)


def _points():
    rng = np.random.default_rng(5)
    points = np.column_stack([np.linspace(-2.0, 2.0, 80), rng.normal(0, 0.08, 80)])
    labels = (points[:, 0] > 0).astype(int)
    return points, labels


def test_transformer_clone_feature_names_and_typed_result():
    points, _ = _points()
    estimator = SplineEmbeddingTransformer(n_centroids=8, random_state=0)
    assert clone(estimator).get_params() == estimator.get_params()
    estimator.fit(points)
    transformed = estimator.transform(points)
    result = estimator.transform_result(points)
    assert transformed.shape[0] == len(points)
    assert transformed.shape[1] == len(estimator.get_feature_names_out())
    assert result.route_id.shape == (len(points),)
    assert estimator.get_feature_names_out()[0].startswith("route_")


def test_classifier_delegates_to_estimator_and_uses_estimator_parameter():
    points, labels = _points()
    downstream = LogisticRegression(max_iter=200)
    model = SplineEmbeddingClassifier(
        estimator=downstream, n_centroids=8, random_state=0,
    ).fit(points, labels)
    assert model.estimator is downstream
    assert model.estimator_ is not downstream
    assert model.predict(points).shape == labels.shape
    assert model.predict_proba(points).shape == (len(points), 2)
