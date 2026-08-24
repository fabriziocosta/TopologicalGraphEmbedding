import numpy as np
from sklearn.ensemble import RandomForestClassifier

from spline_sklearn import SplineGraphClassifier


def test_spline_graph_classifier_defaults_to_random_forest_and_normal_frame():
    rng = np.random.default_rng(0)
    points = np.column_stack([
        np.linspace(-2.0, 2.0, 80),
        rng.normal(0.0, 0.08, 80),
    ])
    labels = (points[:, 0] > 0.0).astype(int)

    model = SplineGraphClassifier(n_centroids=8, random_state=0).fit(points, labels)

    assert isinstance(model.classifier_, RandomForestClassifier)
    normal = model.transform_normal(points)
    assert normal.shape == (len(points), points.shape[1] - 1)
