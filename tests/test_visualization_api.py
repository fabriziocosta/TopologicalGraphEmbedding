import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from topological_graph_embedding import SplineGraphEmbedding
from topological_graph_embedding.datasets import generate_synthetic_datasets
from topological_graph_embedding.metro import MetroLayout


def test_metro_layout_consumes_embedding_result():
    points = generate_synthetic_datasets(n=100, noise=0.03, random_state=1)["y"]
    model = SplineGraphEmbedding(n_centroids=12, random_state=0).fit(points)
    result = model.transform(points)
    layout = MetroLayout(model, random_state=0).fit(result)
    assert layout.transform_points(result).shape == (len(points), 2)
    assert layout.transform_points_3d(result).shape == (len(points), 3)


def test_plot_network_uses_public_name():
    points = generate_synthetic_datasets(n=80, noise=0.03, random_state=1)["circle"]
    model = SplineGraphEmbedding(n_centroids=16, random_state=0).fit(points)
    axis = model.plot_network(points)
    assert axis is not None
    plt.close(axis.figure)
