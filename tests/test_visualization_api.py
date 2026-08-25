import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from topological_graph_embedding import SplineGraphEmbedding
from topological_graph_embedding.datasets import generate_synthetic_datasets
from topological_graph_embedding.visualization import MetroLayout
from topological_graph_embedding.visualization.plots import (
    plot_embedding_row,
    plot_metro_points,
    route_colors,
)
from topological_graph_embedding.visualization.reduction import fit_reducer


def test_metro_layout_consumes_embedding_result():
    points = generate_synthetic_datasets(n=100, noise=0.03, random_state=1)["y"]
    model = SplineGraphEmbedding(n_centroids=12, random_state=0).fit(points)
    result = model.transform(points)
    layout = MetroLayout(model, random_state=0).fit(result)
    displayed = layout.transform_points(result)
    assert displayed.shape == (len(points), 2)
    assert layout.transform_points_3d(result).shape == (len(points), 3)
    for node, radius in layout.junction_radii_.items():
        center = layout.station_positions_[node]
        assert np.all(np.linalg.norm(displayed - center, axis=1) >= radius)


def test_plot_network_uses_public_name():
    points = generate_synthetic_datasets(n=80, noise=0.03, random_state=1)["circle"]
    model = SplineGraphEmbedding(n_centroids=16, random_state=0).fit(points)
    axis = model.plot_network(points)
    assert axis is not None
    plt.close(axis.figure)


def test_plot_embedding_row_ignores_unresolved_station_ids():
    points = generate_synthetic_datasets(n=100, noise=0.03, random_state=1)["y"]
    model = SplineGraphEmbedding(
        backbone_initialization="topological", n_centroids=12, random_state=0,
    ).fit(points)
    result = model.transform(points)
    model.endpoint_node_ids_ = [*model.endpoint_node_ids_, None]

    figure, axes = plt.subplots(1, 4)
    plot_embedding_row(
        axes, points, np.zeros(len(points), dtype=int), model, result,
        projected_title="data", graph_title="embedding",
        metro_lines_title="lines", metro_points_title="points",
    )
    plt.close(figure)


def test_metro_points_use_classes_or_route_ids_as_the_sole_color_source():
    points = generate_synthetic_datasets(n=100, noise=0.03, random_state=1)["y"]
    model = SplineGraphEmbedding(n_centroids=12, random_state=0).fit(points)
    result = model.transform(points)

    fallback_figure, fallback_axis = plt.subplots()
    plot_metro_points(
        fallback_axis, np.zeros(len(points), dtype=int), model, result, "route colors",
    )
    fallback_figure.canvas.draw()
    fallback_artist = fallback_axis.collections[-1]
    expected_route_colors = route_colors(model)[result.route_id].copy()
    expected_route_colors[:, 3] *= 0.60
    np.testing.assert_allclose(fallback_artist.get_facecolors(), expected_route_colors)
    plt.close(fallback_figure)

    class_figure, class_axis = plt.subplots()
    class_labels = np.arange(len(points)) % 2
    plot_metro_points(class_axis, class_labels, model, result, "class colors")
    class_figure.canvas.draw()
    class_artist = class_axis.collections[-1]
    class_facecolors = class_artist.get_facecolors()
    assert np.array_equal(class_facecolors[0], class_facecolors[2])
    assert not np.array_equal(class_facecolors[0], class_facecolors[1])
    assert class_artist.get_edgecolors().size == 0
    plt.close(class_figure)

    regression_figure, regression_axis = plt.subplots()
    regression_labels = np.linspace(0.0, 1.0, len(points))
    plot_metro_points(
        regression_axis, regression_labels, model, result, "regression colors",
    )
    regression_figure.canvas.draw()
    regression_artist = regression_axis.collections[-1]
    assert regression_artist.get_cmap().name == "pale_blue_to_ink"
    assert len(regression_figure.axes) == 2
    plt.close(regression_figure)


def test_classical_mds_reducer_supports_out_of_sample_transform():
    points = generate_synthetic_datasets(n=80, noise=0.03, random_state=2)["y"]
    points = np.column_stack([points, points[:, 0] ** 2])
    reducer = fit_reducer(points, method="mds", random_state=0)

    displayed = reducer.transform(points)
    held_out = reducer.transform(points[:5] + 0.01)

    assert displayed.shape == (len(points), 2)
    assert held_out.shape == (5, 2)
    assert np.all(np.isfinite(displayed))
    assert np.all(np.isfinite(held_out))
