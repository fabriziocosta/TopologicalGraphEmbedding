import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, to_rgba
from sklearn.datasets import load_digits

from skeletalembedding import SkeletalEmbedding
from skeletalembedding.datasets import generate_synthetic_datasets
from skeletalembedding.visualization import MetroLayout, plot_spline_3d
from skeletalembedding.visualization.plots import (
    metro_line_colors,
    plot_embedding_row,
    plot_metro_lines,
    plot_metro_points,
    route_colors,
)
from skeletalembedding.visualization.reduction import fit_reducer
from skeletalembedding.visualization.workflows.synthetic import fit_datasets


def test_metro_layout_consumes_embedding_result():
    points = generate_synthetic_datasets(n=100, noise=0.03, random_state=1)["star"]
    model = SkeletalEmbedding(n_centroids=12, random_state=0).fit(points)
    result = model.transform(points)
    layout = MetroLayout(model, random_state=0).fit(result)
    displayed = layout.transform_points(result)
    assert displayed.shape == (len(points), 2)
    assert layout.transform_points_3d(result).shape == (len(points), 3)
    for node, radius in layout.junction_radii_.items():
        center = layout.station_positions_[node]
    assert np.all(np.linalg.norm(displayed - center, axis=1) >= radius)


def test_binary_tree_workflow_keeps_the_fitted_graph_acyclic():
    points = generate_synthetic_datasets(
        n=1000,
        noise=0.045,
        random_state=0,
        binary_tree_depth=3,
    )["binary-tree"]
    models, _, summary = fit_datasets(
        {"binary-tree": points},
        n_centroids=32,
        persistence_threshold=4.0,
        persistence_max_points=60,
        spline_smoothing=0.005,
        max_cycles=5,
    )

    model = models["binary-tree"]
    assert summary[0]["cycles"] == 0
    assert model.initialization == "legacy_coarsen"
    assert model.n_centroids == 64
    assert all(not route.closed for route in model.routes_)


def test_plot_network_uses_public_name():
    points = generate_synthetic_datasets(n=80, noise=0.03, random_state=1)["circle"]
    model = SkeletalEmbedding(n_centroids=16, random_state=0).fit(points)
    axis = model.plot_network(points)
    assert axis is not None
    plt.close(axis.figure)


def test_plot_embedding_row_ignores_unresolved_station_ids():
    points = generate_synthetic_datasets(n=100, noise=0.03, random_state=1)["star"]
    model = SkeletalEmbedding(
        initialization="skeletal", n_centroids=12, random_state=0,
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
    points = generate_synthetic_datasets(n=100, noise=0.03, random_state=1)["star"]
    model = SkeletalEmbedding(n_centroids=12, random_state=0).fit(points)
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


def test_metro_lines_use_opaque_saturated_route_colors():
    points = generate_synthetic_datasets(n=100, noise=0.03, random_state=1)["circle"]
    model = SkeletalEmbedding(n_centroids=12, random_state=0).fit(points)

    figure, axis = plt.subplots()
    plot_metro_lines(axis, model, "saturated lines")
    figure.canvas.draw()

    line_colors = np.asarray([to_rgba(line.get_color()) for line in axis.lines])
    expected = metro_line_colors(model)[:len(line_colors)]
    np.testing.assert_allclose(line_colors, expected)
    assert np.all(line_colors[:, 3] == 1.0)
    assert np.all(rgb_to_hsv(line_colors[:, :3])[:, 1] >= 0.65)
    assert not np.array_equal(line_colors[0], route_colors(model)[0])
    plt.close(figure)


def test_categorical_metro_lines_match_their_route_majority_class():
    points = generate_synthetic_datasets(n=100, noise=0.03, random_state=1)["circle"]
    model = SkeletalEmbedding(n_centroids=12, random_state=0).fit(points)
    result = model.transform(points)
    labels = np.zeros(len(points), dtype=int)
    labels[:10] = 1

    figure, axes = plt.subplots(1, 4)
    plot_embedding_row(
        axes, points, labels, model, result,
        projected_title="data", graph_title="embedding",
        metro_lines_title="lines", metro_points_title="points",
    )
    figure.canvas.draw()

    point_colors = axes[3].collections[-1].get_facecolors()
    for route, line in enumerate(axes[2].lines):
        members = np.flatnonzero(result.route_id == route)
        majority = np.bincount(labels[members]).argmax()
        expected = point_colors[members[labels[members] == majority][0]]
        np.testing.assert_allclose(to_rgba(line.get_color()), expected)
    plt.close(figure)


def test_classical_mds_reducer_supports_out_of_sample_transform():
    points = generate_synthetic_datasets(n=80, noise=0.03, random_state=2)["star"]
    points = np.column_stack([points, points[:, 0] ** 2])
    reducer = fit_reducer(points, method="mds", random_state=0)

    displayed = reducer.transform(points)
    held_out = reducer.transform(points[:5] + 0.01)

    assert displayed.shape == (len(points), 2)
    assert held_out.shape == (5, 2)
    assert np.all(np.isfinite(displayed))
    assert np.all(np.isfinite(held_out))


def test_plot_spline_3d_renders_pca_skeleton_cross_sections():
    rng = np.random.default_rng(4)
    parameter = np.linspace(0.0, 2.0 * np.pi, 140)
    points = np.column_stack([
        np.cos(parameter),
        np.sin(parameter),
        0.35 * np.sin(2.0 * parameter),
    ]) + 0.025 * rng.normal(size=(len(parameter), 3))
    model = SkeletalEmbedding(n_centroids=18, random_state=0).fit(points)
    result = model.transform(points)

    figure = plot_spline_3d(
        model,
        result,
        n_spline_samples=5,
        ellipse_samples=12,
        show_observations=False,
        show_nodes=False,
        show_reduced_graph=False,
    )
    line_traces = [trace for trace in figure.data if getattr(trace, "mode", None) == "lines"]
    assert len(line_traces) == len(model.routes_) * 6
    assert figure.layout.scene.xaxis.title.text == "PCA component 1"
    assert figure.layout.scene.yaxis.title.text == "PCA component 2"
    assert figure.layout.scene.zaxis.title.text == "PCA component 3"
    assert any(np.ptp(np.asarray(trace.z, dtype=float)) > 1e-8 for trace in line_traces)


def test_plot_spline_3d_renders_a_volumetric_junction_ellipsoid():
    planar = generate_synthetic_datasets(
        n=500, noise=0.045, random_state=0,
    )["figure-eight"]
    points = np.column_stack([
        planar,
        np.random.default_rng(3).normal(scale=0.045, size=len(planar)),
    ])
    model = SkeletalEmbedding(
        n_centroids=36,
        persistence_threshold=4.0,
        persistence_max_points=500,
        max_cycles=4,
        random_state=0,
        standardize=False,
        initialization="skeletal",
    ).fit(points)
    result = model.transform(points)

    figure = plot_spline_3d(
        model,
        result,
        ellipse_samples=16,
        show_observations=False,
    )
    ellipsoids = [trace for trace in figure.data if trace.name == "junction ellipsoid"]
    assert model.realized_cycle_count_ == 2
    assert len(model.junction_node_ids_) == 1
    assert len(model.routes_) == 2
    assert all(route.closed for route in model.routes_)
    assert len(ellipsoids) == 1
    assert any(trace.name == "reduced graph" for trace in figure.data)
    for coordinate in (ellipsoids[0].x, ellipsoids[0].y, ellipsoids[0].z):
        assert np.ptp(np.asarray(coordinate, dtype=float)) > 1e-8


def test_high_dimensional_backbone_does_not_drop_the_detected_junction():
    points = load_digits().data
    model = SkeletalEmbedding(
        n_centroids=36,
        max_cycles=4,
        random_state=0,
        initialization="skeletal",
    ).fit(points)
    result = model.transform(points)

    assert len(model.landmark_graph_._components()) == 1
    assert all(value == 0 for value in model.junction_degree_shortfall_.values())
    assert np.all(result.route_id >= 0)
