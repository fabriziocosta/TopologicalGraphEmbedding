from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from notebooks.spline_visualization import (
    _target_scatter,
    evaluate_route_classification,
    evaluate_route_regression,
    plot_embedding_row,
    plot_graph_embedding,
    plot_metro_points,
    spline_colors,
)
from metro_layout import MetroSplineLayout
from topological_spline_graph import SkeletonGraph, SplineCurve
from topological_spline_graph import spline_normal_coordinates, spline_normal_frames


def test_continuous_targets_use_pale_blue_to_ink_cmap():
    figure, axis = plt.subplots()
    artist = _target_scatter(
        axis, np.arange(24), np.zeros(24), np.linspace(0.0, 1.0, 24),
    )

    assert np.allclose(artist.get_cmap()(0.0), to_rgba('#c7e5f2'))
    assert np.allclose(artist.get_cmap()(1.0), to_rgba('#050b12'))
    plt.close(figure)


def _model():
    node_positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([-4.0, 0.0]),
        2: np.array([4.0, 0.0]),
        3: np.array([0.0, 4.0]),
    }
    graph = SkeletonGraph(node_positions)
    chains = []
    splines = []
    for nodes in ([0, 1], [0, 2], [0, 3]):
        for left, right in zip(nodes[:-1], nodes[1:]):
            graph.add_edge(left, right)
        samples = np.asarray([node_positions[node] for node in nodes])
        chains.append({"nodes": list(nodes), "closed": False})
        splines.append(
            SplineCurve(
                samples=samples,
                t_values=np.linspace(0.0, 1.0, len(samples)),
                closed=False,
            )
        )
    return SimpleNamespace(
        _fitted=True,
        graph_=graph,
        chains_=chains,
        splines_=splines,
        scale_=np.ones(2),
        mean_=np.zeros(2),
        junction_nodes_=[0],
        endpoint_nodes_=[1, 2, 3],
    )


def test_metro_point_nodes_are_hidden_by_default_and_opt_in():
    model = _model()
    result = {
        "highway_id": np.zeros(4, dtype=int),
        "t": np.linspace(0.1, 0.9, 4),
        "residual_vector": np.zeros((4, 2)),
    }
    layout = MetroSplineLayout(model, random_state=0).fit(result)

    figure, axis = plt.subplots()
    plot_metro_points(axis, np.zeros(4), model, result, "points", layout=layout)
    assert len(axis.patches) == 0
    assert len(axis.collections) == 1
    plt.close(figure)

    figure, axis = plt.subplots()
    plot_metro_points(
        axis, np.zeros(4), model, result, "points", layout=layout, show_nodes=True,
    )
    assert len(axis.patches) == 1
    assert len(axis.collections) == 2
    plt.close(figure)


def test_embedding_row_reuses_the_supplied_metro_layout():
    model = _model()
    result = {
        "highway_id": np.zeros(4, dtype=int),
        "t": np.linspace(0.1, 0.9, 4),
        "residual_vector": np.zeros((4, 2)),
    }
    layout = MetroSplineLayout(model, random_state=0).fit(result)
    figure, axes = plt.subplots(1, 4)

    returned = plot_embedding_row(
        axes,
        np.asarray([
            [-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.0, 0.0],
        ]),
        np.zeros(4),
        model,
        result,
        "projected",
        "graph",
        "lines",
        "points",
        layout=layout,
    )

    assert returned is layout
    assert [axis.get_title() for axis in axes] == [
        "projected", "graph", "lines", "points",
    ]
    plt.close(figure)


def test_graph_embedding_uses_circles_for_closed_highways():
    model = _model()
    model.splines_[1] = SplineCurve(
        samples=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        t_values=np.linspace(0.0, 1.0, 3, endpoint=False),
        closed=True,
    )
    result = {
        "highway_id": np.asarray([0, 1, 2, 1], dtype=int),
        "t": np.linspace(0.1, 0.9, 4),
    }

    figure, axis = plt.subplots()
    colors = spline_colors(model)
    plot_graph_embedding(axis, np.zeros(4), model, result, "graph", colors=colors)

    marker_collections = axis.collections[1:]
    assert len(marker_collections) == 2
    assert len(marker_collections[0].get_paths()[0].vertices) == 5  # square
    assert len(marker_collections[1].get_paths()[0].vertices) > 5  # circle
    assert np.allclose(marker_collections[1].get_facecolors()[0], colors[1])
    plt.close(figure)


def test_spline_normal_coordinates_are_perpendicular_to_tangent():
    model = _model()
    result = {
        "highway_id": np.zeros(6, dtype=int),
        "t": np.linspace(0.05, 0.95, 6),
        "residual_vector": np.column_stack([
            np.zeros(6), np.linspace(-1.0, 1.0, 6),
        ]),
    }
    frames = spline_normal_frames(model, result)
    coordinates = spline_normal_coordinates(model, result)

    assert frames.shape == (6, 2, 1)
    assert coordinates.shape == (6, 1)
    tangents = model.splines_[0].tangent(result["t"])
    assert np.allclose(np.sum(tangents * frames[:, :, 0], axis=1), 0.0, atol=1e-8)


def test_route_classification_reports_normalized_accuracy_and_purity():
    model = _model()
    labels = np.tile([0, 1], 30)
    result = {
        "highway_id": np.zeros(len(labels), dtype=int),
        "t": np.linspace(0.01, 0.99, len(labels)),
        "residual_vector": np.column_stack([
            np.zeros(len(labels)), np.where(labels == 0, -1.0, 1.0),
        ]),
    }

    metrics = evaluate_route_classification(model, result, labels, random_state=0)

    assert len(metrics) == len(model.splines_)
    assert metrics[0]["valid"]
    assert np.isclose(metrics[0]["route_purity"], 0.5)
    assert 0.0 <= metrics[0]["normalized_accuracy"] <= 1.0


def test_normalized_accuracy_maps_balanced_random_guessing_to_zero():
    model = _model()
    labels = np.tile([0, 1], 30)
    result = {
        "highway_id": np.zeros(len(labels), dtype=int),
        "t": np.linspace(0.01, 0.99, len(labels)),
        "residual_vector": np.column_stack([
            np.zeros(len(labels)), np.where(labels == 0, -1.0, 1.0),
        ]),
    }

    metrics = evaluate_route_classification(
        model, result, labels,
        classifier=DummyClassifier(strategy="most_frequent"),
        random_state=0,
    )

    assert metrics[0]["valid"]
    assert np.isclose(metrics[0]["normalized_accuracy"], 0.0)


def test_normalized_accuracy_rewards_separation_along_the_route():
    model = _model()
    labels = np.repeat([0, 1], 30)
    result = {
        "highway_id": np.zeros(len(labels), dtype=int),
        "t": np.linspace(0.01, 0.99, len(labels)),
        "residual_vector": np.zeros((len(labels), 2)),
    }

    metrics = evaluate_route_classification(
        model, result, labels,
        classifier=DecisionTreeClassifier(max_depth=1, random_state=0),
        random_state=0,
    )

    assert metrics[0]["valid"]
    assert metrics[0]["accuracy"] > 0.9
    assert metrics[0]["normalized_accuracy"] > 0.9


def test_pure_route_is_perfect_even_without_two_class_cv():
    model = _model()
    labels = np.zeros(6, dtype=int)
    result = {
        "highway_id": np.zeros(len(labels), dtype=int),
        "t": np.linspace(0.01, 0.99, len(labels)),
        "residual_vector": np.column_stack([
            np.zeros(len(labels)), np.ones(len(labels)),
        ]),
    }

    metrics = evaluate_route_classification(model, result, labels, random_state=0)

    assert metrics[0]["valid"]
    assert metrics[0]["status"] == "pure route"
    assert metrics[0]["route_purity"] == 1.0
    assert metrics[0]["normalized_accuracy"] == 1.0


def test_route_regression_reports_rank_correlation():
    model = _model()
    targets = np.linspace(-1.0, 1.0, 40)
    result = {
        "highway_id": np.zeros(len(targets), dtype=int),
        "t": np.linspace(0.01, 0.99, len(targets)),
        "residual_vector": np.column_stack([
            np.zeros(len(targets)), targets,
        ]),
    }

    metrics = evaluate_route_regression(model, result, targets, random_state=0)

    assert len(metrics) == len(model.splines_)
    assert metrics[0]["valid"]
    assert metrics[0]["n_splits"] == 10
    assert -1.0 <= metrics[0]["rank_correlation"] <= 1.0


def test_graph_embedding_places_route_score_pies_past_one():
    model = _model()
    count = 12
    result = {
        "highway_id": np.zeros(count, dtype=int),
        "t": np.linspace(0.05, 0.95, count),
    }
    metrics = [
        {
            "highway_id": route,
            "valid": True,
            "normalized_accuracy": 0.75,
        }
        for route in range(len(model.splines_))
    ]

    figure, axis = plt.subplots()
    plot_graph_embedding(
        axis, np.zeros(count), model, result, "graph",
        route_metrics=metrics,
    )

    assert axis.get_xlim()[1] > 1.0
    assert len(axis.child_axes) == len(model.splines_)
    plt.close(figure)
