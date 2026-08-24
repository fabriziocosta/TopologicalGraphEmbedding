"""Static plotting and route-level evaluation for spline embeddings."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import type_of_target

from ..results import EmbeddingResult
from .metro import MetroLayout

_REGRESSION_CMAP = LinearSegmentedColormap.from_list(
    'pale_blue_to_ink',
    [
        '#c7e5f2',
        '#8fb8d8',
        '#3e709d',
        '#123957',
        '#050b12',
    ],
)


def evaluate_route_classification(
    model: Any,
    result: EmbeddingResult,
    labels,
    classifier=None,
    n_splits: int = 10,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    """Score how well each spline position predicts a categorical target.

    A classifier is evaluated out of fold using the longitudinal coordinate
    and the spline-normal coordinates. ``normalized_accuracy`` maps uniform
    random guessing across the global ``K`` classes to 0 and perfect
    prediction to 1: ``(accuracy - 1/K) / (1 - 1/K)``. ``route_purity`` is
    retained as a diagnostic, and a one-class route is explicitly perfect even
    when it has too few samples for cross-validation.
    """
    target = np.asarray(labels)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 1:
        return []
    route_ids = np.asarray(result.route_id, dtype=int)
    if len(target) != len(route_ids):
        raise ValueError('labels and result must contain the same number of observations')
    if _target_is_continuous(target):
        return []
    if type_of_target(target) not in {'binary', 'multiclass'}:
        return []
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2')
    global_classes = np.unique(target)
    n_global_classes = len(global_classes)
    if n_global_classes == 0:
        return []
    chance = 0.0 if n_global_classes == 1 else 1.0 / n_global_classes

    normal_coordinates = model.normal_coordinates(result)
    longitudinal = np.asarray(result.position, dtype=float).reshape(-1, 1)
    coordinates = (
        np.column_stack([longitudinal, normal_coordinates])
        if normal_coordinates.shape[1]
        else longitudinal
    )
    base_classifier = (
        clone(classifier)
        if classifier is not None
        else RandomForestClassifier(random_state=random_state)
    )
    if classifier is not None and hasattr(base_classifier, 'get_params'):
        parameters = base_classifier.get_params(deep=False)
        if parameters.get('random_state') is None and 'random_state' in parameters:
            base_classifier.set_params(random_state=random_state)

    metrics = []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for route in range(len(model.routes_)):
        members = np.flatnonzero(route_ids == route)
        route_target = target[members]
        classes, counts = np.unique(route_target, return_counts=True)
        metric = {
            'route_id': route,
            'n_samples': len(members),
            'n_classes': len(classes),
            'n_splits': int(n_splits),
            'valid': False,
            'route_purity': np.nan,
            'accuracy': np.nan,
            'normalized_accuracy': np.nan,
            'status': 'unavailable',
        }
        if not len(members):
            metric['status'] = 'empty route'
            metrics.append(metric)
            continue

        route_purity = float(np.max(counts) / len(members))
        metric['route_purity'] = route_purity
        if len(classes) == 1:
            metric.update({
                'valid': True,
                'accuracy': 1.0,
                'normalized_accuracy': 1.0,
                'status': 'pure route',
            })
            metrics.append(metric)
            continue
        if len(members) < n_splits:
            metric['status'] = f'insufficient samples for {n_splits}-fold classification'
            metrics.append(metric)
            continue

        predictions = np.empty(len(members), dtype=target.dtype)
        route_coordinates = coordinates[members]
        for train, test in splitter.split(route_coordinates):
            estimator = clone(base_classifier)
            estimator.fit(route_coordinates[train], route_target[train])
            predictions[test] = estimator.predict(route_coordinates[test])
        accuracy = float(accuracy_score(route_target, predictions))
        normalized_accuracy = float(np.clip(
            (accuracy - chance) / max(1.0 - chance, 1e-12), 0.0, 1.0,
        ))
        metric.update({
            'valid': True,
            'accuracy': accuracy,
            'normalized_accuracy': normalized_accuracy,
            'status': 'ok',
        })
        metrics.append(metric)
    return metrics


def evaluate_route_regression(
    model: Any,
    result: EmbeddingResult,
    targets,
    regressor=None,
    n_splits: int = 10,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate continuous targets in each spline-normal hyperplane.

    ``rank_correlation`` is Spearman's rank correlation between out-of-fold
    predictions and targets.  It captures whether the regressor preserves
    ordering, which is the single regression indicator shown in the graph
    embedding panel.
    """
    target = np.asarray(targets, dtype=float)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 1:
        return []
    route_ids = np.asarray(result.route_id, dtype=int)
    if len(target) != len(route_ids):
        raise ValueError('targets and result must contain the same number of observations')
    if not np.all(np.isfinite(target)):
        return []
    if not _target_is_continuous(target):
        return []
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2')

    normal_coordinates = model.normal_coordinates(result)
    if normal_coordinates.shape[1] == 0:
        return []
    base_regressor = (
        clone(regressor)
        if regressor is not None
        else RandomForestRegressor(random_state=random_state)
    )
    if regressor is not None and hasattr(base_regressor, 'get_params'):
        parameters = base_regressor.get_params(deep=False)
        if parameters.get('random_state') is None and 'random_state' in parameters:
            base_regressor.set_params(random_state=random_state)

    metrics = []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for route in range(len(model.routes_)):
        members = np.flatnonzero(route_ids == route)
        metric = {
            'route_id': route,
            'n_samples': len(members),
            'n_splits': int(n_splits),
            'valid': False,
            'rank_correlation': np.nan,
            'status': 'unavailable',
        }
        if len(members) < n_splits:
            metric['status'] = f'insufficient samples for {n_splits}-fold regression'
            metrics.append(metric)
            continue

        route_target = target[members]
        predictions = np.empty(len(members), dtype=float)
        for train, test in splitter.split(normal_coordinates[members]):
            estimator = clone(base_regressor)
            estimator.fit(normal_coordinates[members][train], route_target[train])
            predictions[test] = estimator.predict(normal_coordinates[members][test])
        rank_correlation = float(spearmanr(route_target, predictions).statistic)
        metric.update({
            'valid': True,
            'rank_correlation': rank_correlation,
            'status': 'ok',
        })
        metrics.append(metric)
    return metrics


def evaluate_route_target(
    model: Any,
    result: EmbeddingResult,
    targets,
    n_splits: int = 10,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    """Select route classification or regression metrics from target type."""
    if _target_is_continuous(targets):
        return evaluate_route_regression(
            model, result, targets, n_splits=n_splits, random_state=random_state,
        )
    return evaluate_route_classification(
        model, result, targets, n_splits=n_splits, random_state=random_state,
    )


def _target_is_continuous(labels) -> bool:
    values = np.asarray(labels)
    if not np.issubdtype(values.dtype, np.number):
        return False
    unique = np.unique(values)
    if len(unique) <= 12:
        return False
    return not np.allclose(unique, np.round(unique)) or len(unique) > 20


def _target_scatter(axis, x, y, labels, **kwargs):
    """Scatter targets with categorical or regression-appropriate colors."""
    values = np.asarray(labels)
    continuous = _target_is_continuous(values)
    if not np.issubdtype(values.dtype, np.number):
        _, values = np.unique(values, return_inverse=True)
    scatter_kwargs = {
        'c': values,
        'cmap': _REGRESSION_CMAP if continuous else 'tab10',
        **kwargs,
    }
    if continuous:
        scatter_kwargs.update(vmin=float(np.min(values)), vmax=float(np.max(values)))
    artist = axis.scatter(x, y, **scatter_kwargs)
    if continuous:
        axis.figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04, label='target')
    return artist


def route_colors(model: Any) -> np.ndarray:
    """Return stable route colors shared by every display mode."""
    count = max(1, len(model.routes_))
    return plt.get_cmap('tab20', count)(np.arange(count))


def plot_labeled_graph(axis, points, labels, model, title, colors=None):
    """Plot a fitted graph directly in a two-dimensional feature space."""
    if colors is None:
        colors = route_colors(model)
    _target_scatter(axis, points[:, 0], points[:, 1], labels, s=10, alpha=0.30)
    for index, spline in enumerate(model.routes_):
        curve = spline.samples * model.scale_ + model.mean_
        if spline.closed:
            curve = np.vstack([curve, curve[0]])
        axis.plot(curve[:, 0], curve[:, 1], color=colors[index], linewidth=2.6)
    _plot_stations(axis, model, transform=lambda values: values * model.scale_ + model.mean_)
    axis.set_title(title)
    axis.set_aspect('equal', adjustable='datalim')
    axis.set_xlabel('feature 1')
    axis.set_ylabel('feature 2')


def plot_projected_graph(axis, points, labels, model, title, reducer, colors=None):
    """Plot a fitted high-dimensional graph after applying a supplied reducer."""
    if colors is None:
        colors = route_colors(model)
    displayed_points = reducer.transform(points)
    _target_scatter(
        axis, displayed_points[:, 0], displayed_points[:, 1], labels,
        s=10, alpha=0.30,
    )
    for index, spline in enumerate(model.routes_):
        curve = spline.samples * model.scale_ + model.mean_
        curve = reducer.transform(curve)
        if spline.closed:
            curve = np.vstack([curve, curve[0]])
        axis.plot(curve[:, 0], curve[:, 1], color=colors[index], linewidth=2.6)
    _plot_stations(axis, model, transform=lambda values: reducer.transform(
        values * model.scale_ + model.mean_
    ))
    axis.set_title(title)
    axis.set_aspect('equal', adjustable='datalim')
    reducer_name = getattr(reducer, 'display_name_', type(reducer).__name__)
    axis.set_xlabel(f'{reducer_name} component 1')
    axis.set_ylabel(f'{reducer_name} component 2')


def _metric_value(metric, name):
    if metric is None:
        return None
    if isinstance(metric, dict):
        value = metric.get(name)
    else:
        value = getattr(metric, name, None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _plot_route_score_pies(axis, model, route_metrics):
    """Place one normalized route score just beyond ``t=1`` per route."""
    if route_metrics is None:
        return
    if isinstance(route_metrics, dict):
        by_route = route_metrics
    else:
        by_route = {
            metric.get('route_id', route): metric
            for route, metric in enumerate(route_metrics)
        }
    if not by_route:
        return

    # The bounds are in the graph embedding's data coordinates, so each inset
    # stays attached to its route row when the figure is resized. Correct
    # the y extent for the axis data aspect; a square data box would otherwise
    # be rendered as a very flat rectangle when there are many route rows.
    is_regression = any(
        isinstance(metric, dict) and 'rank_correlation' in metric
        for metric in by_route.values()
    )
    metric_name = 'rank_correlation' if is_regression else 'normalized_accuracy'
    pie_left = 1.08
    pie_width = 0.04
    origin = axis.transData.transform((0.0, 0.0))
    x_offset = axis.transData.transform((pie_width, 0.0))[0] - origin[0]
    y_offset = axis.transData.transform((0.0, 1.0))[1] - origin[1]
    pie_height = abs(x_offset) / max(abs(y_offset), 1e-12)
    for route in range(len(model.routes_)):
        metric = by_route.get(route)
        value = _metric_value(metric, metric_name)
        if value is None:
            continue
        inset = axis.inset_axes(
            [pie_left, route - pie_height / 2.0, pie_width, pie_height],
            transform=axis.transData,
        )
        inset.set_aspect('equal')
        inset.axis('off')
        # A negative rank correlation is worse than a random ordering and is
        # therefore rendered as an empty score wedge.
        value = float(np.clip(value, 0.0, 1.0))
        inset.pie(
            [value, 1.0 - value],
            colors=['tab:blue', '0.86'],
            startangle=90,
            counterclock=False,
            wedgeprops={'linewidth': 0.35, 'edgecolor': 'white'},
        )


def plot_graph_embedding(
    axis, labels, model, result, title, colors=None, jitter_seed=0,
    classification_metrics=None, route_metrics=None,
):
    """Plot longitudinal coordinates and one optional route score per spline."""
    if colors is None:
        colors = route_colors(model)
    if route_metrics is None:
        route_metrics = classification_metrics
    colors = np.asarray(colors)
    graph_coordinate = result.route_id + 0.08 * np.random.default_rng(jitter_seed).normal(
        size=len(result.position)
    )
    _target_scatter(
        axis, result.position, graph_coordinate, labels, s=10, alpha=0.45,
    )
    route_rows = np.arange(len(model.routes_))
    closed = np.asarray([spline.closed for spline in model.routes_], dtype=bool)
    for marker, mask in (('s', ~closed), ('o', closed)):
        if np.any(mask):
            axis.scatter(
                np.full(np.count_nonzero(mask), -0.045), route_rows[mask],
                c=colors[mask], marker=marker, s=55,
                edgecolors='black', linewidths=0.5, zorder=4,
            )
    axis.set_title(title)
    axis.set_xlabel('longitudinal coordinate t')
    axis.set_ylabel('route id')
    axis.set_yticks(route_rows)
    has_metrics = route_metrics is not None and len(route_metrics) > 0
    axis.set_xlim(-0.09, 1.16 if has_metrics else 1.0)
    _plot_route_score_pies(axis, model, route_metrics)


def _format_metro_axis(axis, title):
    axis.set_title(title)
    axis.set_aspect('equal', adjustable='datalim')
    axis.set_xlabel('metro-map axis 1')
    axis.set_ylabel('metro-map axis 2')


def _plot_metro_stations(axis, model, layout):
    station_positions = layout.node_positions()
    # Junction stations are drawn in data coordinates so their size remains
    # meaningful as the map is zoomed.  Routes have already been clipped to
    # these circumferences by MetroLayout.
    for node in model.junctions_:
        if node not in station_positions:
            continue
        radius = layout.junction_radii_.get(node, 0.72)
        axis.add_patch(
            Circle(
                station_positions[node],
                radius=radius,
                facecolor='white',
                edgecolor='black',
                linewidth=1.2,
                zorder=6,
            )
        )
    endpoints = np.asarray([
        station_positions[node] for node in model.endpoints_ if node in station_positions
    ])
    if len(endpoints):
        axis.scatter(
            endpoints[:, 0], endpoints[:, 1], color='white', edgecolor='black',
            marker='s', linewidth=1.2, s=48, zorder=7,
        )


def plot_metro_lines(axis, model, title, layout=None, colors=None):
    """Plot only metro routes and their station markers."""
    if layout is None:
        layout = MetroLayout(model, random_state=0).fit()
    if colors is None:
        colors = route_colors(model)
    for index, curve in enumerate(layout.transform_splines()):
        if model.routes_[index].closed:
            curve = np.vstack([curve, curve[0]])
        axis.plot(
            curve[:, 0], curve[:, 1],
            color=colors[index], linewidth=2.6, solid_capstyle='round', zorder=3,
        )
    _plot_metro_stations(axis, model, layout)
    _format_metro_axis(axis, title)


def plot_metro_points(
    axis, labels, model, result, title, layout=None, show_nodes=False, colors=None,
    residual_width=0.02,
):
    """Plot observations in the fitted metro-map coordinates.

    Set ``show_nodes=True`` to draw the junction discs and endpoint markers
    on top of the observations.  They are hidden by default so this panel
    focuses on the point distribution.  Point fill colors encode ``labels``;
    route-colored outlines keep the spline assignment visible as well.
    """
    if layout is None:
        layout = MetroLayout(
            model, random_state=0, residual_width=residual_width,
        ).fit(result)
    displayed_points = layout.transform_points(result)
    if colors is None:
        colors = route_colors(model)
    route_ids = np.asarray(result.route_id, dtype=int)
    route_edges = colors[np.clip(route_ids, 0, len(colors) - 1)]
    _target_scatter(
        axis, displayed_points[:, 0], displayed_points[:, 1], labels,
        s=14, alpha=0.60, edgecolors=route_edges, linewidths=0.4, zorder=2,
    )
    if show_nodes:
        _plot_metro_stations(axis, model, layout)
    _format_metro_axis(axis, title)


def plot_metro_graph(
    axis, labels, model, result, title, layout=None, colors=None, residual_width=0.02,
):
    """Plot the combined metro routes, observations, and stations."""
    if layout is None:
        layout = MetroLayout(
            model, random_state=0, residual_width=residual_width,
        ).fit(result)
    if colors is None:
        colors = route_colors(model)
    plot_metro_points(axis, labels, model, result, title, layout=layout, colors=colors)
    plot_metro_lines(axis, model, title, layout=layout, colors=colors)


def plot_embedding_row(
    axes,
    points,
    labels,
    model,
    result,
    projected_title,
    graph_title,
    metro_lines_title,
    metro_points_title,
    reducer=None,
    colors=None,
    jitter_seed=0,
    layout=None,
    show_metro_nodes=False,
    classification_metrics=None,
    route_metrics=None,
    metro_residual_width=0.02,
):
    """Render the standard four-panel embedding row used by the notebooks.

    The first panel uses ``reducer`` when supplied and otherwise plots the
    original two-dimensional points.  The metro layout is fitted once and
    shared by both metro panels.  Junction and endpoint markers in the
    point-only panel are opt-in through ``show_metro_nodes``.
    """
    axes = np.asarray(axes, dtype=object).reshape(-1)
    if len(axes) != 4:
        raise ValueError("axes must contain exactly four plotting axes")
    if reducer is None:
        plot_labeled_graph(axes[0], points, labels, model, projected_title, colors=colors)
    else:
        plot_projected_graph(
            axes[0], points, labels, model, projected_title, reducer, colors=colors,
        )
    plot_graph_embedding(
        axes[1], labels, model, result, graph_title,
        colors=colors, jitter_seed=jitter_seed,
        classification_metrics=classification_metrics,
        route_metrics=route_metrics,
    )
    if layout is None:
        layout = MetroLayout(
            model, random_state=0, residual_width=metro_residual_width,
        ).fit(result)
    plot_metro_lines(
        axes[2], model, metro_lines_title, layout=layout, colors=colors,
    )
    plot_metro_points(
        axes[3], labels, model, result, metro_points_title,
        layout=layout, show_nodes=show_metro_nodes, colors=colors,
    )
    return layout


def _plot_stations(axis, model, transform):
    junctions = np.asarray([model.landmark_graph_.nodes[node] for node in model.junctions_])
    endpoints = np.asarray([model.landmark_graph_.nodes[node] for node in model.endpoints_])
    if len(junctions):
        junctions = transform(junctions)
        axis.scatter(
            junctions[:, 0], junctions[:, 1], color='white', edgecolor='black',
            marker='o', linewidth=1.2, s=60, zorder=5,
        )
    if len(endpoints):
        endpoints = transform(endpoints)
        axis.scatter(
            endpoints[:, 0], endpoints[:, 1],
            color='white', edgecolor='black', marker='s', linewidth=1.2, s=48, zorder=5,
        )


__all__ = [
    'evaluate_route_classification',
    'evaluate_route_regression',
    'evaluate_route_target',
    'plot_embedding_row',
    'plot_graph_embedding',
    'plot_labeled_graph',
    'plot_metro_graph',
    'plot_metro_lines',
    'plot_metro_points',
    'plot_projected_graph',
    'route_colors',
]
