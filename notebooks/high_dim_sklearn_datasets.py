"""Reusable helpers for the high-dimensional scikit-learn notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
)
if __package__:
    from .dimensionality_reduction import fit_reducer
    from .spline_visualization import evaluate_route_target, plot_embedding_row
else:
    from notebooks.dimensionality_reduction import fit_reducer
    from notebooks.spline_visualization import evaluate_route_target, plot_embedding_row
from topological_graph_embedding.topological_spline_graph import TopologicalSplineGraph


def _sample_dataset(points, labels, n, random_state):
    """Limit a loaded dataset to ``n`` deterministic, randomly selected rows."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    if n is None or n >= len(points):
        return points, labels
    if n < 1:
        raise ValueError('n must be positive or None')
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(points), size=n, replace=False))
    return points[indices], labels[indices]


def build_datasets(n=600, random_state=0):
    """Load real, higher-dimensional datasets shipped with scikit-learn.

    ``n`` is treated as a maximum because the built-in datasets have different
    numbers of observations.  The diabetes target is continuous; the other
    datasets have class labels.
    """
    loaded = {
        'digits': load_digits(),
        'wine': load_wine(),
        'breast-cancer': load_breast_cancer(),
        'diabetes': load_diabetes(),
    }
    return {
        name: _sample_dataset(dataset.data, dataset.target, n, random_state + index)
        for index, (name, dataset) in enumerate(loaded.items())
    }


def fit_datasets(
    datasets,
    n_centroids=36,
    persistence_threshold=None,
    spline_smoothing=0.02,
    max_cycles=4,
):
    """Fit the graph model to each ``(points, labels)`` dataset."""
    models = {}
    embeddings = {}
    summary = []
    for index, (name, (points, labels)) in enumerate(datasets.items()):
        model = TopologicalSplineGraph(
            n_centroids=n_centroids,
            persistence_threshold=persistence_threshold,
            spline_smoothing=spline_smoothing,
            max_cycles=max_cycles,
            random_state=10 + index,
        )
        result = model.fit_transform(points)
        model.route_metrics_ = evaluate_route_target(
            model, result, labels, n_splits=10, random_state=10 + index,
        )
        models[name] = model
        embeddings[name] = result
        valid_metrics = [
            metric for metric in model.route_metrics_ if metric['valid']
        ]
        score_key = (
            'rank_correlation'
            if valid_metrics and 'rank_correlation' in valid_metrics[0]
            else 'normalized_accuracy'
        )
        score_label = (
            'rank correlation' if score_key == 'rank_correlation'
            else 'normalized accuracy'
        )
        summary.append({
            'dataset': name,
            'dimensions': points.shape[1],
            'cycles': model.cycle_count_,
            'junctions': len(model.junction_nodes_),
            'endpoints': len(model.endpoint_nodes_),
            'spline_chains': len(model.splines_),
            'median_residual': float(np.median(result['residual_norm'])),
            'score_type': score_label,
            'route_score': (
                float(np.nanmean([metric[score_key] for metric in valid_metrics]))
                if valid_metrics else None
            ),
        })
    return models, embeddings, summary


def _plot_summary(summary, figure_dir):
    columns = [
        'dataset', 'dimensions', 'cycles', 'junctions',
        'endpoints', 'spline_chains', 'median_residual',
        'score_type', 'route_score',
    ]
    values = [[
        f'{row[column]:.2f}' if isinstance(row[column], (float, np.floating))
        else row[column]
        for column in columns
    ] for row in summary]
    figure, axis = plt.subplots(figsize=(14, 3.0))
    axis.axis('off')
    table = axis.table(cellText=values, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    figure.tight_layout()
    figure.savefig(figure_dir / 'high_dim_sklearn_summary.png', dpi=160, bbox_inches='tight')
    plt.show()


def run_demo(
    project_root='.',
    n=600,
    n_centroids=36,
    persistence_threshold=None,
    spline_smoothing=0.02,
    max_cycles=4,
    reducer='umap',
):
    """Build, fit, plot, and summarize the high-dimensional datasets."""
    datasets = build_datasets(n=n)
    models, embeddings, summary = fit_datasets(
        datasets,
        n_centroids=n_centroids,
        persistence_threshold=persistence_threshold,
        spline_smoothing=spline_smoothing,
        max_cycles=max_cycles,
    )

    figure, axes = plt.subplots(len(datasets), 4, figsize=(30, 4 * len(datasets)))
    for row, (name, (points, labels)) in enumerate(datasets.items()):
        reducer_model = fit_reducer(points, method=reducer, random_state=0)
        result = embeddings[name]
        plot_embedding_row(
            axes[row], points, labels, models[name], result,
            projected_title=f'{name}: {reducer.upper()} view and spline graph',
            graph_title=f'{name}: graph embedding',
            metro_lines_title=f'{name}: metro-map lines',
            metro_points_title=f'{name}: metro-map points',
            reducer=reducer_model,
            jitter_seed=row,
            route_metrics=models[name].route_metrics_,
        )

    figure.suptitle('High-dimensional scikit-learn datasets', fontsize=16, y=0.995)
    figure.tight_layout()
    figure_dir = Path(project_root) / 'notebooks' / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / 'high_dim_sklearn_datasets.png', dpi=160, bbox_inches='tight')
    plt.show()
    _plot_summary(summary, figure_dir)
    return datasets, models, embeddings, summary


def display_interactive_controls(datasets):
    """Display controls for refitting one high-dimensional dataset."""
    from html import escape
    from io import BytesIO

    import ipywidgets as widgets
    from IPython.display import display

    dataset_selector = widgets.Dropdown(options=list(datasets), value=list(datasets)[0], description='dataset')
    centroid_slider = widgets.IntSlider(
        value=36, min=8, max=64, step=4, description='centroids', continuous_update=False,
    )
    smoothing_slider = widgets.FloatSlider(
        value=0.02, min=0.0, max=0.20, step=0.005, readout_format='.3f',
        description='smoothing', continuous_update=False,
    )
    cycles_slider = widgets.IntSlider(
        value=4, min=0, max=8, step=1, description='max cycles', continuous_update=False,
    )
    threshold_mode = widgets.Dropdown(
        options=[('automatic', 'auto'), ('manual', 'manual')],
        value='auto', description='H1 threshold',
    )
    threshold_slider = widgets.FloatSlider(
        value=0.25, min=0.0, max=1.5, step=0.025, readout_format='.3f',
        description='manual value', continuous_update=False,
    )
    merge_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.025, readout_format='.3f',
        description='merge (0=auto)', continuous_update=False,
    )
    reducer_selector = widgets.Dropdown(
        options=[('UMAP (default)', 'umap'), ('PCA', 'pca')],
        value='umap', description='2D reducer',
    )
    fit_button = widgets.Button(description='Refit selected dataset', button_style='primary')
    plot_output = widgets.Image(format='png')
    plot_output.layout.width = '100%'
    metrics_output = widgets.HTML()
    last_render_key = None

    def refit_selected_dataset(_=None):
        nonlocal last_render_key
        render_key = (
            dataset_selector.value,
            centroid_slider.value,
            smoothing_slider.value,
            threshold_mode.value,
            threshold_slider.value,
            merge_slider.value,
            reducer_selector.value,
        )
        if render_key == last_render_key:
            return
        last_render_key = render_key
        name = dataset_selector.value
        points, labels = datasets[name]
        threshold = None if threshold_mode.value == 'auto' else threshold_slider.value
        merge_distance = None if merge_slider.value == 0 else merge_slider.value
        model = TopologicalSplineGraph(
            n_centroids=centroid_slider.value,
            persistence_threshold=threshold,
            spline_smoothing=smoothing_slider.value,
            max_cycles=cycles_slider.value,
            random_state=0,
            merge_junction_distance=merge_distance,
        )
        result = model.fit_transform(points)
        model.route_metrics_ = evaluate_route_target(
            model, result, labels, n_splits=10, random_state=0,
        )
        reducer_method = reducer_selector.value
        reducer = fit_reducer(points, method=reducer_method, random_state=0)
        figure, axes = plt.subplots(1, 4, figsize=(30, 5))
        plot_embedding_row(
            axes, points, labels, model, result,
            projected_title=f'{name}: {reducer_method.upper()} view',
            graph_title=f'{name}: graph embedding',
            metro_lines_title=f'{name}: metro-map lines',
            metro_points_title=f'{name}: metro-map points',
            reducer=reducer,
            jitter_seed=0,
            route_metrics=(
                model.route_metrics_
                if hasattr(model, 'route_metrics_') else None
            ),
        )
        figure.tight_layout()
        image_buffer = BytesIO()
        figure.savefig(image_buffer, format='png', dpi=160, bbox_inches='tight')
        plt.close(figure)
        plot_output.value = image_buffer.getvalue()
        valid_metrics = [metric for metric in model.route_metrics_ if metric['valid']]
        score_key = (
            'rank_correlation'
            if valid_metrics and 'rank_correlation' in valid_metrics[0]
            else 'normalized_accuracy'
        )
        score_label = (
            'rank correlation' if score_key == 'rank_correlation'
            else 'normalized accuracy'
        )
        metrics = {
            'dimensions': points.shape[1],
            'cycles': model.cycle_count_,
            'junctions': len(model.junction_nodes_),
            'endpoints': len(model.endpoint_nodes_),
            'chains': len(model.splines_),
            'median_residual': float(np.median(result['residual_norm'])),
            'score_type': score_label,
            'route_score': [
                metric[score_key] for metric in valid_metrics
            ],
        }
        metrics_output.value = f'<pre>{escape(str(metrics))}</pre>'

    display(widgets.VBox([
        widgets.HBox([dataset_selector, centroid_slider, cycles_slider]),
        widgets.HBox([
            smoothing_slider, threshold_mode, threshold_slider, merge_slider,
            reducer_selector,
        ]),
        fit_button,
        plot_output,
        metrics_output,
    ]))
    # Render once before attaching the handler so widget initialization cannot
    # invoke the same render path a second time.
    refit_selected_dataset()
    fit_button.on_click(refit_selected_dataset)
