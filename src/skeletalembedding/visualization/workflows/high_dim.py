"""High-dimensional skeletal-embedding visualization workflow."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
)

from skeletalembedding import SkeletalEmbedding
from skeletalembedding.visualization.plots import (
    evaluate_route_target,
    plot_embedding_row,
)
from skeletalembedding.visualization.reduction import fit_reducer


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
        model = SkeletalEmbedding(
            initialization="skeletal",
            n_centroids=n_centroids,
            persistence_threshold=persistence_threshold,
            spline_smoothing=spline_smoothing,
            max_cycles=max_cycles,
            random_state=10 + index,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Topological landmark constraints could not all be realized by the routing substrate\\.",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="The sparse landmark graph could not realize all requested cycles.*",
                category=RuntimeWarning,
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
            'cycles': model.realized_cycle_count_,
            'junctions': len(model.junctions_),
            'endpoints': len(model.endpoints_),
            'spline_chains': len(model.splines_),
            'ribs': len(model.rib_paths_),
            'median_residual': float(np.median(result.residual_norm)),
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
        'endpoints', 'spline_chains', 'ribs', 'median_residual',
        'score_type', 'route_score',
    ]
    values = [[
        f'{row[column]:.4f}' if column == 'median_residual'
        else f'{row[column]:.2f}' if isinstance(row[column], (float, np.floating))
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
    metro_residual_width=0.02,
    umap_n_neighbors=15,
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

    figure, axes = plt.subplots(
        len(datasets), 4, figsize=(26, 4 * len(datasets)), squeeze=False,
    )
    for row, (name, (points, labels)) in enumerate(datasets.items()):
        reducer_model = fit_reducer(
            points,
            method=reducer,
            random_state=0,
            n_neighbors=umap_n_neighbors,
        )
        result = embeddings[name]
        plot_embedding_row(
            axes[row], points, labels, models[name], result,
            projected_title=f'{name}: {reducer.upper()} view and skeletal spline network',
            graph_title=f'{name}: skeleton embedding',
            metro_lines_title=f'{name}: metro-map lines',
            metro_points_title=f'{name}: metro-map points',
            reducer=reducer_model,
            jitter_seed=row,
            route_metrics=models[name].route_metrics_,
            metro_residual_width=metro_residual_width,
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
    """Display the shared interactive viewer for high-dimensional datasets."""
    from .interactive import display_interactive_viewer

    return display_interactive_viewer(
        datasets,
        default_reducer="umap",
        default_n_centroids=32,
        default_max_cycles=5,
    )
