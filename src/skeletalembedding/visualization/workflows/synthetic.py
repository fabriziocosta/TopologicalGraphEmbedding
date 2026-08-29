"""Synthetic route-embedding visualization workflow."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from skeletalembedding import SkeletalEmbedding
from skeletalembedding.datasets import (
    generate_synthetic_datasets,
    noisy_hypercube,
)
from skeletalembedding.visualization.plots import plot_embedding_row


def build_datasets(
    n=500,
    noise=0.045,
    hypercube_dim=3,
    hypercube_noise=0.055,
    random_state=0,
    polygon_sides=5,
    binary_tree_depth=3,
    star_branches=4,
):
    """Build the 2D synthetic datasets and the noisy hypercube dataset."""
    datasets = generate_synthetic_datasets(
        n=n,
        noise=noise,
        random_state=random_state,
        polygon_sides=polygon_sides,
        binary_tree_depth=binary_tree_depth,
        star_branches=star_branches,
    )
    datasets['hypercube'] = noisy_hypercube(
        n=n,
        dim=hypercube_dim,
        noise=hypercube_noise,
        rng=np.random.default_rng(7),
    )
    return datasets


def fit_datasets(
    datasets,
    n_centroids=32,
    persistence_threshold=4.0,
    spline_smoothing=0.02,
    max_cycles=5,
    persistence_max_points=300,
):
    """Fit one graph per dataset and return models, projections, and a summary."""
    models = {}
    projections = {}
    summary = []

    for index, (name, points) in enumerate(datasets.items()):
        is_binary_tree = name == 'binary-tree'
        model = SkeletalEmbedding(
            initialization="legacy_coarsen" if is_binary_tree else "skeletal",
            n_centroids=max(n_centroids, 64) if is_binary_tree else n_centroids,
            persistence_threshold=persistence_threshold,
            persistence_max_points=persistence_max_points,
            spline_smoothing=spline_smoothing,
            max_cycles=0 if is_binary_tree else max_cycles,
            random_state=index,
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
            models[name] = model.fit(points)
        projections[name] = model.transform(points)
        summary.append({
            'dataset': name,
            'cycles': model.realized_cycle_count_,
            'face_cycles': getattr(model, 'face_cycle_count_', 0),
            'junctions': len(model.junctions_),
            'endpoints': len(model.endpoints_),
            'spline_chains': len(model.routes_),
            'median_residual': float(np.median(projections[name].residual_norm)),
        })
    return models, projections, summary


def _plot_summary(summary, figure_dir, filename='summary_table.png'):
    columns = [
        'dataset', 'cycles', 'face_cycles', 'junctions', 'endpoints',
        'spline_chains', 'median_residual',
    ]
    table_values = [[
        f'{row[column]:.4f}' if column == 'median_residual' else row[column]
        for column in columns
    ] for row in summary]
    figure, axis = plt.subplots(figsize=(12, 3.2))
    axis.axis('off')
    table = axis.table(cellText=table_values, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    figure.tight_layout()
    figure.savefig(figure_dir / filename, dpi=160, bbox_inches='tight')
    plt.show()


def run_static_demo(
    project_root='.',
    hypercube_dim=3,
    n=500,
    noise=0.045,
    hypercube_noise=0.055,
    n_centroids=32,
    persistence_threshold=4.0,
    spline_smoothing=0.02,
    max_cycles=5,
    metro_residual_width=0.02,
    polygon_sides=5,
    binary_tree_depth=3,
    star_branches=4,
    persistence_max_points=300,
):
    """Build, fit, plot, and summarize all static synthetic datasets."""
    project_root = Path(project_root)
    datasets = build_datasets(
        n,
        noise,
        hypercube_dim,
        hypercube_noise,
        polygon_sides=polygon_sides,
        binary_tree_depth=binary_tree_depth,
        star_branches=star_branches,
    )
    models, projections, summary = fit_datasets(
        datasets,
        n_centroids=n_centroids,
        persistence_threshold=persistence_threshold,
        persistence_max_points=persistence_max_points,
        spline_smoothing=spline_smoothing,
        max_cycles=max_cycles,
    )

    figure, axes = plt.subplots(
        len(datasets), 4, figsize=(26, 4 * len(datasets)), squeeze=False,
    )
    reducers = {
        name: PCA(n_components=2, random_state=0).fit(points) if points.shape[1] > 2 else None
        for name, points in datasets.items()
    }
    for row, (name, points) in enumerate(datasets.items()):
        labels = np.zeros(len(points), dtype=int)
        model = models[name]
        result = projections[name]
        title = name.replace('-', ' ').title()
        if reducers[name] is not None:
            title += ' (PCA view)'
        plot_embedding_row(
            axes[row], points, labels, model, result,
            projected_title=title,
            graph_title=f'{name}: graph embedding',
            metro_lines_title=f'{name}: metro-map lines',
            metro_points_title=f'{name}: metro-map points',
            reducer=reducers[name],
            jitter_seed=row,
            metro_residual_width=metro_residual_width,
        )

    figure.suptitle(
        'Synthetic distributions embedded with spline routes', fontsize=16, y=0.995,
    )
    figure.tight_layout()
    figure_dir = project_root / 'notebooks' / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / 'synthetic_distributions.png', dpi=160, bbox_inches='tight')
    plt.show()
    _plot_summary(summary, figure_dir)
    return datasets, models, projections, summary


def display_interactive_controls(datasets):
    """Display the shared interactive viewer for synthetic datasets."""
    from .interactive import display_interactive_viewer

    return display_interactive_viewer(
        datasets,
        default_reducer="pca",
        default_n_centroids=32,
        default_max_cycles=5,
        default_persistence_max_points=300,
        default_persistence_threshold=4.0,
    )
