"""Reusable plotting and widget helpers for the synthetic-distribution notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

if __package__:
    from .spline_visualization import plot_embedding_row
else:
    from notebooks.spline_visualization import plot_embedding_row
from topological_graph_embedding.synthetic_datasets import generate_datasets, noisy_hypercube
from topological_graph_embedding.topological_spline_graph import TopologicalSplineGraph


def build_datasets(
    n=500,
    noise=0.045,
    hypercube_dim=4,
    hypercube_noise=0.055,
    random_state=0,
):
    """Build the 2D synthetic datasets and the noisy hypercube dataset."""
    datasets = generate_datasets(n=n, noise=noise, random_state=random_state)
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
    persistence_threshold=None,
    spline_smoothing=0.02,
    max_cycles=5,
):
    """Fit one graph per dataset and return models, projections, and a summary."""
    models = {}
    projections = {}
    summary = []

    for index, (name, points) in enumerate(datasets.items()):
        model = TopologicalSplineGraph(
            n_centroids=n_centroids,
            persistence_threshold=persistence_threshold,
            spline_smoothing=spline_smoothing,
            max_cycles=max_cycles,
            random_state=index,
        )
        models[name] = model.fit(points)
        projections[name] = model.transform(points)
        summary.append({
            'dataset': name,
            'cycles': model.cycle_count_,
            'junctions': len(model.junction_nodes_),
            'endpoints': len(model.endpoint_nodes_),
            'spline_chains': len(model.splines_),
            'median_residual': float(np.median(projections[name]['residual_norm'])),
        })
    return models, projections, summary


def _plot_summary(summary, figure_dir):
    columns = ['dataset', 'cycles', 'junctions', 'endpoints', 'spline_chains', 'median_residual']
    table_values = [[row[column] for column in columns] for row in summary]
    figure, axis = plt.subplots(figsize=(12, 3.2))
    axis.axis('off')
    table = axis.table(cellText=table_values, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    figure.tight_layout()
    figure.savefig(figure_dir / 'summary_table.png', dpi=160, bbox_inches='tight')
    plt.show()


def run_static_demo(
    project_root='.',
    hypercube_dim=4,
    n=500,
    noise=0.045,
    hypercube_noise=0.055,
    n_centroids=32,
    persistence_threshold=None,
    spline_smoothing=0.02,
    max_cycles=5,
):
    """Build, fit, plot, and summarize all static synthetic datasets."""
    project_root = Path(project_root)
    datasets = build_datasets(n, noise, hypercube_dim, hypercube_noise)
    models, projections, summary = fit_datasets(
        datasets,
        n_centroids=n_centroids,
        persistence_threshold=persistence_threshold,
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
        )

    figure.suptitle(
        'Synthetic distributions embedded with spline highways', fontsize=16, y=0.995,
    )
    figure.tight_layout()
    figure_dir = project_root / 'notebooks' / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / 'synthetic_distributions.png', dpi=160, bbox_inches='tight')
    plt.show()
    _plot_summary(summary, figure_dir)
    return datasets, models, projections, summary


def display_interactive_controls(datasets):
    """Display widgets for refitting a selected dataset with chosen parameters."""
    from html import escape
    from io import BytesIO

    import ipywidgets as widgets
    from IPython.display import display

    dataset_selector = widgets.Dropdown(options=list(datasets), value='line', description='dataset')
    centroid_slider = widgets.IntSlider(
        value=32, min=8, max=64, step=4, description='centroids', continuous_update=False,
    )
    smoothing_slider = widgets.FloatSlider(
        value=0.02, min=0.0, max=0.20, step=0.005, readout_format='.3f',
        description='smoothing', continuous_update=False,
    )
    cycles_slider = widgets.IntSlider(
        value=5, min=0, max=8, step=1, description='max cycles', continuous_update=False,
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
        )
        if render_key == last_render_key:
            return
        last_render_key = render_key
        name = dataset_selector.value
        points = datasets[name]
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
        reducer = PCA(n_components=2, random_state=0).fit(points) if points.shape[1] > 2 else None
        labels = np.zeros(len(points), dtype=int)
        figure, axes = plt.subplots(1, 4, figsize=(26, 5))
        projected_title = f"{name}: {'PCA view' if reducer is not None else 'fitted graph'}"
        plot_embedding_row(
            axes, points, labels, model, result,
            projected_title=projected_title,
            graph_title=f'{name}: graph embedding',
            metro_lines_title=f'{name}: metro-map lines',
            metro_points_title=f'{name}: metro-map points',
            reducer=reducer,
            jitter_seed=0,
        )
        figure.tight_layout()
        image_buffer = BytesIO()
        figure.savefig(image_buffer, format='png', dpi=160, bbox_inches='tight')
        plt.close(figure)
        plot_output.value = image_buffer.getvalue()
        metrics = {
            'cycles': model.cycle_count_,
            'junctions': len(model.junction_nodes_),
            'endpoints': len(model.endpoint_nodes_),
            'chains': len(model.splines_),
            'median_residual': float(np.median(result['residual_norm'])),
        }
        metrics_output.value = f'<pre>{escape(str(metrics))}</pre>'

    display(widgets.VBox([
        widgets.HBox([dataset_selector, centroid_slider, cycles_slider]),
        widgets.HBox([smoothing_slider, threshold_mode, threshold_slider, merge_slider]),
        fit_button,
        plot_output,
        metrics_output,
    ]))
    # Render once before attaching the handler so widget initialization cannot
    # invoke the same render path a second time.
    refit_selected_dataset()
    fit_button.on_click(refit_selected_dataset)
