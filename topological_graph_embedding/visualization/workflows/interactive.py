"""Unified ipywidgets viewer for the notebook workflows."""

from __future__ import annotations

from html import escape
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from topological_graph_embedding import SplineGraphEmbedding
from topological_graph_embedding.visualization.plots import (
    evaluate_route_target,
    plot_embedding_row,
)
from topological_graph_embedding.visualization.reduction import fit_reducer


def _unpack_dataset(value: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalize raw point clouds and ``(points, labels)`` datasets."""
    if isinstance(value, tuple) and len(value) == 2:
        points, labels = value
    else:
        points = value
        labels = np.zeros(len(points), dtype=int)
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("interactive datasets must be two-dimensional with at least two features")
    if len(labels) != len(points):
        raise ValueError("dataset labels must have the same length as its points")
    return points, labels


def _section(widgets: Any, title: str, *children: Any) -> Any:
    row_layout = widgets.Layout(
        display="flex",
        flex_flow="row wrap",
        align_items="center",
        gap="10px",
    )
    return widgets.VBox(
        [
            widgets.HTML(value=f"<b>{title}</b>"),
            widgets.HBox(list(children), layout=row_layout),
        ],
        layout=widgets.Layout(
            border="1px solid #dddddd",
            padding="7px 10px",
            margin="3px 0",
            width="100%",
        ),
    )


def display_interactive_viewer(
    datasets: dict[str, Any],
    *,
    default_reducer: str | None = None,
    default_n_centroids: int = 32,
    default_max_cycles: int = 5,
    random_state: int = 0,
) -> None:
    """Display the shared interactive graph-embedding viewer.

    ``datasets`` may map names to raw point clouds or to ``(points, labels)``
    pairs. The viewer uses the same controls and rendering path for both forms.
    The graph is refit only after pressing the render button, matching the
    computationally expensive nature of the controls.
    """
    if not datasets:
        raise ValueError("datasets must contain at least one named dataset")

    import ipywidgets as widgets
    from IPython.display import display

    names = list(datasets)
    first_points, _ = _unpack_dataset(datasets[names[0]])
    if default_reducer is None:
        default_reducer = "umap" if first_points.shape[1] > 2 else "native"
    default_reducer = str(default_reducer).lower()
    if default_reducer not in {"native", "pca", "umap"}:
        raise ValueError("default_reducer must be 'native', 'pca', or 'umap'")

    style = {"description_width": "initial"}
    control_width = widgets.Layout(width="270px")
    dataset_selector = widgets.Dropdown(
        options=names,
        value=names[0],
        description="dataset",
        style=style,
        layout=widgets.Layout(width="390px"),
    )
    centroid_slider = widgets.IntSlider(
        value=int(default_n_centroids), min=8, max=64, step=4,
        description="centroids", continuous_update=False,
        style=style, layout=control_width,
    )
    smoothing_slider = widgets.FloatSlider(
        value=0.02, min=0.0, max=0.20, step=0.005,
        readout_format=".3f", description="spline smoothing",
        continuous_update=False, style=style, layout=control_width,
    )
    cycles_slider = widgets.IntSlider(
        value=int(default_max_cycles), min=0, max=8, step=1,
        description="max cycles", continuous_update=False,
        style=style, layout=control_width,
    )
    threshold_mode = widgets.Dropdown(
        options=[("automatic", "auto"), ("manual", "manual")],
        value="auto", description="H1 threshold", style=style,
        layout=widgets.Layout(width="230px"),
    )
    threshold_slider = widgets.FloatSlider(
        value=0.25, min=0.0, max=1.5, step=0.025,
        readout_format=".3f", description="manual threshold",
        continuous_update=False, style=style, layout=control_width,
    )
    merge_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.025,
        readout_format=".3f", description="junction merge (0=auto)",
        continuous_update=False, style=style, layout=control_width,
    )
    reducer_selector = widgets.Dropdown(
        options=[
            ("Original first 2 features", "native"),
            ("PCA", "pca"),
            ("UMAP", "umap"),
        ],
        value=default_reducer,
        description="display reducer",
        style=style,
        layout=widgets.Layout(width="285px"),
    )
    neighbors_slider = widgets.IntSlider(
        value=15, min=2, max=100, step=1,
        description="UMAP neighbors", continuous_update=False,
        style=style, layout=control_width,
    )
    dispersion_slider = widgets.FloatSlider(
        value=0.02, min=0.0, max=0.20, step=0.005,
        readout_format=".3f", description="metro dispersion",
        continuous_update=False, style=style, layout=control_width,
    )
    fit_button = widgets.Button(
        description="Render selected dataset",
        button_style="primary",
        layout=widgets.Layout(width="240px"),
    )
    plot_output = widgets.Image(format="png")
    plot_output.layout.width = "100%"
    metrics_output = widgets.HTML()
    last_render_key: tuple[Any, ...] | None = None

    def update_reducer_controls(*_: Any) -> None:
        neighbors_slider.disabled = reducer_selector.value != "umap"

    def update_neighbor_bounds(points: np.ndarray) -> None:
        maximum = max(2, min(100, len(points) - 1))
        neighbors_slider.max = maximum
        neighbors_slider.value = min(neighbors_slider.value, maximum)

    def render_selected_dataset(_=None) -> None:
        nonlocal last_render_key
        name = dataset_selector.value
        points, labels = _unpack_dataset(datasets[name])
        update_neighbor_bounds(points)
        render_key = (
            name,
            centroid_slider.value,
            smoothing_slider.value,
            cycles_slider.value,
            threshold_mode.value,
            threshold_slider.value,
            merge_slider.value,
            reducer_selector.value,
            neighbors_slider.value,
            dispersion_slider.value,
        )
        if render_key == last_render_key:
            return
        last_render_key = render_key

        threshold = None if threshold_mode.value == "auto" else threshold_slider.value
        merge_distance = None if merge_slider.value == 0 else merge_slider.value
        model = SplineGraphEmbedding(
            n_centroids=centroid_slider.value,
            persistence_threshold=threshold,
            spline_smoothing=smoothing_slider.value,
            max_cycles=cycles_slider.value,
            random_state=random_state,
            merge_junction_distance=merge_distance,
        )
        result = model.fit_transform(points)
        model.route_metrics_ = evaluate_route_target(
            model, result, labels, n_splits=10, random_state=random_state,
        )

        reducer_method = reducer_selector.value
        reducer = None if reducer_method == "native" else fit_reducer(
            points,
            method=reducer_method,
            random_state=random_state,
            n_neighbors=neighbors_slider.value,
        )
        display_name = {
            "native": "original feature view",
            "pca": "PCA view",
            "umap": "UMAP view",
        }[reducer_method]
        figure, axes = plt.subplots(1, 4, figsize=(30, 5))
        plot_embedding_row(
            axes, points, labels, model, result,
            projected_title=f"{name}: {display_name}",
            graph_title=f"{name}: graph embedding",
            metro_lines_title=f"{name}: metro-map lines",
            metro_points_title=f"{name}: metro-map points",
            reducer=reducer,
            jitter_seed=0,
            route_metrics=model.route_metrics_,
            metro_residual_width=dispersion_slider.value,
        )
        figure.tight_layout()
        image_buffer = BytesIO()
        figure.savefig(image_buffer, format="png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        plot_output.value = image_buffer.getvalue()

        valid_metrics = [metric for metric in model.route_metrics_ if metric["valid"]]
        score_key = (
            "rank_correlation"
            if valid_metrics and "rank_correlation" in valid_metrics[0]
            else "normalized_accuracy"
        )
        metrics = {
            "dimensions": points.shape[1],
            "cycles": model.realized_cycle_count_,
            "junctions": len(model.junctions_),
            "endpoints": len(model.endpoints_),
            "chains": len(model.routes_),
            "median_residual": round(float(np.median(result.residual_norm)), 4),
            "display_reducer": display_name,
            "umap_neighbors": neighbors_slider.value,
            "metro_dispersion": dispersion_slider.value,
            "route_score": [metric[score_key] for metric in valid_metrics],
        }
        metrics_output.value = f"<pre>{escape(str(metrics))}</pre>"

    reducer_selector.observe(update_reducer_controls, names="value")
    update_reducer_controls()
    display(
        widgets.VBox(
            [
                _section(widgets, "Data", dataset_selector),
                _section(widgets, "Graph fitting", centroid_slider, smoothing_slider),
                _section(
                    widgets,
                    "Topology",
                    cycles_slider,
                    threshold_mode,
                    threshold_slider,
                    merge_slider,
                ),
                _section(
                    widgets,
                    "Display",
                    reducer_selector,
                    neighbors_slider,
                    dispersion_slider,
                ),
                fit_button,
                plot_output,
                metrics_output,
            ],
            layout=widgets.Layout(width="100%"),
        )
    )
    render_selected_dataset()
    fit_button.on_click(render_selected_dataset)


__all__ = ["display_interactive_viewer"]
