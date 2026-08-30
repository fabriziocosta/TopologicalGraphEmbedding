"""Unified ipywidgets viewer for the notebook workflows."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from html import escape
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skeletalembedding import SkeletalEmbedding
from skeletalembedding.visualization.plots import (
    evaluate_route_target,
    plot_embedding_row,
)
from skeletalembedding.visualization.reduction import fit_reducer


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


def _subsample_dataset(value: Any, n_points: int) -> Any:
    """Keep a deterministic, evenly spaced subset of a dataset."""
    points, labels = _unpack_dataset(value)
    if len(points) <= n_points:
        return value
    indices = np.linspace(0, len(points) - 1, n_points, dtype=int)
    if isinstance(value, tuple) and len(value) == 2:
        return points[indices], labels[indices]
    return points[indices]


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
    default_persistence_max_points: int = 60,
    default_persistence_threshold: float | None = None,
    dataset_factory: Callable[[int, float], dict[str, Any]] | None = None,
    default_n_points: int | None = None,
    default_noise: float = 0.045,
) -> None:
    """Display the shared interactive skeleton-embedding viewer.

    ``datasets`` may map names to raw point clouds or to ``(points, labels)``
    pairs. The viewer uses the same controls and rendering path for both forms.
    The graph is refit only after pressing the render button, matching the
    computationally expensive nature of the controls. When ``dataset_factory``
    is supplied, the Data section also exposes controls for regenerating the
    point clouds with a different sample percentage or noise level.
    """
    if not datasets:
        raise ValueError("datasets must contain at least one named dataset")

    import ipywidgets as widgets
    from IPython.display import display

    names = list(datasets)
    first_points, _ = _unpack_dataset(datasets[names[0]])
    initial_n_points = len(first_points) if default_n_points is None else int(default_n_points)
    if initial_n_points < 1:
        raise ValueError("default_n_points must be positive")
    if default_noise < 0:
        raise ValueError("default_noise must be non-negative")
    if default_reducer is None:
        default_reducer = "umap" if first_points.shape[1] > 2 else "native"
    default_reducer = str(default_reducer).lower()
    if default_reducer not in {"native", "pca", "mds", "umap"}:
        raise ValueError("default_reducer must be 'native', 'pca', 'mds', or 'umap'")

    style = {"description_width": "initial"}
    control_width = widgets.Layout(width="270px")
    dataset_selector = widgets.Dropdown(
        options=names,
        value=names[0],
        description="dataset",
        style=style,
        layout=widgets.Layout(width="390px"),
    )
    points_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=100,
        step=10,
        description="points (%)",
        continuous_update=False,
        style=style,
        layout=control_width,
    ) if default_n_points is not None else None
    noise_slider = widgets.FloatSlider(
        value=float(default_noise),
        min=0.0,
        max=max(0.20, float(default_noise)),
        step=0.005,
        readout_format=".3f",
        description="noise",
        continuous_update=False,
        style=style,
        layout=control_width,
    ) if dataset_factory is not None else None
    initialization_selector = widgets.Dropdown(
        options=[
            ("skeletal (topology-aware)", "skeletal"),
            ("legacy coarsen", "legacy_coarsen"),
        ],
        value="skeletal",
        description="initialization",
        style=style,
        layout=widgets.Layout(width="300px"),
    )
    centroid_slider = widgets.IntSlider(
        value=int(default_n_centroids), min=8, max=64, step=4,
        description="centroids", continuous_update=False,
        style=style, layout=control_width,
    )
    backbone_nodes_slider = widgets.IntSlider(
        value=0, min=0, max=128, step=1,
        description="backbone nodes (0=auto)", continuous_update=False,
        style=style, layout=control_width,
    )
    model_neighbors_slider = widgets.IntSlider(
        value=6, min=2, max=50, step=1,
        description="model kNN neighbors", continuous_update=False,
        style=style, layout=control_width,
    )
    topology_neighbors_slider = widgets.IntSlider(
        value=6, min=2, max=50, step=1,
        description="topology kNN neighbors", continuous_update=False,
        style=style, layout=control_width,
    )
    mutual_knn_checkbox = widgets.Checkbox(
        value=True, description="mutual kNN", style=style, layout=control_width,
    )
    add_mst_checkbox = widgets.Checkbox(
        value=True, description="add Euclidean MST", style=style, layout=control_width,
    )
    use_mip_checkbox = widgets.Checkbox(
        value=True, description="use MIP backbone selection", style=style,
        layout=control_width,
    )
    smoothing_slider = widgets.FloatSlider(
        value=0.02, min=0.0, max=0.20, step=0.005,
        readout_format=".3f", description="spline smoothing",
        continuous_update=False, style=style, layout=control_width,
    )
    spline_control_selector = widgets.Dropdown(
        options=[("support points", "support"), ("backbone anchors", "backbone")],
        value="support", description="spline controls", style=style,
        layout=control_width,
    )
    routing_tangent_slider = widgets.FloatSlider(
        value=1.0, min=0.0, max=5.0, step=0.1,
        readout_format=".1f", description="routing tangent weight",
        continuous_update=False, style=style, layout=control_width,
    )
    routing_length_slider = widgets.FloatSlider(
        value=1.0, min=0.0, max=5.0, step=0.1,
        readout_format=".1f", description="routing length weight",
        continuous_update=False, style=style, layout=control_width,
    )
    routing_density_slider = widgets.FloatSlider(
        value=0.5, min=0.0, max=5.0, step=0.1,
        readout_format=".1f", description="routing density weight",
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
    persistence_cap_slider = widgets.IntSlider(
        value=max(1, min(int(default_persistence_max_points), 1000)),
        min=1, max=1000, step=10,
        description="topology subsample cap", continuous_update=False,
        style=style, layout=control_width,
    )
    detect_cycles_checkbox = widgets.Checkbox(
        value=True, description="detect cycles", style=style, layout=control_width,
    )
    detect_junctions_checkbox = widgets.Checkbox(
        value=True, description="detect junctions", style=style, layout=control_width,
    )
    local_pca_checkbox = widgets.Checkbox(
        value=True, description="use local PCA", style=style, layout=control_width,
    )
    tangent_boundary_checkbox = widgets.Checkbox(
        value=True, description="tangent boundary conditions", style=style,
        layout=control_width,
    )
    merge_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.025,
        readout_format=".3f", description="junction merge (0=auto)",
        continuous_update=False, style=style, layout=control_width,
    )
    local_pca_neighbors_slider = widgets.IntSlider(
        value=20, min=2, max=50, step=1,
        description="local PCA neighbors", continuous_update=False,
        style=style, layout=control_width,
    )
    branch_angle_slider = widgets.FloatSlider(
        value=45.0, min=5.0, max=180.0, step=5.0,
        readout_format=".0f", description="maximum branch angle",
        continuous_update=False, style=style, layout=control_width,
    )
    electrical_metric_selector = widgets.Dropdown(
        options=[
            ("none", "none"),
            ("effective resistance", "effective_resistance"),
            ("edge leverage", "edge_leverage"),
            ("aggregate current", "aggregate_current"),
        ],
        value="none", description="electrical diagnostic", style=style,
        layout=widgets.Layout(width="285px"),
    )
    electrical_weight_slider = widgets.FloatSlider(
        value=1.0, min=0.0, max=5.0, step=0.1,
        readout_format=".1f", description="electrical routing weight",
        continuous_update=False, style=style, layout=control_width,
    )
    kron_checkbox = widgets.Checkbox(
        value=False, description="Kron reduction", style=style, layout=control_width,
    )
    residual_dim_slider = widgets.IntSlider(
        value=0, min=0, max=8, step=1,
        description="maximum residual dimension", continuous_update=False,
        style=style, layout=control_width,
    )
    residual_bandwidth_slider = widgets.FloatSlider(
        value=0.1, min=0.01, max=1.0, step=0.01,
        readout_format=".2f", description="residual PCA bandwidth",
        continuous_update=False, style=style, layout=control_width,
    )
    residual_smoothness_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.05,
        readout_format=".2f", description="residual basis smoothness",
        continuous_update=False, style=style, layout=control_width,
    )
    coverage_checkbox = widgets.Checkbox(
        value=False, description="coverage refinement", style=style, layout=control_width,
    )
    coverage_mode = widgets.Dropdown(
        options=[
            ("automatic tolerance", "auto"),
            ("absolute tolerance", "absolute"),
            ("relative tolerance", "relative"),
        ],
        value="auto", description="coverage tolerance", style=style,
        layout=widgets.Layout(width="285px"),
    )
    coverage_tolerance_slider = widgets.FloatSlider(
        value=0.25, min=0.0, max=2.0, step=0.025,
        readout_format=".3f", description="coverage tolerance value",
        continuous_update=False, style=style, layout=control_width,
    )
    coverage_quantile_slider = widgets.FloatSlider(
        value=0.95, min=0.5, max=1.0, step=0.01,
        readout_format=".2f", description="coverage quantile",
        continuous_update=False, style=style, layout=control_width,
    )
    coverage_iterations_slider = widgets.IntSlider(
        value=10, min=1, max=20, step=1,
        description="coverage iterations", continuous_update=False,
        style=style, layout=control_width,
    )
    coverage_ribs_slider = widgets.IntSlider(
        value=0, min=0, max=30, step=1,
        description="maximum ribs (0=unlimited)", continuous_update=False,
        style=style, layout=control_width,
    )
    coverage_selection_selector = widgets.Dropdown(
        options=[("greedy", "greedy"), ("MIP", "mip")],
        value="greedy", description="rib selection", style=style,
        layout=control_width,
    )
    rib_candidate_type_selector = widgets.Dropdown(
        options=[
            ("transverse", "transverse"),
            ("parallel", "parallel"),
            ("both", "both"),
        ],
        value="transverse", description="rib candidate type", style=style,
        layout=control_width,
    )
    coverage_gain_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.01,
        readout_format=".2f", description="minimum rib gain",
        continuous_update=False, style=style, layout=control_width,
    )
    coverage_candidates_slider = widgets.IntSlider(
        value=20, min=1, max=50, step=1,
        description="rib candidates / iteration", continuous_update=False,
        style=style, layout=control_width,
    )
    coverage_spacing_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=2.0, step=0.025,
        readout_format=".3f", description="candidate spacing (0=auto)",
        continuous_update=False, style=style, layout=control_width,
    )
    coverage_min_error_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=2.0, step=0.025,
        readout_format=".3f", description="minimum seed error (0=auto)",
        continuous_update=False, style=style, layout=control_width,
    )
    coverage_length_penalty_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=2.0, step=0.05,
        readout_format=".2f", description="rib length penalty",
        continuous_update=False, style=style, layout=control_width,
    )
    coverage_rib_penalty_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=2.0, step=0.05,
        readout_format=".2f", description="rib count penalty",
        continuous_update=False, style=style, layout=control_width,
    )
    coverage_junction_penalty_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=2.0, step=0.05,
        readout_format=".2f", description="junction penalty",
        continuous_update=False, style=style, layout=control_width,
    )
    stability_checkbox = widgets.Checkbox(
        value=False, description="stability selection", style=style, layout=control_width,
    )
    stability_runs_slider = widgets.IntSlider(
        value=5, min=1, max=30, step=1,
        description="subsample runs", continuous_update=False,
        style=style, layout=control_width,
    )
    stability_fraction_slider = widgets.FloatSlider(
        value=0.7, min=0.1, max=1.0, step=0.05,
        readout_format=".2f", description="subsample fraction",
        continuous_update=False, style=style, layout=control_width,
    )
    stability_support_slider = widgets.FloatSlider(
        value=0.75, min=0.0, max=1.0, step=0.05,
        readout_format=".2f", description="minimum stability support",
        continuous_update=False, style=style, layout=control_width,
    )
    stability_jitter_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.05,
        readout_format=".2f", description="stability jitter",
        continuous_update=False, style=style, layout=control_width,
    )
    stability_residual_checkbox = widgets.Checkbox(
        value=False, description="stable residual subspaces", style=style,
        layout=control_width,
    )
    rib_stability_runs_slider = widgets.IntSlider(
        value=0, min=0, max=30, step=1,
        description="rib stability runs (0=off)", continuous_update=False,
        style=style, layout=control_width,
    )
    rib_min_support_slider = widgets.FloatSlider(
        value=0.6, min=0.0, max=1.0, step=0.05,
        readout_format=".2f", description="minimum rib support",
        continuous_update=False, style=style, layout=control_width,
    )
    reducer_selector = widgets.Dropdown(
        options=[
            ("Original first 2 features", "native"),
            ("PCA", "pca"),
            ("Classical MDS", "mds"),
            ("UMAP", "umap"),
        ],
        value=default_reducer,
        description="display reducer",
        style=style,
        layout=widgets.Layout(width="285px"),
    )
    umap_neighbors_slider = widgets.IntSlider(
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
    last_data_key: tuple[Any, ...] | None = (
        (initial_n_points, float(default_noise))
        if dataset_factory is not None
        else ((100,) if points_slider is not None else None)
    )
    active_datasets = datasets

    def update_reducer_controls(*_: Any) -> None:
        umap_neighbors_slider.disabled = reducer_selector.value != "umap"

    def update_backbone_controls(*_: Any) -> None:
        backbone_nodes_slider.disabled = initialization_selector.value != "skeletal"

    def update_coverage_controls(*_: Any) -> None:
        coverage_tolerance_slider.disabled = coverage_mode.value == "auto"

    def update_dataset_bounds(points: np.ndarray) -> None:
        maximum = max(2, min(50, len(points) - 1))
        for slider in (
            model_neighbors_slider,
            topology_neighbors_slider,
            local_pca_neighbors_slider,
        ):
            slider.max = maximum
            slider.value = min(slider.value, maximum)
        umap_maximum = max(2, min(100, len(points) - 1))
        umap_neighbors_slider.max = umap_maximum
        umap_neighbors_slider.value = min(umap_neighbors_slider.value, umap_maximum)
        persistence_cap_slider.max = max(1, min(1000, len(points)))
        persistence_cap_slider.value = min(
            persistence_cap_slider.value, persistence_cap_slider.max,
        )
        residual_dim_slider.max = max(0, min(8, points.shape[1] - 1))
        residual_dim_slider.value = min(
            residual_dim_slider.value, residual_dim_slider.max,
        )
        backbone_nodes_slider.max = max(2, min(128, len(points)))
        backbone_nodes_slider.value = min(
            backbone_nodes_slider.value, backbone_nodes_slider.max,
        )

    def render_selected_dataset(_=None) -> None:
        nonlocal active_datasets, last_data_key, last_render_key
        if dataset_factory is not None:
            n_points = max(1, round(
                initial_n_points * int(points_slider.value) / 100,
            ))
            data_key = (n_points, float(noise_slider.value))
            if data_key != last_data_key:
                active_datasets = dataset_factory(*data_key)
                if not active_datasets:
                    raise ValueError("dataset_factory must return at least one named dataset")
                dataset_selector.options = list(active_datasets)
                if dataset_selector.value not in active_datasets:
                    dataset_selector.value = next(iter(active_datasets))
                last_data_key = data_key
        elif points_slider is not None:
            point_fraction = int(points_slider.value)
            data_key = (point_fraction,)
            if data_key != last_data_key:
                active_datasets = {
                    name: _subsample_dataset(
                        value,
                        max(1, round(
                            len(_unpack_dataset(value)[0]) * point_fraction / 100,
                        )),
                    )
                    for name, value in datasets.items()
                }
                last_data_key = data_key
        else:
            data_key = None
        name = dataset_selector.value
        points, labels = _unpack_dataset(active_datasets[name])
        update_dataset_bounds(points)
        render_key = (
            name,
            data_key,
            initialization_selector.value,
            centroid_slider.value,
            backbone_nodes_slider.value,
            model_neighbors_slider.value,
            topology_neighbors_slider.value,
            mutual_knn_checkbox.value,
            add_mst_checkbox.value,
            use_mip_checkbox.value,
            smoothing_slider.value,
            spline_control_selector.value,
            routing_length_slider.value,
            routing_tangent_slider.value,
            routing_density_slider.value,
            cycles_slider.value,
            threshold_mode.value,
            threshold_slider.value,
            persistence_cap_slider.value,
            detect_cycles_checkbox.value,
            detect_junctions_checkbox.value,
            local_pca_checkbox.value,
            tangent_boundary_checkbox.value,
            merge_slider.value,
            local_pca_neighbors_slider.value,
            branch_angle_slider.value,
            electrical_metric_selector.value,
            electrical_weight_slider.value,
            kron_checkbox.value,
            residual_dim_slider.value,
            residual_bandwidth_slider.value,
            residual_smoothness_slider.value,
            coverage_checkbox.value,
            coverage_mode.value,
            coverage_tolerance_slider.value,
            coverage_quantile_slider.value,
            coverage_iterations_slider.value,
            coverage_ribs_slider.value,
            coverage_selection_selector.value,
            rib_candidate_type_selector.value,
            coverage_gain_slider.value,
            coverage_candidates_slider.value,
            coverage_spacing_slider.value,
            coverage_min_error_slider.value,
            coverage_length_penalty_slider.value,
            coverage_rib_penalty_slider.value,
            coverage_junction_penalty_slider.value,
            stability_checkbox.value,
            stability_runs_slider.value,
            stability_fraction_slider.value,
            stability_support_slider.value,
            stability_jitter_slider.value,
            stability_residual_checkbox.value,
            rib_stability_runs_slider.value,
            rib_min_support_slider.value,
            reducer_selector.value,
            umap_neighbors_slider.value,
            dispersion_slider.value,
        )
        if render_key == last_render_key:
            return

        threshold = (
            default_persistence_threshold
            if threshold_mode.value == "auto"
            else threshold_slider.value
        )
        merge_distance = None if merge_slider.value == 0 else merge_slider.value
        electrical_metric = electrical_metric_selector.value
        coverage_tolerance = (
            coverage_tolerance_slider.value
            if coverage_mode.value == "absolute"
            else None
        )
        coverage_relative_tolerance = (
            coverage_tolerance_slider.value
            if coverage_mode.value == "relative"
            else None
        )
        coverage_max_ribs = (
            None if coverage_ribs_slider.value == 0 else coverage_ribs_slider.value
        )
        n_backbone_nodes = (
            None
            if initialization_selector.value != "skeletal"
            or backbone_nodes_slider.value == 0
            else int(backbone_nodes_slider.value)
        )
        model = SkeletalEmbedding(
            initialization=initialization_selector.value,
            n_centroids=centroid_slider.value,
            n_backbone_nodes=n_backbone_nodes,
            n_neighbors=model_neighbors_slider.value,
            topology_neighbors=topology_neighbors_slider.value,
            mutual_knn=mutual_knn_checkbox.value,
            add_mst=add_mst_checkbox.value,
            use_mip=use_mip_checkbox.value,
            persistence_threshold=threshold,
            persistence_max_points=persistence_cap_slider.value,
            spline_smoothing=smoothing_slider.value,
            spline_control_mode=spline_control_selector.value,
            routing_length_weight=routing_length_slider.value,
            routing_tangent_weight=routing_tangent_slider.value,
            routing_density_weight=routing_density_slider.value,
            max_cycles=cycles_slider.value,
            detect_cycles=detect_cycles_checkbox.value,
            detect_junctions=detect_junctions_checkbox.value,
            use_local_pca=local_pca_checkbox.value,
            random_state=random_state,
            merge_junction_distance=merge_distance,
            local_pca_neighbors=local_pca_neighbors_slider.value,
            max_branch_angle_degrees=branch_angle_slider.value,
            use_tangent_boundary_conditions=tangent_boundary_checkbox.value,
            use_effective_resistance=electrical_metric in {
                "effective_resistance", "edge_leverage",
            },
            use_electrical_flow=electrical_metric == "aggregate_current",
            use_kron_reduction=kron_checkbox.value,
            routing_resistance_weight=(
                electrical_weight_slider.value
                if electrical_metric in {"effective_resistance", "edge_leverage"}
                else 0.0
            ),
            routing_current_weight=(
                electrical_weight_slider.value
                if electrical_metric == "aggregate_current" else 0.0
            ),
            max_residual_dim=residual_dim_slider.value,
            residual_pca_bandwidth=residual_bandwidth_slider.value,
            residual_subspace_smoothness=residual_smoothness_slider.value,
            coverage_refinement=coverage_checkbox.value,
            coverage_error_tolerance=coverage_tolerance,
            coverage_relative_tolerance=coverage_relative_tolerance,
            coverage_quantile=coverage_quantile_slider.value,
            coverage_max_iterations=coverage_iterations_slider.value,
            coverage_max_ribs=coverage_max_ribs,
            coverage_selection=coverage_selection_selector.value,
            rib_candidate_type=rib_candidate_type_selector.value,
            coverage_min_gain=coverage_gain_slider.value,
            coverage_max_candidates_per_iteration=coverage_candidates_slider.value,
            coverage_candidate_spacing=(
                None if coverage_spacing_slider.value == 0 else coverage_spacing_slider.value
            ),
            coverage_min_error=(
                None if coverage_min_error_slider.value == 0 else coverage_min_error_slider.value
            ),
            coverage_length_penalty=coverage_length_penalty_slider.value,
            coverage_rib_penalty=coverage_rib_penalty_slider.value,
            coverage_junction_penalty=coverage_junction_penalty_slider.value,
            stability_selection=stability_checkbox.value,
            stability_runs=stability_runs_slider.value,
            stability_fraction=stability_fraction_slider.value,
            stability_min_support=stability_support_slider.value,
            stability_jitter=stability_jitter_slider.value,
            stability_residual_subspaces=stability_residual_checkbox.value,
            rib_stability_runs=(
                None if rib_stability_runs_slider.value == 0 else rib_stability_runs_slider.value
            ),
            rib_min_support=rib_min_support_slider.value,
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
            model, result, labels, n_splits=10, random_state=random_state,
        )

        reducer_method = reducer_selector.value
        reducer = None if reducer_method == "native" else fit_reducer(
            points,
            method=reducer_method,
            random_state=random_state,
            n_neighbors=umap_neighbors_slider.value,
        )
        display_name = {
            "native": "original feature view",
            "pca": "PCA view",
            "mds": "classical MDS view",
            "umap": "UMAP view",
        }[reducer_method]
        figure, axes = plt.subplots(1, 4, figsize=(30, 5))
        plot_embedding_row(
            axes, points, labels, model, result,
            projected_title=f"{name}: {display_name}",
            graph_title=f"{name}: skeleton embedding",
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
            "initialization": model.initialization,
            "dimensions": points.shape[1],
            "model_neighbors": model.n_neighbors,
            "topology_neighbors": model.topology_neighbors_,
            "mutual_knn": model.mutual_knn,
            "add_mst": model.add_mst,
            "cycles": model.realized_cycle_count_,
            "backbone_nodes": len(model.backbone_graph_.nodes),
            "junctions": len(model.junctions_),
            "endpoints": len(model.endpoints_),
            "chains": len(model.splines_),
            "ribs": len(model.rib_paths_),
            "median_residual": round(float(np.median(result.residual_norm)), 4),
            "post_pca_error": round(float(model.reconstruction_error_), 4),
            "coverage_iterations": model.coverage_iterations_,
            "stability": model.stability_summary_.get("enabled", False),
            "display_reducer": display_name,
            "umap_neighbors": umap_neighbors_slider.value,
            "metro_dispersion": dispersion_slider.value,
            "route_score": [metric[score_key] for metric in valid_metrics],
        }
        metrics_output.value = f"<pre>{escape(str(metrics))}</pre>"
        last_render_key = render_key

    reducer_selector.observe(update_reducer_controls, names="value")
    initialization_selector.observe(update_backbone_controls, names="value")
    coverage_mode.observe(update_coverage_controls, names="value")
    update_reducer_controls()
    update_backbone_controls()
    update_coverage_controls()
    data_controls = [dataset_selector]
    if points_slider is not None:
        data_controls.append(points_slider)
    if noise_slider is not None:
        data_controls.append(noise_slider)
    display(
        widgets.VBox(
            [
                _section(widgets, "Data", *data_controls),
                _section(
                    widgets,
                    "Graph fitting",
                    initialization_selector,
                    centroid_slider,
                    backbone_nodes_slider,
                    model_neighbors_slider,
                    topology_neighbors_slider,
                    mutual_knn_checkbox,
                    add_mst_checkbox,
                    use_mip_checkbox,
                    smoothing_slider,
                    spline_control_selector,
                    routing_length_slider,
                    routing_tangent_slider,
                    routing_density_slider,
                ),
                _section(
                    widgets,
                    "Topology",
                    cycles_slider,
                    threshold_mode,
                    threshold_slider,
                    persistence_cap_slider,
                    detect_cycles_checkbox,
                    detect_junctions_checkbox,
                    local_pca_checkbox,
                    tangent_boundary_checkbox,
                    merge_slider,
                    local_pca_neighbors_slider,
                    branch_angle_slider,
                ),
                _section(
                    widgets,
                    "Electrical diagnostics",
                    electrical_metric_selector,
                    electrical_weight_slider,
                    kron_checkbox,
                ),
                _section(
                    widgets,
                    "Residual and coverage",
                    residual_dim_slider,
                    residual_bandwidth_slider,
                    residual_smoothness_slider,
                    coverage_checkbox,
                    coverage_mode,
                    coverage_tolerance_slider,
                    coverage_quantile_slider,
                    coverage_iterations_slider,
                    coverage_ribs_slider,
                    coverage_selection_selector,
                    rib_candidate_type_selector,
                    coverage_gain_slider,
                    coverage_candidates_slider,
                    coverage_spacing_slider,
                    coverage_min_error_slider,
                    coverage_length_penalty_slider,
                    coverage_rib_penalty_slider,
                    coverage_junction_penalty_slider,
                ),
                _section(
                    widgets,
                    "Stability and subsampling",
                    stability_checkbox,
                    stability_runs_slider,
                    stability_fraction_slider,
                    stability_support_slider,
                    stability_jitter_slider,
                    stability_residual_checkbox,
                    rib_stability_runs_slider,
                    rib_min_support_slider,
                ),
                _section(
                    widgets,
                    "Display",
                    reducer_selector,
                    umap_neighbors_slider,
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
