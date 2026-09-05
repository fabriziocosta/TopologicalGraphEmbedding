"""Interactive kNN substrate and fitted SkeletalEmbedding workflow."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph

from skeletalembedding import SkeletalEmbedding
from skeletalembedding._topology import _euclidean_mst_edges, _local_scale
from skeletalembedding.datasets import generate_synthetic_datasets

N_SAMPLES = 500
DATASET_NOISE = 0.045

datasets = {}
dataset_names = []
knn_graph_cache = {}
spline_model_cache = {}
DEFAULT_KNN_NEIGHBORS = 6
KNN_MIN = 1
KNN_MAX = 30
RANDOM_STATE = 7

def symmetric_knn_graph(
    points,
    n_neighbors=DEFAULT_KNN_NEIGHBORS,
    mutual_knn=True,
):
    """Return an undirected union or mutual observation kNN graph."""
    directed = kneighbors_graph(
        points,
        n_neighbors=n_neighbors,
        mode='distance',
        include_self=False,
    )
    union = directed.maximum(directed.T)
    if not mutual_knn:
        return union.tocsr()
    mutual_mask = directed.astype(bool).multiply(directed.T.astype(bool))
    return union.multiply(mutual_mask).tocsr()



def get_knn_graph(dataset_name, n_neighbors, mutual_knn, add_mst):
    key = (dataset_name, int(n_neighbors), bool(mutual_knn), bool(add_mst))
    if key not in knn_graph_cache:
        points = datasets[dataset_name]
        graph = symmetric_knn_graph(
            points,
            n_neighbors=int(n_neighbors),
            mutual_knn=bool(mutual_knn),
        )
        mst_edges = np.empty((0, 2), dtype=int)
        if add_mst:
            mst_edges, mst_lengths = _euclidean_mst_edges(points)
            rows = np.concatenate([mst_edges[:, 0], mst_edges[:, 1]])
            columns = np.concatenate([mst_edges[:, 1], mst_edges[:, 0]])
            weights = np.concatenate([mst_lengths, mst_lengths])
            mst_graph = coo_matrix(
                (weights, (rows, columns)),
                shape=(len(points), len(points)),
            ).tocsr()
            graph = graph.maximum(mst_graph)
        knn_graph_cache[key] = (graph, mst_edges)
    return knn_graph_cache[key]

def build_datasets(n=N_SAMPLES, noise=0.045, random_state=RANDOM_STATE):
    """Build the 2D synthetic datasets used by the kNN/backbone viewer."""
    return generate_synthetic_datasets(
        n=int(n), noise=float(noise), random_state=int(random_state),
    )


def _set_dataset_size(point_fraction):
    """Regenerate the cached clouds when the point-count multiplier changes."""
    global datasets, dataset_names, knn_graph_cache, spline_model_cache

    target = max(2, round(N_SAMPLES * float(point_fraction)))
    current = len(next(iter(datasets.values()))) if datasets else None
    if current == target:
        return

    datasets = build_datasets(
        target, noise=DATASET_NOISE, random_state=RANDOM_STATE,
    )
    dataset_names = list(datasets)
    knn_graph_cache = {}
    spline_model_cache = {}


def undirected_edges(graph):
    """Extract each symmetric sparse-graph edge once."""
    coo = graph.tocoo()
    keep = coo.row < coo.col
    edges = np.column_stack([coo.row[keep], coo.col[keep]])
    distances = np.asarray(coo.data[keep], dtype=float)
    return edges.astype(int), distances


def _display_projection(points):
    """Return a consistent two-dimensional display projection."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError('points must be a two-dimensional array')

    dimension = points.shape[1]
    if dimension == 1:
        center = np.zeros(1, dtype=float)
        basis = None
    elif dimension == 2:
        center = np.zeros(2, dtype=float)
        basis = None
    else:
        center = points.mean(axis=0)
        _, _, components = np.linalg.svd(points - center, full_matrices=False)
        basis = components[:2]

    def project(values):
        values = np.asarray(values, dtype=float)
        original_shape = values.shape
        flat = values.reshape(-1, dimension)
        if dimension == 1:
            displayed = np.column_stack((flat[:, 0], np.zeros(len(flat))))
        elif dimension == 2:
            displayed = flat
        else:
            displayed = (flat - center) @ basis.T
        return displayed.reshape(original_shape[:-1] + (2,))

    return project


def _limits(points, pad=0.12):
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    return (minimum[0] - pad * span[0], maximum[0] + pad * span[0],
            minimum[1] - pad * span[1], maximum[1] + pad * span[1])


def _style_axis(axis, points):
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlim(_limits(points)[0:2])
    axis.set_ylim(_limits(points)[2:4])
    axis.set_xlabel('feature 1')
    axis.set_ylabel('feature 2')
    axis.grid(alpha=0.15, linewidth=0.6)


def get_spline_embedding(
    dataset_name, n_centroids, n_neighbors, mutual_knn, add_mst,
    max_cycles, spline_smoothing, persistence_max_points,
    electrical_metric, electrical_weight, max_residual_dim,
    coverage_refinement, coverage_tolerance, coverage_max_iterations,
    stability_selection, stability_runs, stability_fraction,
    backbone_simplification, n_backbone_nodes, junction_confidence=0.7,
    use_multiresolution=True, hierarchy_max_levels=8,
    hierarchy_target_size=1000, hierarchy_min_reduction=0.15,
    representative_method='medoid', hierarchy_distance_quantile=0.1,
    hierarchy_local_neighbors=10, backbone_max_representatives=2000,
    backbone_consensus_levels=3,
):
    """Fit and cache the downstream topology + MIP pipeline."""
    key = (
        dataset_name, int(n_centroids), int(n_neighbors), bool(mutual_knn),
        bool(add_mst), int(max_cycles),
        float(spline_smoothing), int(persistence_max_points),
        str(electrical_metric), float(electrical_weight),
        int(max_residual_dim), bool(coverage_refinement),
        None if coverage_tolerance is None else float(coverage_tolerance),
        int(coverage_max_iterations), bool(stability_selection),
        int(stability_runs), float(stability_fraction),
        float(backbone_simplification),
        None if n_backbone_nodes is None else int(n_backbone_nodes),
        float(junction_confidence),
        bool(use_multiresolution), int(hierarchy_max_levels),
        int(hierarchy_target_size), float(hierarchy_min_reduction),
        str(representative_method), float(hierarchy_distance_quantile),
        int(hierarchy_local_neighbors), int(backbone_max_representatives),
        int(backbone_consensus_levels),
    )
    if key not in spline_model_cache:
        use_resistance = electrical_metric in {'effective resistance', 'edge leverage'}
        use_current = electrical_metric == 'aggregate current'
        model = SkeletalEmbedding(
            n_centroids=int(n_centroids),
            n_backbone_nodes=(
                None if n_backbone_nodes in (None, 0)
                else int(n_backbone_nodes)
            ),
            topology_neighbors=int(n_neighbors),
            mutual_knn=bool(mutual_knn),
            add_mst=bool(add_mst),
            max_cycles=int(max_cycles),
            spline_smoothing=float(spline_smoothing),
            spline_control_mode='backbone',
            merge_junction_distance=(
                float(backbone_simplification) * _local_scale(datasets[dataset_name])
            ),
            junction_confidence=float(junction_confidence),
            random_state=RANDOM_STATE,
            standardize=False,
            persistence_max_points=int(persistence_max_points),
            use_effective_resistance=use_resistance,
            use_electrical_flow=use_current,
            routing_resistance_weight=(
                float(electrical_weight) if use_resistance else 0.0
            ),
            routing_current_weight=(
                float(electrical_weight) if use_current else 0.0
            ),
            max_residual_dim=int(max_residual_dim),
            coverage_refinement=bool(coverage_refinement),
            coverage_error_tolerance=coverage_tolerance,
            coverage_max_iterations=int(coverage_max_iterations),
            stability_selection=bool(stability_selection),
            stability_runs=int(stability_runs),
            stability_fraction=float(stability_fraction),
            use_multiresolution=bool(use_multiresolution),
            hierarchy_max_levels=int(hierarchy_max_levels),
            hierarchy_target_size=int(hierarchy_target_size),
            hierarchy_min_reduction=float(hierarchy_min_reduction),
            representative_method=str(representative_method),
            hierarchy_distance_quantile=float(hierarchy_distance_quantile),
            hierarchy_local_neighbors=int(hierarchy_local_neighbors),
            backbone_max_representatives=int(backbone_max_representatives),
            backbone_consensus_levels=int(backbone_consensus_levels),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='Topological landmark constraints could not all be realized.*',
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                'ignore',
                message='The sparse landmark graph could not realize all requested cycles.*',
                category=RuntimeWarning,
            )
            model.fit(datasets[dataset_name])
        spline_model_cache[key] = model
    return spline_model_cache[key]


def _plot_backbone(axis, points, model, project):
    """Draw the observations faintly with the fitted backbone graph."""
    display_points = project(points)
    backbone_graph = getattr(model, 'backbone_graph_', None)
    if backbone_graph is not None and backbone_graph.edges:
        backbone_segments = project(np.asarray([
            [backbone_graph.nodes[left], backbone_graph.nodes[right]]
            for left, right in backbone_graph.edges
        ]))
        axis.add_collection(LineCollection(
            backbone_segments, colors='black', linewidths=2.5,
            alpha=0.88, zorder=3,
        ))

    if backbone_graph is not None and backbone_graph.nodes:
        node_points = project(np.asarray(list(backbone_graph.nodes.values()), dtype=float))
        axis.scatter(
            node_points[:, 0], node_points[:, 1], s=34,
            facecolors='white', edgecolors='#111111', linewidths=1.0,
            marker='o', label='backbone nodes', zorder=6,
        )

    axis.scatter(display_points[:, 0], display_points[:, 1], s=5, c='#a9cbe0', alpha=0.65, zorder=0)
    axis.plot([], [], color='black', linewidth=2.5, label='backbone edges')
    axis.legend(loc='best', fontsize=7, frameon=False)
    _style_axis(axis, display_points)


def _plot_splines(axis, points, model, project):
    """Draw the observations faintly with fitted splines, without graph edges."""
    display_points = project(points)
    spline_colors = plt.get_cmap('tab10', max(1, len(model.splines_)))
    backbone_count = int(
        getattr(model, 'backbone_element_count_', len(model.splines_))
    )
    for route_id, spline in enumerate(model.splines_):
        samples = project(np.asarray(spline.samples, dtype=float))
        if len(samples) == 0:
            continue
        if spline.closed:
            samples = np.vstack([samples, samples[0]])
        is_rib = route_id >= backbone_count
        axis.plot(
            samples[:, 0], samples[:, 1], color=spline_colors(route_id),
            linewidth=1.0 if is_rib else 3.0, alpha=0.92, zorder=4,
            linestyle='-',
        )

    axis.scatter(display_points[:, 0], display_points[:, 1], s=5, c='#a9cbe0', alpha=0.65, zorder=0)
    axis.plot([], [], color='#4c78a8', linewidth=3.0, label='backbone splines')
    if backbone_count < len(model.splines_):
        axis.plot(
            [], [], color='#4c78a8', linewidth=1.0,
            label='coverage ribs',
        )
    axis.legend(loc='best', fontsize=7, frameon=False)
    _style_axis(axis, display_points)


def _plot_hierarchy(axis, points, model, project, display_level=0):
    """Show one representative level, its ancestry, and resolution evidence."""
    levels = list(getattr(model, 'levels_', []))
    if not levels:
        displayed = project(points)
        axis.scatter(
            displayed[:, 0], displayed[:, 1], s=8, color='#4c78a8', alpha=0.75,
        )
        axis.set_title('MILK hierarchy unavailable')
        _style_axis(axis, displayed)
        return

    level_index = int(np.clip(display_level, 0, len(levels) - 1))
    level = levels[level_index]
    displayed_points = project(points)
    representative_points = project(level.points)

    axis.scatter(
        displayed_points[:, 0], displayed_points[:, 1], s=5,
        color='#b8c8d3', alpha=0.35, linewidths=0, zorder=0,
    )

    # At a coarse level, connect each displayed representative to its child
    # representatives. A deterministic cap keeps the panel responsive for
    # dense interactive datasets while preserving the representative pattern.
    if level_index > 0:
        finer = levels[level_index - 1]
        parent_indices = np.asarray(finer.parent_indices, dtype=int)
        child_ids = np.arange(len(finer.points))
        if len(child_ids) > 1200:
            child_ids = np.linspace(0, len(child_ids) - 1, 1200, dtype=int)
        ancestry_segments = []
        for child in child_ids:
            parent = int(parent_indices[child])
            if 0 <= parent < len(level.points):
                ancestry_segments.append([finer.points[child], level.points[parent]])
        if ancestry_segments:
            axis.add_collection(LineCollection(
                project(np.asarray(ancestry_segments)),
                colors='#9aaab5', linewidths=0.45, alpha=0.28, zorder=1,
            ))

    descendant_counts = np.asarray(
        [len(descendants) for descendants in level.descendant_indices],
        dtype=float,
    )
    marker_sizes = 18.0 + 14.0 * np.log1p(descendant_counts)
    selected = level_index == int(getattr(model, 'selected_backbone_level_', -1))
    axis.scatter(
        representative_points[:, 0], representative_points[:, 1],
        s=marker_sizes, c=descendant_counts, cmap='viridis',
        alpha=0.95, edgecolors=('#d95f02' if selected else 'white'),
        linewidths=(1.5 if selected else 0.35), zorder=3,
    )

    topology = getattr(model, 'topology_by_level_', {}).get(level_index, {})
    cycle_count = topology.get('cycle_count', 'untested')
    junction_count = len(topology.get('junctions', [])) if topology else '—'
    target_size = getattr(model, 'hierarchy_target_size', '—')
    selected_level = getattr(model, 'selected_backbone_level_', 0)
    level_sizes = getattr(model, 'hierarchy_sizes_', [len(level.points)])
    axis.set_title(
        f'MILK-inspired hierarchy: level {level_index}\n'
        f'{len(level.points)} representatives · H1 {cycle_count} · '
        f'junctions {junction_count}',
        pad=10,
    )
    axis.text(
        0.02, 0.98,
        f"sizes: {' → '.join(map(str, level_sizes))}\n"
        f'target: {target_size} · selected backbone: L{selected_level}\n'
        f'orange edge = selected level',
        transform=axis.transAxes, va='top', ha='left', fontsize=8,
        bbox={'facecolor': 'white', 'alpha': 0.82, 'edgecolor': 'none'},
        zorder=5,
    )
    _style_axis(axis, displayed_points)


def render_view(
    dataset_name, n_centroids, n_neighbors, mutual_knn, add_mst,
    max_cycles, spline_smoothing, persistence_max_points,
    electrical_metric, electrical_weight, max_residual_dim,
    coverage_refinement, coverage_tolerance, coverage_max_iterations,
    stability_selection, stability_runs, stability_fraction,
    backbone_simplification, n_backbone_nodes, num_points_fraction=1.0,
    junction_confidence=0.7, use_multiresolution=True, hierarchy_max_levels=8,
    hierarchy_target_size=1000, hierarchy_min_reduction=0.15,
    representative_method='medoid', hierarchy_distance_quantile=0.1,
    hierarchy_local_neighbors=10, backbone_max_representatives=2000,
    backbone_consensus_levels=3, hierarchy_display_level=0,
):
    _set_dataset_size(num_points_fraction)
    points = datasets[dataset_name]
    project = _display_projection(points)
    display_points = project(points)
    graph, mst_edges = get_knn_graph(
        dataset_name, n_neighbors, mutual_knn, add_mst
    )
    fine_edges, _ = undirected_edges(graph)
    spline_model = get_spline_embedding(
        dataset_name, n_centroids, n_neighbors, mutual_knn, add_mst,
        max_cycles, spline_smoothing, persistence_max_points,
        electrical_metric, electrical_weight, max_residual_dim,
        coverage_refinement, coverage_tolerance, coverage_max_iterations,
        stability_selection, stability_runs, stability_fraction,
        backbone_simplification, n_backbone_nodes,
        junction_confidence,
        use_multiresolution, hierarchy_max_levels, hierarchy_target_size,
        hierarchy_min_reduction, representative_method,
        hierarchy_distance_quantile, hierarchy_local_neighbors,
        backbone_max_representatives, backbone_consensus_levels,
    )

    fig, axes = plt.subplots(1, 4, figsize=(28, 6), constrained_layout=True)
    fine_segments = display_points[fine_edges]
    axes[0].add_collection(LineCollection(
        fine_segments,
        colors='#6f9fba',
        linewidths=1.05,
        alpha=0.72,
        zorder=1,
    ))
    axes[0].scatter(
        display_points[:, 0], display_points[:, 1],
        s=11, c='#1e5878', alpha=0.86, linewidths=0, zorder=2,
    )
    axes[0].plot([], [], color='#6f9fba', linewidth=1.05, label='kNN links')
    graph_kind = 'mutual' if mutual_knn else 'symmetric'
    graph_description = graph_kind + (' + MST' if add_mst else '')
    if add_mst and len(mst_edges):
        axes[0].add_collection(LineCollection(
            display_points[mst_edges],
            colors='#d95f59',
            linewidths=1.4,
            alpha=0.9,
            zorder=3,
        ))
        axes[0].plot([], [], color='#d95f59', linewidth=1.4, label='MST edges')
    axes[0].legend(loc='best', fontsize=8, frameon=False)
    axes[0].set_title(
        f'{dataset_name}: {n_neighbors}-NN graph\n'
        f'({graph_description})',
        pad=10,
    )
    _style_axis(axes[0], display_points)

    _plot_hierarchy(
        axes[1], points, spline_model, project, hierarchy_display_level,
    )

    _plot_backbone(axes[2], points, spline_model, project)
    mip_status = str(spline_model.mip_status_).split(':', 1)[0]
    axes[2].set_title(
        f'{dataset_name}: fitted backbone\n'
        f'{len(spline_model.backbone_graph_.nodes)} nodes, '
        f'{len(spline_model.backbone_graph_.edges)} edges ({mip_status})\n'
        f'level {getattr(spline_model, "selected_backbone_level_", 0)} '
        f'of {len(getattr(spline_model, "levels_", [None])) - 1}',
        pad=10,
    )
    _style_axis(axes[2], display_points)

    _plot_splines(axes[3], points, spline_model, project)
    total_routes = len(spline_model.splines_)
    backbone_count = int(
        getattr(spline_model, 'backbone_element_count_', total_routes)
    )
    rib_count = total_routes - backbone_count
    axes[3].set_title(
        f'{dataset_name}: fitted splines\n'
        f'{total_routes} routes: {backbone_count} backbones {rib_count} ribs',
        pad=10,
    )

    fig.suptitle(
        'Observation graph → MILK hierarchy → fitted backbone → fitted splines',
        fontsize=15,
    )
    from IPython.display import display
    display(fig)
    plt.close(fig)


def display_interactive_controls(
    *, n_samples=500, noise=0.045, random_state=7,
):
    """Display the focused kNN/backbone viewer with SkeletalEmbedding controls."""
    global datasets, dataset_names, N_SAMPLES, DATASET_NOISE, RANDOM_STATE
    global knn_graph_cache, spline_model_cache
    import ipywidgets as widgets
    from IPython.display import display

    N_SAMPLES = int(n_samples)
    DATASET_NOISE = float(noise)
    RANDOM_STATE = int(random_state)
    datasets = build_datasets(N_SAMPLES, noise=noise, random_state=RANDOM_STATE)
    dataset_names = list(datasets)
    knn_graph_cache = {}
    spline_model_cache = {}

    dataset_selector = widgets.Dropdown(
        options=dataset_names,
        value=dataset_names[0],
        description='dataset',
        style={'description_width': 'initial'},
    )
    point_fraction_slider = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=10.0,
        step=0.1,
        readout_format='.1f',
        continuous_update=False,
        description='num points (× base)',
        style={'description_width': 'initial'},
    )
    centroid_slider = widgets.IntSlider(
        value=32,
        min=8,
        max=96,
        step=4,
        continuous_update=False,
        description='embedding centroids',
        style={'description_width': 'initial'},
    )
    backbone_nodes_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=30,
        step=1,
        continuous_update=False,
        description='backbone nodes (0=auto)',
        style={'description_width': 'initial'},
    )
    backbone_simplification_slider = widgets.FloatSlider(
        value=2.0,
        min=0.0,
        max=12.0,
        step=0.5,
        readout_format='.1f',
        continuous_update=False,
        description='junction merge radius',
        style={'description_width': 'initial'},
    )
    junction_confidence_slider = widgets.FloatSlider(
        value=0.7,
        min=0.3,
        max=0.95,
        step=0.05,
        readout_format='.2f',
        continuous_update=False,
        description='branch sensitivity (lower = more)',
        style={'description_width': 'initial'},
    )
    hierarchy_switch = widgets.Checkbox(
        value=True,
        description='MILK hierarchy',
        style={'description_width': 'initial'},
    )
    hierarchy_target_slider = widgets.IntSlider(
        value=min(1000, max(32, int(n_samples) // 4)),
        min=16,
        max=max(2000, int(n_samples) * 2),
        step=16,
        continuous_update=False,
        description='hierarchy target size',
        style={'description_width': 'initial'},
    )
    hierarchy_levels_slider = widgets.IntSlider(
        value=8,
        min=0,
        max=12,
        step=1,
        continuous_update=False,
        description='hierarchy max levels',
        style={'description_width': 'initial'},
    )
    hierarchy_reduction_slider = widgets.FloatSlider(
        value=0.15,
        min=0.01,
        max=0.9,
        step=0.01,
        readout_format='.2f',
        continuous_update=False,
        description='minimum reduction',
        style={'description_width': 'initial'},
    )
    hierarchy_quantile_slider = widgets.FloatSlider(
        value=0.1,
        min=0.01,
        max=0.5,
        step=0.01,
        readout_format='.2f',
        continuous_update=False,
        description='local distance quantile',
        style={'description_width': 'initial'},
    )
    hierarchy_neighbors_slider = widgets.IntSlider(
        value=10,
        min=2,
        max=30,
        step=1,
        continuous_update=False,
        description='hierarchy local neighbors',
        style={'description_width': 'initial'},
    )
    hierarchy_representative_selector = widgets.Dropdown(
        options=[('exact medoid', 'medoid'), ('approximate medoid', 'approx_medoid')],
        value='medoid',
        description='representatives',
        style={'description_width': 'initial'},
    )
    hierarchy_backbone_cap_slider = widgets.IntSlider(
        value=2000,
        min=32,
        max=5000,
        step=32,
        continuous_update=False,
        description='backbone representative cap',
        style={'description_width': 'initial'},
    )
    hierarchy_consensus_slider = widgets.IntSlider(
        value=3,
        min=1,
        max=6,
        step=1,
        continuous_update=False,
        description='stable levels to combine',
        style={'description_width': 'initial'},
    )
    hierarchy_display_level_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=8,
        step=1,
        continuous_update=False,
        description='display hierarchy level',
        style={'description_width': 'initial'},
    )
    max_cycles_slider = widgets.IntSlider(
        value=3,
        min=0,
        max=6,
        step=1,
        continuous_update=False,
        description='max cycles (skeleton)',
        style={'description_width': 'initial'},
    )
    persistence_cap_slider = widgets.IntSlider(
        value=60,
        min=10,
        max=N_SAMPLES,
        step=10,
        continuous_update=False,
        description='topology subsample cap',
        style={'description_width': 'initial'},
    )
    spline_smoothing_slider = widgets.FloatSlider(
        value=0.02,
        min=0.0,
        max=2.0,
        step=0.001,
        readout_format='.3f',
        continuous_update=False,
        description='spline smoothing',
        style={'description_width': 'initial'},
    )
    electrical_metric_selector = widgets.Dropdown(
        options=[
            ('edge leverage', 'edge leverage'),
            ('effective resistance', 'effective resistance'),
            ('aggregate current', 'aggregate current'),
            ('off', 'none'),
        ],
        value='edge leverage',
        description='electrical coloring',
        style={'description_width': 'initial'},
    )
    electrical_weight_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=2.0,
        step=0.1,
        readout_format='.1f',
        continuous_update=False,
        description='electrical route weight',
        style={'description_width': 'initial'},
    )
    max_residual_dim_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=1,
        step=1,
        continuous_update=False,
        description='maximum residual dimension',
        style={'description_width': 'initial'},
    )
    coverage_switch = widgets.Checkbox(
        value=True,
        description='coverage refinement',
        style={'description_width': 'initial'},
    )
    coverage_tolerance_slider = widgets.FloatSlider(
        value=0.25,
        min=0.0,
        max=2.0,
        step=0.025,
        readout_format='.3f',
        continuous_update=False,
        description='coverage tolerance',
        style={'description_width': 'initial'},
    )
    coverage_iterations_slider = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        continuous_update=False,
        description='coverage iterations',
        style={'description_width': 'initial'},
    )
    stability_switch = widgets.Checkbox(
        value=True,
        description='stability selection',
        style={'description_width': 'initial'},
    )
    stability_runs_slider = widgets.IntSlider(
        value=5,
        min=1,
        max=30,
        step=1,
        continuous_update=False,
        description='subsample runs',
        style={'description_width': 'initial'},
    )
    stability_fraction_slider = widgets.FloatSlider(
        value=0.7,
        min=0.1,
        max=1.0,
        step=0.05,
        readout_format='.2f',
        continuous_update=False,
        description='subsample fraction',
        style={'description_width': 'initial'},
    )
    knn_slider = widgets.IntSlider(
        value=DEFAULT_KNN_NEIGHBORS,
        min=KNN_MIN,
        max=KNN_MAX,
        step=1,
        continuous_update=False,
        description='base k (NN)',
        style={'description_width': 'initial'},
    )
    mutual_switch = widgets.Checkbox(
        value=False,
        description='mutual nearest neighbors',
        style={'description_width': 'initial'},
    )
    mst_switch = widgets.Checkbox(
        value=False,
        description='add Euclidean MST',
        style={'description_width': 'initial'},
    )
    status_style = (
        '<style>@keyframes skeletalembedding-spin {'
        'from { transform: rotate(0deg); }'
        'to { transform: rotate(360deg); }'
        '}</style>'
    )
    render_status = widgets.HTML(
        value=status_style + '<span style="color:#4c78a8">● Ready</span>',
        layout=widgets.Layout(min_width='180px'),
    )

    # Keep each control wide enough for its description and slider.  Without
    # a fixed flex basis, long labels can shrink the slider to a tiny stub.
    for control in (
        dataset_selector, point_fraction_slider, knn_slider, mutual_switch,
        mst_switch, centroid_slider, backbone_nodes_slider,
        backbone_simplification_slider, junction_confidence_slider,
        max_cycles_slider, persistence_cap_slider, spline_smoothing_slider,
        electrical_metric_selector, electrical_weight_slider,
        max_residual_dim_slider, coverage_switch, coverage_tolerance_slider,
        coverage_iterations_slider, stability_switch, stability_runs_slider,
        stability_fraction_slider, hierarchy_switch, hierarchy_target_slider,
        hierarchy_levels_slider, hierarchy_reduction_slider,
        hierarchy_quantile_slider, hierarchy_neighbors_slider,
        hierarchy_representative_selector, hierarchy_backbone_cap_slider,
        hierarchy_consensus_slider, hierarchy_display_level_slider,
    ):
        control.layout = widgets.Layout(width='500px', flex='0 0 500px')
    hierarchy_header = widgets.HTML(
        value='<b>MILK-inspired hierarchy</b><br>'
              '<small>Recursive local-scale grouping with medoid representatives</small>',
        layout=widgets.Layout(width='500px', flex='0 0 500px'),
    )
    render_status.layout = widgets.Layout(width='280px', flex='0 0 280px')

    controls = widgets.HBox(
        [
            dataset_selector, point_fraction_slider, knn_slider, mutual_switch,
            mst_switch, centroid_slider,
            backbone_nodes_slider,
            backbone_simplification_slider,
            junction_confidence_slider,
            max_cycles_slider, persistence_cap_slider, spline_smoothing_slider,
            electrical_metric_selector, electrical_weight_slider,
            max_residual_dim_slider, coverage_switch, coverage_tolerance_slider,
            coverage_iterations_slider, stability_switch, stability_runs_slider,
            stability_fraction_slider, hierarchy_header, hierarchy_switch,
            hierarchy_target_slider, hierarchy_levels_slider,
            hierarchy_reduction_slider, hierarchy_quantile_slider,
            hierarchy_neighbors_slider, hierarchy_representative_selector,
            hierarchy_backbone_cap_slider, hierarchy_consensus_slider,
            hierarchy_display_level_slider,
            render_status,
        ],
        layout=widgets.Layout(
            display='flex', flex_flow='row wrap', gap='18px', width='100%',
        ),
    )

    def render_with_status(**kwargs):
        render_status.value = status_style + (
            '<span style="color:#b26a00">'
            '<span style="display:inline-block; animation:skeletalembedding-spin 1s linear infinite">'
            '⟳</span> Rendering…</span>'
        )
        try:
            render_view(**kwargs)
        except Exception:
            render_status.value = status_style + '<span style="color:#b00020">✖ Render failed</span>'
            raise
        point_count = len(datasets[kwargs['dataset_name']])
        render_status.value = status_style + (
            f'<span style="color:#2e7d32">● Ready · {point_count:,} points</span>'
        )

    control_map = {
        'dataset_name': dataset_selector,
        'num_points_fraction': point_fraction_slider,
        'n_centroids': centroid_slider,
        'n_backbone_nodes': backbone_nodes_slider,
        'n_neighbors': knn_slider,
        'mutual_knn': mutual_switch,
        'add_mst': mst_switch,
        'backbone_simplification': backbone_simplification_slider,
        'junction_confidence': junction_confidence_slider,
        'max_cycles': max_cycles_slider,
        'spline_smoothing': spline_smoothing_slider,
        'persistence_max_points': persistence_cap_slider,
        'electrical_metric': electrical_metric_selector,
        'electrical_weight': electrical_weight_slider,
        'max_residual_dim': max_residual_dim_slider,
        'coverage_refinement': coverage_switch,
        'coverage_tolerance': coverage_tolerance_slider,
        'coverage_max_iterations': coverage_iterations_slider,
        'stability_selection': stability_switch,
        'stability_runs': stability_runs_slider,
        'stability_fraction': stability_fraction_slider,
        'use_multiresolution': hierarchy_switch,
        'hierarchy_target_size': hierarchy_target_slider,
        'hierarchy_max_levels': hierarchy_levels_slider,
        'hierarchy_min_reduction': hierarchy_reduction_slider,
        'hierarchy_distance_quantile': hierarchy_quantile_slider,
        'hierarchy_local_neighbors': hierarchy_neighbors_slider,
        'representative_method': hierarchy_representative_selector,
        'backbone_max_representatives': hierarchy_backbone_cap_slider,
        'backbone_consensus_levels': hierarchy_consensus_slider,
        'hierarchy_display_level': hierarchy_display_level_slider,
    }
    output = widgets.Output()

    def render_from_controls(_change=None):
        kwargs = {
            name: control.value for name, control in control_map.items()
        }
        output.clear_output(wait=True)
        with output:
            render_with_status(**kwargs)

    for control in control_map.values():
        control.observe(render_from_controls, names='value')

    # Display before starting the initial fit so the status indicator is
    # visible during expensive first renders.
    container = widgets.VBox([controls, output])
    display(container)
    render_from_controls()


__all__ = [
    "build_datasets",
    "display_interactive_controls",
    "get_knn_graph",
    "render_view",
]
