"""Interactive kNN substrate and SkeletalEmbedding backbone workflow."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

from skeletalembedding import SkeletalEmbedding
from skeletalembedding._topology import _euclidean_mst_edges, _local_scale
from skeletalembedding.datasets import generate_synthetic_datasets

N_SAMPLES = 500

datasets = {}
dataset_names = []
knn_graph_cache = {}
spline_model_cache = {}
DEFAULT_KNN_NEIGHBORS = 6
KNN_MIN = 2
KNN_MAX = 12
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
    """Build the 2D synthetic datasets used by the coarsening viewer."""
    return generate_synthetic_datasets(
        n=int(n), noise=float(noise), random_state=int(random_state),
    )


def undirected_edges(graph):
    """Extract each symmetric sparse-graph edge once."""
    coo = graph.tocoo()
    keep = coo.row < coo.col
    edges = np.column_stack([coo.row[keep], coo.col[keep]])
    distances = np.asarray(coo.data[keep], dtype=float)
    return edges.astype(int), distances


def coarsen_knn_graph(points, graph, n_centroids):
    """Fit centroids and contract the observation kNN edges between clusters."""
    model = KMeans(
        n_clusters=int(n_centroids),
        n_init=10,
        random_state=RANDOM_STATE,
    )
    labels = model.fit_predict(points)
    centers = model.cluster_centers_
    sizes = np.bincount(labels, minlength=len(centers))

    fine_edges, _ = undirected_edges(graph)
    left = labels[fine_edges[:, 0]]
    right = labels[fine_edges[:, 1]]
    crosses_cluster = left != right
    pairs = np.sort(
        np.column_stack([left[crosses_cluster], right[crosses_cluster]]),
        axis=1,
    )
    if len(pairs):
        pair_codes = pairs[:, 0] * len(centers) + pairs[:, 1]
        unique_codes, support = np.unique(pair_codes, return_counts=True)
        coarse_edges = np.column_stack(
            np.unravel_index(unique_codes, (len(centers), len(centers)))
        )
    else:
        coarse_edges = np.empty((0, 2), dtype=int)
        support = np.empty(0, dtype=int)
    return centers, sizes, coarse_edges.astype(int), support.astype(int)


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
    initialization, use_mip, max_cycles, spline_smoothing, persistence_max_points,
    electrical_metric, electrical_weight, max_residual_dim,
    coverage_refinement, coverage_tolerance, coverage_max_iterations,
    stability_selection, stability_runs, stability_fraction,
    backbone_simplification, n_backbone_nodes,
):
    """Fit and cache the downstream skeletal pipeline."""
    key = (
        dataset_name, int(n_centroids), int(n_neighbors), bool(mutual_knn),
        bool(add_mst), str(initialization), bool(use_mip), int(max_cycles),
        float(spline_smoothing), int(persistence_max_points),
        str(electrical_metric), float(electrical_weight),
        int(max_residual_dim), bool(coverage_refinement),
        None if coverage_tolerance is None else float(coverage_tolerance),
        int(coverage_max_iterations), bool(stability_selection),
        int(stability_runs), float(stability_fraction),
        float(backbone_simplification),
        None if n_backbone_nodes is None else int(n_backbone_nodes),
    )
    if key not in spline_model_cache:
        use_resistance = electrical_metric in {'effective resistance', 'edge leverage'}
        use_current = electrical_metric == 'aggregate current'
        if initialization == 'auto':
            initialization = 'legacy_coarsen' if dataset_name == 'binary-tree' else 'skeletal'
        model = SkeletalEmbedding(
            n_centroids=int(n_centroids),
            n_backbone_nodes=(
                None if initialization != 'skeletal' or n_backbone_nodes in (None, 0)
                else int(n_backbone_nodes)
            ),
            initialization=initialization,
            topology_neighbors=int(n_neighbors),
            mutual_knn=bool(mutual_knn),
            add_mst=bool(add_mst),
            max_cycles=int(max_cycles),
            use_mip=bool(use_mip),
            spline_smoothing=float(spline_smoothing),
            spline_control_mode='backbone',
            merge_junction_distance=(
                float(backbone_simplification) * _local_scale(datasets[dataset_name])
            ),
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


def _routing_metric(model, metric):
    if metric == 'effective resistance':
        return getattr(model, 'effective_resistance_', {}), 'effective resistance'
    if metric == 'edge leverage':
        return getattr(model, 'edge_leverage_', {}), 'edge leverage'
    if metric == 'aggregate current':
        return getattr(model, 'electrical_traffic_', {}), 'aggregate current'
    return {}, 'routing substrate'


def _plot_spline_pipeline(axis, points, model, electrical_metric):
    """Draw substrate, simplified backbone, support paths, and splines."""
    routing_graph = getattr(model, 'routing_graph_', None)
    metric_values, metric_label = _routing_metric(model, electrical_metric)
    if routing_graph is not None and routing_graph.edges:
        edges = list(routing_graph.edges)
        segments = np.asarray([
            [routing_graph.points[left], routing_graph.points[right]]
            for left, right in edges
        ])
        if metric_values:
            values = np.asarray([metric_values.get(edge, 0.0) for edge in edges], dtype=float)
            finite = np.isfinite(values)
            safe_values = np.where(finite, values, 0.0)
            low = float(np.min(safe_values))
            high = float(np.max(safe_values))
            if high <= low + 1e-12:
                scaled = np.zeros_like(safe_values)
                norm = Normalize(vmin=low, vmax=low + 1.0)
            else:
                scaled = (safe_values - low) / (high - low)
                norm = Normalize(vmin=low, vmax=high)
            cmap = plt.get_cmap('magma')
            collection = LineCollection(
                segments, colors=cmap(norm(safe_values)),
                linewidths=0.35 + 1.8 * scaled, alpha=0.42, zorder=1,
            )
            axis.add_collection(collection)
            scalar = ScalarMappable(norm=norm, cmap=cmap)
            scalar.set_array(safe_values)
            axis.figure.colorbar(scalar, ax=axis, fraction=0.046, pad=0.04, label=metric_label)
        else:
            axis.add_collection(LineCollection(
                segments, colors='#a7adb3', linewidths=0.45, alpha=0.30, zorder=1,
            ))

    landmark_graph = getattr(model, 'landmark_graph_', None)
    if landmark_graph is not None and landmark_graph.edges:
        backbone_segments = np.asarray([
            [landmark_graph.nodes[left], landmark_graph.nodes[right]]
            for left, right in landmark_graph.edges
        ])
        axis.add_collection(LineCollection(
            backbone_segments, colors='black', linewidths=4.0,
            alpha=0.80, zorder=3,
        ))

    route_colors = plt.get_cmap('tab10', max(1, len(model.splines_)))
    for route_id, (chain, spline) in enumerate(zip(model.route_chains_, model.splines_)):
        color = route_colors(route_id)
        support = np.asarray(chain.get('points', []), dtype=float)
        if len(support) >= 2:
            axis.plot(
                support[:, 0], support[:, 1], color=color, linewidth=0.9,
                linestyle=':', alpha=0.58, zorder=4,
            )
        samples = np.asarray(spline.samples, dtype=float)
        if chain.get('closed') and len(samples):
            samples = np.vstack([samples, samples[0]])
        if len(samples):
            axis.plot(
                samples[:, 0], samples[:, 1], color=color, linewidth=2.7,
                alpha=0.95, zorder=5,
            )

    backbone_nodes = getattr(model, 'backbone_graph_', None)
    if backbone_nodes is not None and backbone_nodes.nodes:
        node_points = np.asarray(list(backbone_nodes.nodes.values()), dtype=float)
        axis.scatter(
            node_points[:, 0], node_points[:, 1], s=34,
            facecolors='white', edgecolors='#111111', linewidths=1.0,
            marker='o', label='backbone nodes', zorder=6,
        )

    junctions = getattr(model, 'junction_regions_', [])
    endpoints = getattr(model, 'endpoint_regions_', [])
    if junctions:
        centers = np.asarray([region.center for region in junctions])
        axis.scatter(
            centers[:, 0], centers[:, 1], s=62, c='#111111',
            marker='o', edgecolors='white', linewidths=0.8,
            label='junctions', zorder=7,
        )
    if endpoints:
        centers = np.asarray([region.center for region in endpoints])
        axis.scatter(
            centers[:, 0], centers[:, 1], s=62, facecolors='white',
            edgecolors='#111111', linewidths=1.2, marker='o',
            label='endpoints', zorder=7,
        )

    axis.scatter(points[:, 0], points[:, 1], s=5, c='#a9cbe0', alpha=1.0, zorder=0)
    axis.plot([], [], color='black', linewidth=4.0, label='simplified backbone')
    axis.plot([], [], color='#4c78a8', linewidth=2.7, label='fitted spline')
    axis.plot([], [], color='#4c78a8', linewidth=0.9, linestyle=':', label='dense support path')
    axis.legend(loc='best', fontsize=7, frameon=False)
    axis.text(
        0.02, 0.98,
        f'cycles: {model.realized_cycle_count_}/{model.requested_cycle_count_}\n'
        f'landmark edges: {len(model.landmark_graph_.edges)}  |  '
        f'skeleton splines: {len(model.splines_)}  |  ribs: {len(model.rib_paths_)}',
        transform=axis.transAxes, va='top', fontsize=8,
        bbox={'facecolor': 'white', 'alpha': 0.78, 'edgecolor': 'none', 'pad': 3},
        zorder=8,
    )
    _style_axis(axis, points)


def render_view(
    dataset_name, n_centroids, n_neighbors, mutual_knn, add_mst,
    initialization, use_mip, max_cycles, spline_smoothing, persistence_max_points,
    electrical_metric, electrical_weight, max_residual_dim,
    coverage_refinement, coverage_tolerance, coverage_max_iterations,
    stability_selection, stability_runs, stability_fraction,
    backbone_simplification, n_backbone_nodes,
):
    points = datasets[dataset_name]
    graph, mst_edges = get_knn_graph(
        dataset_name, n_neighbors, mutual_knn, add_mst
    )
    fine_edges, _ = undirected_edges(graph)
    centers, sizes, coarse_edges, support = coarsen_knn_graph(
        points, graph, n_centroids
    )
    spline_model = get_spline_embedding(
        dataset_name, n_centroids, n_neighbors, mutual_knn, add_mst,
        initialization, use_mip, max_cycles, spline_smoothing, persistence_max_points,
        electrical_metric, electrical_weight, max_residual_dim,
        coverage_refinement, coverage_tolerance, coverage_max_iterations,
        stability_selection, stability_runs, stability_fraction,
        backbone_simplification, n_backbone_nodes,
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)
    fine_segments = points[fine_edges]
    axes[0].add_collection(LineCollection(
        fine_segments,
        colors='#6f9fba',
        linewidths=1.05,
        alpha=0.72,
        zorder=1,
    ))
    axes[0].scatter(
        points[:, 0], points[:, 1],
        s=11, c='#1e5878', alpha=0.86, linewidths=0, zorder=2,
    )
    axes[0].plot([], [], color='#6f9fba', linewidth=1.05, label='kNN links')
    graph_kind = 'mutual' if mutual_knn else 'symmetric'
    graph_description = graph_kind + (' + MST' if add_mst else '')
    if add_mst and len(mst_edges):
        axes[0].add_collection(LineCollection(
            points[mst_edges],
            colors='#d95f59',
            linewidths=1.4,
            alpha=0.9,
            zorder=3,
        ))
        axes[0].plot([], [], color='#d95f59', linewidth=1.4, label='MST edges')
    axes[0].legend(loc='best', fontsize=8, frameon=False)
    axes[0].set_title(f'{dataset_name}: {n_neighbors}-NN graph ({graph_description})')
    _style_axis(axes[0], points)

    # Keep the original cloud faintly visible so the coarsening is easy to follow.
    axes[1].scatter(
        points[:, 0], points[:, 1],
        s=7, c='#d9e6ed', alpha=0.48, linewidths=0, zorder=1,
    )
    if len(coarse_edges):
        coarse_segments = centers[coarse_edges]
        relative_support = support / max(float(support.max()), 1.0)
        axes[1].add_collection(LineCollection(
            coarse_segments,
            colors='#356f8f',
            linewidths=0.8 + 2.2 * relative_support,
            alpha=0.82,
            zorder=2,
        ))
    axes[1].scatter(
        centers[:, 0], centers[:, 1],
        s=24 + 90 * sizes / max(float(sizes.max()), 1.0),
        c='#d95f59', edgecolors='#7f2d2a', linewidths=0.7,
        alpha=0.95, zorder=3,
    )
    axes[1].set_title(
        f'{dataset_name}: {len(centers)} coarsened centroids + edges ({graph_description})'
    )
    _style_axis(axes[1], points)

    _plot_spline_pipeline(axes[2], points, spline_model, electrical_metric)
    axes[2].set_title(
        f'{dataset_name}: {spline_model.initialization} backbone, '
        f'{len(spline_model.backbone_graph_.nodes)} nodes, '
        f'simplification ({backbone_simplification:g}× local scale) + '
        f'skeleton splines ({len(spline_model.rib_paths_)} ribs)'
    )

    fig.suptitle(
        'Observation graph → k-means contraction → skeleton network',
        fontsize=15,
    )
    from IPython.display import display
    display(fig)
    plt.close(fig)


def display_interactive_controls(
    *, n_samples=500, noise=0.045, random_state=7,
):
    """Display the focused kNN/coarsening viewer with SkeletalEmbedding controls."""
    global datasets, dataset_names, N_SAMPLES, RANDOM_STATE, knn_graph_cache, spline_model_cache
    import ipywidgets as widgets
    from IPython.display import display

    N_SAMPLES = int(n_samples)
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
    initialization_selector = widgets.Dropdown(
        options=[
            ('automatic (legacy tree / skeletal otherwise)', 'auto'),
            ('skeletal (topology-aware)', 'skeletal'),
            ('legacy coarsen', 'legacy_coarsen'),
        ],
        value='auto',
        description='initialization',
        style={'description_width': 'initial'},
    )
    centroid_slider = widgets.IntSlider(
        value=32,
        min=8,
        max=96,
        step=4,
        continuous_update=False,
        description='coarsened k-means',
        style={'description_width': 'initial'},
    )
    backbone_nodes_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=N_SAMPLES,
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
        description='backbone simplification',
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
    use_mip_switch = widgets.Checkbox(
        value=True,
        description='use MIP backbone selection',
        style={'description_width': 'initial'},
    )
    spline_smoothing_slider = widgets.FloatSlider(
        value=0.02,
        min=0.0,
        max=0.03,
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

    controls = widgets.HBox(
        [
            dataset_selector, initialization_selector, knn_slider, mutual_switch,
            mst_switch, use_mip_switch, centroid_slider,
            backbone_nodes_slider,
            backbone_simplification_slider,
            max_cycles_slider, persistence_cap_slider, spline_smoothing_slider,
            electrical_metric_selector, electrical_weight_slider,
            max_residual_dim_slider, coverage_switch, coverage_tolerance_slider,
            coverage_iterations_slider, stability_switch, stability_runs_slider,
            stability_fraction_slider,
        ],
        layout=widgets.Layout(display='flex', flex_flow='row wrap', gap='18px'),
    )

    def update_backbone_nodes_control(*_):
        is_skeletal = (
            initialization_selector.value == 'skeletal'
            or (
                initialization_selector.value == 'auto'
                and dataset_selector.value != 'binary-tree'
            )
        )
        backbone_nodes_slider.disabled = not is_skeletal

    initialization_selector.observe(update_backbone_nodes_control, names='value')
    dataset_selector.observe(update_backbone_nodes_control, names='value')
    update_backbone_nodes_control()

    output = widgets.interactive_output(
        render_view,
        {
            'dataset_name': dataset_selector,
            'n_centroids': centroid_slider,
            'n_backbone_nodes': backbone_nodes_slider,
            'n_neighbors': knn_slider,
            'mutual_knn': mutual_switch,
            'add_mst': mst_switch,
            'initialization': initialization_selector,
            'use_mip': use_mip_switch,
            'backbone_simplification': backbone_simplification_slider,
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
        },
    )
    display(controls, output)


__all__ = [
    "build_datasets",
    "display_interactive_controls",
    "get_knn_graph",
    "render_view",
]
