"""Interactive Plotly views for spline route embeddings."""

# Plotly's public API intentionally accepts regular dictionaries for nested
# marker, line, and scene specifications.
# ruff: noqa: C408

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.colors import qualitative
except ImportError:  # pragma: no cover - exercised only without the optional dependency.
    go = None
    qualitative = None

from .._frames import _normal_frame
from ..results import EmbeddingResult
from .metro import MetroLayout
from .plots import route_colors


def _require_plotly() -> None:
    if go is None or qualitative is None:
        raise ImportError(
            "plot_spline_3d requires Plotly; install the project requirements "
            "or run `python -m pip install plotly`."
        )


def _plotly_color(color: Any) -> str:
    """Convert an RGB/RGBA array in the matplotlib convention to Plotly RGBA."""
    values = np.asarray(color, dtype=float).reshape(-1)
    if len(values) < 3:
        return "rgb(80, 120, 180)"
    rgb = np.clip(np.rint(values[:3] * 255.0), 0, 255).astype(int)
    alpha = float(values[3]) if len(values) > 3 else 1.0
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.3f})"


def _label_groups(labels: Any, count: int) -> list[tuple[str, np.ndarray]]:
    if labels is None:
        return [("observations", np.ones(count, dtype=bool))]
    values = np.asarray(labels)
    if len(values) != count:
        raise ValueError("labels must have the same length as result.position")
    unique = list(dict.fromkeys(values.tolist()))
    return [(str(value), values == value) for value in unique]


def _fit_pca_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic three-component PCA display transform."""
    points = np.asarray(points, dtype=float)
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    components = np.asarray(components[:3], dtype=float)
    for index in range(len(components)):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0.0:
            components[index] *= -1.0
    if len(components) < 3:
        components = np.pad(components, ((0, 3 - len(components)), (0, 0)))
    return center, components


def _pca_transform(
    points: np.ndarray,
    center: np.ndarray,
    components: np.ndarray,
    z_scale: float = 1.0,
) -> np.ndarray:
    """Project points into the PCA display coordinates, padding to 3D."""
    transformed = (np.asarray(points, dtype=float) - center) @ components.T
    transformed[:, 2] *= float(z_scale)
    return transformed


def _cross_section(
    model: Any,
    result: EmbeddingResult,
    route: int,
    t: float,
    ellipse_samples: int,
    ellipse_bandwidth: float,
    ellipse_scale: float,
) -> np.ndarray:
    """Return one one-standard-deviation ellipse in original feature space."""
    spline = model.routes_[route]
    scale = np.asarray(getattr(model, "scale_", 1.0), dtype=float)
    mean = np.asarray(getattr(model, "mean_", 0.0), dtype=float)
    center = np.asarray(spline.evaluate(t), dtype=float) * scale + mean
    tangent = np.asarray(spline.tangent(t), dtype=float) * scale
    normal = _normal_frame(tangent)
    if normal.shape[1] == 0:
        return np.repeat(center[None, :], ellipse_samples, axis=0)

    route_ids = np.asarray(result.route_id, dtype=int)
    members = np.flatnonzero(route_ids == route)
    if not len(members):
        axes = np.zeros((normal.shape[1], 2), dtype=float)
    else:
        values = np.asarray(result.position, dtype=float)[members]
        distances = np.abs(values - float(t))
        if spline.closed:
            distances = np.minimum(distances, 1.0 - distances)
        weights = np.exp(-0.5 * (distances / ellipse_bandwidth) ** 2)
        if not np.any(weights > 1e-12):
            weights[int(np.argmin(distances))] = 1.0
        residual = np.asarray(result.residual, dtype=float)[members]
        coordinates = residual @ normal
        local_mean = np.average(coordinates, axis=0, weights=weights)
        centered_coordinates = coordinates - local_mean
        weighted = centered_coordinates * np.sqrt(weights)[:, None]
        covariance = (weighted.T @ weighted) / max(float(np.sum(weights)), 1e-12)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        axes = np.zeros((normal.shape[1], 2), dtype=float)
        count = min(2, normal.shape[1])
        axes[:, :count] = (
            eigenvectors[:, :count]
            * (ellipse_scale * np.sqrt(eigenvalues[:count]))[None, :]
        )
        axes = normal @ axes

    angles = np.linspace(0.0, 2.0 * np.pi, ellipse_samples, endpoint=True)
    return center + np.cos(angles)[:, None] * axes[:, 0] + np.sin(angles)[:, None] * axes[:, 1]


def _junction_ellipsoid(
    model: Any,
    original_points: np.ndarray,
    center_standardized: np.ndarray,
    pca_center: np.ndarray,
    pca_components: np.ndarray,
    z_scale: float,
    ellipse_samples: int,
    ellipsoid_scale: float,
) -> np.ndarray:
    """Build a 3D one-standard-deviation ellipsoid at a graph singularity."""
    scale = np.asarray(getattr(model, "scale_", 1.0), dtype=float)
    mean = np.asarray(getattr(model, "mean_", 0.0), dtype=float)
    center = np.asarray(center_standardized, dtype=float) * scale + mean
    original_points = np.asarray(original_points, dtype=float)
    standardized = (original_points - mean) / scale
    display_points = _pca_transform(
        original_points, pca_center, pca_components, z_scale,
    )
    display_center = _pca_transform(
        center[None, :], pca_center, pca_components, z_scale,
    )[0]
    distances = np.linalg.norm(standardized - center_standardized, axis=1)
    local_scale = float(getattr(model, "local_scale_", np.median(distances)))
    radius = max(2.5 * local_scale, 1e-8)
    members = np.flatnonzero(distances <= radius)
    minimum_members = max(8, 3 * original_points.shape[1])
    if len(members) < minimum_members:
        members = np.argsort(distances)[:minimum_members]
    local = display_points[members] - display_center
    weights = np.exp(-0.5 * (distances[members] / max(radius * 0.55, 1e-8)) ** 2)
    weighted = local * np.sqrt(weights)[:, None]
    covariance = (weighted.T @ weighted) / max(float(np.sum(weights)), 1e-12)
    # Keep the junction volume genuinely three-dimensional even when the
    # third coordinate is a very small, intentionally added noise dimension.
    display_spread = np.std(display_points, axis=0)
    floor = (0.12 * local_scale * np.maximum(display_spread, 1e-12)) ** 2
    covariance = covariance + np.diag(floor)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order] * (
        ellipsoid_scale * np.sqrt(np.maximum(eigenvalues[order], 1e-12))
    )[None, :]

    azimuth = np.linspace(0.0, 2.0 * np.pi, ellipse_samples, endpoint=True)
    elevation = np.linspace(0.0, np.pi, max(8, ellipse_samples // 2))
    unit = np.stack(
        [
            np.sin(elevation)[:, None] * np.cos(azimuth)[None, :],
            np.sin(elevation)[:, None] * np.sin(azimuth)[None, :],
            np.broadcast_to(np.cos(elevation)[:, None], (len(elevation), len(azimuth))),
        ],
        axis=-1,
    )
    return display_center + unit @ axes.T


def plot_spline_3d(
    model: Any,
    result: EmbeddingResult,
    labels: Any = None,
    layout: MetroLayout | None = None,
    colors: np.ndarray | None = None,
    title: str = "Spline skeleton in 3D PCA space",
    z_scale: float = 1.0,
    point_size: float = 4.0,
    show_nodes: bool = True,
    n_spline_samples: int = 18,
    ellipse_samples: int = 32,
    ellipse_bandwidth: float = 0.08,
    ellipse_scale: float = 1.0,
    junction_ellipsoid_scale: float = 1.0,
    show_observations: bool = True,
    show_reduced_graph: bool = True,
) -> Any:
    """Return an interactive 3D PCA view of the learned thick-bone skeleton.

    The observations and fitted routes are projected into the first three
    principal components of the input represented by ``result``.  Each route
    receives ``n_spline_samples`` normalized-position cross-sections.  A
    cross-section is a one-standard-deviation ellipse of the local residuals
    in the feature-space hyperplane perpendicular to the route tangent; the
    ellipse is then projected into the same ambient PCA coordinates.

    ``layout`` is retained as a compatibility argument but is no longer used:
    this view is a feature-space PCA rendering rather than a metro schematic.
    """
    _require_plotly()
    if z_scale <= 0.0:
        raise ValueError("z_scale must be positive")
    if n_spline_samples < 1:
        raise ValueError("n_spline_samples must be positive")
    if ellipse_samples < 4:
        raise ValueError("ellipse_samples must be at least 4")
    if ellipse_bandwidth <= 0.0 or not np.isfinite(ellipse_bandwidth):
        raise ValueError("ellipse_bandwidth must be finite and positive")
    if ellipse_scale < 0.0 or not np.isfinite(ellipse_scale):
        raise ValueError("ellipse_scale must be finite and non-negative")
    if junction_ellipsoid_scale < 0.0 or not np.isfinite(junction_ellipsoid_scale):
        raise ValueError("junction_ellipsoid_scale must be finite and non-negative")

    del layout
    original_points = np.asarray(result.projected, dtype=float) + np.asarray(
        result.residual, dtype=float,
    )
    pca_center, pca_components = _fit_pca_3d(original_points)
    points = _pca_transform(original_points, pca_center, pca_components, z_scale)
    route_ids = np.asarray(result.route_id, dtype=int)
    values = np.asarray(result.position, dtype=float)
    residual_norm = np.asarray(result.residual_norm, dtype=float)
    count = len(values)

    colors_by_route = route_colors(model) if colors is None else np.asarray(colors)
    if len(colors_by_route) < len(model.routes_):
        raise ValueError("colors must contain at least one color per spline")

    figure = go.Figure()
    if labels is None:
        groups = [
            (f"spline {route}", route_ids == route)
            for route in range(len(model.routes_))
        ]
    else:
        groups = _label_groups(labels, count)
    target_colors = qualitative.Dark24
    raw_labels = np.zeros(count, dtype=object) if labels is None else np.asarray(labels)
    customdata = np.column_stack([
        raw_labels,
        route_ids,
        values,
        residual_norm,
    ])
    if show_observations:
        for group_index, (label, members) in enumerate(groups):
            marker_color = (
                _plotly_color(colors_by_route[group_index])
                if labels is None
                else target_colors[group_index % len(target_colors)]
            )
            figure.add_trace(
                go.Scatter3d(
                    x=points[members, 0],
                    y=points[members, 1],
                    z=points[members, 2],
                    mode="markers",
                    name=label,
                    legendgroup="observations",
                    marker=dict(size=point_size, color=marker_color, opacity=0.72),
                    customdata=customdata[members],
                    hovertemplate=(
                        "label: %{customdata[0]}<br>"
                        "spline: %{customdata[1]}<br>"
                        "t: %{customdata[2]:.3f}<br>"
                        "residual norm: %{customdata[3]:.3f}<br>"
                        "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

    scale = np.asarray(getattr(model, "scale_", 1.0), dtype=float)
    mean = np.asarray(getattr(model, "mean_", 0.0), dtype=float)
    if show_reduced_graph:
        reduced_graph = getattr(model, "landmark_graph_", None)
        if reduced_graph is not None:
            for edge_index, (left, right) in enumerate(reduced_graph.edges):
                graph_edge = np.asarray([
                    reduced_graph.nodes[left], reduced_graph.nodes[right],
                ], dtype=float) * scale + mean
                graph_edge = _pca_transform(
                    graph_edge, pca_center, pca_components, z_scale,
                )
                figure.add_trace(
                    go.Scatter3d(
                        x=graph_edge[:, 0],
                        y=graph_edge[:, 1],
                        z=graph_edge[:, 2],
                        mode="lines",
                        name="reduced graph" if edge_index == 0 else "reduced graph edge",
                        legendgroup="reduced-graph",
                        showlegend=edge_index == 0,
                        # This is an auxiliary landmark graph, not a fitted
                        # spline. Keep it visually subordinate when enabled.
                        line=dict(color="rgba(70, 70, 70, 0.45)", width=1, dash="dot"),
                        hoverinfo="name",
                    )
                )
    for route, spline in enumerate(model.routes_):
        # Keep the user-controlled station count for tangent-space sections,
        # but draw the centerline densely enough that spline smoothness is not
        # hidden by a low-resolution polygonal rendering.
        section_t = np.linspace(0.0, 1.0, n_spline_samples, endpoint=not spline.closed)
        centerline_t = np.linspace(
            0.0, 1.0, max(64, 4 * n_spline_samples), endpoint=not spline.closed,
        )
        centerline = np.asarray(spline.evaluate(centerline_t), dtype=float) * scale + mean
        curve = _pca_transform(centerline, pca_center, pca_components, z_scale)
        if spline.closed:
            curve = np.vstack([curve, curve[0]])
        figure.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=curve[:, 2],
                mode="lines",
                name=f"spline {route}",
                legendgroup=f"spline-{route}",
                line=dict(color=_plotly_color(colors_by_route[route]), width=6),
                hoverinfo="name",
            )
        )

        sections = []
        for t in section_t:
            section = _cross_section(
                model,
                result,
                route,
                float(t),
                ellipse_samples,
                ellipse_bandwidth,
                ellipse_scale,
            )
            section = _pca_transform(section, pca_center, pca_components, z_scale)
            sections.append(section)

        section_grid = np.asarray(sections, dtype=float)
        if spline.closed:
            section_grid = np.vstack([section_grid, section_grid[:1]])
        route_rgb = np.asarray(colors_by_route[route], dtype=float)[:3]
        route_hex = "#" + "".join(f"{int(np.clip(value, 0.0, 1.0) * 255):02x}" for value in route_rgb)
        figure.add_trace(
            go.Surface(
                x=section_grid[:, :, 0],
                y=section_grid[:, :, 1],
                z=section_grid[:, :, 2],
                name=f"spline {route} · 1σ body",
                legendgroup=f"spline-{route}",
                showlegend=False,
                opacity=0.28,
                colorscale=[[0.0, route_hex], [1.0, route_hex]],
                showscale=False,
                hoverinfo="skip",
            )
        )

        for section in sections:
            figure.add_trace(
                go.Scatter3d(
                    x=section[:, 0],
                    y=section[:, 1],
                    z=section[:, 2],
                    mode="lines",
                    name=f"spline {route} · 1σ section",
                    legendgroup=f"spline-{route}",
                    showlegend=False,
                    line=dict(
                        color=_plotly_color(np.r_[route_rgb, 0.32]),
                        width=1,
                    ),
                    hoverinfo="name",
                )
            )

    if show_nodes:
        station_nodes = getattr(model, "landmark_graph_", None)
        if station_nodes is None:
            junction_centers_original = np.empty((0, original_points.shape[1]), dtype=float)
            junctions = np.empty((0, 3), dtype=float)
            endpoints = np.empty((0, 3), dtype=float)
        else:
            junction_ids = getattr(
                model, "junction_node_ids_", getattr(model, "junctions_", [])
            )
            endpoint_ids = getattr(
                model, "endpoint_node_ids_", getattr(model, "endpoints_", [])
            )
            junctions = np.asarray([
                station_nodes.nodes[node] for node in junction_ids
                if node in station_nodes.nodes
            ], dtype=float).reshape(-1, original_points.shape[1])
            endpoints = np.asarray([
                station_nodes.nodes[node] for node in endpoint_ids
                if node in station_nodes.nodes
            ], dtype=float).reshape(-1, original_points.shape[1])
            junction_centers_original = junctions * scale + mean
            if len(junctions):
                junctions = _pca_transform(junction_centers_original, pca_center, pca_components, z_scale)
            else:
                junctions = np.empty((0, 3), dtype=float)
            if len(endpoints):
                endpoints = _pca_transform(endpoints * scale + mean, pca_center, pca_components, z_scale)
            else:
                endpoints = np.empty((0, 3), dtype=float)
        if len(junctions):
            junction_scale = np.asarray(getattr(model, "scale_", 1.0), dtype=float)
            junction_mean = np.asarray(getattr(model, "mean_", 0.0), dtype=float)
            for junction_index, junction in enumerate(junction_centers_original):
                ellipsoid = _junction_ellipsoid(
                    model,
                    original_points,
                    (junction - junction_mean) / junction_scale,
                    pca_center,
                    pca_components,
                    z_scale,
                    ellipse_samples,
                    junction_ellipsoid_scale,
                )
                figure.add_trace(
                    go.Surface(
                        x=ellipsoid[:, :, 0],
                        y=ellipsoid[:, :, 1],
                        z=ellipsoid[:, :, 2],
                        name="junction ellipsoid",
                        legendgroup="stations",
                        showlegend=junction_index == 0,
                        opacity=0.42,
                        colorscale=[[0.0, "#303030"], [1.0, "#b0b0b0"]],
                        showscale=False,
                        hoverinfo="name",
                    )
                )
        if len(endpoints):
            figure.add_trace(
                go.Scatter3d(
                    x=endpoints[:, 0], y=endpoints[:, 1], z=endpoints[:, 2],
                    mode="markers", name="endpoints", legendgroup="stations",
                    marker=dict(size=8, symbol="square", color="white", line=dict(color="black", width=2)),
                    hoverinfo="name",
                )
            )

    figure.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(itemsizing="constant"),
        scene=dict(
            xaxis_title="PCA component 1",
            yaxis_title="PCA component 2",
            zaxis_title="PCA component 3",
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.35, z=1.15)),
        ),
    )
    return figure


__all__ = ["plot_spline_3d"]
