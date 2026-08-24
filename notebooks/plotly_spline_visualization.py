"""Interactive Plotly views for the spline graph embedding."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.colors import qualitative
except ImportError:  # pragma: no cover - exercised only without the optional dependency.
    go = None
    qualitative = None

from topological_graph_embedding.metro_layout import MetroSplineLayout
if __package__:
    from .spline_visualization import spline_colors
else:
    from notebooks.spline_visualization import spline_colors


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
        raise ValueError("labels must have the same length as result['t']")
    unique = list(dict.fromkeys(values.tolist()))
    return [(str(value), values == value) for value in unique]


def plot_spline_3d(
    model: Any,
    result: dict[str, np.ndarray],
    labels: Any = None,
    layout: MetroSplineLayout | None = None,
    colors: np.ndarray | None = None,
    title: str = "Spline graph: metro plane + local residual plane",
    z_scale: float = 1.0,
    point_size: float = 4.0,
    show_nodes: bool = True,
) -> Any:
    """Return an interactive 3D spline map.

    The fitted metro layout is the XY plane and every spline is drawn at
    ``z=0``.  For each observation, the first locally fitted residual PCA
    coordinate is the signed lateral offset in XY and the second is the Z
    coordinate.  ``z_scale`` is a display-only multiplier for the height axis.
    """
    _require_plotly()
    if z_scale <= 0.0:
        raise ValueError("z_scale must be positive")
    if layout is None:
        layout = MetroSplineLayout(model, random_state=0).fit(result)
    points = layout.transform_points_3d(result)
    points[:, 2] *= float(z_scale)
    highway_ids = np.asarray(result["highway_id"], dtype=int)
    values = np.asarray(result["t"], dtype=float)
    residual_norm = np.asarray(result.get("residual_norm", np.zeros(len(values))), dtype=float)
    count = len(values)

    route_colors = spline_colors(model) if colors is None else np.asarray(colors)
    if len(route_colors) < len(model.splines_):
        raise ValueError("colors must contain at least one color per spline")

    figure = go.Figure()
    groups = _label_groups(labels, count)
    target_colors = qualitative.Dark24
    raw_labels = np.zeros(count, dtype=object) if labels is None else np.asarray(labels)
    customdata = np.column_stack([
        raw_labels,
        highway_ids,
        values,
        residual_norm,
    ])
    for group_index, (label, members) in enumerate(groups):
        marker_color = target_colors[group_index % len(target_colors)]
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
                    "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    for route, curve in enumerate(layout.transform_splines()):
        if model.splines_[route].closed:
            curve = np.vstack([curve, curve[0]])
        figure.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=np.zeros(len(curve)),
                mode="lines",
                name=f"spline {route}",
                legendgroup=f"spline-{route}",
                line=dict(color=_plotly_color(route_colors[route]), width=6),
                hoverinfo="name",
            )
        )

    if show_nodes:
        stations = layout.node_positions()
        junctions = np.asarray([
            stations[node] for node in model.junction_nodes_ if node in stations
        ])
        endpoints = np.asarray([
            stations[node] for node in model.endpoint_nodes_ if node in stations
        ])
        if len(junctions):
            figure.add_trace(
                go.Scatter3d(
                    x=junctions[:, 0], y=junctions[:, 1], z=np.zeros(len(junctions)),
                    mode="markers", name="junctions", legendgroup="stations",
                    marker=dict(size=8, color="white", line=dict(color="black", width=2)),
                    hoverinfo="name",
                )
            )
        if len(endpoints):
            figure.add_trace(
                go.Scatter3d(
                    x=endpoints[:, 0], y=endpoints[:, 1], z=np.zeros(len(endpoints)),
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
            xaxis_title="metro-map axis 1",
            yaxis_title="metro-map axis 2",
            zaxis_title="local residual PC2",
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.35, z=1.15)),
        ),
    )
    return figure


__all__ = ["plot_spline_3d"]
