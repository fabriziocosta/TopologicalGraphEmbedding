"""Matplotlib rendering of a fitted route network in feature space."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._topology import _as_point_cloud


def plot_network(
    model: Any,
    X: np.ndarray | None = None,
    ax: Any = None,
    show_projections: bool = False,
    max_projection_lines: int = 150,
    title: str | None = None,
) -> Any:
    """Plot a fitted two-dimensional route network and its observations."""
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Call fit before plot_network")
    if model.n_features_in_ != 2:
        raise ValueError("plot_network is only available for two-dimensional data")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError("plot_network requires matplotlib") from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    data = model._original_X_ if X is None else _as_point_cloud(X)
    colors = (
        "#123f5a", "#8c2f39", "#246a5b", "#5a3d78",
        "#a24b2a", "#365486", "#665126", "#3d566e",
    )
    ax.scatter(
        data[:, 0], data[:, 1], s=10, alpha=0.22,
        color="#a9cbe0", label="observations",
    )
    for index, route in enumerate(model.routes_):
        curve = route.samples * model.scale_ + model.mean_
        if route.closed:
            curve = np.vstack([curve, curve[0]])
        ax.plot(
            curve[:, 0], curve[:, 1], linewidth=2.8,
            color=colors[index % len(colors)],
            label="route" if index == 0 else None,
        )
    junction_ids = getattr(model, "junction_node_ids_", model.junctions_)
    endpoint_ids = getattr(model, "endpoint_node_ids_", model.endpoints_)
    junctions = np.asarray([
        model.landmark_graph_.nodes[node] for node in junction_ids
    ])
    endpoints = np.asarray([
        model.landmark_graph_.nodes[node] for node in endpoint_ids
    ])
    if len(junctions):
        junctions = junctions * model.scale_ + model.mean_
        ax.scatter(
            junctions[:, 0], junctions[:, 1], s=70,
            color="tab:red", zorder=5, label="junction",
        )
    if len(endpoints):
        endpoints = endpoints * model.scale_ + model.mean_
        ax.scatter(
            endpoints[:, 0], endpoints[:, 1], s=65, marker="s",
            color="tab:orange", zorder=5, label="endpoint",
        )
    if show_projections and len(data):
        result = model.transform(data)
        count = min(max_projection_lines, len(data))
        indices = np.linspace(0, len(data) - 1, count, dtype=int)
        for index in indices:
            ax.plot(
                [data[index, 0], result.projected[index, 0]],
                [data[index, 1], result.projected[index, 1]],
                color="0.35", linewidth=0.35, alpha=0.35,
            )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return ax


__all__ = ["plot_network"]
