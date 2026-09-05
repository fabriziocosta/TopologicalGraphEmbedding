"""Optional 2D/3D plots of hierarchy ancestry and skeletal refinement."""

from __future__ import annotations

import numpy as np


def plot_hierarchy(model, *, show_ancestry=False, max_points=5000, figsize=None):
    """Return a Matplotlib figure and axes in original feature coordinates.

    Plotting may subsample observations and ancestry lines, never fitted
    representatives. Marker area encodes original descendant count.
    """
    import matplotlib.pyplot as plt

    if model.n_features_in_ not in (2, 3):
        raise ValueError("plot_hierarchy requires 2D or 3D fitted observations")
    if max_points < 1:
        raise ValueError("max_points must be positive")
    dimension = model.n_features_in_
    count = len(model.levels_) + 2
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    figure = plt.figure(figsize=figsize or (5 * columns, 4 * rows))
    axes = [
        figure.add_subplot(
            rows, columns, i + 1, projection="3d" if dimension == 3 else None
        )
        for i in range(count)
    ]

    def coordinates(points):
        return (np.asarray(points) * model.scale_ + model.mean_).T

    original = model.levels_[0].points
    sample = np.linspace(
        0, len(original) - 1, min(max_points, len(original)), dtype=int
    )
    for index, (level, ax) in enumerate(zip(model.levels_, axes)):
        ax.scatter(*coordinates(original[sample]), s=2, color="0.8", alpha=0.35)
        sizes = 10 + 8 * np.log1p([len(ids) for ids in level.descendant_indices])
        ids = sample if index == 0 else np.arange(len(level.points))
        ax.scatter(*coordinates(level.points[ids]), s=sizes[ids], color="tab:blue")
        if show_ancestry and index > 0:
            fine = model.levels_[index - 1]
            take = np.linspace(
                0, len(fine.points) - 1, min(max_points, len(fine.points)), dtype=int
            )
            for child in take:
                ax.plot(
                    *coordinates(
                        [fine.points[child], level.points[fine.parent_indices[child]]]
                    ),
                    color="0.6",
                    alpha=0.3,
                    linewidth=0.5,
                )
        topology = model.topology_by_level_.get(index, {})
        for region in topology.get("junctions", []):
            ax.scatter(*coordinates([region.center]), marker="x", color="tab:red", s=70)
        for region in topology.get("endpoints", []):
            ax.scatter(
                *coordinates([region.center]), marker="s", color="tab:green", s=30
            )
        suffix = " • selected" if index == model.selected_backbone_level_ else ""
        ax.set_title(
            f"Level {index}: {len(level.points)} representatives{suffix}\n"
            f"H1: {topology.get('cycle_count', 'untested')}"
        )
    for ax in axes[-2:]:
        ax.scatter(*coordinates(original[sample]), s=2, color="0.8", alpha=0.35)
    for support in model.coarse_backbone_paths_.values():
        axes[-2].plot(*coordinates(support), color="tab:blue")
    axes[-2].set_title("Coarse backbone")
    for route, kind in zip(model.routes_, model.element_types_):
        axes[-1].plot(
            *coordinates(route.samples),
            color="tab:orange" if kind == "rib" else "tab:blue",
        )
    axes[-1].set_title("Refined backbone and ribs")
    figure.tight_layout()
    return figure, np.asarray(axes)
