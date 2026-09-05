"""Visualization tools for route networks and embedding results.

The static and interactive plotting modules are loaded lazily so importing the
core package does not require Matplotlib, Plotly, or scikit-learn.
"""

from .metro import MetroLayout

_PLOT_NAMES = (
    "evaluate_route_classification",
    "evaluate_route_regression",
    "evaluate_route_target",
    "metro_line_colors",
    "plot_embedding_row",
    "plot_graph_embedding",
    "plot_labeled_graph",
    "plot_metro_graph",
    "plot_metro_lines",
    "plot_metro_points",
    "plot_projected_graph",
    "plot_spline_3d",
    "metro_line_colors",
    "route_colors",
)
_PLOT_NAME_SET = set(_PLOT_NAMES)
_INTERACTIVE_NAME_SET = {"plot_spline_3d"}

__all__ = [
    "MetroLayout",
    "evaluate_route_classification",
    "evaluate_route_regression",
    "evaluate_route_target",
    "plot_embedding_row",
    "plot_graph_embedding",
    "plot_hierarchy",
    "plot_labeled_graph",
    "plot_metro_graph",
    "plot_metro_lines",
    "plot_metro_points",
    "plot_projected_graph",
    "plot_spline_3d",
    "route_colors",
]


def __getattr__(name: str):
    if name == "plot_hierarchy":
        from .hierarchy import plot_hierarchy

        return plot_hierarchy
    if name in _PLOT_NAME_SET:
        if name in _INTERACTIVE_NAME_SET:
            from . import interactive

            return getattr(interactive, name)
        from . import plots

        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
