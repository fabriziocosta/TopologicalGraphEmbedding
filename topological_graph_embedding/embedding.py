"""Public spline graph embedding estimator."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._curves import _SCIPY_IMPORT_ERROR, _fit_curve
from ._frames import (
    _fit_normal_frame_grid,
    spline_normal_coordinates as _spline_normal_coordinates,
)
from ._topology import (
    _as_point_cloud,
    _estimate_persistence,
    _extract_chains,
    _is_nearly_linear,
    _kmeans,
    _local_scale,
    _merge_nearby_junctions,
    _minimum_spanning_tree,
    _ordered_path_graph,
    _project_to_principal_line,
    _prune_short_terminal_branches,
    _symmetric_knn_edges,
    _LandmarkGraph,
    _standardize,
)
from .results import EmbeddingResult

Array = np.ndarray

class SplineGraphEmbedding:
    """Fit a smooth graph of spline highways to a point cloud."""

    def __init__(
        self,
        n_centroids: int = 32,
        persistence_threshold: float | None = None,
        spline_smoothing: float = 0.02,
        max_cycles: int = 5,
        random_state: int = 0,
        standardize: bool = True,
        merge_junction_distance: float | None = None,
        prune_short_branches: bool = True,
        prune_branch_factor: float = 0.5,
        persistence_max_points: int = 60,
        spline_samples_per_node: int = 12,
        linear_structure_tolerance: float = 0.12,
        topology_neighbors: int = 6,
    ) -> None:
        if n_centroids < 3:
            raise ValueError("n_centroids must be at least 3")
        if max_cycles < 0:
            raise ValueError("max_cycles must be non-negative")
        if persistence_max_points < 1:
            raise ValueError("persistence_max_points must be at least 1")
        if spline_samples_per_node < 1:
            raise ValueError("spline_samples_per_node must be at least 1")
        if topology_neighbors < 2:
            raise ValueError("topology_neighbors must be at least 2")
        if spline_smoothing < 0 or not np.isfinite(spline_smoothing):
            raise ValueError("spline_smoothing must be finite and non-negative")
        if prune_branch_factor < 0 or not np.isfinite(prune_branch_factor):
            raise ValueError("prune_branch_factor must be finite and non-negative")
        if linear_structure_tolerance < 0 or not np.isfinite(linear_structure_tolerance):
            raise ValueError("linear_structure_tolerance must be finite and non-negative")
        if persistence_threshold is not None and (
            persistence_threshold < 0 or not np.isfinite(persistence_threshold)
        ):
            raise ValueError("persistence_threshold must be finite and non-negative")
        if merge_junction_distance is not None and (
            merge_junction_distance < 0 or not np.isfinite(merge_junction_distance)
        ):
            raise ValueError("merge_junction_distance must be finite and non-negative")
        self.n_centroids = int(n_centroids)
        self.persistence_threshold = persistence_threshold
        self.spline_smoothing = float(spline_smoothing)
        self.max_cycles = int(max_cycles)
        self.random_state = int(random_state)
        self.standardize = bool(standardize)
        self.merge_junction_distance = merge_junction_distance
        self.prune_short_branches = prune_short_branches
        self.prune_branch_factor = float(prune_branch_factor)
        self.persistence_max_points = int(persistence_max_points)
        self.spline_samples_per_node = int(spline_samples_per_node)
        self.linear_structure_tolerance = float(linear_structure_tolerance)
        self.topology_neighbors = int(topology_neighbors)
        self._fitted = False

    def fit(self, X: Array | Sequence[Sequence[float]]) -> "SplineGraphEmbedding":
        original = _as_point_cloud(X)
        if original.shape[0] < 3:
            raise ValueError("X must contain at least three observations for fitting")
        if self.standardize:
            points, mean, scale = _standardize(original)
        else:
            points = original.copy()
            mean = np.zeros(original.shape[1])
            scale = np.ones(original.shape[1])
        self.n_features_in_ = points.shape[1]
        self.mean_ = mean
        self.scale_ = scale
        self._original_X_ = original

        self.persistence_diagram_, self.persistence_backend_ = _estimate_persistence(
            points, max_points=self.persistence_max_points, random_state=self.random_state
        )
        self.local_scale_ = _local_scale(points)
        if self.persistence_threshold is None:
            threshold = 4.0 * self.local_scale_
            self.persistence_threshold_ = float(threshold)
        else:
            self.persistence_threshold_ = float(self.persistence_threshold)
        if len(self.persistence_diagram_):
            lifetimes = self.persistence_diagram_[:, 1] - self.persistence_diagram_[:, 0]
            significant = np.isfinite(lifetimes) & (lifetimes >= self.persistence_threshold_)
            self.persistent_cycle_count_ = int(np.sum(significant))
        else:
            self.persistent_cycle_count_ = 0
        self.requested_cycle_count_ = min(self.max_cycles, self.persistent_cycle_count_)

        self.centroids_ = _kmeans(points, self.n_centroids, self.random_state)
        self.linear_structure_ = _is_nearly_linear(self.centroids_, self.linear_structure_tolerance)
        if self.linear_structure_:
            # A noisy sample of a line can produce a branched MST and a
            # borderline H1 bar.  A PCA-ordered chain is the appropriate
            # low-complexity skeleton for this geometry.
            self.requested_cycle_count_ = 0
            self.merge_junction_distance_ = 0.0
            graph = _ordered_path_graph(self.centroids_)
        else:
            graph = _minimum_spanning_tree(self.centroids_)
            if self.prune_short_branches:
                _prune_short_terminal_branches(graph, self.prune_branch_factor)
            if self.merge_junction_distance is None:
                merge_distance = 13.0 * self.local_scale_
            else:
                merge_distance = self.merge_junction_distance
            self.merge_junction_distance_ = float(merge_distance)
            graph = _merge_nearby_junctions(graph, merge_distance)
        self.landmark_graph_ = graph
        self.topology_candidate_edges_ = _symmetric_knn_edges(
            self.landmark_graph_, self.topology_neighbors,
        )

        target_cycles = self.requested_cycle_count_
        while self.landmark_graph_.cycle_rank() < target_cycles:
            candidate = self._best_cycle_edge()
            if candidate is None:
                break
            left, right, weight = candidate
            self.landmark_graph_.add_edge(left, right, weight)

        self.realized_cycle_count_ = self.landmark_graph_.cycle_rank()
        self.topology_shortfall_ = max(0, target_cycles - self.realized_cycle_count_)
        if self.topology_shortfall_:
            warnings.warn(
                "The sparse landmark graph could not realize all requested cycles "
                f"({self.topology_shortfall_} shortfall).",
                RuntimeWarning,
                stacklevel=2,
            )
        self.junctions_ = [node for node in self.landmark_graph_.nodes if self.landmark_graph_.degree(node) >= 3]
        self.endpoints_ = [node for node in self.landmark_graph_.nodes if self.landmark_graph_.degree(node) == 1]
        self.route_chains_ = _extract_chains(self.landmark_graph_)
        if not self.route_chains_:
            # Pruning can collapse a very small or fully duplicated graph to a
            # single geometric landmark.  Keep the transform contract valid by
            # retaining one degenerate highway instead of returning -1 IDs.
            nodes = sorted(self.landmark_graph_.nodes)
            if nodes:
                self.route_chains_ = [{"nodes": [nodes[0]], "closed": False}]
        for chain in self.route_chains_:
            chain["points"] = np.asarray([self.landmark_graph_.nodes[node] for node in chain["nodes"]])
            if self.linear_structure_:
                chain["points"] = _project_to_principal_line(chain["points"])
        self.routes_ = [
            _fit_curve(
                chain["points"],
                closed=chain["closed"],
                smoothing=self.spline_smoothing,
                sample_count=max(64, len(chain["nodes"]) * self.spline_samples_per_node),
            )
            for chain in self.route_chains_
        ]
        self._anchor_closed_junctions()
        if _SCIPY_IMPORT_ERROR is not None:
            warnings.warn(
                "SciPy is unavailable; spline highways use the NumPy fallback: "
                f"{_SCIPY_IMPORT_ERROR}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.route_backends_ = [spline.backend for spline in self.routes_]
        self.normal_frame_grids_ = [
            _fit_normal_frame_grid(spline) for spline in self.routes_
        ]
        self._fitted = True
        return self

    def _anchor_closed_junctions(self) -> None:
        """Make closed spline samples pass exactly through graph junctions."""
        for chain, spline in zip(self.route_chains_, self.routes_):
            if not chain["closed"] or len(spline.samples) == 0:
                continue
            nodes = list(chain["nodes"])
            if len(nodes) > 1 and nodes[0] == nodes[-1]:
                nodes = nodes[:-1]
            used_samples: set[int] = set()
            for node_index, node in enumerate(nodes):
                if self.landmark_graph_.degree(node) < 3 or node_index >= len(spline.t_values):
                    continue
                parameter = float(spline.t_values[node_index]) % 1.0
                sample = int(np.rint(parameter * len(spline.samples))) % len(spline.samples)
                if sample in used_samples:
                    continue
                spline.samples[sample] = self.landmark_graph_.nodes[node]
                used_samples.add(sample)

    def _best_cycle_edge(self) -> tuple[int, int, float] | None:
        graph = self.landmark_graph_
        best: tuple[float, int, int, float] | None = None
        existing = set(graph.edges)
        for left, right in sorted(self.topology_candidate_edges_):
            if graph._key(left, right) in existing:
                continue
            direct = float(np.linalg.norm(graph.nodes[left] - graph.nodes[right]))
            if direct <= max(1e-10, 0.25 * self.local_scale_):
                continue
            path_distance, path = graph.shortest_path(left, right)
            if len(path) < 4:
                continue
            score = path_distance / direct
            candidate = (score, left, right, direct)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            return None
        _, left, right, direct = best
        return left, right, direct

    def transform(self, X: Array | Sequence[Sequence[float]]) -> dict[str, Array]:
        if not self._fitted:
            raise RuntimeError("Call fit before transform")
        original = _as_point_cloud(X)
        if original.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        points = (original - self.mean_) / self.scale_
        best_distance = np.full(len(points), np.inf)
        route_id = np.full(len(points), -1, dtype=int)
        t = np.zeros(len(points))
        projection = np.zeros_like(points)
        for highway, spline in enumerate(self.routes_):
            candidate_projection, candidate_t, candidate_d2 = spline.project(points)
            improved = candidate_d2 < best_distance
            best_distance[improved] = candidate_d2[improved]
            route_id[improved] = highway
            t[improved] = candidate_t[improved]
            projection[improved] = candidate_projection[improved]
        self.invalid_projection_count_ = int(np.sum(route_id < 0))
        if self.invalid_projection_count_:
            raise RuntimeError(
                "The fitted spline graph could not project all observations; "
                f"{self.invalid_projection_count_} observations have no highway."
            )
        projected_original = projection * self.scale_ + self.mean_
        residual = original - projected_original
        tangent = np.zeros_like(points)
        for highway, spline in enumerate(self.routes_):
            members = np.flatnonzero(route_id == highway)
            if len(members):
                tangent[members] = spline.tangent(t[members])
        result = {
            "route_id": route_id,
            "t": t,
            "projection": projected_original,
            "residual": residual,
            "residual_norm": np.linalg.norm(residual, axis=1),
            # Tangents are stored in the standardized fitting coordinates so
            # they can define the correct high-dimensional normal hyperplane.
            "tangent": tangent,
        }
        return result

    def fit_transform(self, X: Array | Sequence[Sequence[float]]) -> dict[str, Array]:
        return self.fit(X).transform(X)

    def plot_network(
        self,
        X: Array | Sequence[Sequence[float]] | None = None,
        ax: Any = None,
        show_projections: bool = False,
        max_projection_lines: int = 150,
        title: str | None = None,
    ) -> Any:
        """Plot a fitted 2D graph and optionally return the Matplotlib axes."""
        if not self._fitted:
            raise RuntimeError("Call fit before plot_network")
        if self.n_features_in_ != 2:
            raise ValueError("plot is only available for two-dimensional data")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on environment.
            raise ImportError("plot requires matplotlib") from exc
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        data = self._original_X_ if X is None else _as_point_cloud(X)
        ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.22, color="tab:blue", label="observations")
        for index, spline in enumerate(self.routes_):
            curve = spline.samples * self.scale_ + self.mean_
            if spline.closed:
                curve = np.vstack([curve, curve[0]])
            ax.plot(curve[:, 0], curve[:, 1], linewidth=2.8, label="spline" if index == 0 else None)
        junctions = np.asarray([self.landmark_graph_.nodes[node] for node in self.junctions_])
        endpoints = np.asarray([self.landmark_graph_.nodes[node] for node in self.endpoints_])
        if len(junctions):
            junctions = junctions * self.scale_ + self.mean_
            ax.scatter(junctions[:, 0], junctions[:, 1], s=70, color="tab:red", zorder=5, label="junction")
        if len(endpoints):
            endpoints = endpoints * self.scale_ + self.mean_
            ax.scatter(endpoints[:, 0], endpoints[:, 1], s=65, marker="s", color="tab:orange", zorder=5, label="endpoint")
        if show_projections:
            result = self.transform(data)
            count = min(max_projection_lines, len(data))
            indices = np.linspace(0, len(data) - 1, count, dtype=int)
            for index in indices:
                ax.plot(
                    [data[index, 0], result.projected[index, 0]],
                    [data[index, 1], result.projected[index, 1]],
                    color="0.35",
                    linewidth=0.35,
                    alpha=0.35,
                )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
        if title:
            ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        return ax


