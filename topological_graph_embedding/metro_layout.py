"""Schematic, metro-map-style layouts for fitted spline graphs."""

from __future__ import annotations

from typing import Any

import numpy as np


class MetroSplineLayout:
    """Lay out a fitted spline graph as a readable 2D schematic.

    The layout deliberately does not project the observations from feature
    space.  It uses graph connectivity, spline arc lengths, and a small set of
    preferred directions to produce a map-like drawing.  Observations are then
    placed at their longitudinal position on the assigned route, with their
    distance from that route shown as a lateral offset.  By default, a
    locally smoothed PCA frame of residual vectors chooses the lateral side;
    ``route_pca`` is available when one stable frame per route is preferred.
    """

    def __init__(
        self,
        model: Any,
        random_state: int = 0,
        layout_iterations: int = 280,
        angle_weight: float = 0.22,
        parallel_spacing: float = 1.00,
        residual_quantile: float = 0.95,
        residual_width: float = 0.02,
        circle_samples: int = 96,
        junction_radius: float = 0.72,
        junction_radius_per_branch: float = 0.10,
        junction_port_spacing: float = 0.42,
        # Keep the fixed-size junction offset visually subordinate to the
        # routes.  This scales schematic distances, not lane spacing or discs.
        layout_scale: float = 2.0,
        residual_frame: str = "local_pca",
        local_pca_bins: int = 8,
        local_pca_window: int = 24,
    ) -> None:
        self.model = model
        self.random_state = int(random_state)
        self.layout_iterations = int(layout_iterations)
        self.angle_weight = float(angle_weight)
        self.parallel_spacing = float(parallel_spacing)
        self.residual_quantile = float(residual_quantile)
        self.residual_width = float(residual_width)
        self.circle_samples = max(32, int(circle_samples))
        self.junction_radius = float(junction_radius)
        self.junction_radius_per_branch = float(junction_radius_per_branch)
        self.junction_port_spacing = float(junction_port_spacing)
        self.layout_scale = float(layout_scale)
        self.residual_frame = str(residual_frame).lower()
        self.local_pca_bins = max(1, int(local_pca_bins))
        self.local_pca_window = max(4, int(local_pca_window))
        if self.junction_radius <= 0.0:
            raise ValueError("junction_radius must be positive")
        if self.junction_radius_per_branch < 0.0:
            raise ValueError("junction_radius_per_branch must be non-negative")
        if self.junction_port_spacing < 0.0:
            raise ValueError("junction_port_spacing must be non-negative")
        if self.layout_scale <= 0.0:
            raise ValueError("layout_scale must be positive")
        if self.residual_frame not in {"local_pca", "route_pca"}:
            raise ValueError("residual_frame must be 'local_pca' or 'route_pca'")

    def fit(self, result: dict[str, np.ndarray] | None = None) -> "MetroSplineLayout":
        """Fit the schematic layout and optionally learn residual-side signs."""
        if not getattr(self.model, "_fitted", False):
            raise RuntimeError("Fit the TopologicalSplineGraph before its metro layout")
        if not 0.0 < self.residual_quantile <= 1.0:
            raise ValueError("residual_quantile must be in (0, 1]")

        self.graph_ = self.model.graph_
        self.node_ids_ = sorted(self.graph_.nodes)
        self.node_index_ = {node: index for index, node in enumerate(self.node_ids_)}
        self.route_lengths_ = np.asarray(
            [self._spline_length(spline) for spline in self.model.splines_],
            dtype=float,
        )
        positive_lengths = self.route_lengths_[self.route_lengths_ > 1e-12]
        self.reference_length_ = float(np.median(positive_lengths)) if len(positive_lengths) else 1.0

        self._initial_node_positions = self._force_layout()
        self.source_route_angles_ = self._source_route_angles()
        self.source_route_direction_bins_ = self._source_route_direction_bins()
        station_ids = [node for node in self.node_ids_ if self.graph_.degree(node) != 2]
        self.station_ids_ = station_ids
        self.station_positions_ = {
            node: self._initial_node_positions[self.node_index_[node]].copy()
            for node in station_ids
        }
        self._optimise_station_positions()
        self._layout_cycles()
        self.route_paths_ = self._build_route_paths()
        self._configure_junction_discs()
        self.route_paths_ = self._clip_routes_at_junctions(self.route_paths_)
        self.residual_scale_ = self._set_residual_scale(result)
        self._fit_residual_axes(result)
        self._fitted = True
        return self

    @staticmethod
    def _spline_length(spline: Any) -> float:
        samples = np.asarray(spline.samples, dtype=float)
        if spline.closed and len(samples) > 1:
            samples = np.vstack([samples, samples[0]])
        return float(np.sum(np.linalg.norm(np.diff(samples, axis=0), axis=1)))

    def _edge_targets(self) -> dict[tuple[int, int], float]:
        weights = np.asarray(list(self.graph_.edges.values()), dtype=float)
        positive = weights[weights > 1e-12]
        reference = float(np.median(positive)) if len(positive) else 1.0
        return {
            edge: self.layout_scale * max(0.8, float(weight) / reference * 3.0)
            for edge, weight in self.graph_.edges.items()
        }

    def _force_layout(self) -> np.ndarray:
        """Create a source-aware initial layout for the graph nodes."""
        source_positions = self._source_layout_positions()
        self.source_layout_positions_ = source_positions
        if source_positions is not None:
            return source_positions
        return self._topology_force_layout()

    def _source_layout_positions(self) -> np.ndarray | None:
        """Project fitted graph coordinates into the schematic's two axes."""
        count = len(self.node_ids_)
        if count == 0:
            return np.empty((0, 2), dtype=float)
        coordinates = np.asarray(
            [self.graph_.nodes[node] for node in self.node_ids_],
            dtype=float,
        )
        if coordinates.ndim != 2 or coordinates.shape[1] == 0:
            return None
        centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
        if coordinates.shape[1] <= 2:
            positions = np.zeros((count, 2), dtype=float)
            positions[:, : coordinates.shape[1]] = centered
        else:
            _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
            if not len(singular_values) or singular_values[0] <= 1e-12:
                return None
            components = components[:2].copy()
            for component in range(len(components)):
                pivot = int(np.argmax(np.abs(components[component])))
                if components[component, pivot] < 0.0:
                    components[component] *= -1.0
            positions = centered @ components.T
            if positions.shape[1] < 2:
                positions = np.pad(positions, ((0, 0), (0, 2 - positions.shape[1])))

        source_lengths = np.asarray(
            [
                np.linalg.norm(self.graph_.nodes[left] - self.graph_.nodes[right])
                for left, right in self.graph_.edges
            ],
            dtype=float,
        )
        source_lengths = source_lengths[source_lengths > 1e-12]
        if not len(source_lengths):
            return None
        targets = self._edge_targets()
        target_scale = float(np.median(list(targets.values()))) if targets else 3.0
        source_scale = float(np.median(source_lengths))
        positions *= target_scale / max(source_scale, 1e-12)
        return positions

    def _topology_force_layout(self) -> np.ndarray:
        """Create a topology-only fallback layout for the graph nodes."""
        count = len(self.node_ids_)
        if count == 0:
            return np.empty((0, 2), dtype=float)
        if count == 1:
            return np.zeros((1, 2), dtype=float)

        targets = self._edge_targets()
        typical = float(np.median(list(targets.values()))) if targets else 3.0
        radius = max(2.0, typical * np.sqrt(count) / 2.0)
        angles = 2.0 * np.pi * np.arange(count) / count
        positions = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        rng = np.random.default_rng(self.random_state)
        positions += 0.05 * typical * rng.normal(size=positions.shape)

        edges = [
            (self.node_index_[left], self.node_index_[right], target)
            for (left, right), target in targets.items()
        ]
        repulsion = 0.9 * typical * typical
        for iteration in range(max(1, self.layout_iterations)):
            delta = positions[:, None, :] - positions[None, :, :]
            distance_squared = np.sum(delta * delta, axis=2)
            distance_squared[np.diag_indices(count)] = np.inf
            distance = np.sqrt(np.maximum(distance_squared, 1e-12))
            displacement = np.sum(
                delta / distance[:, :, None] * (repulsion / distance_squared)[:, :, None],
                axis=1,
            )
            for left, right, target in edges:
                vector = positions[left] - positions[right]
                length = max(float(np.linalg.norm(vector)), 1e-8)
                correction = 0.75 * (length - target) * vector / length
                displacement[left] -= correction
                displacement[right] += correction
            displacement -= 0.08 * positions
            temperature = 0.12 * (1.0 - 0.7 * iteration / max(1, self.layout_iterations))
            step = np.clip(temperature * displacement, -0.45 * typical, 0.45 * typical)
            positions += step
            positions -= np.mean(positions, axis=0)
        return positions

    @staticmethod
    def _direction_bin(angle: float) -> int:
        """Return the nearest one of the eight canonical line directions."""
        return int(np.floor(angle / (np.pi / 4.0) + 0.5)) % 8

    @staticmethod
    def _canonical_directions() -> np.ndarray:
        return np.column_stack([
            np.cos(np.arange(8) * np.pi / 4.0),
            np.sin(np.arange(8) * np.pi / 4.0),
        ])

    def _source_route_angles(self) -> dict[int, float]:
        """Return oriented source-space angles for every open route."""
        angles: dict[int, float] = {}
        for route, left, right, _ in self._open_routes():
            vector = np.asarray(self.graph_.nodes[right] - self.graph_.nodes[left], dtype=float)
            if vector.ndim != 1 or len(vector) == 0:
                continue
            if len(vector) > 2:
                source = self.source_layout_positions_
                if source is None:
                    continue
                left_index = self.node_index_[left]
                right_index = self.node_index_[right]
                vector = source[right_index] - source[left_index]
            else:
                vector = vector[:2]
            if np.linalg.norm(vector) <= 1e-12:
                continue
            angle = np.arctan2(vector[1], vector[0])
            angles[route] = self._direction_bin(angle) * np.pi / 4.0
        return angles

    def _source_route_direction_bins(self) -> dict[tuple[int, int], int]:
        """Return source-derived outward direction bins for route incidences."""
        bins: dict[tuple[int, int], int] = {}
        for route, angle in self.source_route_angles_.items():
            base = self._direction_bin(angle)
            bins[route, 0] = base
            bins[route, 1] = (base + 4) % 8
        return bins

    def _open_routes(self) -> list[tuple[int, int, int, float]]:
        routes = []
        for index, chain in enumerate(self.model.chains_):
            if chain["closed"]:
                continue
            nodes = chain["nodes"]
            if len(nodes) < 2 or nodes[0] == nodes[-1]:
                continue
            routes.append((index, int(nodes[0]), int(nodes[-1]), float(self.route_lengths_[index])))
        return routes

    def _optimise_station_positions(self) -> None:
        if len(self.station_ids_) < 2:
            return
        routes = self._open_routes()
        if not routes:
            return
        station_index = {node: index for index, node in enumerate(self.station_ids_)}
        initial = np.asarray([self.station_positions_[node] for node in self.station_ids_])
        scale = max(self.reference_length_, 1e-8)
        targets = [
            self.layout_scale * max(0.8, length / scale * 3.0)
            for _, _, _, length in routes
        ]
        scale *= self.layout_scale
        route_ids = []
        route_indices = [
            (station_index[left], station_index[right], target)
            for (_, left, right, _), target in zip(routes, targets)
            if left in station_index and right in station_index
        ]
        for route, left, right, _ in routes:
            if left in station_index and right in station_index:
                route_ids.append(route)
        if not route_indices:
            return
        def objective(flat: np.ndarray) -> float:
            positions = flat.reshape(-1, 2)
            value = 0.0
            for route_id, (left, right, target) in zip(route_ids, route_indices):
                vector = positions[right] - positions[left]
                length = max(float(np.linalg.norm(vector)), 1e-8)
                value += ((length - target) / target) ** 2
                angle = np.arctan2(vector[1], vector[0])
                value += self.angle_weight * 0.5 * (1.0 - np.cos(8.0 * angle))
                source_angle = self.source_route_angles_.get(route_id)
                if source_angle is not None:
                    value += 0.60 * 0.5 * (1.0 - np.cos(angle - source_angle))
            for left in range(len(positions)):
                for right in range(left):
                    distance = float(np.linalg.norm(positions[left] - positions[right]))
                    minimum = 0.45 * scale
                    if distance < minimum:
                        value += 0.25 * ((minimum - distance) / minimum) ** 2
            value += 0.70 * float(np.sum((positions - initial) ** 2)) / (scale * scale)
            return value

        try:
            from scipy.optimize import minimize

            fitted = minimize(
                objective,
                initial.ravel(),
                method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-8},
            )
            optimized = fitted.x.reshape(-1, 2)
            if np.isfinite(fitted.fun):
                self.station_positions_ = {
                    node: optimized[index].copy()
                    for index, node in enumerate(self.station_ids_)
                }
        except Exception:
            # The force-directed initialization is still a useful schematic if
            # scipy.optimize is unavailable or an optimization fails.
            return

    def _cycle_basis(self) -> list[list[tuple[int, int, int]]]:
        """Return a simple cycle basis of the station-level route graph."""
        routes = self._open_routes()
        adjacency: dict[int, list[tuple[int, int]]] = {node: [] for node in self.station_ids_}
        for route_index, left, right, _ in routes:
            adjacency[left].append((right, route_index))
            adjacency[right].append((left, route_index))

        parent: dict[int, int | None] = {}
        parent_edge: dict[int, int] = {}
        depth: dict[int, int] = {}
        visited: set[int] = set()
        cycles: list[list[tuple[int, int, int]]] = []

        def visit(node: int, level: int) -> None:
            visited.add(node)
            depth[node] = level
            for neighbor, route_index in adjacency[node]:
                if neighbor not in visited:
                    parent[neighbor] = node
                    parent_edge[neighbor] = route_index
                    visit(neighbor, level + 1)
                    continue
                if parent_edge.get(node) == route_index or depth.get(neighbor, level) >= level:
                    continue
                descendant_path = [node]
                while descendant_path[-1] != neighbor:
                    ancestor = parent.get(descendant_path[-1])
                    if ancestor is None:
                        descendant_path = []
                        break
                    descendant_path.append(ancestor)
                if len(descendant_path) < 2:
                    continue
                cycle_nodes = [neighbor] + list(reversed(descendant_path[:-1]))
                cycle: list[tuple[int, int, int]] = []
                for left, right in zip(cycle_nodes[:-1], cycle_nodes[1:]):
                    child = right if parent.get(right) == left else left
                    cycle.append((parent_edge[child], left, right))
                cycle.append((route_index, node, neighbor))
                if len({route for route, _, _ in cycle}) >= 2:
                    cycles.append(cycle)

        for root in self.station_ids_:
            if root not in visited:
                parent[root] = None
                visit(root, 0)
        return cycles

    def _layout_cycles(self) -> None:
        """Give every detected cycle a smooth circular metro outline."""
        self.cycle_route_paths_: dict[int, np.ndarray] = {}
        route_records = {
            route_index: (left, right, length)
            for route_index, left, right, length in self._open_routes()
        }
        for loop_index, cycle in enumerate(self._cycle_basis()):
            route_ids = [route for route, _, _ in cycle]
            if any(route in self.cycle_route_paths_ for route in route_ids):
                continue
            cycle_nodes = [left for _, left, _ in cycle]
            if len(cycle_nodes) < 2:
                continue
            sample_count = self.circle_samples
            center = np.mean([self.station_positions_[node] for node in cycle_nodes], axis=0)
            perimeter = self.layout_scale * max(
                sum(route_records[route][2] for route in route_ids), 3.0
            )
            radius = perimeter / (2.0 * np.pi)
            rotation = 2.0 * np.pi * (loop_index % 8) / 8.0
            theta = rotation + 2.0 * np.pi * np.arange(sample_count) / sample_count
            circle = center + radius * np.column_stack([np.cos(theta), np.sin(theta)])
            node_indices = np.rint(
                np.arange(len(cycle_nodes)) * sample_count / len(cycle_nodes)
            ).astype(int) % sample_count
            node_positions = dict(zip(cycle_nodes, node_indices))
            for node, vertex in node_positions.items():
                self.station_positions_[node] = circle[vertex]

            for route_index, left, right in cycle:
                start = node_positions[left]
                end = node_positions[right]
                step_count = (end - start) % sample_count
                indices = [(start + step) % sample_count for step in range(step_count + 1)]
                path = circle[indices]
                route_left, route_right, _ = route_records[route_index]
                if (route_left, route_right) != (left, right):
                    path = path[::-1]
                self.cycle_route_paths_[route_index] = path

    def _build_route_paths(self) -> list[np.ndarray]:
        paths = []
        loop_index = 0
        for index, chain in enumerate(self.model.chains_):
            nodes = [int(node) for node in chain["nodes"]]
            if not chain["closed"]:
                if index in self.cycle_route_paths_:
                    cycle_path = np.asarray(self.cycle_route_paths_[index], dtype=float).copy()
                    # A junction can participate in more than one cycle.  The
                    # cycle pass may therefore have moved its temporary arc
                    # endpoint more than once.  Re-anchor every arc to the
                    # final station positions before clipping it at the
                    # station disc; otherwise a route can jump through a
                    # junction circle or start on a stale cycle position.
                    if len(cycle_path) >= 2:
                        cycle_path[0] = self.station_positions_.get(nodes[0], cycle_path[0])
                        cycle_path[-1] = self.station_positions_.get(nodes[-1], cycle_path[-1])
                    paths.append(cycle_path)
                    continue
                if len(nodes) < 2:
                    paths.append(np.zeros((2, 2), dtype=float))
                else:
                    start_preferred = None
                    if self.graph_.degree(nodes[0]) >= 3:
                        start_preferred = self.source_route_direction_bins_.get((index, 0))
                    end_outward = None
                    if self.graph_.degree(nodes[-1]) >= 3:
                        end_outward = self.source_route_direction_bins_.get((index, 1))
                    end_preferred = None if end_outward is None else (end_outward + 4) % 8
                    paths.append(self._octilinear_path(
                        self.station_positions_.get(
                            nodes[0], self._initial_node_positions[self.node_index_[nodes[0]]]
                        ),
                        self.station_positions_.get(
                            nodes[-1], self._initial_node_positions[self.node_index_[nodes[-1]]]
                        ),
                        preferred_direction=start_preferred,
                        terminal_direction=end_preferred,
                    ))
                continue

            unique_nodes = nodes[:-1] if len(nodes) > 1 and nodes[0] == nodes[-1] else nodes
            sample_count = self.circle_samples
            if unique_nodes:
                available = np.asarray([
                    self._initial_node_positions[self.node_index_[node]] for node in unique_nodes
                ])
                center = np.mean(available, axis=0)
            else:
                center = np.zeros(2, dtype=float)
            perimeter = self.layout_scale * max(self.route_lengths_[index], 3.0)
            radius = perimeter / (2.0 * np.pi)

            rotation = 2.0 * np.pi * (loop_index % 8) / 8.0
            theta = rotation + 2.0 * np.pi * np.arange(sample_count) / sample_count
            node_indices = np.rint(
                np.arange(len(unique_nodes)) * sample_count / len(unique_nodes)
            ).astype(int) % sample_count
            attached_junctions = list(dict.fromkeys(
                node for node in unique_nodes if self.graph_.degree(node) >= 3
            ))
            if attached_junctions:
                # A junction belonging to a cycle is an intersection on the
                # cycle, not a separate station beside it.  Keep the first
                # already-placed intersection fixed and move the cycle center
                # so the corresponding cycle vertex passes through it.  Other
                # attached junctions are then moved onto their cycle vertices;
                # the connecting routes are built from these final positions.
                anchor = attached_junctions[0]
                anchor_vertex = node_indices[unique_nodes.index(anchor)]
                anchor_direction = np.asarray([
                    np.cos(theta[anchor_vertex]), np.sin(theta[anchor_vertex]),
                ])
                center = self.station_positions_[anchor] - radius * anchor_direction
                theta = rotation + 2.0 * np.pi * np.arange(sample_count) / sample_count
            paths.append(center + radius * np.column_stack([np.cos(theta), np.sin(theta)]))
            circle = paths[-1]
            for node in attached_junctions:
                vertex = node_indices[unique_nodes.index(node)]
                self.station_positions_[node] = circle[vertex].copy()
            loop_index += 1
        paths = self._separate_attached_cycles(paths)
        paths = self._rebuild_open_routes(paths)
        paths = self._separate_parallel_routes(paths)
        return self._extend_junction_stubs(paths)

    def _separate_attached_cycles(self, paths: list[np.ndarray]) -> list[np.ndarray]:
        """Keep distinct cycle junction discs separated without breaking loops."""
        placements: list[tuple[int, list[int]]] = []
        for route, chain in enumerate(self.model.chains_):
            if not chain["closed"] or len(paths[route]) < 2:
                continue
            unique_nodes = list(chain["nodes"])
            if len(unique_nodes) > 1 and unique_nodes[0] == unique_nodes[-1]:
                unique_nodes = unique_nodes[:-1]
            attached = list(dict.fromkeys(
                node for node in unique_nodes if self.graph_.degree(node) >= 3
            ))
            if attached:
                placements.append((route, attached))

        for left_index, (left_route, left_nodes) in enumerate(placements):
            for right_route, right_nodes in placements[left_index + 1 :]:
                if set(left_nodes) & set(right_nodes):
                    continue
                left_node = left_nodes[0]
                right_node = right_nodes[0]
                left = np.asarray(self.station_positions_[left_node], dtype=float)
                right = np.asarray(self.station_positions_[right_node], dtype=float)
                vector = right - left
                distance = float(np.linalg.norm(vector))
                left_radius = self.junction_radius + self.junction_radius_per_branch * max(
                    0, self.graph_.degree(left_node) - 3
                )
                right_radius = self.junction_radius + self.junction_radius_per_branch * max(
                    0, self.graph_.degree(right_node) - 3
                )
                required = left_radius + right_radius + 0.20
                if distance >= required:
                    continue
                if distance <= 1e-10:
                    vector = np.asarray([1.0, 0.0])
                    distance = 1.0
                shift = 0.5 * (required - distance) * vector / distance
                for node in left_nodes:
                    self.station_positions_[node] -= shift
                paths[left_route] = np.asarray(paths[left_route]) - shift
                for node in right_nodes:
                    self.station_positions_[node] += shift
                paths[right_route] = np.asarray(paths[right_route]) + shift
        return paths

    def _rebuild_open_routes(self, paths: list[np.ndarray]) -> list[np.ndarray]:
        """Rebuild connectors after cycle-attached stations have moved."""
        for index, chain in enumerate(self.model.chains_):
            if chain["closed"] or index in self.cycle_route_paths_:
                continue
            nodes = [int(node) for node in chain["nodes"]]
            if len(nodes) < 2:
                continue
            start_node, end_node = nodes[0], nodes[-1]
            start = self.station_positions_.get(
                start_node, self._initial_node_positions[self.node_index_[start_node]]
            )
            end = self.station_positions_.get(
                end_node, self._initial_node_positions[self.node_index_[end_node]]
            )
            start_preferred = None
            if self.graph_.degree(start_node) >= 3:
                start_preferred = self.source_route_direction_bins_.get((index, 0))
            end_outward = None
            if self.graph_.degree(end_node) >= 3:
                end_outward = self.source_route_direction_bins_.get((index, 1))
            end_preferred = None if end_outward is None else (end_outward + 4) % 8
            paths[index] = self._octilinear_path(
                start,
                end,
                preferred_direction=start_preferred,
                terminal_direction=end_preferred,
            )
        return paths

    def _junction_incidences(self) -> dict[int, list[tuple[int, int]]]:
        """Return ``(route, side)`` incidences for every junction station."""
        incidences: dict[int, list[tuple[int, int]]] = {
            node: [] for node in self.graph_.nodes if self.graph_.degree(node) >= 3
        }
        for route, chain in enumerate(self.model.chains_):
            if chain["closed"] or len(chain["nodes"]) < 2:
                continue
            left, right = int(chain["nodes"][0]), int(chain["nodes"][-1])
            if left in incidences:
                incidences[left].append((route, 0))
            if right in incidences:
                incidences[right].append((route, 1))
        return incidences

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _allocate_port_angles(self, angles: np.ndarray, radius: float) -> np.ndarray:
        """Separate crowded incident directions while preserving their order."""
        count = len(angles)
        if count <= 1 or self.junction_port_spacing <= 0.0:
            return np.mod(angles, 2.0 * np.pi)

        # A physical spacing is converted to an angular gap.  The cap keeps
        # high-degree junctions feasible even when their ideal ports collide.
        minimum_gap = min(
            self.junction_port_spacing / max(radius, 1e-8),
            0.86 * 2.0 * np.pi / count,
        )
        normalized = np.mod(angles, 2.0 * np.pi)
        order = np.argsort(normalized)
        sorted_angles = normalized[order]
        gaps = np.diff(np.r_[sorted_angles, sorted_angles[0] + 2.0 * np.pi])
        cut = int(np.argmax(gaps))
        cyclic_order = np.r_[order[cut + 1:], order[:cut + 1]]
        base = np.asarray([normalized[index] for index in cyclic_order], dtype=float)
        for index in range(1, count):
            while base[index] < base[index - 1] + minimum_gap:
                base[index] += minimum_gap
        if base[-1] - base[0] > 2.0 * np.pi - minimum_gap:
            center = float(np.mean(base))
            base = center + (np.arange(count) - 0.5 * (count - 1)) * minimum_gap
        assigned = np.empty(count, dtype=float)
        assigned[cyclic_order] = np.mod(base, 2.0 * np.pi)
        return assigned

    def _configure_junction_discs(self) -> None:
        """Allocate adaptive station discs and source-aware lane ports."""
        incidences = self._junction_incidences()
        self.junction_radii_ = {}
        self.junction_ports_ = {}
        self.route_junction_ports_ = {}
        directions = self._canonical_directions()
        for node, members in incidences.items():
            degree = self.graph_.degree(node)
            radius = self.junction_radius + self.junction_radius_per_branch * max(0, degree - 3)
            offsets: list[float] = []
            for route, side in members:
                path = np.asarray(self.route_paths_[route], dtype=float)
                oriented = path if side == 0 else path[::-1]
                direction_bin = self.source_route_direction_bins_.get((route, side))
                if direction_bin is None:
                    direction = oriented[1] - oriented[0]
                    direction_bin = self._direction_bin(np.arctan2(direction[1], direction[0]))
                direction = directions[int(direction_bin) % 8]
                normal = np.asarray([-direction[1], direction[0]])
                center = np.asarray(self.station_positions_[node], dtype=float)
                offsets.append(float((oriented[0] - center) @ normal))
            radius = max(
                radius,
                max((abs(offset) for offset in offsets), default=0.0)
                / np.sin(np.pi / 8.0 - 1e-3)
                + 0.12,
                getattr(self, "_required_junction_radii_", {}).get(node, 0.0),
            )
            self.junction_radii_[node] = radius
            center = np.asarray(self.station_positions_[node], dtype=float)
            for route, side in members:
                path = np.asarray(self.route_paths_[route], dtype=float)
                oriented = path if side == 0 else path[::-1]
                direction_bin = self.source_route_direction_bins_.get((route, side))
                if direction_bin is None:
                    direction = oriented[1] - oriented[0]
                    direction_bin = self._direction_bin(np.arctan2(direction[1], direction[0]))
                direction = directions[int(direction_bin) % 8]
                normal = np.asarray([-direction[1], direction[0]])
                lateral_offset = float((oriented[0] - center) @ normal)
                along = np.sqrt(max(radius * radius - lateral_offset * lateral_offset, 0.0))
                port = center + lateral_offset * normal + along * direction
                self.junction_ports_[node, route, side] = port
                self.route_junction_ports_[route, side] = (node, port)

    @staticmethod
    def _circle_intersection(
        start: np.ndarray,
        end: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> tuple[float, np.ndarray] | None:
        vector = end - start
        offset = start - center
        coefficients = (
            float(vector @ vector),
            float(2.0 * (offset @ vector)),
            float(offset @ offset - radius * radius),
        )
        if coefficients[0] <= 1e-14:
            return None
        roots = np.roots(coefficients)
        valid = sorted(
            float(root.real)
            for root in roots
            if abs(float(root.imag)) <= 1e-8 and -1e-9 <= float(root.real) <= 1.0 + 1e-9
        )
        if not valid:
            return None
        fraction = float(np.clip(valid[0], 0.0, 1.0))
        return fraction, start + fraction * vector

    def _clip_route_endpoint(
        self,
        path: np.ndarray,
        node: int,
        port: np.ndarray,
        side: int,
    ) -> np.ndarray:
        """Clip one route endpoint at the junction circumference.

        The un-clipped route contains an interior station point followed by a
        deliberately extended, outward canonical stub.  Keep only the part
        after the first exit from the disc; replacing the interior point while
        retaining the old prefix would make the route run back through the
        junction.
        """
        oriented = np.asarray(path, dtype=float)
        reversed_path = side == 1
        if reversed_path:
            oriented = oriented[::-1]

        center = np.asarray(self.station_positions_[node], dtype=float)
        radius = float(self.junction_radii_[node])
        distances = np.linalg.norm(oriented - center, axis=1)
        crossing_index = None
        crossing = np.asarray(port, dtype=float)
        for index in range(len(oriented) - 1):
            start_distance = float(distances[index])
            end_distance = float(distances[index + 1])
            if start_distance <= radius + 1e-8 and end_distance >= radius - 1e-8:
                intersection = self._circle_intersection(
                    oriented[index], oriented[index + 1], center, radius,
                )
                if intersection is not None:
                    crossing = intersection[1]
                crossing_index = index
                break

        if crossing_index is None:
            # This is only a defensive fallback for a degenerate route.  The
            # endpoint-clearance pass normally guarantees an outward crossing.
            crossing_index = 0
        # The first stub is constructed from the same canonical direction and
        # lateral offset used to calculate ``port``.  Numerical differences in
        # the line-circle intersection are therefore negligible; use the
        # supplied port so the public port and clipped path agree exactly.
        if crossing_index == 0 and np.linalg.norm(crossing - port) <= 1e-6:
            crossing = np.asarray(port, dtype=float)
        clipped = np.asarray([crossing, *oriented[crossing_index + 1 :]], dtype=float)
        cleaned = [clipped[0]]
        for point in clipped[1:]:
            if np.linalg.norm(point - cleaned[-1]) > 1e-10:
                cleaned.append(point)
        clipped = np.asarray(cleaned, dtype=float)
        return clipped[::-1] if reversed_path else clipped

    def _clip_routes_at_junctions(self, paths: list[np.ndarray]) -> list[np.ndarray]:
        """Make every incident route begin at a junction circumference port."""
        clipped_paths = [np.asarray(path, dtype=float).copy() for path in paths]
        for (route, side), (node, port) in self.route_junction_ports_.items():
            clipped_paths[route] = self._clip_route_endpoint(
                clipped_paths[route], node, port, side
            )
        return clipped_paths

    @classmethod
    def _octilinear_path(
        cls,
        start: np.ndarray,
        end: np.ndarray,
        preferred_direction: int | None = None,
        terminal_direction: int | None = None,
    ) -> np.ndarray:
        """Connect two stations using only canonical octilinear segments."""
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        displacement = end - start
        if np.linalg.norm(displacement) <= 1e-10:
            return np.asarray([start, end])
        directions = cls._canonical_directions()

        if preferred_direction is not None or terminal_direction is not None:
            stub_length = min(0.8, max(0.18, 0.25 * float(np.linalg.norm(displacement))))
            connector_start = start.copy()
            connector_end = end.copy()
            prefix = [start]
            suffix = [end]
            if preferred_direction is not None:
                connector_start = start + stub_length * directions[int(preferred_direction) % 8]
                prefix.append(connector_start)
            if terminal_direction is not None:
                connector_end = end - stub_length * directions[int(terminal_direction) % 8]
                suffix.insert(0, connector_end)
            connector = cls._octilinear_path(connector_start, connector_end)
            result = prefix + [point for point in connector[1:-1]] + suffix
            cleaned = [result[0]]
            for point in result[1:]:
                if np.linalg.norm(point - cleaned[-1]) > 1e-10:
                    cleaned.append(point)
            return np.asarray(cleaned, dtype=float)

        for index in range(8):
            first = directions[index]
            second = directions[(index + 1) % 8]
            coefficients = np.linalg.solve(np.column_stack([first, second]), displacement)
            if np.all(coefficients >= -1e-9):
                if coefficients[0] <= 1e-9 or coefficients[1] <= 1e-9:
                    return np.asarray([start, end])
                elbow = start + coefficients[0] * first
                return np.asarray([start, elbow, end])
        # A zero-length numerical fallback is the only case where a raw
        # segment is unavoidable; non-degenerate routes are covered by the
        # eight adjacent direction cones above.
        return np.asarray([start, end])

    def _separate_parallel_routes(self, paths: list[np.ndarray]) -> list[np.ndarray]:
        """Assign source-ordered, centered offsets to parallel route groups.

        Opposite rays are separate local lane groups.  Thus one route leaving
        to the right stays on the centerline while two routes leaving to the
        left are centered around that same centerline.  This is the useful
        junction interpretation of parallel lanes: routes on the same side
        are spaced, while a through route is not displaced merely because
        another route leaves in the opposite direction.
        """
        base_paths = [np.asarray(path, dtype=float).copy() for path in paths]
        endpoint_groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for index, chain in enumerate(self.model.chains_):
            if chain["closed"] or index in self.cycle_route_paths_:
                continue
            if len(paths[index]) < 2 or len(chain["nodes"]) < 2:
                continue
            left, right = int(chain["nodes"][0]), int(chain["nodes"][-1])
            for side, (station, other) in enumerate(((left, right), (right, left))):
                direction_bin = self.source_route_direction_bins_.get((index, side))
                if direction_bin is None:
                    oriented = paths[index] if side == 0 else paths[index][::-1]
                    direction = oriented[1] - oriented[0]
                    direction_bin = self._direction_bin(np.arctan2(direction[1], direction[0]))
                # Keep opposite rays separate.  A single route on a ray then
                # receives the zero offset, even if parallel routes leave on
                # the opposite side of the junction.
                direction_family = int(direction_bin) % 8
                endpoint_groups.setdefault((station, direction_family), []).append((index, side))

        endpoint_offsets: dict[tuple[int, int], np.ndarray] = {}
        self._required_junction_radii_: dict[int, float] = {}
        for group_key, members in endpoint_groups.items():
            if len(members) < 2:
                continue
            line_angle = group_key[1] * (np.pi / 4.0)
            normal = np.asarray([-np.sin(line_angle), np.cos(line_angle)])
            ordered_members = sorted(
                members,
                key=lambda member: float(
                    self._source_lateral_position(member[0], member[1], normal, base_paths)
                ),
            )
            for lane, (route_index, side) in enumerate(ordered_members):
                centered_lane = lane - 0.5 * (len(members) - 1)
                endpoint_offsets[(route_index, side)] = (
                    centered_lane * self.parallel_spacing * normal
                )
            self._required_junction_radii_[group_key[0]] = max(
                self._required_junction_radii_.get(group_key[0], 0.0),
                max(
                    float(np.linalg.norm(endpoint_offsets[(route_index, side)]))
                    for route_index, side in members
                ) / np.sin(np.pi / 8.0 - 1e-3) + 0.12,
            )

        # Translate each route once. If both ends are junctions, the first
        # available lane assignment becomes the route-wide offset, preserving
        # a continuous lane instead of independently shifting both ends.
        self.route_lane_offsets_: dict[int, np.ndarray] = {}
        for route in range(len(paths)):
            candidates = [
                endpoint_offsets.get((route, 0)),
                endpoint_offsets.get((route, 1)),
            ]
            chosen = next((offset for offset in candidates if offset is not None), None)
            if chosen is not None:
                self.route_lane_offsets_[route] = np.asarray(chosen, dtype=float)

        # A terminal marker belongs to its lane. Apply each shift once so
        # marker positions do not depend on route iteration order.
        endpoint_shifts: dict[int, np.ndarray] = {}
        for route_index, offset in self.route_lane_offsets_.items():
            chain = self.model.chains_[route_index]
            left, right = int(chain["nodes"][0]), int(chain["nodes"][-1])
            if self.graph_.degree(left) == 1:
                endpoint_shifts[left] = endpoint_shifts.get(left, np.zeros(2)) + offset
            if self.graph_.degree(right) == 1:
                endpoint_shifts[right] = endpoint_shifts.get(right, np.zeros(2)) + offset

        for station, shift in endpoint_shifts.items():
            self.station_positions_[station] = self.station_positions_[station] + shift

        placed_paths = [path.copy() for path in base_paths]
        for index, lane_offset in self.route_lane_offsets_.items():
            if np.linalg.norm(lane_offset) <= 1e-10:
                continue
            translated = base_paths[index] + lane_offset
            cleaned = [translated[0]]
            for point in translated[1:]:
                if np.linalg.norm(point - cleaned[-1]) > 1e-10:
                    cleaned.append(point)
            candidate = np.asarray(cleaned)
            placed_paths[index] = candidate
        self._ensure_endpoint_clearance(placed_paths)
        paths[:] = placed_paths
        return paths

    def _extend_junction_stubs(self, paths: list[np.ndarray]) -> list[np.ndarray]:
        """Carry every incident route outside its junction disc before trim.

        A station point is retained internally so lane offsets can be measured
        from the junction center.  The first and last visible segments are
        then extended beyond the disc along their source-derived octilinear
        directions.  This both makes the circumference a real empty offset
        and gives clipping an unambiguous outward crossing.
        """
        directions = self._canonical_directions()
        incidence = self._junction_incidences()
        by_route: dict[int, dict[int, tuple[int, int]]] = {}
        for node, members in incidence.items():
            for route, side in members:
                by_route.setdefault(route, {})[side] = (node, side)

        for route, sides in by_route.items():
            chain = self.model.chains_[route]
            if chain["closed"] or route in self.cycle_route_paths_ or len(paths[route]) < 2:
                continue
            oriented_path = np.asarray(paths[route], dtype=float)
            left, right = int(chain["nodes"][0]), int(chain["nodes"][-1])
            start = oriented_path[0].copy()
            end = oriented_path[-1].copy()

            start_stub = None
            start_disc = None
            if 0 in sides:
                direction_bin = self.source_route_direction_bins_.get((route, 0))
                if direction_bin is not None:
                    center = np.asarray(self.station_positions_[left], dtype=float)
                    direction = directions[int(direction_bin) % 8]
                    normal = np.asarray([-direction[1], direction[0]])
                    lateral = float((start - center) @ normal)
                    radius = max(
                        self.junction_radius
                        + self.junction_radius_per_branch * max(0, self.graph_.degree(left) - 3),
                        self._required_junction_radii_.get(left, 0.0),
                    )
                    travel = np.sqrt(max(radius * radius - lateral * lateral, 0.0)) + 0.20
                    start_stub = center + lateral * normal + travel * direction
                    start_disc = (center, radius)

            end_stub = None
            end_disc = None
            if 1 in sides:
                outward_bin = self.source_route_direction_bins_.get((route, 1))
                if outward_bin is not None:
                    center = np.asarray(self.station_positions_[right], dtype=float)
                    outward = directions[int(outward_bin) % 8]
                    normal = np.asarray([-outward[1], outward[0]])
                    lateral = float((end - center) @ normal)
                    radius = max(
                        self.junction_radius
                        + self.junction_radius_per_branch * max(0, self.graph_.degree(right) - 3),
                        self._required_junction_radii_.get(right, 0.0),
                    )
                    travel = np.sqrt(max(radius * radius - lateral * lateral, 0.0)) + 0.20
                    end_stub = center + lateral * normal + travel * outward
                    end_disc = (center, radius)

            if start_stub is None and end_stub is None:
                continue
            connector_start = start_stub if start_stub is not None else start
            connector_end = end_stub if end_stub is not None else end
            if start_disc is not None and end_disc is not None:
                connector = self._junction_clear_connector(
                    connector_start,
                    connector_end,
                    [start_disc, end_disc],
                )
            else:
                connector = self._octilinear_path(connector_start, connector_end)
            rebuilt = []
            if start_stub is not None:
                rebuilt.extend([start, start_stub])
            else:
                rebuilt.append(start)
            rebuilt.extend(connector[1:-1])
            if end_stub is not None:
                rebuilt.extend([end_stub, end])
            else:
                rebuilt.append(end)
            cleaned = [rebuilt[0]]
            for point in rebuilt[1:]:
                if np.linalg.norm(point - cleaned[-1]) > 1e-10:
                    cleaned.append(point)
            paths[route] = np.asarray(cleaned, dtype=float)
        return paths

    @staticmethod
    def _polyline_clears_discs(
        path: np.ndarray,
        discs: list[tuple[np.ndarray, float]],
    ) -> bool:
        """Return whether every segment of ``path`` stays outside discs."""
        for start, end in zip(path[:-1], path[1:]):
            vector = end - start
            denominator = max(float(vector @ vector), 1e-12)
            for center, radius in discs:
                fraction = np.clip(float((center - start) @ vector) / denominator, 0.0, 1.0)
                closest = start + fraction * vector
                if np.linalg.norm(closest - center) < radius - 1e-8:
                    return False
        return True

    @classmethod
    def _junction_clear_connector(
        cls,
        start: np.ndarray,
        end: np.ndarray,
        discs: list[tuple[np.ndarray, float]],
    ) -> np.ndarray:
        """Connect two outward stubs without passing through either disc."""
        direct = cls._octilinear_path(start, end)
        if cls._polyline_clears_discs(direct, discs):
            return direct

        centers = np.asarray([center for center, _ in discs], dtype=float)
        radii = [radius for _, radius in discs]
        margin = 0.35
        right = float(np.max(centers[:, 0] + radii) + margin)
        left = float(np.min(centers[:, 0] - radii) - margin)
        top = float(np.max(centers[:, 1] + radii) + margin)
        bottom = float(np.min(centers[:, 1] - radii) - margin)
        vertical_target = top if end[1] >= centers[-1, 1] else bottom
        for x_far in (right, left):
            candidate = np.asarray([
                start,
                [x_far, start[1]],
                [x_far, vertical_target],
                [end[0], vertical_target],
                end,
            ], dtype=float)
            if cls._polyline_clears_discs(candidate, discs):
                return candidate

        # The fallback remains octilinear and is only reachable for unusual
        # overlapping disc configurations where no rectangular detour was
        # found within the finite schematic box.
        return direct

    def _source_lateral_position(
        self,
        route: int,
        side: int,
        normal: np.ndarray,
        base_paths: list[np.ndarray],
    ) -> float:
        """Return a stable source-space ordering value for one incidence."""
        chain = self.model.chains_[route]
        left, right = int(chain["nodes"][0]), int(chain["nodes"][-1])
        station = left if side == 0 else right
        other = right if side == 0 else left
        source = self.source_layout_positions_
        if source is not None:
            vector = source[self.node_index_[other]] - source[self.node_index_[station]]
        else:
            oriented = base_paths[route] if side == 0 else base_paths[route][::-1]
            vector = oriented[-1] - oriented[0]
        return float(vector @ normal)

    def _ensure_endpoint_clearance(self, paths: list[np.ndarray]) -> None:
        """Move terminal markers beyond the planned junction discs."""
        directions = self._canonical_directions()
        for route, chain in enumerate(self.model.chains_):
            if chain["closed"] or len(chain["nodes"]) < 2:
                continue
            left, right = int(chain["nodes"][0]), int(chain["nodes"][-1])
            for side, (junction, endpoint) in enumerate(((left, right), (right, left))):
                if self.graph_.degree(junction) < 3 or self.graph_.degree(endpoint) != 1:
                    continue
                if len(paths[route]) < 2:
                    continue
                oriented = paths[route] if side == 0 else paths[route][::-1]
                center = np.asarray(self.station_positions_[junction], dtype=float)
                direction_bin = self.source_route_direction_bins_.get((route, side))
                if direction_bin is None:
                    direction = oriented[1] - oriented[0]
                    direction_bin = self._direction_bin(np.arctan2(direction[1], direction[0]))
                current = np.asarray(oriented[-1], dtype=float)
                radius = max(
                    self.junction_radius
                    + self.junction_radius_per_branch * max(0, self.graph_.degree(junction) - 3),
                    self._required_junction_radii_.get(junction, 0.0),
                )
                route_target = self.layout_scale * max(
                    0.8,
                    float(self.route_lengths_[route]) / max(self.reference_length_, 1e-8) * 3.0,
                )
                direction = directions[int(direction_bin) % 8]
                normal = np.asarray([-direction[1], direction[0]])
                # Use the assigned lane's lateral position, not an arbitrary
                # residual from the source embedding.  A singleton lane is
                # therefore exactly on the circle centerline; only parallel
                # lane assignments displace it.
                lane_offset = self.route_lane_offsets_.get(route)
                lateral = 0.0 if lane_offset is None else float(lane_offset @ normal)
                required = radius + 0.12 + route_target
                outer_radius = max(required, abs(lateral) + 0.20)
                along = np.sqrt(max(outer_radius * outer_radius - lateral * lateral, 0.0))
                new_endpoint = center + lateral * normal + along * direction
                oriented[-1] = new_endpoint
                paths[route] = oriented if side == 0 else oriented[::-1]
                self.station_positions_[endpoint] = new_endpoint.copy()

    def _set_residual_scale(self, result: dict[str, np.ndarray] | None) -> float:
        if result is None:
            return self.residual_width
        residual = self._standardized_residual(result)
        finite = residual[np.isfinite(residual)]
        if not len(finite) or np.max(finite) <= 1e-12:
            return self.residual_width
        reference = max(float(np.quantile(finite, self.residual_quantile)), 1e-8)
        return self.residual_width * self.reference_length_ / reference

    def _standardized_residual(self, result: dict[str, np.ndarray]) -> np.ndarray:
        if "residual_vector" in result:
            vector = np.asarray(result["residual_vector"], dtype=float)
            return np.linalg.norm(vector / self.model.scale_, axis=1)
        return np.asarray(result["residual_norm"], dtype=float)

    @staticmethod
    def _principal_components(
        vectors: np.ndarray,
        count: int = 2,
        fallback: np.ndarray | None = None,
        require_separation: bool = False,
    ) -> np.ndarray | None:
        """Return deterministic, consistently oriented residual PCA axes."""
        count = max(1, int(count))
        vectors = np.asarray(vectors, dtype=float)
        fallback_array = None if fallback is None else np.asarray(fallback, dtype=float)
        if fallback_array is not None:
            if fallback_array.ndim == 1:
                fallback_array = fallback_array[None, :]
            if fallback_array.ndim != 2:
                fallback_array = None
        if vectors.ndim != 2 or len(vectors) < 2:
            if fallback_array is None:
                return None
            axes = np.zeros((count, fallback_array.shape[1]), dtype=float)
            rows = min(count, len(fallback_array))
            axes[:rows] = fallback_array[:rows]
            norms = np.linalg.norm(axes, axis=1)
            axes /= np.maximum(norms[:, None], 1e-12)
            return axes
        centered = vectors - np.mean(vectors, axis=0, keepdims=True)
        if not np.all(np.isfinite(centered)):
            return None if fallback_array is None else fallback_array.copy()
        _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
        if not len(components) or not len(singular_values) or singular_values[0] <= 1e-10:
            return None if fallback_array is None else fallback_array.copy()
        if (
            require_separation
            and len(singular_values) > 1
            and singular_values[0] <= 1.15 * max(singular_values[1], 1e-12)
        ):
            return None if fallback_array is None else fallback_array.copy()

        available = min(count, len(components))
        candidate = np.asarray(components[:available], dtype=float).copy()

        axes = np.zeros((count, vectors.shape[1]), dtype=float)
        for index in range(count):
            if index < available:
                axis = candidate[index]
            elif fallback_array is not None and index < len(fallback_array):
                axis = fallback_array[index]
            else:
                continue
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-10:
                continue
            axis = axis / norm
            reference = None
            if fallback_array is not None and index < len(fallback_array):
                reference = fallback_array[index]
                if np.linalg.norm(reference) > 1e-10 and float(axis @ reference) < 0.0:
                    axis *= -1.0
            if reference is None or np.linalg.norm(reference) <= 1e-10:
                pivot = int(np.argmax(np.abs(axis)))
                if axis[pivot] < 0.0:
                    axis *= -1.0
            axes[index] = axis
        if not np.any(np.linalg.norm(axes, axis=1) > 1e-10):
            return None
        return axes

    @classmethod
    def _principal_axis(
        cls,
        vectors: np.ndarray,
        fallback: np.ndarray | None = None,
        require_separation: bool = False,
    ) -> np.ndarray | None:
        """Return the first deterministic PCA direction for residual vectors."""
        components = cls._principal_components(
            vectors, count=1, fallback=fallback,
            require_separation=require_separation,
        )
        if components is None or not len(components):
            return None
        axis = components[0]
        return axis if np.linalg.norm(axis) > 1e-10 else None

    @staticmethod
    def _orient_axis(axis: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
        """Orient a PCA axis consistently with a route or neighboring frame."""
        axis = np.asarray(axis, dtype=float).copy()
        if reference is not None and float(axis @ reference) < 0.0:
            axis *= -1.0
            return axis
        if reference is None:
            pivot = int(np.argmax(np.abs(axis)))
            if axis[pivot] < 0.0:
                axis *= -1.0
        return axis

    def _fit_residual_axes(self, result: dict[str, np.ndarray] | None) -> None:
        """Fit route-wide and locally varying residual frames.

        The route-wide PCA is a stable fallback.  For the default local frame,
        overlapping windows in longitudinal coordinate ``t`` get their own
        first residual component.  Orienting neighboring components against
        one another prevents the usual sign flips of independently fitted
        principal components.
        """
        self.residual_axes_: dict[int, np.ndarray] = {}
        self.residual_axis_knots_: dict[int, dict[str, np.ndarray]] = {}
        self.residual_plane_axes_: dict[int, np.ndarray] = {}
        self.residual_plane_knots_: dict[int, dict[str, np.ndarray]] = {}
        if result is None or "residual_vector" not in result:
            return
        vectors = np.asarray(result["residual_vector"], dtype=float) / self.model.scale_
        highway_ids = np.asarray(result["highway_id"], dtype=int)
        values = np.asarray(result["t"], dtype=float)
        if vectors.ndim != 2 or len(vectors) != len(highway_ids):
            return

        for route in range(len(self.model.splines_)):
            member_indices = np.flatnonzero(highway_ids == route)
            members = vectors[member_indices]
            if len(members) < 2:
                continue
            global_plane = self._principal_components(members, count=2)
            if global_plane is None:
                continue
            global_axis = global_plane[0]
            self.residual_axes_[route] = global_axis
            self.residual_plane_axes_[route] = global_plane
            if self.residual_frame != "local_pca" or len(members) < 6:
                continue

            order = np.argsort(values[member_indices], kind="mergesort")
            ordered_t = values[member_indices][order]
            ordered_vectors = members[order]
            bin_count = min(self.local_pca_bins, max(1, len(members) // 4))
            centers = np.linspace(0, len(members) - 1, bin_count).round().astype(int)
            half_window = max(2, self.local_pca_window // 2)
            knot_t: list[float] = []
            knot_axes: list[np.ndarray] = []
            knot_planes: list[np.ndarray] = []
            previous = global_plane
            for center in centers:
                left = max(0, int(center) - half_window)
                right = min(len(members), int(center) + half_window + 1)
                local_plane = self._principal_components(
                    ordered_vectors[left:right], count=2, fallback=previous,
                    require_separation=True,
                )
                if local_plane is None:
                    local_plane = global_plane.copy()
                local_axis = local_plane[0]
                knot_t.append(float(np.mean(ordered_t[left:right])))
                knot_axes.append(local_axis)
                knot_planes.append(local_plane)
                previous = local_plane
            if knot_axes:
                self.residual_axis_knots_[route] = {
                    "t": np.asarray(knot_t, dtype=float),
                    "axes": np.asarray(knot_axes, dtype=float),
                }
                self.residual_plane_knots_[route] = {
                    "t": np.asarray(knot_t, dtype=float),
                    "axes": np.asarray(knot_planes, dtype=float),
                }

    def _residual_axes_for(self, route: int, values: np.ndarray) -> np.ndarray | None:
        """Return one consistently oriented residual axis per query point."""
        fallback = self.residual_axes_.get(route)
        if fallback is None:
            return None
        values = np.asarray(values, dtype=float)
        if self.residual_frame != "local_pca":
            return np.repeat(fallback[None, :], len(values), axis=0)
        knots = self.residual_axis_knots_.get(route)
        if knots is None or not len(knots["t"]):
            return np.repeat(fallback[None, :], len(values), axis=0)
        indices = np.searchsorted(knots["t"], values, side="left")
        indices = np.clip(indices, 0, len(knots["t"]) - 1)
        if len(knots["t"]) > 1:
            previous = np.clip(indices - 1, 0, len(knots["t"]) - 1)
            use_previous = np.abs(values - knots["t"][previous]) < np.abs(
                values - knots["t"][indices]
            )
            indices[use_previous] = previous[use_previous]
        axes = np.asarray(knots["axes"][indices], dtype=float)
        valid = np.linalg.norm(axes, axis=1) > 1e-10
        axes[~valid] = fallback
        axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
        return axes

    def _residual_plane_axes_for(self, route: int, values: np.ndarray) -> np.ndarray | None:
        """Return the local two-dimensional residual plane for query points."""
        fallback = self.residual_plane_axes_.get(route)
        if fallback is None:
            return None
        values = np.asarray(values, dtype=float)
        if self.residual_frame != "local_pca":
            return np.repeat(fallback[None, :, :], len(values), axis=0)
        knots = self.residual_plane_knots_.get(route)
        if knots is None or not len(knots["t"]):
            return np.repeat(fallback[None, :, :], len(values), axis=0)
        indices = np.searchsorted(knots["t"], values, side="left")
        indices = np.clip(indices, 0, len(knots["t"]) - 1)
        if len(knots["t"]) > 1:
            previous = np.clip(indices - 1, 0, len(knots["t"]) - 1)
            use_previous = np.abs(values - knots["t"][previous]) < np.abs(
                values - knots["t"][indices]
            )
            indices[use_previous] = previous[use_previous]
        axes = np.asarray(knots["axes"][indices], dtype=float)
        valid = np.linalg.norm(axes, axis=2) > 1e-10
        for component in range(axes.shape[1]):
            axes[~valid[:, component], component] = fallback[component]
        axes /= np.maximum(np.linalg.norm(axes, axis=2, keepdims=True), 1e-12)
        return axes

    @staticmethod
    def _polyline_position_and_tangent(
        vertices: np.ndarray,
        values: np.ndarray,
        closed: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.asarray(vertices, dtype=float)
        if closed:
            starts = vertices
            ends = np.roll(vertices, -1, axis=0)
        else:
            starts = vertices[:-1]
            ends = vertices[1:]
        vectors = ends - starts
        lengths = np.linalg.norm(vectors, axis=1)
        lengths[lengths < 1e-12] = 1e-12
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total = cumulative[-1]
        distances = np.mod(values, 1.0) * total if closed else np.clip(values, 0.0, 1.0) * total
        indices = np.searchsorted(cumulative[1:], distances, side="right")
        indices = np.clip(indices, 0, len(vectors) - 1)
        local = (distances - cumulative[indices]) / lengths[indices]
        positions = starts[indices] + local[:, None] * vectors[indices]
        tangent = vectors[indices] / lengths[indices, None]
        return positions, tangent

    def transform_splines(self) -> list[np.ndarray]:
        """Return 2D schematic samples corresponding to every fitted spline."""
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Call fit before transform_splines")
        result = []
        for path, spline in zip(self.route_paths_, self.model.splines_):
            values = np.linspace(0.0, 1.0, len(spline.samples), endpoint=not spline.closed)
            points, _ = self._polyline_position_and_tangent(path, values, spline.closed)
            result.append(points)
        return result

    def transform_points(self, result: dict[str, np.ndarray]) -> np.ndarray:
        """Map observations to route position plus a local residual strip.

        Longitudinal position comes from the spline parameter.  The lateral
        side comes from a PCA direction fitted in a neighborhood of that
        parameter, while the lateral magnitude is the robustly clipped
        residual norm.  This keeps the map readable for high-dimensional data
        and avoids a discontinuous sign choice at spline bends.
        """
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Call fit before transform_points")
        highway_ids = np.asarray(result["highway_id"], dtype=int)
        values = np.asarray(result["t"], dtype=float)
        residual = self._standardized_residual(result)
        if "residual_vector" in result and not self.residual_axes_:
            self._fit_residual_axes(result)
        points = np.zeros((len(highway_ids), 2), dtype=float)
        for route, path in enumerate(self.route_paths_):
            members = highway_ids == route
            if not np.any(members):
                continue
            spline = self.model.splines_[route]
            positions, tangent = self._polyline_position_and_tangent(path, values[members], spline.closed)
            normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
            axes = self._residual_axes_for(route, values[members])
            if axes is not None and "residual_vector" in result:
                vector = np.asarray(result["residual_vector"], dtype=float)[members] / self.model.scale_
                signed_component = np.sum(vector * axes, axis=1)
                side = np.sign(signed_component)
                zero = side == 0.0
                if np.any(zero):
                    route_axis = self.residual_axes_.get(route)
                    if route_axis is not None:
                        side[zero] = np.sign(vector[zero] @ route_axis)
            else:
                side = np.where(np.flatnonzero(members) % 2 == 0, -1.0, 1.0)
            side[side == 0.0] = 1.0
            offset = side * np.minimum(residual[members], 1.0 / max(self.residual_scale_, 1e-8))
            points[members] = positions + self.residual_scale_ * offset[:, None] * normal
        return points

    def transform_points_3d(self, result: dict[str, np.ndarray]) -> np.ndarray:
        """Map observations to a 3D metro view with a local residual plane.

        The spline network lies on ``z=0``.  The first local residual PCA
        coordinate becomes the lateral XY offset from the route and the
        second becomes height.  Residual components beyond this displayed
        plane remain represented by ``residual_norm`` in the input result.
        """
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Call fit before transform_points_3d")
        if "residual_vector" not in result:
            points = self.transform_points(result)
            return np.column_stack([points, np.zeros(len(points), dtype=float)])
        if not self.residual_plane_axes_:
            self._fit_residual_axes(result)

        highway_ids = np.asarray(result["highway_id"], dtype=int)
        values = np.asarray(result["t"], dtype=float)
        vectors = np.asarray(result["residual_vector"], dtype=float) / self.model.scale_
        points = np.zeros((len(highway_ids), 3), dtype=float)
        limit = 1.0 / max(self.residual_scale_, 1e-8)
        for route, path in enumerate(self.route_paths_):
            members = highway_ids == route
            if not np.any(members):
                continue
            spline = self.model.splines_[route]
            positions, tangent = self._polyline_position_and_tangent(
                path, values[members], spline.closed,
            )
            normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
            axes = self._residual_plane_axes_for(route, values[members])
            if axes is None:
                fallback = self.transform_points(result)[members]
                points[members, :2] = fallback
                continue
            scores = np.einsum(
                "nd,nkd->nk", vectors[members], axes, optimize=True,
            )
            lateral = np.clip(scores[:, 0], -limit, limit)
            points[members, :2] = positions + self.residual_scale_ * lateral[:, None] * normal
            if axes.shape[1] > 1:
                points[members, 2] = self.residual_scale_ * np.clip(
                    scores[:, 1], -limit, limit,
                )
        return points

    def node_positions(self) -> dict[int, np.ndarray]:
        """Return schematic positions for junction and endpoint stations."""
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Call fit before node_positions")
        return {node: position.copy() for node, position in self.station_positions_.items()}


__all__ = ["MetroSplineLayout"]

