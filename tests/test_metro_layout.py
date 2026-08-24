from types import SimpleNamespace

import numpy as np

from metro_layout import MetroSplineLayout
from topological_spline_graph import SkeletonGraph, SplineCurve


def _model(node_positions, chain_nodes):
    graph = SkeletonGraph(node_positions)
    chains = []
    splines = []
    for nodes in chain_nodes:
        for left, right in zip(nodes[:-1], nodes[1:]):
            graph.add_edge(left, right)
        samples = np.asarray([node_positions[node] for node in nodes], dtype=float)
        chains.append({"nodes": list(nodes), "closed": False})
        splines.append(
            SplineCurve(
                samples=samples,
                t_values=np.linspace(0.0, 1.0, len(samples)),
                closed=False,
            )
        )
    return SimpleNamespace(
        _fitted=True,
        graph_=graph,
        chains_=chains,
        splines_=splines,
        scale_=np.ones(samples.shape[1]),
    )


def _angle_bin(vector):
    angle = np.arctan2(vector[1], vector[0])
    return int(np.floor(angle / (np.pi / 4.0) + 0.5)) % 8


def _incidence_path(layout, model, route, side):
    path = layout.route_paths_[route]
    return path if side == 0 else path[::-1]


def test_source_order_and_octilinear_junction_exits():
    model = _model(
        {
            0: np.array([0.0, 0.0]),
            1: np.array([-4.0, 0.0]),
            2: np.array([4.0, 0.0]),
            3: np.array([0.0, 4.0]),
        },
        [[0, 1], [0, 2], [0, 3]],
    )
    layout = MetroSplineLayout(model, random_state=0).fit()

    for route in range(len(model.chains_)):
        path = _incidence_path(layout, model, route, 0)
        direction = path[1] - path[0]
        assert _angle_bin(direction) == layout.source_route_direction_bins_[route, 0]

    center = layout.station_positions_[0]
    assert layout.station_positions_[1][0] < center[0]
    assert layout.station_positions_[2][0] > center[0]
    assert layout.station_positions_[3][1] > center[1]


def test_opposite_horizontal_rays_use_the_centerline_reference():
    model = _model(
        {
            0: np.array([0.0, 0.0]),
            1: np.array([5.0, 0.0]),
            2: np.array([-5.0, -1.0]),
            3: np.array([-5.0, 1.0]),
            4: np.array([0.0, 5.0]),
        },
        [[0, 1], [0, 2], [0, 3], [0, 4]],
    )
    layout = MetroSplineLayout(model, parallel_spacing=1.0, random_state=0).fit()
    center = layout.station_positions_[0]

    right_delta = layout.junction_ports_[0, 0, 0] - center
    left_deltas = sorted(
        float((layout.junction_ports_[0, route, 0] - center)[1])
        for route in (1, 2)
    )
    assert np.isclose(right_delta[1], 0.0, atol=1e-8)
    assert np.allclose(left_deltas, [-0.5, 0.5], atol=1e-8)
    expected_radius = 0.5 / np.sin(np.pi / 8.0 - 1e-3) + 0.12
    assert np.isclose(layout.junction_radii_[0], expected_radius, atol=1e-8)


def test_clipped_routes_do_not_reenter_a_junction_disc():
    model = _model(
        {
            0: np.array([0.0, 0.0]),
            1: np.array([-5.0, -1.0]),
            2: np.array([5.0, 0.0]),
            3: np.array([0.0, 5.0]),
        },
        [[0, 1], [0, 2], [0, 3]],
    )
    layout = MetroSplineLayout(model, random_state=0).fit()
    center = layout.station_positions_[0]
    radius = layout.junction_radii_[0]
    for route in range(len(model.chains_)):
        path = layout.route_paths_[route]
        for start, end in zip(path[:-1], path[1:]):
            vector = end - start
            fraction = np.clip(
                float((center - start) @ vector) / max(float(vector @ vector), 1e-12),
                0.0,
                1.0,
            )
            closest = start + fraction * vector
            assert np.linalg.norm(closest - center) >= radius - 1e-8


def test_directional_lanes_stay_inside_disjoint_octilinear_sectors():
    nodes = {0: np.array([0.0, 0.0])}
    chains = []
    next_node = 1
    for direction_bin in range(8):
        direction = np.array([
            np.cos(direction_bin * np.pi / 4.0),
            np.sin(direction_bin * np.pi / 4.0),
        ])
        normal = np.array([-direction[1], direction[0]])
        for lateral in (-1.5, -0.5, 0.5, 1.5):
            nodes[next_node] = 5.0 * direction + lateral * normal
            chains.append([0, next_node])
            next_node += 1

    model = _model(nodes, chains)
    layout = MetroSplineLayout(model, parallel_spacing=1.0, random_state=0).fit()
    center = layout.station_positions_[0]
    for (node, route, side), port in layout.junction_ports_.items():
        if node != 0:
            continue
        direction_bin = layout.source_route_direction_bins_[route, side]
        actual = np.arctan2(*(port - center)[::-1])
        expected = direction_bin * np.pi / 4.0
        error = abs((actual - expected + np.pi) % (2.0 * np.pi) - np.pi)
        assert error < np.pi / 8.0

    assert layout.junction_radii_[0] > layout.junction_radius


def test_cycle_junctions_are_on_cycles_and_connectors_use_final_intersections():
    node_positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([-1.0, 0.0]),
        2: np.array([0.0, 1.0]),
        3: np.array([1.0, 0.0]),
        4: np.array([6.0, 0.0]),
        5: np.array([5.0, 0.0]),
        6: np.array([6.0, 1.0]),
        7: np.array([7.0, 0.0]),
        8: np.array([3.0, 0.0]),
    }
    graph = SkeletonGraph(node_positions)
    chains = []
    splines = []

    def add_chain(nodes, closed):
        for left, right in zip(nodes[:-1], nodes[1:]):
            graph.add_edge(left, right)
        samples = np.asarray([node_positions[node] for node in nodes], dtype=float)
        chains.append({"nodes": list(nodes), "closed": closed})
        splines.append(
            SplineCurve(
                samples=samples,
                t_values=np.linspace(0.0, 1.0, len(samples), endpoint=not closed),
                closed=closed,
            )
        )

    add_chain([0, 1, 2, 3, 0], True)
    add_chain([4, 5, 6, 7, 4], True)
    add_chain([0, 8, 4], False)
    model = SimpleNamespace(
        _fitted=True,
        graph_=graph,
        chains_=chains,
        splines_=splines,
        scale_=np.ones(2),
    )
    layout = MetroSplineLayout(model, random_state=0).fit()

    for route, node in ((0, 0), (1, 4)):
        cycle = layout.route_paths_[route]
        assert np.min(np.linalg.norm(cycle - layout.station_positions_[node], axis=1)) < 1e-8

    connector = layout.route_paths_[2]
    assert np.isclose(
        np.linalg.norm(connector[0] - layout.station_positions_[0]),
        layout.junction_radii_[0],
        atol=1e-8,
    )
    assert np.isclose(
        np.linalg.norm(connector[-1] - layout.station_positions_[4]),
        layout.junction_radii_[4],
        atol=1e-8,
    )
    assert np.linalg.norm(layout.station_positions_[4] - layout.station_positions_[0]) > 2.0


def test_parallel_lanes_are_centered_and_endpoints_clear_the_disc():
    model = _model(
        {
            0: np.array([0.0, 0.0]),
            1: np.array([5.0, -1.0]),
            2: np.array([5.0, 1.0]),
            3: np.array([0.0, 5.0]),
        },
        [[0, 1], [0, 2], [0, 3]],
    )
    layout = MetroSplineLayout(model, parallel_spacing=1.0, random_state=0).fit()

    ports = [layout.junction_ports_[0, route, 0] for route in range(2)]
    offsets = sorted(float(port[1] - layout.station_positions_[0][1]) for port in ports)
    assert np.allclose(offsets, [-0.5, 0.5], atol=1e-8)
    assert np.isclose(offsets[1] - offsets[0], 1.0, atol=1e-8)

    radius = layout.junction_radii_[0]
    for endpoint in (1, 2):
        assert np.linalg.norm(layout.station_positions_[endpoint] - layout.station_positions_[0]) > radius + 0.1


def test_layout_scale_spreads_the_map_without_changing_lane_clearance():
    model = _model(
        {
            0: np.array([0.0, 0.0]),
            1: np.array([5.0, -1.0]),
            2: np.array([5.0, 1.0]),
            3: np.array([0.0, 5.0]),
        },
        [[0, 1], [0, 2], [0, 3]],
    )
    compact = MetroSplineLayout(model, parallel_spacing=1.0, layout_scale=1.0).fit()
    spacious = MetroSplineLayout(model, parallel_spacing=1.0, layout_scale=2.0).fit()

    assert np.isclose(compact.junction_radii_[0], spacious.junction_radii_[0])
    compact_offsets = sorted(
        float((compact.junction_ports_[0, route, 0] - compact.station_positions_[0])[1])
        for route in (0, 1)
    )
    spacious_offsets = sorted(
        float((spacious.junction_ports_[0, route, 0] - spacious.station_positions_[0])[1])
        for route in (0, 1)
    )
    assert np.allclose(compact_offsets, spacious_offsets, atol=1e-8)
    assert np.linalg.norm(spacious.station_positions_[1] - spacious.station_positions_[0]) > (
        np.linalg.norm(compact.station_positions_[1] - compact.station_positions_[0])
    )


def test_high_dimensional_source_layout_uses_deterministic_pca_axes():
    model = _model(
        {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([-3.0, 0.0, 1.0]),
            2: np.array([3.0, 0.0, -1.0]),
            3: np.array([0.0, 3.0, 0.5]),
        },
        [[0, 1], [0, 2], [0, 3]],
    )
    first = MetroSplineLayout(model, random_state=0).fit()
    second = MetroSplineLayout(model, random_state=0).fit()

    assert first.source_layout_positions_.shape == (4, 2)
    assert np.all(np.isfinite(first.source_layout_positions_))
    assert np.allclose(first.source_layout_positions_, second.source_layout_positions_)
    assert first.source_route_direction_bins_ == second.source_route_direction_bins_


def test_local_residual_pca_changes_frame_along_a_route():
    model = _model(
        {
            0: np.array([0.0, 0.0]),
            1: np.array([6.0, 0.0]),
        },
        [[0, 1]],
    )
    t = np.linspace(0.03, 0.97, 24)
    residual_vector = np.zeros((len(t), 2))
    residual_vector[:12, 1] = np.where(np.arange(12) % 2 == 0, -1.0, 1.0)
    residual_vector[12:, 0] = np.where(np.arange(12) % 2 == 0, -1.0, 1.0)
    result = {
        "highway_id": np.zeros(len(t), dtype=int),
        "t": t,
        "residual_vector": residual_vector,
    }

    layout = MetroSplineLayout(
        model, random_state=0, local_pca_bins=4, local_pca_window=6,
    ).fit(result)
    knots = layout.residual_axis_knots_[0]

    assert knots["axes"].shape == (4, 2)
    assert abs(knots["axes"][0, 1]) > abs(knots["axes"][0, 0])
    assert abs(knots["axes"][-1, 0]) > abs(knots["axes"][-1, 1])
    displayed = layout.transform_points(result)
    assert np.all(np.isfinite(displayed))


def test_route_pca_is_available_as_a_stable_local_frame_fallback():
    model = _model(
        {
            0: np.array([0.0, 0.0]),
            1: np.array([6.0, 0.0]),
        },
        [[0, 1]],
    )
    t = np.linspace(0.05, 0.95, 12)
    residual_vector = np.column_stack([np.zeros(len(t)), np.linspace(-1.0, 1.0, len(t))])
    result = {
        "highway_id": np.zeros(len(t), dtype=int),
        "t": t,
        "residual_vector": residual_vector,
    }
    layout = MetroSplineLayout(model, residual_frame="route_pca").fit(result)

    assert layout.residual_axes_[0].shape == (2,)
    assert not layout.residual_axis_knots_


def test_transform_points_3d_uses_a_local_residual_plane():
    model = _model(
        {
            0: np.array([0.0, 0.0]),
            1: np.array([6.0, 0.0]),
        },
        [[0, 1]],
    )
    model.scale_ = np.ones(3)
    t = np.linspace(0.03, 0.97, 24)
    residual_vector = np.zeros((len(t), 3))
    residual_vector[:12, 1] = np.where(np.arange(12) % 2 == 0, -1.0, 1.0)
    residual_vector[:12, 2] = np.linspace(-0.6, 0.6, 12)
    residual_vector[12:, 1] = np.linspace(-0.6, 0.6, 12)
    residual_vector[12:, 2] = np.where(np.arange(12) % 2 == 0, -1.0, 1.0)
    result = {
        "highway_id": np.zeros(len(t), dtype=int),
        "t": t,
        "residual_vector": residual_vector,
    }

    layout = MetroSplineLayout(model, local_pca_bins=4, local_pca_window=6).fit(result)
    displayed = layout.transform_points_3d(result)

    assert displayed.shape == (len(t), 3)
    assert np.all(np.isfinite(displayed))
    assert layout.residual_plane_axes_[0].shape == (2, 3)
    assert np.any(np.abs(displayed[:, 2]) > 1e-8)


def test_crossing_source_routes_are_retained():
    model = _model(
        {
            0: np.array([-3.0, -3.0]),
            1: np.array([-3.0, 3.0]),
            2: np.array([3.0, 3.0]),
            3: np.array([3.0, -3.0]),
        },
        [[0, 2], [1, 3]],
    )
    layout = MetroSplineLayout(model, random_state=0).fit()

    first = layout.route_paths_[0]
    second = layout.route_paths_[1]

    def cross(left, right):
        return float(left[0] * right[1] - left[1] * right[0])

    def proper_crossing(first_path, second_path):
        for first_start, first_end in zip(first_path[:-1], first_path[1:]):
            first_vector = first_end - first_start
            for second_start, second_end in zip(second_path[:-1], second_path[1:]):
                second_vector = second_end - second_start
                first_a = cross(first_vector, second_start - first_start)
                first_b = cross(first_vector, second_end - first_start)
                second_a = cross(second_vector, first_start - second_start)
                second_b = cross(second_vector, first_end - second_start)
                if first_a * first_b < 0.0 and second_a * second_b < 0.0:
                    return True
        return False

    assert proper_crossing(first, second)
