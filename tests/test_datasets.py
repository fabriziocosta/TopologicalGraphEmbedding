import numpy as np
import pytest

from topological_graph_embedding.datasets import (
    generate_synthetic_datasets,
    noisy_polygon_rays_circles,
)


def test_polygon_rays_circles_is_reproducible_and_configurable():
    first = noisy_polygon_rays_circles(
        n=240,
        noise=0.0,
        n_sides=4,
        radius=1.3,
        circle_radius=0.18,
        rng=np.random.default_rng(12),
    )
    second = noisy_polygon_rays_circles(
        n=240,
        noise=0.0,
        n_sides=4,
        radius=1.3,
        circle_radius=0.18,
        rng=np.random.default_rng(12),
    )

    assert first.shape == (240, 2)
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)


def test_polygon_rays_circles_is_registered_with_synthetic_datasets():
    datasets = generate_synthetic_datasets(n=32, noise=0.02, random_state=3)

    assert "polygon-rays-circles" in datasets
    assert datasets["polygon-rays-circles"].shape == (32, 2)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_sides": 2}, {"radius": 0.0}, {"circle_radius": -0.1}],
)
def test_polygon_rays_circles_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        noisy_polygon_rays_circles(**kwargs)
