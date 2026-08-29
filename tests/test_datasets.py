import numpy as np
import pytest

from skeletalembedding.datasets import (
    generate_synthetic_datasets,
    noisy_binary_tree,
    noisy_polygon_rays_circles,
    noisy_star,
)


def test_binary_tree_is_reproducible_and_depth_configurable():
    first = noisy_binary_tree(
        n=240,
        noise=0.0,
        depth=2,
        rng=np.random.default_rng(12),
    )
    second = noisy_binary_tree(
        n=240,
        noise=0.0,
        depth=2,
        rng=np.random.default_rng(12),
    )
    deeper = noisy_binary_tree(
        n=240,
        noise=0.0,
        depth=3,
        rng=np.random.default_rng(12),
    )

    assert first.shape == (240, 2)
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, deeper)


def test_binary_tree_is_registered_with_configurable_depth():
    datasets = generate_synthetic_datasets(
        n=32,
        noise=0.02,
        random_state=3,
        binary_tree_depth=4,
    )

    assert "binary-tree" in datasets
    assert "branching-tree" not in datasets
    assert datasets["binary-tree"].shape == (32, 2)


@pytest.mark.parametrize("depth", [0, -1, 1.5, True])
def test_binary_tree_rejects_invalid_depth(depth):
    with pytest.raises(ValueError, match="depth"):
        noisy_binary_tree(depth=depth)


def test_star_is_reproducible_and_configurable():
    first = noisy_star(
        n=240,
        noise=0.0,
        branches=4,
        rng=np.random.default_rng(12),
    )
    second = noisy_star(
        n=240,
        noise=0.0,
        branches=4,
        rng=np.random.default_rng(12),
    )
    five_branch = noisy_star(
        n=240,
        noise=0.0,
        branches=5,
        rng=np.random.default_rng(12),
    )

    assert first.shape == (240, 2)
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, five_branch)


@pytest.mark.parametrize("branches", [0, 1, -1, 2.5, True])
def test_star_rejects_invalid_branch_count(branches):
    with pytest.raises(ValueError, match="branches"):
        noisy_star(branches=branches)


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


def test_polygon_side_count_is_exposed_by_dataset_generator():
    with pytest.raises(ValueError, match="n_sides"):
        generate_synthetic_datasets(n=8, polygon_sides=2)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_sides": 2}, {"radius": 0.0}, {"circle_radius": -0.1}],
)
def test_polygon_rays_circles_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        noisy_polygon_rays_circles(**kwargs)
