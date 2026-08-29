import numpy as np
import pytest

from skeletalembedding import SkeletalEmbedding
from skeletalembedding._optimization import select_backbone_mip
from skeletalembedding._topology import CandidatePath
from skeletalembedding.datasets import generate_synthetic_datasets


def _ribbon(seed=2, n=150):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 2.0 * np.pi, n)
    v = rng.uniform(-0.18, 0.18, n)
    return np.column_stack(((1.0 + v) * np.cos(u), (1.0 + v) * np.sin(u), 0.2 * v))


def _torus(seed=2, n=180):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 2.0 * np.pi, n)
    v = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack(((2.0 + 0.6 * np.cos(v)) * np.cos(u),
                            (2.0 + 0.6 * np.cos(v)) * np.sin(u),
                            0.6 * np.sin(v)))


@pytest.mark.parametrize("branches", [3, 4])
def test_y_and_x_keep_intrinsic_branch_count(branches):
    points = generate_synthetic_datasets(
        n=180, noise=0.02, random_state=3, star_branches=branches,
    )["star"]
    model = SkeletalEmbedding(n_centroids=16, random_state=1).fit(points)
    assert len(model.junctions_) == 1
    assert model.junctions_[0].branch_count == branches
    assert all(value == "intrinsic" for value in model.junction_types_.values())


def test_curved_ribbon_uses_residual_pca_before_coverage_ribs():
    points = _ribbon()
    model = SkeletalEmbedding(
        n_centroids=10, max_residual_dim=1, random_state=1,
    ).fit(points)
    assert len(model.rib_paths_) == 0
    assert model.residual_dim_ == 1
    assert model.post_pca_residual_norm_.shape == (len(points),)


def test_torus_coverage_adds_wireframe_cycles_without_changing_persistent_topology():
    model = SkeletalEmbedding(
        n_centroids=10,
        n_neighbors=6,
        max_residual_dim=1,
        coverage_refinement=True,
        coverage_error_tolerance=0.12,
        coverage_max_iterations=2,
        coverage_max_candidates_per_iteration=2,
        random_state=2,
    ).fit(_torus())
    assert model.persistent_cycle_count_ >= 1
    assert len(model.rib_paths_) > 0
    assert model.skeleton_cycle_rank_ >= model.backbone_cycle_rank_
    assert model.skeleton_cycle_rank_ > model.persistent_cycle_count_
    assert all(item["junction_type"] == "coverage" for item in model.coverage_intersections_)


def test_mip_fallback_diagnostics_are_deterministic():
    candidate = CandidatePath(0, 1, [0, 1], 1.0, 1.0, 0.0)
    specifications = [{"kind": "endpoint"}, {"kind": "endpoint"}]
    assert select_backbone_mip([candidate], specifications, 0, use_mip=False) == ({}, "disabled")
    assert select_backbone_mip(
        [candidate], specifications, 0, cycle_class_count=1,
    )[1].startswith("infeasible:missing-persistent-cycle-class")
