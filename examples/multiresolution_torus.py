"""Run with PYTHONPATH=src python examples/multiresolution_torus.py."""

import argparse

import numpy as np

from skeletalembedding import SkeletalEmbedding


def torus_points():
    u, v = np.meshgrid(
        np.linspace(0, 2 * np.pi, 24, endpoint=False),
        np.linspace(0, 2 * np.pi, 16, endpoint=False),
    )
    u, v = u.ravel(), v.ravel()
    return np.column_stack(
        (
            (2 + 0.8 * np.cos(v)) * np.cos(u),
            (2 + 0.8 * np.cos(v)) * np.sin(u),
            0.8 * np.sin(v),
        )
    )


def fit_torus():
    return SkeletalEmbedding(
        hierarchy_target_size=80,
        n_centroids=12,
        standardize=False,
        persistence_max_points=180,
        persistence_threshold=1.5,
        max_cycles=2,
        max_residual_dim=1,
        residual_subspace_smoothness=0.2,
        coverage_refinement=True,
        coverage_max_iterations=1,
        coverage_max_candidates_per_iteration=3,
        coverage_error_tolerance=0.1,
        rib_seed_source="both",
        random_state=0,
    ).fit(torus_points())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", help="Optional output figure path (requires Matplotlib)"
    )
    args = parser.parse_args()
    model = fit_torus()
    print(model.hierarchy_summary_)
    print("Resolution support:", model.cycle_resolution_support_)
    print(
        "Backbone cycles:", model.backbone_cycle_rank_, "Ribs:", len(model.rib_paths_)
    )
    if args.plot:
        from skeletalembedding.visualization import plot_hierarchy

        figure, _ = plot_hierarchy(model, show_ancestry=True)
        figure.savefig(args.plot, dpi=140)
