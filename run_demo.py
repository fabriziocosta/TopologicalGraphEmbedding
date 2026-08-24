"""Run the synthetic benchmark and write plots plus a summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from topological_graph_embedding import SplineGraphEmbedding
from topological_graph_embedding.datasets import generate_synthetic_datasets


def run(output_dir: str | Path = "outputs", n: int = 500, noise: float = 0.045, random_state: int = 0) -> list[dict[str, object]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = generate_synthetic_datasets(n=n, noise=noise, random_state=random_state)
    rows: list[dict[str, object]] = []
    for index, (name, points) in enumerate(datasets.items()):
        model = SplineGraphEmbedding(
            n_centroids=32,
            persistence_threshold=None,
            spline_smoothing=0.02,
            max_cycles=5,
            random_state=random_state + index,
            merge_junction_distance=None,
        )
        transformed = model.fit_transform(points)
        figure_path = output / f"{name}.png"
        ax = model.plot_network(points, show_projections=True, title=name.replace("-", " ").title())
        ax.figure.savefig(figure_path, dpi=160, bbox_inches="tight")
        ax.figure.clf()
        rows.append(
            {
                "dataset": name,
                "inferred_cycles": model.realized_cycle_count_,
                "persistent_h1_bars": len(model.persistence_diagram_),
                "junctions": len(model.junctions_),
                "endpoints": len(model.endpoints_),
                "spline_chains": len(model.routes_),
                "median_projection_residual": float(np.median(transformed.residual_norm)),
                "figure": str(figure_path),
            }
        )

    table_path = output / "summary.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} figures and {table_path}")
    print("dataset,inferred_cycles,junctions,endpoints,spline_chains,median_projection_residual")
    for row in rows:
        print(
            f"{row['dataset']},{row['inferred_cycles']},{row['junctions']},"
            f"{row['endpoints']},{row['spline_chains']},{row['median_projection_residual']:.5f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.045)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    run(args.output_dir, n=args.n, noise=args.noise, random_state=args.random_state)


if __name__ == "__main__":
    main()
