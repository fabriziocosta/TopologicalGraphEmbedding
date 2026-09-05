"""Opt-in hierarchy scaling benchmark; no full-data PH or MIP.

PYTHONPATH=src python benchmarks/multiresolution.py --sizes 10000 100000 1000000
Add --fit-coarse to time a topology/MIP probe capped at 1000 representatives.
Reports process peak RSS (includes earlier runs), construction time, level
sizes, and optional topology input / MIP candidate counts as JSON lines.
"""

import argparse
import json
import resource
import sys
from time import perf_counter

import numpy as np

from skeletalembedding import SkeletalEmbedding
from skeletalembedding._multiresolution import build_hierarchy


def benchmark(n, fit_coarse=False, max_levels=8):
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 2 * np.pi, n)
    points = np.column_stack([np.cos(t), np.sin(t), rng.normal(0, 0.01, n)])
    start = perf_counter()
    levels = build_hierarchy(
        points, representative_method="approx_medoid", max_levels=max_levels
    )
    result = {
        "n": n,
        "hierarchy_seconds": perf_counter() - start,
        "level_sizes": [len(level.points) for level in levels],
    }
    if fit_coarse:
        coarse = levels[-1].points
        if len(coarse) > 1000:
            result["coarse_fit"] = (
                "skipped: hierarchy remains above 1000 representatives"
            )
        else:
            start = perf_counter()
            model = SkeletalEmbedding(use_multiresolution=False, standardize=False).fit(
                coarse
            )
            result.update(
                coarse_fit_seconds=perf_counter() - start,
                topology_input_sizes=model.topology_input_sizes_,
                mip_candidate_count=model.mip_candidate_count_,
            )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["process_peak_rss_mb"] = rss / (
        1024**2 if sys.platform == "darwin" else 1024
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000])
    parser.add_argument("--fit-coarse", action="store_true")
    parser.add_argument("--max-levels", type=int, default=8)
    args = parser.parse_args()
    for n in args.sizes:
        if n < 3:
            parser.error("sizes must be at least 3")
        print(json.dumps(benchmark(n, args.fit_coarse, args.max_levels)), flush=True)
