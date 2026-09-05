# Multiresolution scaling

Run from the repository with NumPy and SciPy installed:

```sh
PYTHONPATH=src python benchmarks/multiresolution.py --sizes 10000 100000 1000000
PYTHONPATH=src python benchmarks/multiresolution.py --sizes 100000 --max-levels 16 --fit-coarse
```

The fixture is a noisy circle in three dimensions. The script reports JSON
lines and uses approximate medoids. The optional coarse fit is capped at
1,000 representatives; it never runs PH or MIP on the original large cloud.
Eight compression steps need not reach the 1,000-representative target.
Increase `--max-levels` to investigate that time/compression tradeoff.

Observed local runs (September 2026, Python 3.14; indicative, not performance
guarantees):

| Original rows | Hierarchy seconds | Final representatives | Levels including original | Process peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.21 | 884 | 7 | 130 |
| 100,000 | 2.24 | 3,785 | 9 | 247 |
| 1,000,000 | 25.79 | 36,726 | 9 | 1,386 |

The 10,000-row case's additional coarse fit took 0.56 seconds, with 884 topology
input points and 3 MIP candidate paths. The 100,000-row coarse fit was skipped
because it exceeded the probe cap. No PH or MIP was requested for the million-row
run. RSS includes interpreter/dependency memory and earlier runs in the same
process; the million-row measurement used a separate process.

These figures cover hierarchy construction and an optional isolated coarse
probe, not full-data spline projection, stability runs, or coverage refinement.
Dense high-dimensional clouds can make exact kNN queries substantially more
expensive. The test suite separately checks that compression never calls the
global pairwise-distance helper and that backbone inputs are representative-sized.
