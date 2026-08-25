# Notebooks

`visualize_synthetic_distributions.ipynb` runs the model on all eight 2D
synthetic distributions plus a noisy 4D hypercube. The notebook is a thin
wrapper around `topological_graph_embedding.visualization.workflows.synthetic`:
edit its parameters and rerun the function call. The hypercube dimension is controlled by `HYPERCUBE_DIM`
(default `4`) and the polygon dataset's number of sides by `POLYGON_SIDES`
(default `5`). The hypercube is displayed through a 2D PCA projection.

Notebook workflows use `backbone_initialization="topological"` explicitly.
This keeps notebook figures on the topology-aware initialization path even
though the library estimator retains `"coarsen"` as its backward-compatible
default. The workflow wrappers also silence the expected constraint-warning
when a noisy interactive refit cannot realize every requested incidence;
direct estimator use still exposes that diagnostic normally.

The synthetic workflow uses `persistence_max_points=300` and a normalized H1
threshold of `4.0` by default. This higher persistence resolution is important
for the polygon-with-circles and hypercube examples; it is still bounded so
the notebook remains practical. Electrical resistance and current terms remain
disabled unless explicitly requested.

The final notebook cell calls the shared `ipywidgets` viewer for selecting a
dataset and interactively refitting it. Controls are grouped into **Data**,
**Graph fitting**, **Topology**, and **Display** sections; the latter includes
the reducer, UMAP neighbor count, and metro dispersion width. Press
**Render selected dataset** after changing the controls.

`visualize_sklearn_toy_datasets.ipynb` applies the same model to scikit-learn's
moons, circles, blobs, classification, and Gaussian-quantile generators. It
shows the fitted graph beside the graph-coordinate embedding `(route_id, position)`.
Its final cell uses the same grouped interactive viewer as the other notebooks
for refitting one selected toy dataset.

`visualize_high_dim_sklearn_datasets.ipynb` applies the model to the built-in
scikit-learn digits, wine, breast-cancer, and diabetes datasets. It shows UMAP
projections by default (with PCA and classical MDS available as options), longitudinal graph coordinates, and a schematic metro-map layout
of the fitted graphs. The metro-map view preserves the fitted graph's broad
source-space placement and branch ordering while simplifying routes, keeps
spline routes straight where possible, uses parallel offset lanes where
needed, starts incident lines at the junction-disc radius, and allows routes
to cross when that preserves the source correspondence. Observations are placed using
their position and residual offset relative to the assigned spline. It
includes separate metro line/station and metro point-only panels, plus the
same grouped interactive viewer, including the UMAP neighbor slider.

Categorical targets use discrete colors; the continuous diabetes target uses a
pale-blue-to-ink-blue gradient and a target colorbar.

`visualize_paul_single_cell.ipynb` applies the model to the Paul et al. mouse
bone-marrow MARS-seq experiment. It downloads the public `paul15.h5` file to
`~/.cache/topologicalgraphembedding/`, preprocesses expression without using
cell labels during fitting, denoises the graph-fitting space with 50 PCA
components, and shows PCA, intrinsic graph coordinates, the fitted spline
graph, broad lineages, and marker-expression panels.

Launch it from the repository root with:

```bash
jupyter notebook notebooks/visualize_synthetic_distributions.ipynb
```

The notebook saves rendered figures under `notebooks/figures/` when executed.

Shared plotting logic is part of the installed package under
`topological_graph_embedding.visualization`: `plots.py` contains the reusable
`plot_embedding_row` four-panel renderer, `metro.py` contains the schematic
layout, `interactive.py` contains the Plotly view, and
`workflows/interactive.py` contains the shared notebook control panel. The
metro-map point panel hides junction and endpoint markers by default; pass
`show_metro_nodes=True` to display them.

The notebooks import the installed `topological_graph_embedding` package. The
notebook bootstrap adds the repository root to the import path for local
development, but the package is the sole source of implementation code.

Start Jupyter from either the repository root or this directory:

```bash
jupyter notebook notebooks/visualize_sklearn_toy_datasets.ipynb
# or, from notebooks/: jupyter notebook visualize_sklearn_toy_datasets.ipynb
```

For reusable sklearn pipelines, `topological_graph_embedding.sklearn` provides
`SplineEmbeddingTransformer` and `SplineEmbeddingClassifier`. The classifier combines
spline identity, longitudinal position, and residual coordinates in the local
hyperplane perpendicular to each spline with a configurable downstream
estimator passed through `estimator=`; its default is a random forest. The high-dimensional dataset view
adds one per-route score just past `t=1`: normalized out-of-fold multiclass
accuracy from the route coordinates for categorical targets, or Spearman rank
correlation for regression targets.
