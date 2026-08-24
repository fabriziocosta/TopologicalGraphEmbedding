# Topological graph embedding

This package fits a smooth spline route network to a noisy point cloud. It
builds a sparse landmark kNN topology, fits open or closed spline routes, and
projects observations into a typed embedding result.

Install the core package and development tools with:

```bash
python -m pip install -e ".[dev,notebooks]"
```

Run the demo with:

```bash
python run_demo.py --output-dir outputs
```

## Core API

```python
from topological_graph_embedding import SplineGraphEmbedding

model = SplineGraphEmbedding(n_centroids=32, random_state=0)
result = model.fit_transform(X)

print(model.realized_cycle_count_)
print(result.projected, result.residual_norm)
normal = model.normal_coordinates(result)
model.plot_network(X, show_projections=True)
```

`result` is a frozen `EmbeddingResult` with the fields `route_id`,
`position`, `projected`, `residual`, `residual_norm`, and `tangent`. Results are
attribute-based and do not support dictionary indexing.

Fitted topology diagnostics include `persistent_cycle_count_`,
`requested_cycle_count_`, `realized_cycle_count_`, `topology_shortfall_`,
`persistence_backend_`, `route_backends_`, `routes_`, `route_chains_`,
`junctions_`, and `endpoints_`.

The optional sklearn adapters are available from `topological_graph_embedding.sklearn`:

```python
from topological_graph_embedding.sklearn import (
    SplineEmbeddingClassifier,
    SplineEmbeddingTransformer,
)

embedding = SplineEmbeddingTransformer(n_centroids=32, random_state=0)
X_route = embedding.fit_transform(X)
result = embedding.transform_result(X)

classifier = SplineEmbeddingClassifier(n_centroids=32, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```

Pass a downstream sklearn estimator through `estimator=`. The metro layout,
synthetic datasets, and plotting helpers are optional modules under
`topological_graph_embedding.metro`, `topological_graph_embedding.datasets`,
and the `notebooks` package.

Synthetic data are generated with `generate_synthetic_datasets()`; the
interactive notebooks in `notebooks/` import the installed package and keep
no local copies of the implementation modules.
