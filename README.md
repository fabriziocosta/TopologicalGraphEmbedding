# Topological graph embedding prototype

The prototype fits a compact graph of smooth spline highways to a noisy point
cloud. It estimates the number of H1 cycles, clusters observations into graph
landmarks, builds an MST, adds cycle-closing edges, fits open or periodic
splines, and projects every observation onto the resulting network.

Install the recommended dependencies and run the benchmark:

```bash
python -m pip install -r requirements.txt
python run_demo.py --output-dir outputs
```

The demo writes one PNG per synthetic data set and `outputs/summary.csv`.

For an interactive comparison of the line, Y, X, circle, figure-eight,
branching tree, loop-with-branch, and a noisy 4D hypercube, open
[`notebooks/visualize_synthetic_distributions.ipynb`](notebooks/visualize_synthetic_distributions.ipynb).

For scikit-learn toy distributions and their graph-coordinate embeddings, open
[`notebooks/visualize_sklearn_toy_datasets.ipynb`](notebooks/visualize_sklearn_toy_datasets.ipynb).

For real higher-dimensional scikit-learn datasets (digits, wine, breast cancer,
and diabetes), open
[`notebooks/visualize_high_dim_sklearn_datasets.ipynb`](notebooks/visualize_high_dim_sklearn_datasets.ipynb).
That notebook includes a UMAP projection by default (with PCA available as an
alternative), longitudinal-coordinate, and metro-map views. In the map view,
observations are placed along their assigned spline and offset laterally by a
robust residual distance. The default lateral side uses a locally smoothed PCA
frame of residuals along each spline; use ``MetroSplineLayout(...,
residual_frame="route_pca")`` for a single route-wide frame.

For an interactive 3D view of the digits data, open
[`notebooks/visualize_digits_3d.ipynb`](notebooks/visualize_digits_3d.ipynb).
The spline map stays on `z=0`; the first two local residual PCA coordinates
place observations laterally and vertically around their assigned spline.

For the Paul et al. mouse bone-marrow single-cell experiment, open
[`notebooks/visualize_paul_single_cell.ipynb`](notebooks/visualize_paul_single_cell.ipynb).
The notebook downloads and caches the public `paul15.h5` file outside the
repository, applies standard label-free single-cell preprocessing, and fits
the topological graph embedding.

Minimal API usage:

```python
from topological_spline_graph import TopologicalSplineGraph

model = TopologicalSplineGraph(n_centroids=32, random_state=0)
result = model.fit_transform(X)
print(model.cycle_count_)
print(result["projection"], result["residual_norm"])
```

For a scikit-learn transformer or a spline-aware classifier:

```python
from spline_sklearn import SplineGraphClassifier, SplineGraphTransformer

embedding = SplineGraphTransformer(n_centroids=32, random_state=0)
X_graph = embedding.fit_transform(X)

model = SplineGraphClassifier(n_centroids=32, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

The classifier assigns every observation to its closest spline and gives the
downstream estimator the spline identity, longitudinal coordinate, and
residual coordinates expressed in the local hyperplane perpendicular to the
spline.  Thus points with the same longitudinal coordinate can still be
separated by which spline they occupy and by their off-spline displacement.
Pass any compatible sklearn classifier through `classifier=`; the default is
`RandomForestClassifier`.

The high-dimensional dataset notebook scores categorical routes by normalized
out-of-fold multiclass accuracy using the longitudinal and spline-normal
coordinates, so random assignment is 0 and perfect prediction is 1. A route
with one class is explicitly perfect. Regression routes show one Spearman
rank-correlation score. Continuous target plots use a pale-blue-to-ink-blue
color scale.
