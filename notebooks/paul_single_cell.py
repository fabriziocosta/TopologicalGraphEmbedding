"""Helpers for the Paul et al. hematopoiesis single-cell notebook."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors


DATA_URL = "https://exampledata.scverse.org/scanpy/paul15.h5"
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "topologicalgraphembedding" / "paul15.h5"
CLUSTER_TYPE_NAMES = 6 * ["Ery"] + "MEP Mk GMP GMP DC Baso Baso Mo Mo Neu Neu Eos Lymph".split()
TYPE_ORDER = ["Ery", "MEP", "Mk", "GMP", "DC", "Baso", "Mo", "Neu", "Eos", "Lymph"]
BROAD_LINEAGE = {
    "Ery": "erythroid",
    "MEP": "progenitor",
    "Mk": "megakaryocyte",
    "GMP": "myeloid progenitor",
    "DC": "myeloid",
    "Baso": "myeloid",
    "Mo": "myeloid",
    "Neu": "myeloid",
    "Eos": "myeloid",
    "Lymph": "lymphoid",
}
LINEAGE_ORDER = [
    "erythroid", "progenitor", "megakaryocyte",
    "myeloid progenitor", "myeloid", "lymphoid",
]
LINEAGE_PALETTE = dict(zip(
    LINEAGE_ORDER,
    ["#d95f02", "#7570b3", "#e7298a", "#66a61e", "#1b9e77", "#386cb0"],
))


def download_paul15(cache_path=DEFAULT_CACHE_PATH):
    """Download the public Paul15 HDF5 file once and return its local path."""
    cache_path = Path(cache_path).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        response = requests.get(
            DATA_URL,
            headers={"User-Agent": "TopologicalGraphEmbedding notebook"},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        with cache_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return cache_path


def load_paul15(cache_path=DEFAULT_CACHE_PATH):
    """Load counts, names, and author annotations from the Paul15 HDF5 file."""
    data_path = download_paul15(cache_path)
    with h5py.File(data_path, "r") as handle:
        counts = np.asarray(handle["data.debatched"][()], dtype=np.float64).T
        gene_names = handle["data.debatched_rownames"][()].astype(str)
        cell_names = handle["data.debatched_colnames"][()].astype(str)
        cluster_ids = handle["cluster.id"][()].ravel().astype(int)
        informative_genes = handle["info.genes_strings"][()].astype(str)
    cell_types = np.asarray([CLUSTER_TYPE_NAMES[cluster - 1] for cluster in cluster_ids])
    return {
        "counts": counts,
        "gene_names": gene_names,
        "cell_names": cell_names,
        "cluster_ids": cluster_ids,
        "cell_types": cell_types,
        "informative_genes": informative_genes,
        "data_path": data_path,
    }


def preprocess_paul15(raw, min_detected_genes=300, min_cells_per_gene=5, n_hvg=750):
    """Apply label-free count preprocessing and return the model matrix."""
    counts = raw["counts"][:, np.isin(raw["gene_names"], raw["informative_genes"])]
    gene_names = raw["gene_names"][np.isin(raw["gene_names"], raw["informative_genes"])]
    detected_genes = (counts > 0).sum(axis=1)
    cell_mask = detected_genes >= min_detected_genes
    counts = counts[cell_mask]
    cell_names = raw["cell_names"][cell_mask]
    cluster_ids = raw["cluster_ids"][cell_mask]
    cell_types = raw["cell_types"][cell_mask]

    gene_mask = (counts > 0).sum(axis=0) >= min_cells_per_gene
    counts = counts[:, gene_mask]
    gene_names = gene_names[gene_mask]
    library_size = counts.sum(axis=1)
    normalized = counts / library_size[:, None] * 10_000.0
    log_expression = np.log1p(normalized)
    gene_variance = log_expression.var(axis=0)
    n_hvg = min(n_hvg, log_expression.shape[1])
    hvg_indices = np.argsort(gene_variance)[-n_hvg:][::-1]
    selected_gene_names = gene_names[hvg_indices]
    X_log = log_expression[:, hvg_indices]
    X = (X_log - X_log.mean(axis=0)) / (X_log.std(axis=0) + 1e-12)
    X = np.clip(X, -10.0, 10.0)
    return {
        "X": X,
        "X_log": X_log,
        "selected_gene_names": selected_gene_names,
        "cell_names": cell_names,
        "cluster_ids": cluster_ids,
        "cell_types": cell_types,
        "library_size": library_size,
        "detected_genes": detected_genes[cell_mask],
        "counts_after_qc": counts,
    }


def one_nn_accuracy_2d(coordinates, labels):
    """Leave-one-out 1-NN accuracy for a displayed coordinate system."""
    neighbors = NearestNeighbors(n_neighbors=2).fit(coordinates)
    nearest = neighbors.kneighbors(return_distance=False)[:, 1]
    return float(np.mean(np.asarray(labels)[nearest] == labels))


def _plot_splines_in_pca(ax, model, reducer, color="black"):
    for spline in model.splines_:
        curve = spline.samples * model.scale_ + model.mean_
        curve = reducer.transform(curve)
        if spline.closed:
            curve = np.vstack([curve, curve[0]])
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.0, alpha=0.9)
    junctions = np.asarray([model.graph_.nodes[node] for node in model.junction_nodes_])
    endpoints = np.asarray([model.graph_.nodes[node] for node in model.endpoint_nodes_])
    if len(junctions):
        ax.scatter(*reducer.transform(junctions * model.scale_ + model.mean_).T,
                   color="red", s=38, zorder=4, label="junction")
    if len(endpoints):
        ax.scatter(*reducer.transform(endpoints * model.scale_ + model.mean_).T,
                   color="darkorange", marker="s", s=32, zorder=4, label="endpoint")


def scatter_categories(ax, coordinates, labels, order, palette, size=8, alpha=0.38):
    for label in order:
        mask = np.asarray(labels) == label
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=size,
                   alpha=alpha, color=palette[label], label=label, rasterized=True)


def plot_expression_panels(fig, axes, embedding, expression, gene_names, marker_genes):
    """Plot expression values over graph coordinates for available markers."""
    available = []
    for ax, gene in zip(np.ravel(axes), marker_genes):
        matches = np.flatnonzero(gene_names == gene)
        if not len(matches):
            ax.set_title(f"{gene} (not available)")
            ax.axis("off")
            continue
        available.append(gene)
        values = expression[:, matches[0]]
        order = np.argsort(values)
        scatter = ax.scatter(
            embedding[order, 0], embedding[order, 1], c=values[order],
            s=8, alpha=0.55, cmap="viridis", rasterized=True,
        )
        ax.set_title(gene)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02, label="log1p")
    return available


def plot_graph_coordinate_embedding(ax, result, labels, order, palette):
    """Plot the model's intrinsic (highway, t) coordinates."""
    highway = result["highway_id"]
    jitter = np.random.default_rng(42).normal(0.0, 0.07, len(highway))
    scatter_categories(ax, np.column_stack([result["t"], highway + jitter]), labels,
                       order, palette, size=8, alpha=0.42)
    ax.set_xlabel("position along spline highway (t)")
    ax.set_ylabel("highway id")
    ax.set_yticks(sorted(set(highway)))
    ax.set_aspect("auto")


__all__ = [
    "BROAD_LINEAGE", "DATA_URL", "DEFAULT_CACHE_PATH", "LINEAGE_ORDER",
    "LINEAGE_PALETTE", "TYPE_ORDER", "download_paul15",
    "load_paul15", "one_nn_accuracy_2d", "plot_expression_panels",
    "plot_graph_coordinate_embedding", "preprocess_paul15", "scatter_categories",
    "_plot_splines_in_pca",
]
