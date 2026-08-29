"""Optional notebook-style visualization workflows."""

from .digits import build_dataset_catalog, fit_catalog_entry
from .toy import build_toy_datasets, fit_toy_datasets, render_toy_datasets

__all__ = [
    "build_dataset_catalog",
    "build_toy_datasets",
    "fit_catalog_entry",
    "fit_toy_datasets",
    "render_toy_datasets",
]
