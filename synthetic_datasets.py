"""Compatibility imports for the topological graph embedding package."""

from topological_graph_embedding.synthetic_datasets import (
    DATASET_FACTORIES,
    generate_datasets,
    noisy_hypercube,
)

__all__ = ["DATASET_FACTORIES", "generate_datasets", "noisy_hypercube"]

