"""Public package for topological spline graph embeddings."""

from .embedding import SplineGraphEmbedding
from .results import EmbeddingResult

__all__ = [
    "EmbeddingResult",
    "SplineGraphEmbedding",
]
