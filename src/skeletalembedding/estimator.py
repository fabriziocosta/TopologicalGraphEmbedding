"""Public estimator orchestration for skeletal embeddings."""

from .embedding import SkeletalEmbedding as _SkeletalEmbeddingCore


class SkeletalEmbedding(_SkeletalEmbeddingCore):
    """Learn a stable topology-aware backbone, residual fields, and ribs.

    The computational stages remain implemented by the private pipeline
    helpers; this module is the canonical public estimator entry point.
    """


__all__ = ["SkeletalEmbedding"]
