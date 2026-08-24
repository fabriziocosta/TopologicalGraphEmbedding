"""Optional scikit-learn adapters for :mod:`topological_graph_embedding`."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .embedding import SplineGraphEmbedding
from .results import EmbeddingResult

Array = np.ndarray


class SplineEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Fit a spline route network and expose numeric embedding features."""

    def __init__(
        self,
        n_centroids: int = 32,
        persistence_threshold: float | None = None,
        spline_smoothing: float = 0.02,
        max_cycles: int = 5,
        random_state: int = 0,
        standardize: bool = True,
        merge_junction_distance: float | None = None,
        prune_short_branches: bool = True,
        prune_branch_factor: float = 0.5,
        persistence_max_points: int = 60,
        spline_samples_per_node: int = 12,
        linear_structure_tolerance: float = 0.12,
        topology_neighbors: int = 6,
    ) -> None:
        self.n_centroids = n_centroids
        self.persistence_threshold = persistence_threshold
        self.spline_smoothing = spline_smoothing
        self.max_cycles = max_cycles
        self.random_state = random_state
        self.standardize = standardize
        self.merge_junction_distance = merge_junction_distance
        self.prune_short_branches = prune_short_branches
        self.prune_branch_factor = prune_branch_factor
        self.persistence_max_points = persistence_max_points
        self.spline_samples_per_node = spline_samples_per_node
        self.linear_structure_tolerance = linear_structure_tolerance
        self.topology_neighbors = topology_neighbors

    def _new_embedding(self) -> SplineGraphEmbedding:
        return SplineGraphEmbedding(
            n_centroids=self.n_centroids,
            persistence_threshold=self.persistence_threshold,
            spline_smoothing=self.spline_smoothing,
            max_cycles=self.max_cycles,
            random_state=self.random_state,
            standardize=self.standardize,
            merge_junction_distance=self.merge_junction_distance,
            prune_short_branches=self.prune_short_branches,
            prune_branch_factor=self.prune_branch_factor,
            persistence_max_points=self.persistence_max_points,
            spline_samples_per_node=self.spline_samples_per_node,
            linear_structure_tolerance=self.linear_structure_tolerance,
            topology_neighbors=self.topology_neighbors,
        )

    def fit(
        self,
        X: Array | Sequence[Sequence[float]],
        y: Array | Sequence[Any] | None = None,
    ) -> "SplineEmbeddingTransformer":
        points = check_array(X, ensure_2d=True, dtype=float)
        self._check_feature_count(points, reset=True)
        self.embedding_ = self._new_embedding().fit(points)
        self._fit_feature_names_ = self._make_feature_names()
        return self

    def _check_feature_count(self, points: Array, reset: bool) -> None:
        if reset:
            self.n_features_in_ = points.shape[1]
            return
        check_is_fitted(self, "n_features_in_")
        if points.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {points.shape[1]} features, but this estimator was "
                f"fitted with {self.n_features_in_} features."
            )

    def _make_feature_names(self) -> np.ndarray:
        route_count = len(self.embedding_.routes_)
        names = [f"route_{route}" for route in range(route_count)]
        names.extend(("position", "residual_norm"))
        names.extend(f"residual_{feature}" for feature in range(self.n_features_in_))
        return np.asarray(names, dtype=object)

    def transform_result(self, X: Array | Sequence[Sequence[float]]) -> EmbeddingResult:
        """Return the typed projection result for ``X``."""
        check_is_fitted(self, "embedding_")
        points = check_array(X, ensure_2d=True, dtype=float)
        self._check_feature_count(points, reset=False)
        return self.embedding_.transform(points)

    def _features_from_result(self, result: EmbeddingResult) -> Array:
        route_id = result.route_id
        route_count = len(self.embedding_.routes_)
        route_features = np.zeros((len(route_id), route_count), dtype=float)
        valid = (route_id >= 0) & (route_id < route_count)
        route_features[np.arange(len(route_id))[valid], route_id[valid]] = 1.0
        residual_scaled = result.residual / self.embedding_.scale_
        return np.hstack(
            (
                route_features,
                result.position[:, None],
                result.residual_norm[:, None],
                residual_scaled,
            )
        )

    def transform(self, X: Array | Sequence[Sequence[float]]) -> Array:
        """Return route indicators, position, and residual features."""
        return self._features_from_result(self.transform_result(X))

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, "_fit_feature_names_")
        return self._fit_feature_names_.copy()


class SplineEmbeddingClassifier(ClassifierMixin, SplineEmbeddingTransformer):
    """Classify observations using route and deterministic normal features."""

    def __init__(
        self,
        estimator: Any | None = None,
        n_centroids: int = 32,
        persistence_threshold: float | None = None,
        spline_smoothing: float = 0.02,
        max_cycles: int = 5,
        random_state: int = 0,
        standardize: bool = True,
        merge_junction_distance: float | None = None,
        prune_short_branches: bool = True,
        prune_branch_factor: float = 0.5,
        persistence_max_points: int = 60,
        spline_samples_per_node: int = 12,
        linear_structure_tolerance: float = 0.12,
        topology_neighbors: int = 6,
    ) -> None:
        super().__init__(
            n_centroids=n_centroids,
            persistence_threshold=persistence_threshold,
            spline_smoothing=spline_smoothing,
            max_cycles=max_cycles,
            random_state=random_state,
            standardize=standardize,
            merge_junction_distance=merge_junction_distance,
            prune_short_branches=prune_short_branches,
            prune_branch_factor=prune_branch_factor,
            persistence_max_points=persistence_max_points,
            spline_samples_per_node=spline_samples_per_node,
            linear_structure_tolerance=linear_structure_tolerance,
            topology_neighbors=topology_neighbors,
        )
        self.estimator = estimator

    def fit(
        self,
        X: Array | Sequence[Sequence[float]],
        y: Array | Sequence[Any],
    ) -> "SplineEmbeddingClassifier":
        points, target = check_X_y(X, y, ensure_2d=True, dtype=float, multi_output=True)
        super().fit(points)
        self.estimator_ = (
            clone(self.estimator)
            if self.estimator is not None
            else RandomForestClassifier(random_state=self.random_state)
        )
        result = self.transform_result(points)
        route_count = len(self.embedding_.routes_)
        normal = self.embedding_.normal_coordinates(result)
        classifier_features = np.hstack((self.transform(points)[:, : route_count + 1], normal))
        self.estimator_.fit(classifier_features, target)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        self.n_outputs_ = getattr(self.estimator_, "n_outputs_", 1)
        return self

    def _classifier_features(self, X: Array | Sequence[Sequence[float]]) -> Array:
        result = self.transform_result(X)
        route_count = len(self.embedding_.routes_)
        route_features = self._features_from_result(result)[:, : route_count + 1]
        return np.hstack((route_features, self.embedding_.normal_coordinates(result)))

    def predict(self, X: Array | Sequence[Sequence[float]]) -> Array:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(self._classifier_features(X))

    def fit_predict(self, X: Array | Sequence[Sequence[float]], y: Array | Sequence[Any]) -> Array:
        return self.fit(X, y).predict(X)

    def predict_proba(self, X: Array | Sequence[Sequence[float]]) -> Array:
        check_is_fitted(self, "estimator_")
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError("estimator does not provide predict_proba")
        return self.estimator_.predict_proba(self._classifier_features(X))

    def decision_function(self, X: Array | Sequence[Sequence[float]]) -> Array:
        check_is_fitted(self, "estimator_")
        if not hasattr(self.estimator_, "decision_function"):
            raise AttributeError("estimator does not provide decision_function")
        return self.estimator_.decision_function(self._classifier_features(X))


__all__ = ["SplineEmbeddingTransformer", "SplineEmbeddingClassifier"]
