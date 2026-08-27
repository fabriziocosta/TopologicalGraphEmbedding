"""Optional scikit-learn adapters for :mod:`topological_graph_embedding`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .embedding import SplineGraphEmbedding
from .results import EmbeddingResult

Array = np.ndarray


class SplineEmbeddingTransformer(TransformerMixin, BaseEstimator):
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
        mutual_knn: bool = False,
        add_mst: bool = False,
        max_residual_dim: int = 0,
        residual_pca_bandwidth: float = 0.1,
        residual_subspace_smoothness: float = 0.0,
        backbone_initialization: str = "coarsen",
        detect_cycles: bool = True,
        detect_junctions: bool = True,
        junction_scales: int | Sequence[float] = 6,
        junction_inner_fraction: float = 0.25,
        junction_confidence: float = 0.7,
        use_local_pca: bool = True,
        local_pca_neighbors: int = 20,
        max_branch_angle_degrees: float = 45.0,
        use_effective_resistance: bool = False,
        use_electrical_flow: bool = False,
        use_kron_reduction: bool = False,
        routing_length_weight: float = 1.0,
        routing_tangent_weight: float = 1.0,
        routing_density_weight: float = 0.5,
        routing_resistance_weight: float = 0.0,
        routing_current_weight: float = 0.0,
        use_tangent_boundary_conditions: bool = True,
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
        self.mutual_knn = mutual_knn
        self.add_mst = add_mst
        self.max_residual_dim = max_residual_dim
        self.residual_pca_bandwidth = residual_pca_bandwidth
        self.residual_subspace_smoothness = residual_subspace_smoothness
        self.backbone_initialization = backbone_initialization
        self.detect_cycles = detect_cycles
        self.detect_junctions = detect_junctions
        self.junction_scales = junction_scales
        self.junction_inner_fraction = junction_inner_fraction
        self.junction_confidence = junction_confidence
        self.use_local_pca = use_local_pca
        self.local_pca_neighbors = local_pca_neighbors
        self.max_branch_angle_degrees = max_branch_angle_degrees
        self.use_effective_resistance = use_effective_resistance
        self.use_electrical_flow = use_electrical_flow
        self.use_kron_reduction = use_kron_reduction
        self.routing_length_weight = routing_length_weight
        self.routing_tangent_weight = routing_tangent_weight
        self.routing_density_weight = routing_density_weight
        self.routing_resistance_weight = routing_resistance_weight
        self.routing_current_weight = routing_current_weight
        self.use_tangent_boundary_conditions = use_tangent_boundary_conditions

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
            mutual_knn=self.mutual_knn,
            add_mst=self.add_mst,
            max_residual_dim=self.max_residual_dim,
            residual_pca_bandwidth=self.residual_pca_bandwidth,
            residual_subspace_smoothness=self.residual_subspace_smoothness,
            backbone_initialization=self.backbone_initialization,
            detect_cycles=self.detect_cycles,
            detect_junctions=self.detect_junctions,
            junction_scales=self.junction_scales,
            junction_inner_fraction=self.junction_inner_fraction,
            junction_confidence=self.junction_confidence,
            use_local_pca=self.use_local_pca,
            local_pca_neighbors=self.local_pca_neighbors,
            max_branch_angle_degrees=self.max_branch_angle_degrees,
            use_effective_resistance=self.use_effective_resistance,
            use_electrical_flow=self.use_electrical_flow,
            use_kron_reduction=self.use_kron_reduction,
            routing_length_weight=self.routing_length_weight,
            routing_tangent_weight=self.routing_tangent_weight,
            routing_density_weight=self.routing_density_weight,
            routing_resistance_weight=self.routing_resistance_weight,
            routing_current_weight=self.routing_current_weight,
            use_tangent_boundary_conditions=self.use_tangent_boundary_conditions,
        )

    def fit(
        self,
        X: Array | Sequence[Sequence[float]],
        y: Array | Sequence[Any] | None = None,
    ) -> SplineEmbeddingTransformer:
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
                f"X has {points.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )

    def _make_feature_names(self) -> np.ndarray:
        route_count = len(self.embedding_.routes_)
        names = [f"route_{route}" for route in range(route_count)]
        names.extend(("position", "residual_norm"))
        if self.embedding_.residual_dim_ > 0:
            names.extend(
                f"residual_pca_{feature}"
                for feature in range(self.embedding_.residual_dim_)
            )
        else:
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
        if self.embedding_.residual_dim_ > 0:
            residual_features = result.residual_coordinates
        else:
            residual_features = result.residual / self.embedding_.scale_
        return np.hstack(
            (
                route_features,
                result.position[:, None],
                result.residual_norm[:, None],
                residual_features,
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
        mutual_knn: bool = False,
        add_mst: bool = False,
        max_residual_dim: int = 0,
        residual_pca_bandwidth: float = 0.1,
        residual_subspace_smoothness: float = 0.0,
        backbone_initialization: str = "coarsen",
        detect_cycles: bool = True,
        detect_junctions: bool = True,
        junction_scales: int | Sequence[float] = 6,
        junction_inner_fraction: float = 0.25,
        junction_confidence: float = 0.7,
        use_local_pca: bool = True,
        local_pca_neighbors: int = 20,
        max_branch_angle_degrees: float = 45.0,
        use_effective_resistance: bool = False,
        use_electrical_flow: bool = False,
        use_kron_reduction: bool = False,
        routing_length_weight: float = 1.0,
        routing_tangent_weight: float = 1.0,
        routing_density_weight: float = 0.5,
        routing_resistance_weight: float = 0.0,
        routing_current_weight: float = 0.0,
        use_tangent_boundary_conditions: bool = True,
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
            mutual_knn=mutual_knn,
            add_mst=add_mst,
            max_residual_dim=max_residual_dim,
            residual_pca_bandwidth=residual_pca_bandwidth,
            residual_subspace_smoothness=residual_subspace_smoothness,
            backbone_initialization=backbone_initialization,
            detect_cycles=detect_cycles,
            detect_junctions=detect_junctions,
            junction_scales=junction_scales,
            junction_inner_fraction=junction_inner_fraction,
            junction_confidence=junction_confidence,
            use_local_pca=use_local_pca,
            local_pca_neighbors=local_pca_neighbors,
            max_branch_angle_degrees=max_branch_angle_degrees,
            use_effective_resistance=use_effective_resistance,
            use_electrical_flow=use_electrical_flow,
            use_kron_reduction=use_kron_reduction,
            routing_length_weight=routing_length_weight,
            routing_tangent_weight=routing_tangent_weight,
            routing_density_weight=routing_density_weight,
            routing_resistance_weight=routing_resistance_weight,
            routing_current_weight=routing_current_weight,
            use_tangent_boundary_conditions=use_tangent_boundary_conditions,
        )
        self.estimator = estimator

    def fit(
        self,
        X: Array | Sequence[Sequence[float]],
        y: Array | Sequence[Any],
    ) -> SplineEmbeddingClassifier:
        points, target = check_X_y(X, y, ensure_2d=True, dtype=float, multi_output=True)
        super().fit(points)
        self.estimator_ = (
            clone(self.estimator)
            if self.estimator is not None
            else RandomForestClassifier(random_state=self.random_state)
        )
        result = self.transform_result(points)
        route_count = len(self.embedding_.routes_)
        normal = self._classifier_residual_features(result)
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
        return np.hstack((route_features, self._classifier_residual_features(result)))

    def _classifier_residual_features(self, result: EmbeddingResult) -> Array:
        if self.embedding_.residual_dim_ > 0:
            return result.residual_coordinates
        return self.embedding_.normal_coordinates(result)

    def predict(self, X: Array | Sequence[Sequence[float]]) -> Array:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(self._classifier_features(X))

    def fit_predict(self, X: Array | Sequence[Sequence[float]], y: Array | Sequence[Any]) -> Array:
        return self.fit(X, y).predict(X)

    def predict_proba(self, X: Array | Sequence[Sequence[float]]) -> Array:
        check_is_fitted(self, "estimator_")
        features = self._classifier_features(X)
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError("estimator does not provide predict_proba")
        return self.estimator_.predict_proba(features)

    def decision_function(self, X: Array | Sequence[Sequence[float]]) -> Array:
        check_is_fitted(self, "estimator_")
        features = self._classifier_features(X)
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(features)
        if hasattr(self.estimator_, "predict_proba"):
            probabilities = np.asarray(self.estimator_.predict_proba(features))
            if probabilities.shape[1] == 2:
                return probabilities[:, 1] - probabilities[:, 0]
            return probabilities
        raise AttributeError("estimator does not provide decision_function or predict_proba")


__all__ = ["SplineEmbeddingClassifier", "SplineEmbeddingTransformer"]
