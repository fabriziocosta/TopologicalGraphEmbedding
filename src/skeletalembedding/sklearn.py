"""Optional scikit-learn adapters for :mod:`skeletalembedding`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .estimator import SkeletalEmbedding
from .results import EmbeddingResult

Array = np.ndarray


class SkeletalEmbeddingTransformer(TransformerMixin, BaseEstimator):
    """Fit a skeletal model and expose numeric element-coordinate features."""

    def __init__(
        self,
        n_centroids: int = 32,
        n_backbone_nodes: int | None = None,
        backbone_node_spacing: float | None = None,
        backbone_node_policy: str = "topology_preserving",
        n_neighbors: int = 6,
        persistence_threshold: float | None = None,
        spline_smoothing: float = 0.02,
        spline_control_mode: str = "support",
        max_cycles: int = 5,
        random_state: int = 0,
        standardize: bool = True,
        merge_junction_distance: float | None = None,
        prune_short_branches: bool = True,
        prune_branch_factor: float = 0.5,
        persistence_max_points: int = 60,
        spline_samples_per_node: int = 12,
        linear_structure_tolerance: float = 0.12,
        topology_neighbors: int | None = None,
        mutual_knn: bool = True,
        add_mst: bool = True,
        max_residual_dim: int = 0,
        residual_pca_bandwidth: float = 0.1,
        residual_subspace_smoothness: float = 0.0,
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
        coverage_refinement: bool = False,
        coverage_error_tolerance: float | None = None,
        coverage_relative_tolerance: float | None = None,
        coverage_quantile: float = 0.95,
        coverage_max_iterations: int = 10,
        coverage_max_ribs: int | None = None,
        coverage_max_candidates_per_iteration: int = 20,
        coverage_candidate_spacing: float | None = None,
        coverage_min_error: float | None = None,
        coverage_min_gain: float = 0.0,
        coverage_length_penalty: float = 0.0,
        coverage_rib_penalty: float = 0.0,
        coverage_junction_penalty: float = 0.0,
        coverage_selection: str = "greedy",
        rib_candidate_type: str = "transverse",
        stability_selection: bool = False,
        stability_runs: int = 30,
        stability_fraction: float = 0.7,
        stability_min_support: float = 0.75,
        stability_jitter: float = 0.0,
        rib_stability_runs: int | None = None,
        rib_min_support: float = 0.6,
        stability_residual_subspaces: bool = False,
        use_multiresolution: bool = True,
        hierarchy_max_levels: int = 8,
        hierarchy_target_size: int = 1000,
        hierarchy_min_reduction: float = 0.15,
        representative_method: str = "medoid",
        hierarchy_distance_quantile: float = 0.1,
        hierarchy_local_neighbors: int = 10,
        backbone_level: int | str = "auto",
        backbone_max_representatives: int = 2000,
        backbone_consensus_levels: int = 3,
        route_resolution_weight: float = 0.1,
        rib_resolution_weight: float = 0.1,
        rib_seed_source: str = "both",
        n_jobs: int | None = None,
    ) -> None:
        self.n_centroids = n_centroids
        self.n_backbone_nodes = n_backbone_nodes
        self.backbone_node_spacing = backbone_node_spacing
        self.backbone_node_policy = backbone_node_policy
        self.n_neighbors = n_neighbors
        self.persistence_threshold = persistence_threshold
        self.spline_smoothing = spline_smoothing
        self.spline_control_mode = spline_control_mode
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
        self.coverage_refinement = coverage_refinement
        self.coverage_error_tolerance = coverage_error_tolerance
        self.coverage_relative_tolerance = coverage_relative_tolerance
        self.coverage_quantile = coverage_quantile
        self.coverage_max_iterations = coverage_max_iterations
        self.coverage_max_ribs = coverage_max_ribs
        self.coverage_max_candidates_per_iteration = coverage_max_candidates_per_iteration
        self.coverage_candidate_spacing = coverage_candidate_spacing
        self.coverage_min_error = coverage_min_error
        self.coverage_min_gain = coverage_min_gain
        self.coverage_length_penalty = coverage_length_penalty
        self.coverage_rib_penalty = coverage_rib_penalty
        self.coverage_junction_penalty = coverage_junction_penalty
        self.coverage_selection = coverage_selection
        self.rib_candidate_type = rib_candidate_type
        self.stability_selection = stability_selection
        self.stability_runs = stability_runs
        self.stability_fraction = stability_fraction
        self.stability_min_support = stability_min_support
        self.stability_jitter = stability_jitter
        self.rib_stability_runs = rib_stability_runs
        self.rib_min_support = rib_min_support
        self.stability_residual_subspaces = stability_residual_subspaces
        self.use_multiresolution = use_multiresolution
        self.hierarchy_max_levels = hierarchy_max_levels
        self.hierarchy_target_size = hierarchy_target_size
        self.hierarchy_min_reduction = hierarchy_min_reduction
        self.representative_method = representative_method
        self.hierarchy_distance_quantile = hierarchy_distance_quantile
        self.hierarchy_local_neighbors = hierarchy_local_neighbors
        self.backbone_level = backbone_level
        self.backbone_max_representatives = backbone_max_representatives
        self.backbone_consensus_levels = backbone_consensus_levels
        self.route_resolution_weight = route_resolution_weight
        self.rib_resolution_weight = rib_resolution_weight
        self.rib_seed_source = rib_seed_source
        self.n_jobs = n_jobs

    def _new_embedding(self) -> SkeletalEmbedding:
        return SkeletalEmbedding(
            n_centroids=self.n_centroids,
            n_backbone_nodes=self.n_backbone_nodes,
            backbone_node_spacing=self.backbone_node_spacing,
            backbone_node_policy=self.backbone_node_policy,
            n_neighbors=self.n_neighbors,
            persistence_threshold=self.persistence_threshold,
            spline_smoothing=self.spline_smoothing,
            spline_control_mode=self.spline_control_mode,
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
            coverage_refinement=self.coverage_refinement,
            coverage_error_tolerance=self.coverage_error_tolerance,
            coverage_relative_tolerance=self.coverage_relative_tolerance,
            coverage_quantile=self.coverage_quantile,
            coverage_max_iterations=self.coverage_max_iterations,
            coverage_max_ribs=self.coverage_max_ribs,
            coverage_max_candidates_per_iteration=self.coverage_max_candidates_per_iteration,
            coverage_candidate_spacing=self.coverage_candidate_spacing,
            coverage_min_error=self.coverage_min_error,
            coverage_min_gain=self.coverage_min_gain,
            coverage_length_penalty=self.coverage_length_penalty,
            coverage_rib_penalty=self.coverage_rib_penalty,
            coverage_junction_penalty=self.coverage_junction_penalty,
            coverage_selection=self.coverage_selection,
            rib_candidate_type=self.rib_candidate_type,
            stability_selection=self.stability_selection,
            stability_runs=self.stability_runs,
            stability_fraction=self.stability_fraction,
            stability_min_support=self.stability_min_support,
            stability_jitter=self.stability_jitter,
            rib_stability_runs=self.rib_stability_runs,
            rib_min_support=self.rib_min_support,
            stability_residual_subspaces=self.stability_residual_subspaces,
            use_multiresolution=self.use_multiresolution,
            hierarchy_max_levels=self.hierarchy_max_levels,
            hierarchy_target_size=self.hierarchy_target_size,
            hierarchy_min_reduction=self.hierarchy_min_reduction,
            representative_method=self.representative_method,
            hierarchy_distance_quantile=self.hierarchy_distance_quantile,
            hierarchy_local_neighbors=self.hierarchy_local_neighbors,
            backbone_level=self.backbone_level,
            backbone_max_representatives=self.backbone_max_representatives,
            backbone_consensus_levels=self.backbone_consensus_levels,
            route_resolution_weight=self.route_resolution_weight,
            rib_resolution_weight=self.rib_resolution_weight,
            rib_seed_source=self.rib_seed_source,
            n_jobs=self.n_jobs,
        )

    def fit(
        self,
        X: Array | Sequence[Sequence[float]],
        y: Array | Sequence[Any] | None = None,
    ) -> SkeletalEmbeddingTransformer:
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
        names = [f"skeleton_element_{element}" for element in range(route_count)]
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


class SkeletalEmbeddingClassifier(ClassifierMixin, SkeletalEmbeddingTransformer):
    """Classify observations using route and deterministic normal features."""

    def __init__(
        self,
        estimator: Any | None = None,
        n_centroids: int = 32,
        n_backbone_nodes: int | None = None,
        backbone_node_spacing: float | None = None,
        backbone_node_policy: str = "topology_preserving",
        n_neighbors: int = 6,
        persistence_threshold: float | None = None,
        spline_smoothing: float = 0.02,
        spline_control_mode: str = "support",
        max_cycles: int = 5,
        random_state: int = 0,
        standardize: bool = True,
        merge_junction_distance: float | None = None,
        prune_short_branches: bool = True,
        prune_branch_factor: float = 0.5,
        persistence_max_points: int = 60,
        spline_samples_per_node: int = 12,
        linear_structure_tolerance: float = 0.12,
        topology_neighbors: int | None = None,
        mutual_knn: bool = True,
        add_mst: bool = True,
        max_residual_dim: int = 0,
        residual_pca_bandwidth: float = 0.1,
        residual_subspace_smoothness: float = 0.0,
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
        coverage_refinement: bool = False,
        coverage_error_tolerance: float | None = None,
        coverage_relative_tolerance: float | None = None,
        coverage_quantile: float = 0.95,
        coverage_max_iterations: int = 10,
        coverage_max_ribs: int | None = None,
        coverage_max_candidates_per_iteration: int = 20,
        coverage_candidate_spacing: float | None = None,
        coverage_min_error: float | None = None,
        coverage_min_gain: float = 0.0,
        coverage_length_penalty: float = 0.0,
        coverage_rib_penalty: float = 0.0,
        coverage_junction_penalty: float = 0.0,
        coverage_selection: str = "greedy",
        rib_candidate_type: str = "transverse",
        stability_selection: bool = False,
        stability_runs: int = 30,
        stability_fraction: float = 0.7,
        stability_min_support: float = 0.75,
        stability_jitter: float = 0.0,
        rib_stability_runs: int | None = None,
        rib_min_support: float = 0.6,
        stability_residual_subspaces: bool = False,
        use_multiresolution: bool = True,
        hierarchy_max_levels: int = 8,
        hierarchy_target_size: int = 1000,
        hierarchy_min_reduction: float = 0.15,
        representative_method: str = "medoid",
        hierarchy_distance_quantile: float = 0.1,
        hierarchy_local_neighbors: int = 10,
        backbone_level: int | str = "auto",
        backbone_max_representatives: int = 2000,
        backbone_consensus_levels: int = 3,
        route_resolution_weight: float = 0.1,
        rib_resolution_weight: float = 0.1,
        rib_seed_source: str = "both",
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(
            n_centroids=n_centroids,
            n_backbone_nodes=n_backbone_nodes,
            backbone_node_spacing=backbone_node_spacing,
            backbone_node_policy=backbone_node_policy,
            n_neighbors=n_neighbors,
            persistence_threshold=persistence_threshold,
            spline_smoothing=spline_smoothing,
            spline_control_mode=spline_control_mode,
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
            coverage_refinement=coverage_refinement,
            coverage_error_tolerance=coverage_error_tolerance,
            coverage_relative_tolerance=coverage_relative_tolerance,
            coverage_quantile=coverage_quantile,
            coverage_max_iterations=coverage_max_iterations,
            coverage_max_ribs=coverage_max_ribs,
            coverage_max_candidates_per_iteration=coverage_max_candidates_per_iteration,
            coverage_candidate_spacing=coverage_candidate_spacing,
            coverage_min_error=coverage_min_error,
            coverage_min_gain=coverage_min_gain,
            coverage_length_penalty=coverage_length_penalty,
            coverage_rib_penalty=coverage_rib_penalty,
            coverage_junction_penalty=coverage_junction_penalty,
            coverage_selection=coverage_selection,
            rib_candidate_type=rib_candidate_type,
            stability_selection=stability_selection,
            stability_runs=stability_runs,
            stability_fraction=stability_fraction,
            stability_min_support=stability_min_support,
            stability_jitter=stability_jitter,
            rib_stability_runs=rib_stability_runs,
            rib_min_support=rib_min_support,
            stability_residual_subspaces=stability_residual_subspaces,
            use_multiresolution=use_multiresolution,
            hierarchy_max_levels=hierarchy_max_levels,
            hierarchy_target_size=hierarchy_target_size,
            hierarchy_min_reduction=hierarchy_min_reduction,
            representative_method=representative_method,
            hierarchy_distance_quantile=hierarchy_distance_quantile,
            hierarchy_local_neighbors=hierarchy_local_neighbors,
            backbone_level=backbone_level,
            backbone_max_representatives=backbone_max_representatives,
            backbone_consensus_levels=backbone_consensus_levels,
            route_resolution_weight=route_resolution_weight,
            rib_resolution_weight=rib_resolution_weight,
            rib_seed_source=rib_seed_source,
            n_jobs=n_jobs,
        )
        self.estimator = estimator

    def fit(
        self,
        X: Array | Sequence[Sequence[float]],
        y: Array | Sequence[Any],
    ) -> SkeletalEmbeddingClassifier:
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


__all__ = ["SkeletalEmbeddingClassifier", "SkeletalEmbeddingTransformer"]
