"""Scikit-learn adapters for the topological spline graph.

The graph itself is deliberately inspectable and returns a dictionary from
``transform``.  This module provides the conventional numeric transformer
interface as well as a classifier which composes the graph coordinates with
any scikit-learn-compatible classifier.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .topological_spline_graph import TopologicalSplineGraph, spline_normal_coordinates


Array = np.ndarray


class SplineGraphTransformer(BaseEstimator, TransformerMixin):
    """Fit spline highways and expose graph-aware numeric coordinates.

    Each transformed observation contains a one-hot spline identity, its
    longitudinal coordinate, and its signed residual from the closest spline
    in the original feature space.  The residual coordinates retain the side
    of a spline, unlike a coordinate made from ``t`` alone.
    """

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
        # Keep constructor arguments untouched so sklearn's get_params and
        # clone can reconstruct the estimator exactly.
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

    def _new_graph(self) -> TopologicalSplineGraph:
        return TopologicalSplineGraph(
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
    ) -> "SplineGraphTransformer":
        """Fit the spline graph.  ``y`` is accepted for pipeline compatibility."""
        points = check_array(X, ensure_2d=True, dtype=float)
        self._check_feature_count(points, reset=True)
        self.graph_ = self._new_graph().fit(points)
        self._fit_feature_names_ = self._make_feature_names()
        return self

    def _check_feature_count(self, points: Array, reset: bool) -> None:
        """Validate feature count across sklearn versions."""
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
        route_count = len(self.graph_.splines_)
        names = [f"highway_{route}" for route in range(route_count)]
        names.append("longitudinal_t")
        names.append("residual_norm")
        names.extend(f"residual_{feature}" for feature in range(self.n_features_in_))
        return np.asarray(names, dtype=object)

    def _features_from_result(self, result: dict[str, Array]) -> Array:
        highway_id = np.asarray(result["highway_id"], dtype=int)
        route_count = len(self.graph_.splines_)
        highway_features = np.zeros((len(highway_id), route_count), dtype=float)
        valid = (highway_id >= 0) & (highway_id < route_count)
        highway_features[np.arange(len(highway_id))[valid], highway_id[valid]] = 1.0

        residual = np.asarray(result["residual_vector"], dtype=float)
        # Scale residual coordinates in the same metric used while fitting.
        residual_scaled = residual / self.graph_.scale_
        residual_norm = np.linalg.norm(residual_scaled, axis=1, keepdims=True)
        longitudinal = np.asarray(result["t"], dtype=float)[:, None]
        return np.hstack((highway_features, longitudinal, residual_norm, residual_scaled))

    def transform_graph(
        self,
        X: Array | Sequence[Sequence[float]],
    ) -> dict[str, Array]:
        """Return the raw projection dictionary from the fitted graph."""
        check_is_fitted(self, "graph_")
        points = check_array(X, ensure_2d=True, dtype=float)
        self._check_feature_count(points, reset=False)
        return self.graph_.transform(points)

    def transform(self, X: Array | Sequence[Sequence[float]]) -> Array:
        """Return route identity, longitudinal, and signed residual features."""
        return self._features_from_result(self.transform_graph(X))

    def transform_normal(self, X: Array | Sequence[Sequence[float]]) -> Array:
        """Return residual coordinates in the spline-normal hyperplane frame."""
        return spline_normal_coordinates(self.graph_, self.transform_graph(X))

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return names for the columns emitted by :meth:`transform`."""
        check_is_fitted(self, "_fit_feature_names_")
        return self._fit_feature_names_.copy()


class SplineGraphClassifier(ClassifierMixin, SplineGraphTransformer):
    """Classify observations using spline identity, position, and normal frame.

    Parameters are the same as :class:`SplineGraphTransformer`, plus
    ``classifier``.  If omitted, a :class:`RandomForestClassifier` is used.
    Any estimator implementing sklearn's ``fit`` and ``predict`` interface can
    be supplied, for example ``LogisticRegression`` or ``RandomForestClassifier``.
    The supplied estimator is cloned before fitting.
    """

    def __init__(
        self,
        classifier: Any | None = None,
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
        self.classifier = classifier

    def fit(
        self,
        X: Array | Sequence[Sequence[float]],
        y: Array | Sequence[Any],
    ) -> "SplineGraphClassifier":
        points, target = check_X_y(
            X,
            y,
            ensure_2d=True,
            dtype=float,
            multi_output=True,
        )
        super().fit(points)
        self.classifier_ = clone(self.classifier) if self.classifier is not None else RandomForestClassifier(
            random_state=self.random_state
        )
        # Keep the graph identity and longitudinal coordinate, while using a
        # true local frame for the off-spline component.  This makes the
        # classifier invariant to arbitrary rotations of the original feature
        # axes within the spline-normal hyperplane.
        graph_result = self.transform_graph(points)
        route_features = self._features_from_result(graph_result)
        route_count = len(self.graph_.splines_)
        normal_features = spline_normal_coordinates(self.graph_, graph_result)
        classifier_features = np.hstack((route_features[:, :route_count + 1], normal_features))
        self.classifier_.fit(classifier_features, target)
        if hasattr(self.classifier_, "classes_"):
            self.classes_ = self.classifier_.classes_
        self.n_outputs_ = getattr(self.classifier_, "n_outputs_", 1)
        return self

    def predict(self, X: Array | Sequence[Sequence[float]]) -> Array:
        """Predict labels with the modular downstream classifier."""
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict(self._classifier_features(X))

    def fit_predict(
        self,
        X: Array | Sequence[Sequence[float]],
        y: Array | Sequence[Any],
    ) -> Array:
        """Fit the graph and classifier, then return predictions for ``X``."""
        return self.fit(X, y).predict(X)

    def predict_proba(self, X: Array | Sequence[Sequence[float]]) -> Array:
        """Delegate probability prediction when the classifier supports it."""
        check_is_fitted(self, "classifier_")
        if not hasattr(self.classifier_, "predict_proba"):
            raise AttributeError("classifier does not provide predict_proba")
        return self.classifier_.predict_proba(self._classifier_features(X))

    def decision_function(self, X: Array | Sequence[Sequence[float]]) -> Array:
        """Delegate decision scores when the classifier supports them."""
        check_is_fitted(self, "classifier_")
        if not hasattr(self.classifier_, "decision_function"):
            raise AttributeError("classifier does not provide decision_function")
        return self.classifier_.decision_function(self._classifier_features(X))

    def _classifier_features(self, X: Array | Sequence[Sequence[float]]) -> Array:
        """Build the same route-plus-normal features used during fitting."""
        graph_result = self.transform_graph(X)
        route_features = self._features_from_result(graph_result)
        route_count = len(self.graph_.splines_)
        normal_features = spline_normal_coordinates(self.graph_, graph_result)
        return np.hstack((route_features[:, :route_count + 1], normal_features))
