"""Spline route fitting, evaluation, and projection."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.interpolate import splev, splprep
except ImportError as exc:  # pragma: no cover
    splev = None
    splprep = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

Array = np.ndarray

def _catmull_rom(points: Array, t: Array, closed: bool) -> Array:
    """Fallback cubic curve evaluator used when SciPy is unavailable."""
    n = len(points)
    if n < 3:
        return np.vstack([points[0] + value * (points[-1] - points[0]) for value in t])
    result = []
    if closed:
        for value in t:
            position = (value % 1.0) * n
            index = int(np.floor(position))
            local = position - index
            p0 = points[(index - 1) % n]
            p1 = points[index % n]
            p2 = points[(index + 1) % n]
            p3 = points[(index + 2) % n]
            result.append(
                0.5
                * ((2 * p1) + (-p0 + p2) * local + (2 * p0 - 5 * p1 + 4 * p2 - p3) * local**2
                   + (-p0 + 3 * p1 - 3 * p2 + p3) * local**3)
            )
    else:
        for value in t:
            position = np.clip(value, 0.0, 1.0) * (n - 1)
            index = min(int(np.floor(position)), n - 2)
            local = position - index
            p0 = points[max(index - 1, 0)]
            p1 = points[index]
            p2 = points[index + 1]
            p3 = points[min(index + 2, n - 1)]
            result.append(
                0.5
                * ((2 * p1) + (-p0 + p2) * local + (2 * p0 - 5 * p1 + 4 * p2 - p3) * local**2
                   + (-p0 + 3 * p1 - 3 * p2 + p3) * local**3)
            )
    return np.asarray(result)


@dataclass
class _SplineRoute:
    """Dense, parameterized representation of one fitted route."""

    samples: Array
    t_values: Array
    closed: bool
    tck: Any = None
    backend: str = "numpy"

    def evaluate(self, t: Array | float) -> Array:
        values = np.atleast_1d(np.asarray(t, dtype=float))
        if self.closed:
            values = values % 1.0
        else:
            values = np.clip(values, 0.0, 1.0)
        if self.tck is not None and splev is not None:
            evaluated = np.asarray(splev(values, self.tck)).T
        else:
            evaluated = _catmull_rom(self.samples, values, self.closed)
        if not self.closed:
            evaluated[values <= 0.0] = self.samples[0]
            evaluated[values >= 1.0] = self.samples[-1]
        return evaluated[0] if np.ndim(t) == 0 else evaluated

    def tangent(self, t: Array | float, epsilon: float = 1e-4) -> Array:
        """Return unit tangent vectors in the curve's coordinate system.

        The dense sampled curve is also used as the projection geometry, so a
        centered finite difference gives a consistent tangent for both the
        SciPy and NumPy curve evaluators.  The direction is all that matters
        here; the normal frame is invariant to the tangent's scale.
        """
        values = np.atleast_1d(np.asarray(t, dtype=float))
        if self.closed:
            left = values - epsilon
            right = values + epsilon
        else:
            left = np.clip(values - epsilon, 0.0, 1.0)
            right = np.clip(values + epsilon, 0.0, 1.0)
        derivatives = np.asarray(self.evaluate(right) - self.evaluate(left), dtype=float)
        norms = np.linalg.norm(derivatives, axis=1, keepdims=True)
        invalid = norms[:, 0] < 1e-12
        if np.any(invalid):
            fallback = np.diff(self.samples, axis=0)
            if self.closed:
                fallback = np.vstack([fallback, self.samples[0] - self.samples[-1]])
            fallback_norms = np.linalg.norm(fallback, axis=1)
            valid_fallback = np.flatnonzero(fallback_norms > 1e-12)
            if len(valid_fallback):
                for row in np.flatnonzero(invalid):
                    nearest = int(np.argmin(np.abs(self.t_values[valid_fallback] - values[row])))
                    derivatives[row] = fallback[valid_fallback[nearest]]
                norms = np.linalg.norm(derivatives, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        result = derivatives / norms
        return result[0] if np.ndim(t) == 0 else result

    def project(self, X: Array, batch_size: int = 4096) -> tuple[Array, Array, Array]:
        """Project points onto sampled line segments; return point, t, d2."""
        X = np.asarray(X, dtype=float)
        count = len(self.samples)
        segment_count = count if self.closed else count - 1
        if segment_count <= 0:
            projection = np.repeat(self.samples[:1], len(X), axis=0)
            residual = X - projection
            return projection, np.zeros(len(X)), np.sum(residual * residual, axis=1)
        starts = self.samples[:segment_count]
        ends = self.samples[1:segment_count + 1]
        if self.closed:
            ends = np.vstack([ends, self.samples[0]])
        vectors = ends - starts
        denominators = np.sum(vectors * vectors, axis=1)
        denominators[denominators < 1e-15] = 1.0

        best_d2 = np.full(len(X), np.inf)
        best_projection = np.zeros_like(X)
        best_t = np.zeros(len(X))
        for batch_start in range(0, len(X), max(1, int(batch_size))):
            batch_stop = min(batch_start + max(1, int(batch_size)), len(X))
            batch = X[batch_start:batch_stop]
            batch_best_d2 = np.full(len(batch), np.inf)
            batch_best_projection = np.zeros_like(batch)
            batch_best_t = np.zeros(len(batch))
            for index in range(segment_count):
                offset = batch - starts[index]
                alpha = np.sum(offset * vectors[index], axis=1) / denominators[index]
                alpha = np.clip(alpha, 0.0, 1.0)
                candidate = starts[index] + alpha[:, None] * vectors[index]
                d2 = np.sum((batch - candidate) ** 2, axis=1)
                improved = d2 < batch_best_d2
                batch_best_d2[improved] = d2[improved]
                batch_best_projection[improved] = candidate[improved]
                if self.closed:
                    t = (index + alpha) / segment_count
                else:
                    t = self.t_values[index] + alpha * (
                        self.t_values[index + 1] - self.t_values[index]
                    )
                batch_best_t[improved] = t[improved]
            best_d2[batch_start:batch_stop] = batch_best_d2
            best_projection[batch_start:batch_stop] = batch_best_projection
            best_t[batch_start:batch_stop] = batch_best_t
        return best_projection, best_t, best_d2

def _fit_curve(points: Array, closed: bool, smoothing: float, sample_count: int) -> _SplineRoute:
    points = np.asarray(points, dtype=float)
    if closed and len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 2:
        repeated = np.repeat(points, 2, axis=0)
        return _SplineRoute(repeated, np.array([0.0, 1.0]), False, backend="degenerate")

    differences = np.diff(points, axis=0)
    if closed:
        differences = np.vstack([differences, points[0] - points[-1]])
    segment_lengths = np.linalg.norm(differences, axis=1)
    if np.sum(segment_lengths) <= 1e-12:
        return _SplineRoute(
            np.repeat(points[:1], 2, axis=0),
            np.array([0.0, 1.0]),
            False,
            backend="degenerate",
        )
    if closed:
        t = np.concatenate([[0.0], np.cumsum(segment_lengths[:-1])]) / np.sum(segment_lengths)
    else:
        t = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        t /= t[-1]

    tck = None
    backend = "numpy"
    # FITPACK's parametric spline implementation supports at most ten
    # coordinate dimensions.  High-dimensional embeddings intentionally use
    # the deterministic NumPy fallback without producing one warning per
    # route.
    scipy_supported_dimension = points.shape[1] < 11
    if splprep is not None and len(points) >= 3 and scipy_supported_dimension:
        degree = min(3, len(points) - 1)
        smoothing_factor = max(0.0, float(smoothing)) * len(points)
        for _ in range(8):
            try:
                # FITPACK can report that a small requested smoothing value is
                # numerically unattainable for a short chain. Increase the
                # value for that chain only; other RuntimeWarnings remain
                # visible and ordinary fitting errors still use the fallback.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'error',
                        message='A theoretically impossible result when finding a smoothing spline.*',
                        category=RuntimeWarning,
                    )
                    tck, _ = splprep(
                        points.T,
                        u=t,
                        s=smoothing_factor,
                        per=closed,
                        k=degree,
                    )
                break
            except RuntimeWarning:
                smoothing_factor = max(1e-6, 2.0 * smoothing_factor)
                tck = None
            except Exception as exc:  # noqa: BLE001 - backend failure uses explicit fallback
                warnings.warn(
                    f"SciPy spline fitting failed; using the NumPy fallback: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                tck = None
                break
        if tck is not None:
            backend = "scipy"
    count = max(32, int(sample_count))
    sample_t = np.linspace(0.0, 1.0, count, endpoint=not closed)
    if tck is not None and splev is not None:
        candidate_samples = np.asarray(splev(sample_t, tck)).T
        control_min = np.min(points, axis=0)
        control_max = np.max(points, axis=0)
        control_span = np.maximum(control_max - control_min, 1e-8)
        margin = np.maximum(0.25 * control_span, 1e-3)
        has_overshoot = np.any(candidate_samples < control_min - margin) or np.any(
            candidate_samples > control_max + margin
        )
        if np.all(np.isfinite(candidate_samples)) and not has_overshoot:
            samples = candidate_samples
        else:
            tck = None
            backend = "numpy-overshoot-fallback"
    if tck is None:
        samples = _catmull_rom(points, sample_t, closed)
    if not closed:
        # Smoothing splines can pull their endpoints away from the graph
        # nodes.  Blend that correction over several samples so the route
        # reaches the node without creating a sharp one-segment kink.
        if tck is not None:
            window = min(0.12, max(0.06, 8.0 / max(count - 1, 1)))
            start_weight = 1.0 - np.clip(sample_t / window, 0.0, 1.0) ** 2 * (
                3.0 - 2.0 * np.clip(sample_t / window, 0.0, 1.0)
            )
            end_distance = 1.0 - sample_t
            end_weight = 1.0 - np.clip(end_distance / window, 0.0, 1.0) ** 2 * (
                3.0 - 2.0 * np.clip(end_distance / window, 0.0, 1.0)
            )
            samples = samples + start_weight[:, None] * (points[0] - samples[0])
            samples = samples + end_weight[:, None] * (points[-1] - samples[-1])
            # The sampled curve now includes the endpoint constraints.  Use
            # the dense corrected samples for both plotting and evaluation.
            tck = None
        samples[0] = points[0]
        samples[-1] = points[-1]
    return _SplineRoute(samples, sample_t, closed, tck, backend=backend)
