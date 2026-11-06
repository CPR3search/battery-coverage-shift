"""Reference model implementing the fit(X, y, subjects) / predict(X) interface.

Normalization statistics are fitted on training data only.
"""
import numpy as np

from .registry import register


@register("nearest_centroid")
class NearestCentroid:
    """Dependency-free nearest-centroid classifier used by the smoke test."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.mu = self.sd = None
        self.centroids = {}

    def _norm(self, X, fit):
        if not self.normalize:
            return X
        if fit:  # statistics come from training data only
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        return (X - self.mu) / self.sd

    def fit(self, X, y, subjects=None):
        Xn = self._norm(X, fit=True)
        self.centroids = {c: Xn[y == c].mean(0) for c in np.unique(y)}
        return self

    def predict(self, X):
        Xn = self._norm(X, fit=False)
        classes = sorted(self.centroids)
        d = np.stack([((Xn - self.centroids[c]) ** 2).sum(1) for c in classes], 1)
        return np.array(classes)[d.argmin(1)]
