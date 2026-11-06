"""Feature scaling and prediction clipping for cross-domain use.

`RobustFeatureScaler` floors the scale of near-constant features so an unseen value
cannot produce extreme standardized values. `domain_constant_features` flags
features that are constant within every domain and would therefore act as a domain
label. The scaler and the prediction clip are fit on training data only;
`domain_constant_features` uses no labels but is transductive if given target
domains.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["RobustFeatureScaler", "RangeClippedRegressor", "domain_constant_features"]


class RobustFeatureScaler(BaseEstimator, TransformerMixin):
    """Median/IQR scaling with a scale floor and optional output clipping.

    Parameters
    ----------
    scale_floor
        Minimum usable spread, as a fraction of the feature's absolute median.
        Features below this spread are centred but not scaled. Set to 0 to disable.
    clip
        If not None, standardized values are clipped to +-clip.
    """

    def __init__(self, scale_floor: float = 1e-3, clip: float | None = 10.0) -> None:
        self.scale_floor = scale_floor
        self.clip = clip

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.center_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        # IQR to standard-deviation equivalent under normality
        scale = iqr / 1.349
        # Fall back to std where the IQR is degenerate (e.g. more than 50% ties)
        bad = ~np.isfinite(scale) | (scale <= 0)
        if np.any(bad):
            scale[bad] = X[:, bad].std(axis=0)
        floor = self.scale_floor * np.maximum(np.abs(self.center_), 1.0)
        self.degenerate_ = ~np.isfinite(scale) | (scale <= floor)
        scale = np.where(self.degenerate_, 1.0, scale)
        self.scale_ = scale
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        Z = (X - self.center_) / self.scale_
        if self.clip is not None:
            Z = np.clip(Z, -self.clip, self.clip)
        return Z


class RangeClippedRegressor(BaseEstimator):
    """Wrap a regressor and clip predictions to the observed training target range.

    The range is [min, max] of the training targets, widened by `margin` in the
    target's units. Implemented as a wrapper so calibration residuals and target
    predictions come from the same clipped predictor.
    """

    def __init__(self, estimator, margin: float = 0.1) -> None:
        self.estimator = estimator
        self.margin = margin

    def fit(self, X, y, **fit_params):
        """Fit the wrapped estimator, forwarding fit params (e.g. `groups=` for grouped CV)."""
        y = np.asarray(y, float)
        self.estimator_ = self.estimator.fit(X, y, **fit_params)
        self.y_min_ = float(np.min(y)) - self.margin
        self.y_max_ = float(np.max(y)) + self.margin
        return self

    def predict(self, X):
        """Predict and clip; `n_clipped_` records the count for the latest call only.

        Use `count_clipped` to obtain the count for a specific input.
        """
        p = np.asarray(self.estimator_.predict(X), float)
        self.n_clipped_ = self.count_clipped(X, _raw=p)
        return np.clip(p, self.y_min_, self.y_max_)

    def count_clipped(self, X, _raw=None) -> int:
        """Return how many predictions for `X` fall outside the fitted training range."""
        p = np.asarray(self.estimator_.predict(X), float) if _raw is None else _raw
        return int(np.sum((p < self.y_min_) | (p > self.y_max_)))


def domain_constant_features(X, domains, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of features that are constant within every domain.

    Such a feature carries no within-domain information but can identify the domain.
    """
    X = np.asarray(X, float)
    domains = np.asarray(domains)
    mask = np.ones(X.shape[1], dtype=bool)
    for d in np.unique(domains):
        sub = X[domains == d]
        if len(sub) < 2:
            continue
        spread = sub.max(axis=0) - sub.min(axis=0)
        mask &= spread <= tol * np.maximum(np.abs(np.median(sub, axis=0)), 1.0)
    return mask
