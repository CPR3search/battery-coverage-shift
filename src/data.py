"""Dataset adapters selected by the config field `data.type`.

Every dataset exposes `.load() -> (X, y, groups)`. `SyntheticDataset` serves the
smoke test, `CSVSubjectDataset` loads per-subject CSVs, and `batterylife` loads the
BatteryLife corpus."""
from pathlib import Path

import numpy as np


def build_dataset(cfg: dict):
    cfg = dict(cfg)
    t = cfg.pop("type", "synthetic")
    table = {"synthetic": SyntheticDataset, "csv": CSVSubjectDataset}
    if t == "batterylife":  # imported lazily
        from .data_batterylife import BatteryLifeDataset
        return BatteryLifeDataset(**cfg)
    if t not in table:
        raise ValueError(
            f"unknown data.type '{t}'; options: {sorted([*table, 'batterylife'])}")
    return table[t](**cfg)


class CSVSubjectDataset:
    """Load one CSV per subject, one sample per row.

    The subject id comes from the filename stem, or from `subject_col` if set.

    Config (data block):
      type: csv
      data_dir: data           # folder of *.csv
      label_col: label         # name of the label column
      pattern: "*.csv"
      subject_col: null        # if set, read subject id from this column instead
                               # of the filename
      drop_cols: []            # columns to ignore (e.g. timestamp)
      feature_cols: null       # explicit feature list; else all remaining columns

    Raw time series must be windowed beforehand so that each row is one sample.
    """

    def __init__(self, data_dir="data", label_col="label", pattern="*.csv",
                 subject_col=None, drop_cols=None, feature_cols=None):
        self.data_dir = Path(data_dir)
        self.label_col = label_col
        self.pattern = pattern
        self.subject_col = subject_col
        self.drop_cols = list(drop_cols or [])
        self.feature_cols = feature_cols

    def load(self):
        import pandas as pd

        files = sorted(self.data_dir.glob(self.pattern))
        if not files:
            raise FileNotFoundError(
                f"no CSVs matching {self.pattern} in {self.data_dir.resolve()}")

        X_parts, y_parts, subj_parts = [], [], []
        feat_cols = self.feature_cols
        for f in files:
            df = pd.read_csv(f)
            if self.label_col not in df.columns:
                raise KeyError(f"{f.name}: no label column '{self.label_col}' "
                               f"(found {list(df.columns)})")
            # subject id: from a column, else from the filename stem
            if self.subject_col:
                subj = df[self.subject_col].astype(str).to_numpy()
                ignore = {self.label_col, self.subject_col, *self.drop_cols}
            else:
                subj = np.array([f.stem] * len(df))
                ignore = {self.label_col, *self.drop_cols}
            if feat_cols is None:
                feat_cols = [c for c in df.columns if c not in ignore]
            X_parts.append(df[feat_cols].to_numpy(dtype=float))
            y_parts.append(df[self.label_col].to_numpy())
            subj_parts.append(subj)

        X = np.concatenate(X_parts, 0)
        y_raw = np.concatenate(y_parts, 0)
        subjects = np.concatenate(subj_parts, 0)
        # encode labels to ints if they are strings/categorical
        if y_raw.dtype.kind in "OUS":
            classes, y = np.unique(y_raw, return_inverse=True)
            self.classes_ = classes
        else:
            y = y_raw.astype(int)
        # map subject strings to compact ints for the split helpers
        _, subjects = np.unique(subjects, return_inverse=True)
        return X, y, subjects


class SyntheticDataset:
    """Synthetic binary classification data with a per-subject feature offset."""

    def __init__(self, n_subjects: int = 8, per_subject: int = 60, dim: int = 16,
                 seed: int = 0):
        self.n_subjects, self.per_subject, self.dim, self.seed = (
            n_subjects, per_subject, dim, seed)

    def load(self):
        rng = np.random.default_rng(self.seed)
        X, y, subj = [], [], []
        for s in range(self.n_subjects):
            # Per-subject offset simulates subject shift.
            offset = rng.normal(0, 0.5, self.dim)
            for _ in range(self.per_subject):
                label = rng.integers(0, 2)
                signal = rng.normal(label, 1.0, self.dim) + offset
                X.append(signal); y.append(label); subj.append(s)
        return np.array(X), np.array(y), np.array(subj)
