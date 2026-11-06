"""Adapter for the BatteryLife_Processed corpus (HF, MIT licence).

One pickle per cell:
    {cell_id, form_factor, anode_material, cathode_material, electrolyte_material,
     nominal_capacity_in_Ah, depth_of_charge, depth_of_discharge,
     already_spent_cycles, max/min_voltage_limit_in_V, max/min_current_limit_in_A,
     charge_protocol, discharge_protocol, SOC_interval,
     cycle_data: [ {cycle_number, current_in_A, voltage_in_V,
                    charge_capacity_in_Ah, discharge_capacity_in_Ah, time_in_s,
                    temperature_in_C, internal_resistance_in_ohm, Qdlin}, ... ]}

Labels live in `Life labels/<SUBSET>_labels.json` as {filename: cycle_life}.

Only the first S cycles of each cell are read and reduced to a feature vector, so
no downstream code sees later cycles. Aging-factor metadata is extracted alongside
the features for `conformal.PhysicsWeightedConformal` but kept out of X, so the
predictor cannot read the chemistry label directly. The feature cache key includes
`FEATURE_VERSION`.
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.paths import resolve

CAPACITY_REFERENCES = ("nominal", "window")
FEATURE_VERSION = "v4"



def feature_version(capacity_reference: str = "nominal") -> str:
    """Return the feature-definition tag for the given capacity reference."""
    return FEATURE_VERSION if capacity_reference == "nominal" else f"{FEATURE_VERSION}-window"


# Cross-domain features use only channels present in every subset (voltage,
# discharge capacity, current, time). Temperature and internal resistance are
# missing in several subsets, and their presence alone would identify the source
# dataset, so they are available only as optional within-domain features.

# Capacities are normalised by nominal capacity: absolute capacity spans roughly
# 0.045 Ah to 29 Ah, so raw Ah would act as a proxy for dataset identity.
FEATURE_NAMES = [
    "log_var_dQ",        # log10 var(Q(v)_S - Q(v)_ref) -- the Severson feature
    "log_absmin_dQ",     # log10 |min dQ(v)|
    "log_absmean_dQ",    # log10 |mean dQ(v)|
    "soh_first",         # normalised discharge capacity, first cycle in window
    "soh_last",          # normalised discharge capacity, cycle S
    "soh_fade",          # soh_first - soh_last
    "soh_slope",         # slope of normalised capacity vs cycle number
    "soh_ratio",         # soh_last / soh_first
    "v_mean",            # mean voltage over the window (V)
    "v_range_mean",      # mean per-cycle (max - min) voltage (V)
    "cycle_time_mean",   # mean cycle duration (s)
    "cycle_time_slope",  # slope of cycle duration vs cycle number
]

OPTIONAL_FEATURE_NAMES = ["temp_mean", "temp_max", "ir_mean", "ir_slope"]

# Design temperature per cell, in priority order: the Tongji `_CY<T>-` token, the
# CALB `CALB_<T>_` token, the filename token immediately before the SOC-window token
# (e.g. `..._NMC_-5C_0-100_...`), and finally a per-subset constant from the subset
# README. A generic `_(\d+)C_` pattern is not used because it also matches cell IDs
# (`MICH_02C_...`) and C-rates (`XJTU_2C_...`). The measured temperature channel is
# not used because it is absent or all-NaN in several subsets.
SUBSET_DESIGN_TEMP_C = {
    "MATR": 30.0,     # README: "working temperature of 30 degrees Celsius"
    "HUST": 30.0,     # README: "constant temperature of 30 degrees Celsius"
    "CALCE": 25.0,    # README: "temperature of 25 degrees Celsius"
    "RWTH": 25.0,     # README: "temperature of 25 degrees Celsius"
    "XJTU": 25.0,     # README: "cycled to failure at room temperature" (approximate)
    "NA-ion": 25.0,   # README: 12 protocols "at 25 degrees Celsius"
    "ZN-coin": 25.0,  # README: same protocol "at 25 degrees Celsius"
}
SUBSETS_TEMP_APPROXIMATE = {"XJTU"}   # room temperature, not a stated setpoint

_SOC_WINDOW = re.compile(r"^\d+-\d+$")
_TEMP_TOKEN = re.compile(r"^(-?\d+(?:\.\d+)?)C$")
_TONGJI_TEMP = re.compile(r"_CY(\d+)-")
# CALB_<T>_B<id>.pkl with T in {0, 25, 35, 45}; the token is the cycling temperature
# stated in the subset README, not a batch index.
_CALB_TEMP = re.compile(r"^CALB_(\d+)_")


def _parse_design_temp(filename: str, subset: str) -> tuple[float | None, str]:
    """Return (temperature in C, provenance tag) for one cell; None if unknown."""
    m = _TONGJI_TEMP.search(filename)
    if m:
        return float(m.group(1)), "filename:CY"
    if subset == "CALB":
        m = _CALB_TEMP.match(filename)
        if m:
            return float(m.group(1)), "filename:CALB"

    # BatteryArchive-style: the token just before the SOC window is the temperature.
    tokens = Path(filename).stem.split("_")
    for i, tok in enumerate(tokens):
        if _SOC_WINDOW.match(tok) and i > 0:
            tm = _TEMP_TOKEN.match(tokens[i - 1])
            if tm:
                return float(tm.group(1)), "filename:pre-SOC"

    if subset in SUBSET_DESIGN_TEMP_C:
        prov = ("readme:approx" if subset in SUBSETS_TEMP_APPROXIMATE
                else "readme:setpoint")
        return SUBSET_DESIGN_TEMP_C[subset], prov
    return None, "unknown"

# Aging factors used by the physical distance. Numeric ones need a range for Gower.
AGING_FACTOR_NUMERIC_RANGES = {
    "nominal_capacity_in_Ah": 30.0,   # spans coin cells (~0.045) to large format (~29)
    "design_temp_C": 60.0,            # roughly -10..50 C across the corpus
    "max_voltage_limit_in_V": 2.0,
    "depth_of_discharge": 1.0,
}


@dataclass
class BatteryData:
    """Feature matrix plus the grouping and metadata the protocol needs."""
    X: np.ndarray                      # (n_cells, n_features)
    y: np.ndarray                      # (n_cells,) cycle life
    cells: np.ndarray                  # (n_cells,) cell id -- the grouping unit
    domains: np.ndarray                # (n_cells,) sub-dataset name, for LODO
    meta: list[dict] = field(default_factory=list)  # aging factors, per cell
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def __len__(self) -> int:
        return len(self.y)


# The corpus spells the same electrode material several ways ('LFP', 'LiFePO4'), and a
# categorical Gower term counts any string mismatch as a full unit of distance. Known
# spellings map to a canonical token; unrecognised values pass through unchanged.
_MATERIAL_CANON = {
    "lifepo4": "LFP", "lfp": "LFP",
    "nmc": "NMC", "ncm": "NMC", "limn2o4": "LMO", "lmo": "LMO",
    "nca": "NCA", "licoo2": "LCO", "lco": "LCO",
    "graphite": "graphite", "carbon": "graphite", "c": "graphite",
    "mno2": "MnO2", "zinc": "Zn", "zn": "Zn",
    "unknown": None, "": None,
}


def _canon_material(v):
    if v is None:
        return None
    key = str(v).strip().lower().replace(" ", "").replace("-", "")
    return _MATERIAL_CANON.get(key, str(v).strip())


def _safe_log10(x: float) -> float:
    """Return log10 of a magnitude, floored at 1e-12 so exact zeros do not yield -inf."""
    return float(np.log10(max(abs(x), 1e-12)))


QV_GRID_POINTS = 128


def _qv_curve(cycle: dict, grid: np.ndarray) -> np.ndarray | None:
    """Interpolate discharge capacity onto a normalised voltage grid, Q(v_norm).

    Computed for every subset instead of using `Qdlin`, which only some subsets
    provide. Voltage is min-max normalised per cycle so that chemistries with
    different voltage windows share one grid. Returns None when no usable
    discharge segment exists.
    """
    v = np.asarray(cycle.get("voltage_in_V", []), float)
    q = np.asarray(cycle.get("discharge_capacity_in_Ah", []), float)
    if v.size < 4 or v.size != q.size:
        return None
    # Discharge is identified by increasing discharge capacity; the current sign
    # convention varies across subsets.
    inc = np.diff(q, prepend=q[0]) > 0
    v, q = v[inc], q[inc]
    ok = np.isfinite(v) & np.isfinite(q)
    v, q = v[ok], q[ok]
    if v.size < 4:
        return None
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if not np.isfinite(vmin) or vmax - vmin < 1e-6:
        return None
    vn = (v - vmin) / (vmax - vmin)
    # np.interp needs strictly ascending x: sort, and keep one capacity per duplicate voltage.
    order = np.argsort(vn)
    vn, qs = vn[order], q[order]
    uniq, idx = np.unique(vn, return_index=True)
    if uniq.size < 4:
        return None
    qq = np.maximum.reduceat(qs, idx) if idx.size == uniq.size else qs[idx]
    out = np.interp(grid, uniq, qq)
    return out if np.all(np.isfinite(out)) else None


def _cell_features(cyc: list[dict], n_cycles: int, ref_cycle: int,
                   nominal_capacity: float | None = None) -> np.ndarray | None:
    """Reduce the first `n_cycles` cycles of one cell to a feature vector.

    Returns None if the cell is too short or lacks required fields; such cells are
    skipped by the caller, not imputed.
    """
    if len(cyc) < n_cycles:
        return None
    win = cyc[:n_cycles]

    # Severson-style delta-Q(V) between the window end and a reference cycle
    ref_idx = min(ref_cycle, n_cycles - 1)
    grid = np.linspace(0.0, 1.0, QV_GRID_POINTS)
    q_end = _qv_curve(win[n_cycles - 1], grid)
    q_ref = _qv_curve(win[ref_idx], grid)
    if q_end is None or q_ref is None:
        return None
    dq = q_end - q_ref
    if dq.size == 0 or not np.all(np.isfinite(dq)):
        return None

    # Per-cycle scalars from universal channels only
    caps, vmeans, vranges, durs, nums = [], [], [], [], []
    for c in win:
        dc = np.asarray(c.get("discharge_capacity_in_Ah", []), float)
        dc = dc[np.isfinite(dc)]
        caps.append(float(np.max(dc)) if dc.size else np.nan)
        v = np.asarray(c.get("voltage_in_V", []), float)
        v = v[np.isfinite(v)]
        vmeans.append(float(np.mean(v)) if v.size else np.nan)
        vranges.append(float(np.max(v) - np.min(v)) if v.size else np.nan)
        ts = np.asarray(c.get("time_in_s", []), float)
        ts = ts[np.isfinite(ts)]
        durs.append(float(ts[-1] - ts[0]) if ts.size > 1 else np.nan)
        nums.append(float(c.get("cycle_number", len(nums) + 1)))

    caps = np.asarray(caps); nums = np.asarray(nums)
    if not np.any(np.isfinite(caps)):
        return None
    # Normalise capacity to an SOH-like scale; fall back to the window maximum if
    # the nominal capacity is absent.
    denom = nominal_capacity if (nominal_capacity and nominal_capacity > 0) \
        else float(np.nanmax(caps))
    if not np.isfinite(denom) or denom <= 0:
        return None
    soh = caps / denom
    # Normalise the difference curve by the same denominator so the delta-Q features
    # are dimensionless and not dominated by rated capacity.
    dq = dq / denom

    def _slope(y: np.ndarray) -> float:
        ok = np.isfinite(y) & np.isfinite(nums)
        if ok.sum() < 2:
            return 0.0
        return float(np.polyfit(nums[ok], y[ok], 1)[0])

    fin = soh[np.isfinite(soh)]
    s_first, s_last = float(fin[0]), float(fin[-1])
    feats = [
        _safe_log10(float(np.var(dq))),
        _safe_log10(float(np.min(dq))),
        _safe_log10(float(np.mean(dq))),
        s_first,
        s_last,
        s_first - s_last,
        _slope(soh),
        s_last / s_first if s_first > 1e-9 else np.nan,
        float(np.nanmean(vmeans)) if np.any(np.isfinite(vmeans)) else np.nan,
        float(np.nanmean(vranges)) if np.any(np.isfinite(vranges)) else np.nan,
        float(np.nanmean(durs)) if np.any(np.isfinite(durs)) else np.nan,
        _slope(np.asarray(durs)),
    ]
    out = np.asarray(feats, float)
    return out if np.all(np.isfinite(out)) else None


def _capacity_denominator(obj: dict, reference: str = "nominal") -> float | None:
    """Return the capacity used to normalise the capacity features and delta-Q curve.

    `nominal` uses the recorded nominal capacity. `window` additionally scales it by
    the width of the record's SOC_interval (width 1 if missing or empty), the rule
    BatteryLife uses for its life labels.
    """
    nominal = obj.get("nominal_capacity_in_Ah")
    if reference == "nominal" or not nominal:
        return nominal
    interval = obj.get("SOC_interval")
    width = float(interval[1]) - float(interval[0]) if interval else 1.0
    return float(nominal) * (width if width > 0 else 1.0)


def _cell_meta(obj: dict, subset: str, filename: str = "") -> dict:
    """Return the aging factors used by the physical distance.

    Kept out of the feature matrix X; used only to weight conformal calibration.
    """
    temp, prov = _parse_design_temp(filename, subset)
    return {
        "cathode_material": _canon_material(obj.get("cathode_material")),
        "anode_material": _canon_material(obj.get("anode_material")),
        "form_factor": obj.get("form_factor"),
        "nominal_capacity_in_Ah": obj.get("nominal_capacity_in_Ah"),
        "max_voltage_limit_in_V": obj.get("max_voltage_limit_in_V"),
        "depth_of_discharge": obj.get("depth_of_discharge"),
        "design_temp_C": temp,
        "design_temp_provenance": prov,
    }


def load_labels(root: Path, subset: str) -> dict:
    """Load the label map for a subset, reconciling filename variants.

    Handles '_' versus '-' in the label-file name and '-#N' versus '--N' in keys.
    Exact keys take precedence; variant spellings only fill gaps.
    """
    raw = None
    for name in (subset, subset.replace("_", "-"), subset.replace("-", "_")):
        p = root / "Life labels" / f"{name}_labels.json"
        if p.exists():
            raw = json.loads(p.read_text())
            break
    if raw is None:
        raise FileNotFoundError(
            f"no labels JSON for subset '{subset}' under {root}")

    widened = dict(raw)
    for k, v in raw.items():
        for alt in (k.replace("-#", "--"), k.replace("--", "-#")):
            widened.setdefault(alt, v)
    return widened


class BatteryLifeDataset:
    """Loads one or more BatteryLife sub-datasets into a `BatteryData`.

    Config (data block):
        type: batterylife
        root: BatteryLife_Processed    # relative to $BATTERY_ROOT (src/paths.py)
        subsets: [MATR, HUST, ...]     # each becomes a domain for LODO
        n_cycles: 100                  # observation window S
        ref_cycle: 9                   # 0-based; cycle 10 in Severson's convention
        min_life: null                 # cells with life <= min_life are excluded; defaults to n_cycles
        cache_dir: null                # defaults to <root>/_feature_cache
        capacity_reference: nominal    # nominal (default) or window
    """

    def __init__(self, root: str, subsets: list[str] | None = None,
                 n_cycles: int = 100, ref_cycle: int = 9,
                 min_life: int | None = None,
                 cache_dir: str | None = None,
                 capacity_reference: str = "nominal") -> None:
        if capacity_reference not in CAPACITY_REFERENCES:
            raise ValueError(f"capacity_reference must be one of {CAPACITY_REFERENCES}")
        self.capacity_reference = capacity_reference
        self.root = resolve(root)
        self.subsets = list(subsets) if subsets else None
        self.n_cycles = int(n_cycles)
        self.ref_cycle = int(ref_cycle)
        # A cell that fails within the observation window is not a prediction target.
        self.min_life = int(min_life) if min_life is not None else self.n_cycles
        self.cache_dir = resolve(cache_dir) if cache_dir else self.root / "_feature_cache"
        self.skipped_: dict[str, int] = {}
        self.excluded_short_life_: dict[str, int] = {}

    # -- discovery -------------------------------------------------------
    def available_subsets(self) -> list[str]:
        skip = {"Life labels", "READMEs", "_feature_cache"}
        return sorted(d.name for d in self.root.iterdir()
                      if d.is_dir() and d.name not in skip
                      and not d.name.startswith((".", "_")))

    def _labels(self, subset: str) -> dict:
        return load_labels(self.root, subset)

    # -- loading ---------------------------------------------------------
    def _subset_fingerprint(self, subset: str) -> str:
        """Return a cheap fingerprint (file count and total bytes) for the cache key."""
        files = sorted((self.root / subset).glob("*.pkl"))
        total = sum(f.stat().st_size for f in files)
        return f"n{len(files)}b{total}"

    def _load_subset(self, subset: str) -> BatteryData:
        fp = self._subset_fingerprint(subset)
        cache = (self.cache_dir /
                 f"{subset}_S{self.n_cycles}_r{self.ref_cycle}_"
                 f"L{self.min_life}_{feature_version(self.capacity_reference)}_{fp}.npz")
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            return BatteryData(X=z["X"], y=z["y"], cells=z["cells"],
                               domains=z["domains"], meta=list(z["meta"]))

        labels = self._labels(subset)
        Xs, ys, cells, metas = [], [], [], []
        skipped = 0
        short_life = 0
        for pkl in sorted((self.root / subset).glob("*.pkl")):
            life = labels.get(pkl.name)
            if life is None:
                skipped += 1
                continue
            if float(life) <= self.min_life:
                short_life += 1
                continue
            with open(pkl, "rb") as fh:
                obj = pickle.load(fh)
            feats = _cell_features(obj.get("cycle_data", []), self.n_cycles,
                                   self.ref_cycle,
                                   _capacity_denominator(obj, self.capacity_reference))
            if feats is None:
                skipped += 1
                continue
            Xs.append(feats)
            ys.append(float(life))
            cells.append(obj.get("cell_id", pkl.stem))
            metas.append(_cell_meta(obj, subset, pkl.name))

        self.skipped_[subset] = skipped
        self.excluded_short_life_[subset] = short_life
        if not Xs:
            raise RuntimeError(
                f"subset '{subset}': no usable cells "
                f"(skipped {skipped}; is the download complete, and is "
                f"n_cycles={self.n_cycles} shorter than the shortest cell?)")

        data = BatteryData(
            X=np.vstack(Xs), y=np.asarray(ys, float),
            cells=np.asarray(cells), domains=np.asarray([subset] * len(ys)),
            meta=metas,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=data.X, y=data.y, cells=data.cells,
                            domains=data.domains, meta=np.array(data.meta, dtype=object))
        return data

    def load_full(self) -> BatteryData:
        subsets = self.subsets or self.available_subsets()
        parts = [self._load_subset(s) for s in subsets]
        # `skipped_` and `excluded_short_life_` are filled only on a cache miss, so
        # exclusions are counted against the files on disk instead.
        self.excluded_ = {}
        for s, part in zip(subsets, parts):
            on_disk = len(list((self.root / s).glob("*.pkl")))
            self.excluded_[s] = int(on_disk - len(part.y))
        self.n_excluded_ = int(sum(self.excluded_.values()))
        return BatteryData(
            X=np.vstack([p.X for p in parts]),
            y=np.concatenate([p.y for p in parts]),
            cells=np.concatenate([p.cells for p in parts]),
            domains=np.concatenate([p.domains for p in parts]),
            meta=[m for p in parts for m in p.meta],
        )

    def load(self):
        """Return (X, y, cells); `load_full` also returns domains and metadata."""
        d = self.load_full()
        return d.X, d.y, d.cells
