"""CPMLP life predictor on raw early-cycle curves.

Architecture and curve preprocessing follow BatteryLife (Tan et al., MIT licence) with its
published Li-ion settings. The target is standardized base-ten log life, and early stopping
monitors a held-out tenth of the training cells. The configuration is fixed, not tuned.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

CURVE_LENGTH = 300
WINDOW = 100
PUBLISHED = {"d_model": 32, "d_ff": 256, "e_layers": 12, "d_layers": 7, "dropout": 0.0,
             "learning_rate": 5e-5, "batch_size": 32, "max_epochs": 100, "patience": 5}
# BatteryLife data_loader.py: datasets that discharge before they charge within a cycle.
_DISCHARGE_FIRST = ("RWTH", "CALB_0", "CALB_25", "CALB_45")
_ZN_CHARGE_FIRST = {
    "ZN-coin_402-1_20231209225636_01_1.pkl", "ZN-coin_402-2_20231209225727_01_2.pkl",
    "ZN-coin_402-3_20231209225844_01_3.pkl", "ZN-coin_403-1_20231209225922_01_4.pkl",
    "ZN-coin_428-1_20231212185048_01_2.pkl", "ZN-coin_428-2_20231212185058_01_4.pkl",
    "ZN-coin_429-1_20231212185129_01_5.pkl", "ZN-coin_429-2_20231212185157_01_8.pkl",
    "ZN-coin_430-1_20231212185250_02_6.pkl", "ZN-coin_430-2_20231212185305_02_7.pkl",
    "ZN-coin_430-3_20231212185323_03_2.pkl"}


def _nominal(file_name: str, cell: dict) -> float:
    if file_name.startswith("RWTH"):
        return 1.85
    if file_name.startswith("SNL_18650_NCA_25C_20-80"):
        return 3.2
    return float(cell["nominal_capacity_in_Ah"])


def _discharge_first(file_name: str) -> bool:
    prefix = file_name.split("_")[0]
    if prefix == "CALB":
        prefix = "_".join(file_name.split("_")[:2])
    return prefix in _DISCHARGE_FIRST or (prefix == "ZN-coin" and file_name not in _ZN_CHARGE_FIRST)


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    """Index-based interpolation, including the official grid's end point one past the last sample."""
    raw = np.arange(1, len(values) + 1)
    return np.interp(np.linspace(1, len(values) + 1, num=length, endpoint=True), raw, values)


def _clean(cycle: dict) -> dict:
    """Back-fill rows with negative capacity; drop trailing ones, as the official loader does."""
    keys = ("voltage_in_V", "current_in_A", "charge_capacity_in_Ah", "discharge_capacity_in_Ah")
    rows = np.column_stack([np.asarray(cycle[k], float) for k in keys])
    bad = (rows[:, 2] < 0) | (rows[:, 3] < 0)
    if bad.any():
        good = np.flatnonzero(~bad)
        if good.size == 0:
            rows = rows[:0]
        else:
            rows = rows[:good[-1] + 1]
            nxt = good[np.searchsorted(good, np.arange(len(rows)))]
            rows = rows[nxt]
    return dict(zip(keys, rows.T))


def cycle_curve(cycle: dict, nominal: float, discharge_first: bool,
                length: int = CURVE_LENGTH) -> np.ndarray | None:
    """One cycle as [3, length]: voltage, C-rate, capacity. None if a half is absent."""
    c = _clean(cycle)
    v, i, qc, qd = (c["voltage_in_V"], c["current_in_A"], c["charge_capacity_in_Ah"],
                    c["discharge_capacity_in_Ah"])
    if v.size == 0:
        return None
    rate = i / nominal
    if not (rate >= 0.01).any() or not (rate <= -0.01).any():
        return None
    if discharge_first:
        end = int(np.flatnonzero(rate <= -0.01)[-1])
        keep = np.abs(rate[end:]) > 0.01
        dis = (v[:end], i[:end], qd[:end])
        cha = (v[end:][keep], i[end:][keep], qc[end:][keep])
    else:
        end = int(np.flatnonzero(rate >= 0.01)[-1])
        keep = np.abs(rate[end:]) > 0.01
        cha = (v[:end], i[:end], qc[:end])
        dis = (v[end:][keep], i[end:][keep], qd[end:][keep])
    if min(len(cha[0]), len(dis[0])) == 0:
        return None
    half = length // 2
    volt, curr, cap = (np.concatenate([_resample(cha[k], half), _resample(dis[k], half)])
                       for k in range(3))
    return np.stack([volt / volt.max(), curr / nominal, cap / nominal])


def cell_curves(path: Path, window: int = WINDOW, length: int = CURVE_LENGTH) -> tuple[np.ndarray, int]:
    """[window, 3, length] for one cell, and the number of cycles filled with zeros."""
    with Path(path).open("rb") as fh:
        cell = pickle.load(fh)
    name = Path(path).name
    nominal = _nominal(name, cell)
    first = _discharge_first(name)
    out = np.zeros((window, 3, length), np.float32)
    missing = 0
    cycles = cell["cycle_data"]
    for k in range(window):
        curve = cycle_curve(cycles[k], nominal, first, length) if k < len(cycles) else None
        if curve is None or not np.isfinite(curve).all():
            missing += 1
            continue
        out[k] = curve
    return out, missing


def load_curves(root: Path, domains: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, int]:
    """Curves aligned to the dataset's rows, with the total count of zero-filled cycles."""
    out, missing = [], 0
    for dom, cell in zip(domains, cells):
        arr, m = cell_curves(Path(root) / str(dom) / f"{cell}.pkl")
        out.append(arr)
        missing += m
    return np.stack(out), missing


class _Block(nn.Module):
    """BatteryLife MLPBlock: residual two-layer MLP with both dropouts and a final LayerNorm."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.a = nn.Linear(d_model, d_ff)
        self.drop = nn.Dropout(dropout)
        self.b = nn.Linear(d_ff, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(self.drop(self.b(self.drop(self.a(x).relu()))) + x)


class _Net(nn.Module):
    """BatteryLife CPMLP: per-cycle MLP blocks, learned pooling over cycles, then MLP blocks."""

    def __init__(self, length, window, d_model, d_ff, e_layers, d_layers, dropout) -> None:
        super().__init__()
        self.embed = nn.Linear(length * 3, d_model)
        self.intra = nn.ModuleList(_Block(d_model, d_ff, dropout) for _ in range(e_layers))
        self.pool = nn.Linear(window * d_model, d_model)
        self.inter = nn.ModuleList(_Block(d_model, d_ff, dropout) for _ in range(d_layers))
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.embed(x.flatten(2))
        for blk in self.intra:
            h = blk(h)
        h = self.pool(h.flatten(1))
        for blk in self.inter:
            h = blk(h)
        return self.head(h.relu()).squeeze(-1)


class CPMLPRegressor:
    """Fit on [N, window, 3, length] curves and predict log life, with sklearn's call shape."""

    def __init__(self, seed: int, device: str | None = None, **overrides) -> None:
        self.seed = seed
        self.device = device
        self.config = {**PUBLISHED, **overrides}

    def _network(self, length: int, window: int):
        c = self.config
        return _Net(length, window, c["d_model"], c["d_ff"], c["e_layers"], c["d_layers"], c["dropout"])

    def _device(self) -> str:
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y, groups):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        c, device = self.config, self._device()
        y = np.asarray(y, float)
        groups = np.asarray(groups)
        unique = rng.permutation(np.unique(groups))
        n_stop = max(1, int(round(0.1 * len(unique))))
        stop = np.isin(groups, unique[:n_stop])
        fit = ~stop
        self.mean_, self.std_ = float(y[fit].mean()), float(y[fit].std())
        window = X.shape[1]
        net = self._network(X.shape[3], window).to(device)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        yt = torch.as_tensor((y - self.mean_) / self.std_, dtype=torch.float32, device=device)
        fit_idx, stop_idx = np.flatnonzero(fit), np.flatnonzero(stop)
        opt = torch.optim.Adam(net.parameters(), lr=c["learning_rate"])
        best, best_state, wait = float("inf"), None, 0
        self.history_ = []
        for epoch in range(c["max_epochs"]):
            net.train()
            # One sample per cell and window prefix, with unseen cycles zeroed (as in BatteryLife).
            cell = np.repeat(fit_idx, window)
            seen = np.tile(np.arange(1, window + 1), len(fit_idx))
            order = rng.permutation(len(cell))
            for start in range(0, len(order), c["batch_size"]):
                b = order[start:start + c["batch_size"]]
                xb = Xt[torch.as_tensor(cell[b], device=device)].clone()
                mask = (torch.arange(window, device=device)[None] <
                        torch.as_tensor(seen[b], device=device)[:, None])
                xb *= mask[:, :, None, None]
                loss = torch.nn.functional.mse_loss(net(xb), yt[torch.as_tensor(cell[b], device=device)])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pred = net(Xt[torch.as_tensor(stop_idx, device=device)]).cpu().numpy()
            life, life_hat = 10 ** y[stop_idx], 10 ** (pred * self.std_ + self.mean_)
            mape = float(np.mean(np.abs(life_hat - life) / life))
            self.history_.append(mape)
            if mape < best:
                best, wait = mape, 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()})
            else:
                wait += 1
                if wait >= c["patience"]:
                    break
        net.load_state_dict(best_state)
        self.net_ = net.cpu().eval()
        self.best_stop_mape_ = best
        self.epochs_ = len(self.history_)
        return self

    def predict(self, X):
        device = self._device()
        net = self.net_.to(device)
        out = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                xb = torch.as_tensor(X[start:start + 256], dtype=torch.float32, device=device)
                out.append(net(xb).cpu().numpy())
        self.net_ = net.cpu()
        return np.concatenate(out).astype(float) * self.std_ + self.mean_
