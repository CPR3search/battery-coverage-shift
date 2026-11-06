
Code for the manuscript *"Coverage Loss of Conformal Prediction Intervals for Battery Cycle Life Under Dataset Shift"*.

The study measures whether conformal interval coverage transfers to held-out datasets, using cycle life predicted from the first 100 cycles.

## Installation

Python 3.13.5. Versions are pinned in `requirements.txt`.

```bash
git clone https://github.com/CPR3search/battery-coverage-shift.git
cd battery-coverage-shift
python -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

Two extras are installed only when needed. `huggingface_hub` is for downloading the data.

## Data
[BatteryLife_Processed](https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed)
```bash
export BATTERY_ROOT=$PWD
python -m src.download_batterylife --tier pilot
python -m src.download_batterylife --tier extra
```

`--tier extra` also fetches Stanford and SDU, which the paper does not use. `--tier heldout` fetches the external test datasets.

`BATTERY_ROOT` holds `BatteryLife_Processed/` and `checkpoints/`.

## Experiments

Trains every predictor; it needs the data
(`--tier pilot`, `extra` and `heldout` above) and `torch` for CPMLP.

Development evaluation (leave one dataset out, 9 held-out datasets, 5 seeds):

```bash
make lodo LODO_CONFIG=configs/pilot3_untuned.yaml    LODO_RUN=runs/pilot3_untuned     # ridge
make lodo LODO_CONFIG=configs/pilot3_tuned.yaml      LODO_RUN=runs/pilot3_tuned       # model search
make lodo LODO_CONFIG=configs/final_cpmlp.yaml       LODO_RUN=runs/final_cpmlp        # CPMLP (GPU)
make lodo LODO_CONFIG=configs/sens_window_ridge.yaml LODO_RUN=runs/sens_window_ridge  # capacity-reference sensitivity
make lodo LODO_CONFIG=configs/sens_window_tuned.yaml LODO_RUN=runs/sens_window_tuned
```

External test (sodium-ion, zinc and CALB held out in turn). Each tag can be evaluated once.

```bash
python -m src.final_test_chemistry --tag my_rerun --out-root runs/rerun_chemistry
python -m src.final_test_chemistry --tag my_rerun_window --out-root runs/rerun_chemistry_window \
    --capacity-reference window
```

Then point the report at the new runs, for example
`make diag-numbers ARM_UNTUNED=runs/pilot3_untuned CHEM_UNTUNED=runs/rerun_chemistry_ridge ...`.
Checkpoints are written under `$BATTERY_ROOT/checkpoints`.
