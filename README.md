
Code and records for the manuscript *"Coverage Loss of Conformal Prediction Intervals for Battery Cycle Life Under Dataset Shift"*.

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

