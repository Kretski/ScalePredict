# ScalePredict

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21842461.svg)](https://doi.org/10.5281/zenodo.21842461)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**W-Twin: Forecast-Based Detection of Progressive Neural Network Training Degradation**

ScalePredict monitors neural network training by comparing the observed loss trajectory against a scaling-law baseline (ScalePredict). W-Twin detects progressive degradation that is invisible to classical threshold and CUSUM detectors.

---

## The Problem

Current training monitors are **reactive** — they detect NaN losses, gradient explosions, or hardware failures *after* they occur. Progressive degradation (slowly increasing label noise, gradual weight corruption, subtle data pipeline issues) accumulates undetected until significant GPU compute has been wasted.

## The Solution

W-Twin compares each training step against an **expected trajectory** derived from scaling laws:

```
W(t) = Q(t) · (D(t) − α)

where:
  D(t) = (L_obs(t) − L_pred(t)) / σ_local(t)   ← baseline-normalized deviation
  Q(t) = exp(−MSE_fit / τ)                        ← baseline confidence
  α    = fixed z-score threshold (default: 2.0)
```

Alert fires when `W(t) > 0` for `n_consec` consecutive steps.

---

## Results

From the paper ([Zenodo 10.5281/zenodo.21842461](https://doi.org/10.5281/zenodo.21842461)):

| Experiment | Runs | W-Twin | Threshold | CUSUM |
|---|---|---|---|---|
| Progressive drift detection | 9 | **9/9 (100%)** | 0/9 | 0/9 |
| Mean detection delay | — | **223 ± 11 steps** | — | — |
| False alarm rate | 30 clean | **0/30 (0%)** | 0/30 | 0/30 |
| Abrupt failure (spike) | 2 | 2/2 (+5 steps) | 0/2 | 2/2 (+1 step, faster) |

W-Twin is the **only method that detects progressive drift**. For abrupt failures, CUSUM remains faster.

---

## Installation

```bash
pip install scalepredict
```

For training with PyTorch:
```bash
pip install scalepredict[train]
```

---

## Quick Start

### Python API

```python
from scalepredict.monitor import WTwinMonitor

monitor = WTwinMonitor(
    warmup_steps=100,   # skip LR warmup
    alpha=2.0,          # z-score threshold
    n_consec=5,         # consecutive steps for alert
)

# Stream step-by-step during training
for step, loss in training_loop():
    state = monitor.update(step, loss)
    if state.alert:
        print(f"ALERT at step {step}: W={state.W:.3f}")
        # → rollback, stop, or notify
```

### CLI — monitor a training log CSV

```bash
scalepredict monitor training_log.csv
```

```bash
# Specify column names
scalepredict monitor wandb_export.csv --loss-col train/loss --step-col _step

# Save W-Twin scores to CSV
scalepredict monitor training_log.csv --output wtwin_scores.csv

# Adjust sensitivity
scalepredict monitor training_log.csv --alpha 1.5 --warmup 200
```

### CLI — quick demo (no file needed)

```bash
scalepredict demo
```

---

## Integration Examples

### HuggingFace Trainer callback

```python
from transformers import TrainerCallback
from scalepredict.monitor import WTwinMonitor

class WTwinCallback(TrainerCallback):
    def __init__(self, alert_fn=None):
        self.monitor = WTwinMonitor()
        self.alert_fn = alert_fn or (lambda s: print(f"ALERT at step {s}"))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            st = self.monitor.update(state.global_step, logs["loss"])
            if st.alert:
                self.alert_fn(state.global_step)

# Usage:
trainer = Trainer(..., callbacks=[WTwinCallback()])
```

### Weights & Biases CSV export

```bash
# Export from W&B: Runs → Export → CSV
scalepredict monitor wandb_run_export.csv \
    --loss-col train/loss \
    --step-col _step \
    --warmup 200
```

---

## Architecture

```
ScalePredict
├── scalepredict.monitor      ← W-Twin (core)
│   ├── WTwinMonitor          ← streaming detector
│   ├── PowerLawBaseline      ← ScalePredict baseline (plug-and-play)
│   └── run_ablation()        ← comparative benchmark
├── scalepredict.common       ← shared math (MAD, power-law fit)
└── scalepredict.cli          ← CLI entry point
```

The baseline is **plug-and-play** — replace `PowerLawBaseline` with any causal forecasting model:

```python
from scalepredict.monitor.baseline import BaseBaseline
from scalepredict.monitor import WTwinMonitor

class MyKalmanBaseline(BaseBaseline):
    def fit(self, steps, losses): ...
    def predict(self, t): ...
    @property
    def fit_mse(self): ...
    @property
    def is_fitted(self): ...

monitor = WTwinMonitor(baseline=MyKalmanBaseline())
```

---

## Limitations

- Validated on nano-GPT (842K params) with synthetic byte-level text
- Assumes monotonically decreasing loss during calibration (power-law)
- Failure injection is synthetic (label corruption, weight corruption)
- External validation on public step-level training logs is pending

See Section 8 (Threats to Validity) in the [paper](https://doi.org/10.5281/zenodo.21842461).

---

## Citation

```bibtex
@software{kretski2026wtwin,
  author    = {Kretski, Dimitar},
  title     = {W-Twin: Forecast-Based Detection of Progressive
               Neural Network Training Degradation},
  year      = {2026},
  doi       = {10.5281/zenodo.21842461},
  url       = {https://zenodo.org/records/21842461},
  publisher = {Zenodo}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

Author: Dimitar Kretski, Center for Hydro- and Aerodynamics, Varna, Bulgaria
ORCID: [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)
