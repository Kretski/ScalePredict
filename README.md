# ScalePredict

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21842461.svg)](https://doi.org/10.5281/zenodo.21842461)
[![CI](https://github.com/Kretski/ScalePredict/actions/workflows/ci.yml/badge.svg)](https://github.com/Kretski/ScalePredict/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Detect progressive neural network training degradation before it shows up in your loss curves.**

ScalePredict compares your training loss against a scaling-law baseline at every step. When the trajectory drifts from what is expected, W-Twin raises an alert — before classical threshold and CUSUM detectors notice anything.

---

## 30-second example

```python
from scalepredict.monitor import WTwinMonitor

monitor = WTwinMonitor()

# Drop into any training loop — one line per step
for step, loss in enumerate(your_training_losses, 1):
    state = monitor.update(step, loss)
    if state.alert:
        print(f"⚠ Degradation detected at step {step}  (W={state.W:.2f})")
        # → rollback checkpoint, send alert, or stop run
```

Or from the command line — no code needed:

```bash
pip install git+https://github.com/Kretski/ScalePredict.git
scalepredict monitor training_log.csv
scalepredict demo
```

---

## The problem

Current training monitors are **reactive**: they catch NaN losses, gradient explosions, and hardware failures *after* the damage is done. Progressive degradation — slowly increasing label noise, gradual weight corruption, subtle data pipeline drift — accumulates undetected until significant GPU budget is wasted.

## How it works

W-Twin fits a power-law baseline to early training steps, then at every step computes:

```
W(t) = Q(t) · (D(t) − α)

D(t) = (L_obs(t) − L_pred(t)) / σ_local(t)   ← how far off the expected curve
Q(t) = exp(−MSE_fit / τ)                        ← how much to trust the baseline
α    = 2.0                                       ← detection threshold (z-score)
```

An alert fires when `W(t) > 0` for 5 consecutive steps. No tuning required for basic use.

---

## Results

From the [paper](https://doi.org/10.5281/zenodo.21842461) — real nano-GPT training runs:

| Experiment           | Runs          | W-Twin             | Threshold | CUSUM                 |
| -------------------- | ------------- | ------------------ | --------- | --------------------- |
| Progressive drift    | 9             | **9/9 (100%)**     | 0/9 (0%)  | 0/9 (0%)              |
| Mean detection delay | —             | **223 ± 11 steps** | —         | —                     |
| False alarm rate     | 30 clean runs | **0/30 (0%)**      | 0/30      | 0/30                  |
| Abrupt spike         | 2             | 2/2 (+5 steps)     | 0/2       | 2/2 (+1 step, faster) |

W-Twin is the only method that detects progressive drift. For sudden spikes, CUSUM is faster — both are complementary.

> **Scope:** Results are from controlled nano-GPT experiments with injected failures. External validation on independent architectures and real training logs is ongoing.

---

## Installation

```bash
pip install git+https://github.com/Kretski/ScalePredict.git
```

Dependencies: `numpy`, `scipy` only. No framework lock-in.

---

## Usage

### In a training loop

```python
from scalepredict.monitor import WTwinMonitor

monitor = WTwinMonitor(
    warmup_steps=100,  # skip LR warmup phase
    alpha=2.0,         # detection sensitivity
    n_consec=5,        # steps above threshold before alert
)

for step, loss in training_loop():
    state = monitor.update(step, loss)
    if state.alert:
        print(f"Step {step}: W={state.W:.3f} — possible degradation")

print(f"First alert: {monitor.first_alert_step()}")
```

### HuggingFace Trainer

```python
from transformers import TrainerCallback
from scalepredict.monitor import WTwinMonitor

class WTwinCallback(TrainerCallback):
    def __init__(self):
        self.monitor = WTwinMonitor()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            st = self.monitor.update(state.global_step, logs["loss"])
            if st.alert:
                print(f"⚠ W-Twin alert at step {state.global_step}")

trainer = Trainer(..., callbacks=[WTwinCallback()])
```

### CLI — monitor a CSV log

```bash
scalepredict monitor training_log.csv
scalepredict monitor wandb_export.csv --loss-col train/loss --step-col _step
scalepredict monitor training_log.csv --output wtwin_scores.csv
scalepredict demo
```

### CLI — detect + classify + suggest *(v1.2)*

```bash
# Detect alert AND get failure classification + recommendation
scalepredict suggest training_log.csv

# With full JSON output
scalepredict suggest training_log.csv --json
```

Output example:

```
⚠  W-Twin Alert at step 2390
   Failure type : gradual_drift  (confidence: 51%)
   Suggestion   : consider_lr_reduction
   Action       : manual_review
   Reasoning    : D(t) shows a sustained positive slope — training is
                  gradually deviating from the expected trajectory.
   [EXPERIMENTAL — validate before acting]
```

> `suggest` is advisory only — it does not modify any training state.

---

## API reference

### `WTwinMonitor`

```python
WTwinMonitor(
    warmup_steps=50,   # steps to skip (LR warmup)
    alpha=2.0,         # fixed z-score threshold
    n_consec=5,        # consecutive alerts required
    mad_window=50,     # window for local noise estimate
    tau=1e-3,          # baseline confidence decay
)
```

| Method               | Returns            | Description         |
| -------------------- | ------------------ | ------------------- |
| `update(step, loss)` | `WTwinState`       | Process one step    |
| `first_alert_step()` | `int \| None`      | Step of first alert |
| `history`            | `list[WTwinState]` | Full history        |
| `reset()`            | —                  | Reset state         |

`WTwinState` fields: `step`, `l_obs`, `l_pred`, `D`, `Q`, `T`, `W`, `alert`

### `suggest(monitor)` *(v1.2, experimental)*

```python
from scalepredict.monitor import WTwinMonitor, suggest

monitor = WTwinMonitor()
for step, loss in training_loop():
    monitor.update(step, loss)

s = suggest(monitor)
print(s.failure_type)   # "gradual_drift" | "abrupt_spike" | "uncertain"
print(s.suggestion)     # "consider_lr_reduction" | "consider_rollback" | "manual_review"
print(s.confidence)     # float in [0.5, 0.95]
print(s.as_dict())      # full JSON-serializable output
```

> **Experimental:** classifier uses heuristics from synthetic experiments only.
> Not validated on labeled real failures. All suggestions require human review.

### Custom baseline

```python
from scalepredict.monitor.baseline import BaseBaseline
from scalepredict.monitor import WTwinMonitor

class MyBaseline(BaseBaseline):
    def fit(self, steps, losses): ...
    def predict(self, t): ...
    @property
    def fit_mse(self): return 0.001
    @property
    def is_fitted(self): return True

monitor = WTwinMonitor(baseline=MyBaseline())
```

---

## Reproduce the paper experiments

```bash
git clone https://github.com/Kretski/ScalePredict.git
cd ScalePredict
pip install -e ".[train]"

# Clean run
python examples/train_real.py --mode none --steps 3000 --model-size small --seed 42

# Progressive drift (key result)
python examples/train_real.py --mode progressive_label \
    --failure-step 2000 --steps 3000 --model-size small \
    --seed 42 --ramp-steps 1000 --max-noise-prob 0.5

# Abrupt failure
python examples/train_real.py --mode weight_corrupt \
    --failure-step 2000 --steps 3000 --model-size small --seed 42
```

---

## Limitations

- Validated on nano-GPT (842K parameters) with synthetic byte-level text
- Power-law baseline assumes monotonically decreasing loss
- Failures are injected synthetically
- `suggest()` classifier not validated on labeled real failures
- External validation on independent architectures pending

Full details in Section 8 of the [paper](https://doi.org/10.5281/zenodo.21842461).

---

## Feedback welcome

If you run ScalePredict on your own training logs — whether it works or not — please open an [issue](https://github.com/Kretski/ScalePredict/issues). External validation is the next priority.

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

**Author:** Dimitar Kretski, Center for Hydro- and Aerodynamics, Varna, Bulgaria
ORCID: [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)
