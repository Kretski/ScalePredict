"""
W-Twin Detection Rate Benchmark
=================================
Тества колко рано W-Twin хваща прогресивна деградация при различни оптимизатори.
Три нива на деградация: слаба / умерена / силна.
Сравнение с CUSUM baseline.

Синтетични loss криви, реалистично моделирани.
Injection point: стъпка 2000 (от 3000 общо).
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import Optional
from wtwin import WTwinMonitor

RNG_BASE = 42
N_RUNS = 10          # runs per (optimizer × drift_level)  — като paper
N_STEPS = 3000
FAILURE_STEP = 2000
WARMUP = 100

# ---------------------------------------------------------------------------
# Drift levels — три реалистични сценария
# ---------------------------------------------------------------------------
DRIFT_LEVELS = {
    "weak":     {"drift_rate": 0.0015, "label": "Слаба (drift×0.0015/step)"},
    "moderate": {"drift_rate": 0.003,  "label": "Умерена (drift×0.003/step)"},
    "strong":   {"drift_rate": 0.006,  "label": "Силна (drift×0.006/step)"},
}

# ---------------------------------------------------------------------------
# Loss генератори (same като FA benchmark)
# ---------------------------------------------------------------------------

def power_law(steps, a, b):
    return a * np.power(steps, -b)


def inject_drift(base_loss, steps, failure_step, drift_rate):
    """Прогресивна деградация: loss спира да намалява и започва да расте."""
    drifted = base_loss.copy()
    mask = steps > failure_step
    drifted[mask] += drift_rate * (steps[mask] - failure_step)
    return drifted


OPTIMIZER_PROFILES = {
    "AdamW": {
        "a_range": (3.5, 4.5), "b_range": (0.28, 0.35),
        "noise_std": 0.003,
        "warmup_steps": 100, "alpha": 2.0, "n_consec": 5,
    },
    "SGD+Momentum": {
        "a_range": (4.0, 5.5), "b_range": (0.22, 0.30),
        "noise_std": 0.008,
        "osc_freq": (0.01, 0.03), "osc_amp": (0.01, 0.025),
        "warmup_steps": 50, "alpha": 2.5, "n_consec": 7,
    },
    "Lion": {
        "a_range": (3.0, 4.0), "b_range": (0.30, 0.40),
        "noise_std": 0.004,
        "warmup_steps": 200, "alpha": 2.0, "n_consec": 5,
    },
    "RMSprop": {
        "a_range": (3.8, 5.0), "b_range": (0.25, 0.32),
        "noise_std": 0.006,
        "n_spikes": (2, 8), "spike_amp": (0.01, 0.04),
        "warmup_steps": 100, "alpha": 2.2, "n_consec": 6,
    },
}


def generate_clean_loss(opt_name, steps, seed):
    rng = np.random.default_rng(seed)
    p = OPTIMIZER_PROFILES[opt_name]
    a = rng.uniform(*p["a_range"])
    b = rng.uniform(*p["b_range"])
    base = power_law(steps, a, b)

    # Optimizer-specific noise
    noise = rng.normal(0, p["noise_std"], size=len(steps))

    if opt_name == "SGD+Momentum":
        freq = rng.uniform(*p["osc_freq"])
        amp  = rng.uniform(*p["osc_amp"])
        noise += amp * np.sin(freq * steps) * np.exp(-steps / 2000)

    if opt_name == "Lion":
        early = steps < 200
        base[early] *= rng.uniform(1.1, 1.3)

    if opt_name == "RMSprop":
        n = rng.integers(*p["n_spikes"])
        idxs = rng.integers(100, len(steps), size=n)
        amps = rng.uniform(*p["spike_amp"], size=n)
        for i, a_ in zip(idxs, amps):
            base[i] += a_

    return np.clip(base + noise, 0.01, None)


# ---------------------------------------------------------------------------
# CUSUM baseline
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """Стандартен CUSUM — реактивен, сравнява с локална история."""
    def __init__(self, k=0.5, h=5.0, warmup=100):
        self.k = k
        self.h = h
        self.warmup = warmup
        self.S_pos = 0.0
        self.S_neg = 0.0
        self.history = []
        self._first_alert = None

    def update(self, step, loss):
        self.history.append(loss)
        if len(self.history) < self.warmup:
            return False
        mu = np.mean(self.history[-50:])  # rolling mean
        x  = loss - mu
        self.S_pos = max(0, self.S_pos + x - self.k)
        self.S_neg = max(0, self.S_neg - x - self.k)
        alert = (self.S_pos > self.h) or (self.S_neg > self.h)
        if alert and self._first_alert is None:
            self._first_alert = step
        return alert

    def first_alert_step(self):
        return self._first_alert


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    optimizer: str
    drift_level: str
    run_id: int
    wtwin_detected: bool
    wtwin_step: Optional[int]
    wtwin_delay: Optional[int]   # steps after FAILURE_STEP
    cusum_detected: bool
    cusum_step: Optional[int]
    cusum_delay: Optional[int]


def run_detection(opt_name, drift_level, drift_rate, run_id):
    steps = np.arange(1, N_STEPS + 1, dtype=float)
    seed  = run_id * 17 + 3

    clean  = generate_clean_loss(opt_name, steps, seed)
    losses = inject_drift(clean, steps, FAILURE_STEP, drift_rate)

    cfg = OPTIMIZER_PROFILES[opt_name]

    # W-Twin
    monitor = WTwinMonitor(
        warmup_steps=cfg["warmup_steps"],
        alpha=cfg["alpha"],
        n_consec=cfg["n_consec"],
    )
    for s, l in zip(steps, losses):
        monitor.update(int(s), float(l))

    wt_step  = monitor.first_alert_step()
    wt_det   = wt_step is not None
    wt_delay = (wt_step - FAILURE_STEP) if wt_det else None

    # CUSUM
    cusum = CUSUMDetector(warmup=cfg["warmup_steps"])
    for s, l in zip(steps, losses):
        cusum.update(int(s), float(l))

    cs_step  = cusum.first_alert_step()
    cs_det   = cs_step is not None
    cs_delay = (cs_step - FAILURE_STEP) if cs_det else None

    return DetectionResult(
        optimizer=opt_name,
        drift_level=drift_level,
        run_id=run_id,
        wtwin_detected=wt_det,
        wtwin_step=wt_step,
        wtwin_delay=wt_delay,
        cusum_detected=cs_det,
        cusum_step=cs_step,
        cusum_delay=cs_delay,
    )


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    all_results = []
    for opt_name in OPTIMIZER_PROFILES:
        for drift_level, dcfg in DRIFT_LEVELS.items():
            tag = f"{opt_name} / {drift_level}"
            print(f"  {tag}...", end=" ", flush=True)
            runs = [
                run_detection(opt_name, drift_level, dcfg["drift_rate"], i)
                for i in range(N_RUNS)
            ]
            wt_det = sum(r.wtwin_detected for r in runs)
            cs_det = sum(r.cusum_detected for r in runs)
            print(f"W-Twin={wt_det}/{N_RUNS}  CUSUM={cs_det}/{N_RUNS}")
            all_results.extend(runs)
    return all_results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarize(results, opt_name, drift_level):
    subset = [r for r in results if r.optimizer == opt_name and r.drift_level == drift_level]
    wt_det  = [r for r in subset if r.wtwin_detected]
    cs_det  = [r for r in subset if r.cusum_detected]

    wt_delays = [r.wtwin_delay for r in wt_det if r.wtwin_delay is not None]
    cs_delays = [r.cusum_delay for r in cs_det if r.cusum_delay is not None]

    return {
        "optimizer": opt_name,
        "drift_level": drift_level,
        "n_runs": len(subset),
        "wtwin_detection_rate": len(wt_det) / len(subset),
        "wtwin_detected": len(wt_det),
        "wtwin_mean_delay": round(float(np.mean(wt_delays)), 1) if wt_delays else None,
        "wtwin_std_delay":  round(float(np.std(wt_delays)), 1)  if wt_delays else None,
        "cusum_detection_rate": len(cs_det) / len(subset),
        "cusum_detected": len(cs_det),
        "cusum_mean_delay": round(float(np.mean(cs_delays)), 1) if cs_delays else None,
    }


def print_report(results):
    print("\n" + "=" * 78)
    print("W-Twin DETECTION RATE BENCHMARK  vs  CUSUM baseline")
    print(f"N={N_RUNS} runs × {len(OPTIMIZER_PROFILES)} optimizers × {len(DRIFT_LEVELS)} drift levels")
    print(f"Injection point: step {FAILURE_STEP} / {N_STEPS}")
    print("=" * 78)

    all_summaries = []
    for opt_name in OPTIMIZER_PROFILES:
        print(f"\n{'─'*78}")
        print(f"  {opt_name}")
        print(f"{'─'*78}")
        print(f"  {'Drift Level':<12} {'W-Twin Det':>12} {'W-Twin Delay':>14} {'CUSUM Det':>11} {'CUSUM Delay':>13} {'Winner':>8}")
        print(f"  {'-'*72}")

        for drift_level in DRIFT_LEVELS:
            s = summarize(results, opt_name, drift_level)
            all_summaries.append(s)

            wt_rate = f"{s['wtwin_detected']}/{s['n_runs']} ({s['wtwin_detection_rate']:.0%})"
            cs_rate = f"{s['cusum_detected']}/{s['n_runs']} ({s['cusum_detection_rate']:.0%})"

            wt_delay = f"{s['wtwin_mean_delay']}±{s['wtwin_std_delay']}" if s['wtwin_mean_delay'] else "—"
            cs_delay = f"{s['cusum_mean_delay']}" if s['cusum_mean_delay'] else "—"

            # Winner logic
            wt_dr = s['wtwin_detection_rate']
            cs_dr = s['cusum_detection_rate']
            if wt_dr > cs_dr:
                winner = "W-Twin ✓"
            elif cs_dr > wt_dr:
                winner = "CUSUM ✓"
            elif wt_dr == cs_dr == 0.0:
                winner = "neither"
            else:
                # same rate → compare delay
                wt_d = s['wtwin_mean_delay'] or 9999
                cs_d = s['cusum_mean_delay'] or 9999
                winner = "W-Twin" if wt_d <= cs_d else "CUSUM"

            print(f"  {drift_level:<12} {wt_rate:>12} {wt_delay:>14} {cs_rate:>11} {cs_delay:>13} {winner:>8}")

    print("\n" + "=" * 78)
    print("AGGREGATE  (across all optimizers)")
    print(f"{'─'*78}")

    for drift_level in DRIFT_LEVELS:
        sub = [s for s in all_summaries if s["drift_level"] == drift_level]
        wt_total = sum(s["wtwin_detected"] for s in sub)
        cs_total = sum(s["cusum_detected"] for s in sub)
        total    = sum(s["n_runs"] for s in sub)
        wt_delays_all = [s["wtwin_mean_delay"] for s in sub if s["wtwin_mean_delay"]]
        label = DRIFT_LEVELS[drift_level]["label"]
        mean_d = f"{np.mean(wt_delays_all):.0f} steps" if wt_delays_all else "—"
        print(f"  {label}")
        print(f"    W-Twin: {wt_total}/{total}  |  CUSUM: {cs_total}/{total}  |  W-Twin mean delay: {mean_d}")

    print("=" * 78)
    print("\n⚠  Ограничения:")
    print("   • Синтетични криви — power-law + Gaussian шум (не реални GPU runs)")
    print("   • Деградацията е линейно инжектирана — реалната може да е по-сложна")
    print("   • CUSUM тук е baseline имплементация, не tuned")
    print("   • N=10 runs — малка извадка, CI-та не са изчислени")

    # Save JSON
    with open("/mnt/user-data/outputs/wtwin_detection_rate_results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print("\nResults saved → wtwin_detection_rate_results.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("W-Twin Detection Rate Benchmark")
    print(f"Optimizers : {list(OPTIMIZER_PROFILES.keys())}")
    print(f"Drift levels: {list(DRIFT_LEVELS.keys())}")
    print(f"Runs: {N_RUNS} per cell  |  Steps: {N_STEPS}  |  Injection: step {FAILURE_STEP}\n")

    results = run_benchmark()
    print_report(results)
