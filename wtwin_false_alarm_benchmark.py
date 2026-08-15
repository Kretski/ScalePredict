"""
W-Twin False Alarm Rate Benchmark
==================================
Тества колко шумни са различните оптимизатори при ЧИСТИ runs (без инжектирана деградация).
Цел: false alarm rate — W-Twin не трябва да алармира при нормална тренировка.

Синтетични loss криви, реалистично моделирани за всеки оптимизатор.
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict
from wtwin import WTwinMonitor

RNG = np.random.default_rng(42)
N_RUNS = 30       # runs per optimizer (като в paper-а)
N_STEPS = 3000    # стъпки per run
WARMUP = 100      # общ warmup за всички


# ---------------------------------------------------------------------------
# Синтетични loss криви — реалистични профили по оптимизатор
# ---------------------------------------------------------------------------

def power_law_loss(steps: np.ndarray, a: float, b: float) -> np.ndarray:
    """Базова power-law крива: L(t) = a * t^(-b)"""
    return a * np.power(steps, -b)


def make_adamw_loss(steps: np.ndarray, seed: int) -> np.ndarray:
    """
    AdamW: гладка power-law + малък Gaussian шум.
    Реалистичен профил: бързо намаляване, след това плато.
    """
    rng = np.random.default_rng(seed)
    base = power_law_loss(steps, a=rng.uniform(3.5, 4.5), b=rng.uniform(0.28, 0.35))
    noise = rng.normal(0, 0.003, size=len(steps))
    # Малък LR warmup ефект в началото
    warmup_mask = steps < 100
    base[warmup_mask] *= rng.uniform(1.05, 1.15)
    return np.clip(base + noise, 0.01, None)


def make_sgd_momentum_loss(steps: np.ndarray, seed: int) -> np.ndarray:
    """
    SGD + Momentum: по-шумна крива, с леки осцилации.
    По-бавна конвергенция, по-висок шум.
    """
    rng = np.random.default_rng(seed)
    base = power_law_loss(steps, a=rng.uniform(4.0, 5.5), b=rng.uniform(0.22, 0.30))
    # Осцилации — характерни за SGD
    osc_freq = rng.uniform(0.01, 0.03)
    osc_amp  = rng.uniform(0.01, 0.025)
    oscillation = osc_amp * np.sin(osc_freq * steps) * np.exp(-steps / 2000)
    noise = rng.normal(0, 0.008, size=len(steps))
    return np.clip(base + oscillation + noise, 0.01, None)


def make_lion_loss(steps: np.ndarray, seed: int) -> np.ndarray:
    """
    Lion: агресивна ранна конвергенция, след това гладко плато.
    По-дълъг ефективен warmup.
    """
    rng = np.random.default_rng(seed)
    base = power_law_loss(steps, a=rng.uniform(3.0, 4.0), b=rng.uniform(0.30, 0.40))
    # Lion e по-агресивен в началото
    early = steps < 200
    base[early] *= rng.uniform(1.1, 1.3)
    noise = rng.normal(0, 0.004, size=len(steps))
    return np.clip(base + noise, 0.01, None)


def make_rmsprop_loss(steps: np.ndarray, seed: int) -> np.ndarray:
    """
    RMSprop: адаптивен, но по-нестабилен от Adam при стандартни задачи.
    Умерен шум, понякога micro-spikes.
    """
    rng = np.random.default_rng(seed)
    base = power_law_loss(steps, a=rng.uniform(3.8, 5.0), b=rng.uniform(0.25, 0.32))
    noise = rng.normal(0, 0.006, size=len(steps))
    # Случайни micro-spikes (RMSprop е известен с тях)
    n_spikes = rng.integers(2, 8)
    spike_idx = rng.integers(100, len(steps), size=n_spikes)
    spike_amp = rng.uniform(0.01, 0.04, size=n_spikes)
    for idx, amp in zip(spike_idx, spike_amp):
        base[idx] += amp
    return np.clip(base + noise, 0.01, None)


OPTIMIZER_CONFIGS = {
    "AdamW": {
        "gen_fn": make_adamw_loss,
        "warmup_steps": 100,
        "alpha": 2.0,
        "n_consec": 5,
    },
    "SGD+Momentum": {
        "gen_fn": make_sgd_momentum_loss,
        "warmup_steps": 50,
        "alpha": 2.5,
        "n_consec": 7,
    },
    "Lion": {
        "gen_fn": make_lion_loss,
        "warmup_steps": 200,
        "alpha": 2.0,
        "n_consec": 5,
    },
    "RMSprop": {
        "gen_fn": make_rmsprop_loss,
        "warmup_steps": 100,
        "alpha": 2.2,
        "n_consec": 6,
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    optimizer: str
    run_id: int
    false_alarm: bool
    first_alert_step: int | None
    total_alerts: int


def run_single(optimizer_name: str, cfg: dict, run_id: int) -> RunResult:
    steps = np.arange(1, N_STEPS + 1, dtype=float)
    losses = cfg["gen_fn"](steps, seed=run_id * 100 + 7)

    monitor = WTwinMonitor(
        warmup_steps=cfg["warmup_steps"],
        alpha=cfg["alpha"],
        n_consec=cfg["n_consec"],
    )

    alert_steps = []
    for s, l in zip(steps, losses):
        state = monitor.update(int(s), float(l))
        if state.alert:
            alert_steps.append(int(s))

    first = monitor.first_alert_step()
    return RunResult(
        optimizer=optimizer_name,
        run_id=run_id,
        false_alarm=(first is not None),
        first_alert_step=first,
        total_alerts=len(alert_steps),
    )


def run_benchmark() -> Dict[str, List[RunResult]]:
    results = {}
    for opt_name, cfg in OPTIMIZER_CONFIGS.items():
        print(f"  Testing {opt_name} ({N_RUNS} clean runs)...", end=" ", flush=True)
        runs = [run_single(opt_name, cfg, i) for i in range(N_RUNS)]
        results[opt_name] = runs
        fa_count = sum(r.false_alarm for r in runs)
        print(f"FA={fa_count}/{N_RUNS}")
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: Dict[str, List[RunResult]]):
    print("\n" + "=" * 65)
    print("W-Twin FALSE ALARM RATE BENCHMARK — Clean runs (no injected failure)")
    print(f"N={N_RUNS} runs × {N_STEPS} steps per optimizer")
    print("=" * 65)

    summary = []
    for opt_name, runs in results.items():
        fa_runs   = [r for r in runs if r.false_alarm]
        fa_rate   = len(fa_runs) / N_RUNS
        fa_steps  = [r.first_alert_step for r in fa_runs]
        mean_step = int(np.mean(fa_steps)) if fa_steps else None
        cfg       = OPTIMIZER_CONFIGS[opt_name]

        summary.append({
            "optimizer":   opt_name,
            "fa_count":    len(fa_runs),
            "fa_rate":     fa_rate,
            "mean_fa_step": mean_step,
            "alpha":       cfg["alpha"],
            "n_consec":    cfg["n_consec"],
            "warmup":      cfg["warmup_steps"],
        })

        verdict = "✅ PASS" if fa_rate == 0.0 else ("⚠️  LOW" if fa_rate <= 0.10 else "❌ HIGH")
        step_info = f"  (mean step: {mean_step})" if mean_step else ""
        print(f"\n{verdict}  {opt_name}")
        print(f"       False alarms : {len(fa_runs)}/{N_RUNS}  ({fa_rate:.1%}){step_info}")
        print(f"       α={cfg['alpha']}, n_consec={cfg['n_consec']}, warmup={cfg['warmup_steps']}")

    print("\n" + "-" * 65)
    print("SUMMARY TABLE")
    print(f"{'Optimizer':<16} {'FA Rate':>10} {'FA Count':>10} {'Mean FA Step':>14} {'Verdict':>10}")
    print("-" * 65)
    for s in summary:
        verdict = "PASS" if s["fa_rate"] == 0.0 else ("LOW" if s["fa_rate"] <= 0.10 else "HIGH")
        ms = str(s["mean_fa_step"]) if s["mean_fa_step"] else "—"
        print(f"{s['optimizer']:<16} {s['fa_rate']:>10.1%} {s['fa_count']:>10} {ms:>14} {verdict:>10}")
    print("=" * 65)

    # JSON output за архив
    with open("/mnt/user-data/outputs/wtwin_false_alarm_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved → wtwin_false_alarm_results.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"W-Twin False Alarm Benchmark")
    print(f"Optimizers: {list(OPTIMIZER_CONFIGS.keys())}")
    print(f"Runs per optimizer: {N_RUNS}  |  Steps: {N_STEPS}\n")

    results = run_benchmark()
    print_report(results)
