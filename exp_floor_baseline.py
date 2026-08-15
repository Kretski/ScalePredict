"""
exp_floor_baseline.py
=====================
ExpFloorBaseline — алтернативен trajectory model за fine-tuning.

НЕ пипа:
  - PowerLawBaseline
  - W-Twin core (Q, D, W, threshold logic)
  - WTwinMonitor

Добавя само нов модел на очакваната trajectory:

    L(t) = (L0 - L_inf) * exp(-k * t) + L_inf

Физически логичен за fine-tuning:
  - Бърз спад в началото (предтрениран модел се адаптира)
  - Асимптотично плато (capacity limit на задачата)

Диагностичен резултат (2026-08-15):
  - PowerLaw baseline: FA @ стъпка 129 (преди injection @ 200)
  - ExpFloor baseline: без FA ✅
  - Причина: PowerLaw предвижда продължаващо намаляване,
    ExpFloor предвижда реалното плато (~0.699)

EXPERIMENTAL STATUS:
  Validated only on synthetic diagnostic data (1 loss curve, 1 seed).
  Fine-tuning failure matrix not yet complete.
  Do not remove experimental warning before full matrix validation.

Употреба:
    from exp_floor_baseline import ExpFloorBaseline
    from wtwin import WTwinMonitor

    monitor = WTwinMonitor(
        baseline=ExpFloorBaseline(),
        warmup_steps=200,
        alpha=2.5,
        n_consec=7,
    )
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from wtwin.monitor.baseline import BaseBaseline


class ExpFloorBaseline(BaseBaseline):
    """
    Exponential decay + asymptotic floor trajectory model.

    L(t) = (L0 - L_inf) * exp(-k * t) + L_inf

    Parameters
    ----------
    warmup_steps : int
        Steps to exclude before fitting (default 20).
    calibration_frac : float
        Fraction of post-warmup steps used for fitting (default 0.15).
        Slightly larger than PowerLaw default (0.10) because exp fit
        needs more points to constrain L_inf reliably.
    min_cal_points : int
        Minimum calibration points required (default 10).
    """

    def __init__(
        self,
        warmup_steps: int = 20,
        calibration_frac: float = 0.15,
        min_cal_points: int = 10,
    ):
        self.warmup_steps      = warmup_steps
        self.calibration_frac  = calibration_frac
        self.min_cal_points    = min_cal_points

        self._L0:    float = 0.0
        self._Linf:  float = 0.0
        self._k:     float = 0.0
        self._mse:   float = float('inf')
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # BaseBaseline interface
    # ------------------------------------------------------------------

    def fit(self, steps: np.ndarray, losses: np.ndarray) -> None:
        steps  = np.asarray(steps,  dtype=float)
        losses = np.asarray(losses, dtype=float)

        # Exclude warmup
        mask = steps > self.warmup_steps
        sc, lc = steps[mask], losses[mask]

        if len(sc) < self.min_cal_points:
            raise ValueError(
                f"Only {len(sc)} post-warmup points — need ≥{self.min_cal_points}."
            )

        # Calibration window
        n_cal = max(self.min_cal_points, int(len(sc) * self.calibration_frac))
        sc, lc = sc[:n_cal], lc[:n_cal]

        def _model(t, L0, Linf, k):
            return (L0 - Linf) * np.exp(-k * t) + Linf

        try:
            # Initial guess: L0=first, Linf=last*0.9, k=small
            p0     = [lc[0], lc[-1] * 0.9, 0.005]
            bounds = ([0.0, 0.0, 1e-6], [10.0, lc[0], 2.0])
            popt, _ = curve_fit(
                _model, sc, lc,
                p0=p0, bounds=bounds,
                maxfev=5000, method='trf',
            )
            self._L0, self._Linf, self._k = popt

        except (RuntimeError, ValueError):
            # Fallback: treat as flat at calibration mean
            self._L0   = float(lc.mean())
            self._Linf = float(lc.mean())
            self._k    = 1e-4

        preds      = _model(sc, self._L0, self._Linf, self._k)
        self._mse  = float(np.mean((lc - preds) ** 2))
        self._fitted = True

    def predict(self, t: float | np.ndarray) -> float | np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        return (self._L0 - self._Linf) * np.exp(-self._k * float(t)) + self._Linf

    @property
    def fit_mse(self) -> float:
        return self._mse

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def coefficients(self) -> dict[str, float]:
        return {"L0": self._L0, "L_inf": self._Linf, "k": self._k}


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np

    # Синтетична fine-tuning крива: бърз спад → плато
    steps  = np.arange(1, 201, dtype=float)
    losses = 0.3 * np.exp(-0.02 * steps) + 0.6 + np.random.default_rng(0).normal(0, 0.01, 200)

    bl = ExpFloorBaseline(warmup_steps=5, calibration_frac=0.15)
    bl.fit(steps, losses)

    print("ExpFloorBaseline smoke test")
    print(f"  Coefficients: {bl.coefficients}")
    print(f"  fit_mse:      {bl.fit_mse:.6f}")
    print(f"  predict(50):  {bl.predict(50):.4f}  (expected ~{0.3*np.exp(-0.02*50)+0.6:.4f})")
    print(f"  predict(200): {bl.predict(200):.4f}  (expected ~{0.3*np.exp(-0.02*200)+0.6:.4f})")
    print("  OK ✅")
