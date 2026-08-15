"""
compare_pythia.py
=================
W-Twin vs CUSUM vs Threshold върху реални Pythia runs.
"""
import json
import numpy as np
from wtwin import WTwinMonitor


def run_cusum(points, k=0.5, h=5.0, warmup=100):
    S = 0.0
    first = None
    history = [l for _, l in points[:warmup]]
    for i, (step, loss) in enumerate(points):
        history.append(loss)
        if i < warmup:
            continue
        mu = np.mean(history[-50:])
        S = max(0, S + (loss - mu) - k)
        if S > h and first is None:
            first = step
    return first


def run_threshold(points, multiplier=1.5, warmup=100):
    baseline = np.mean([l for _, l in points[:warmup]])
    for step, loss in points[warmup:]:
        if loss > baseline * multiplier:
            return step
    return None


with open('pythia_multi.json') as f:
    data = json.load(f)

print('=' * 65)
print('W-Twin vs CUSUM vs Threshold — реални Pythia runs')
print('=' * 65)
print()

for label, points in data.items():
    # W-Twin
    wt = WTwinMonitor(
        warmup_steps=100,
        alpha=2.0,
        n_consec=5,
        calibration_frac=0.10,
    )
    for step, loss in points:
        wt.update(int(step), float(loss))
    wt_alert = wt.first_alert_step()
    W_vals   = [s.W for s in wt.history if s.W == s.W]
    W_max    = round(max(W_vals), 2) if W_vals else 0

    # CUSUM
    cs_alert = run_cusum(points)

    # Threshold
    th_alert = run_threshold(points)

    loss_start = round(points[0][1], 3)
    loss_end   = round(points[-1][1], 3)

    wt_str = f'step {wt_alert}' if wt_alert else 'no alert'
    cs_str = f'step {cs_alert}' if cs_alert else 'no alert'
    th_str = f'step {th_alert}' if th_alert else 'no alert'

    print(f'{label}')
    print(f'  loss:      {loss_start} -> {loss_end}')
    print(f'  W-Twin:    {wt_str}   W_max={W_max}')
    print(f'  CUSUM:     {cs_str}')
    print(f'  Threshold: {th_str}')
    print()

print('=' * 65)
print('Очаквано:')
print('  clean_143k      -> всички: no alert')
print('  high_loss runs  -> W-Twin трябва да хване по-рано от threshold')
print('=' * 65)
