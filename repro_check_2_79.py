"""
repro_check_2_79.py
===================
Reproducibility check: high_loss_2.79 (run id: 3f4r65tq)
Цел: възпроизвеждаме W_max=4.89 @ step 2031 от compare_pythia.py

Параметри от оригиналния тест:
  warmup_steps=100, alpha=2.0, n_consec=5, calibration_frac=0.10
  sampling: points[::10] (всяка 10-та точка от пълната история)
"""
import json
import numpy as np
from wtwin import WTwinMonitor

# Зареждаме пълните данни (от pythia_multi.json)
with open('pythia_multi.json') as f:
    data = json.load(f)

pts_full = data['high_loss_2.79']
print('=' * 60)
print('Reproducibility check: high_loss_2.79')
print('=' * 60)
print(f'Points (sampled 1:10): {len(pts_full)}')
print(f'Loss: {pts_full[0][1]:.4f} -> {pts_full[-1][1]:.4f}')
print()

# Конфигурация от compare_pythia.py
configs = {
    'original (warmup=100, alpha=2.0, cal=0.10)': dict(
        warmup_steps=100, alpha=2.0, n_consec=5, calibration_frac=0.10,
    ),
    'early_phase (warmup=50, alpha=2.0, cal=0.15)': dict(
        warmup_steps=50, alpha=2.0, n_consec=5, calibration_frac=0.15,
    ),
}

for cfg_name, cfg in configs.items():
    mon = WTwinMonitor(**cfg)
    for step, loss in pts_full:
        mon.update(int(step), float(loss))

    first = mon.first_alert_step()
    fitted = mon.baseline.is_fitted
    W_vals = [s.W for s in mon.history if s.W == s.W]
    W_max  = round(max(W_vals), 3) if W_vals else None
    W_min  = round(min(W_vals), 3) if W_vals else None

    # Baseline коефициенти
    bl = mon.baseline
    coeffs = bl.coefficients if fitted else None

    print(f'Config: {cfg_name}')
    print(f'  fitted:      {fitted}')
    print(f'  first_alert: {first}')
    print(f'  W_max:       {W_max}')
    print(f'  W_min:       {W_min}')
    if coeffs:
        print(f'  baseline:    {coeffs}')
        print(f'  fit_mse:     {bl.fit_mse:.6f}')

    # W около alert зоната
    print(f'  W trajectory около step 2031:')
    for s in mon.history:
        if 1900 <= s.step <= 2200:
            if s.step % 200 == 31 or abs(s.step - 2031) < 15:
                print(f'    step {s.step:6d}: W={s.W:+8.3f}  D={s.D:+8.3f}  alert={s.alert}')
    print()

print('=' * 60)
print('ОЧАКВАНО: first_alert=2031, W_max≈4.89 при оригиналната конфигурация')
print('Ако числата съвпадат -> reproducible')
print('Ако не -> документираме разликата')
print('=' * 60)
