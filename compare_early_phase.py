"""
compare_early_phase.py
======================
Сравнява ранната фаза (step 1-6000) на трите high-loss Pythia runs.
Цел: разбираме защо W-Twin хваща само 2.79, не 2.87 и 3.02.
"""
import json
import numpy as np
from wtwin import WTwinMonitor

with open('pythia_early.json') as f:
    early = json.load(f)

with open('pythia_multi.json') as f:
    full = json.load(f)

print('=' * 65)
print('Ранна фаза анализ (step 1-6000) — три high-loss runs')
print('=' * 65)
print()

# Сравнение на ранната loss динамика
print('Loss @ ключови стъпки:')
print(f'  {"Step":>6}  {"2.79":>10}  {"2.87":>10}  {"3.02":>10}')
print('  ' + '-' * 42)

checkpoints = [100, 500, 1000, 2000, 3000, 4000, 5000]
for target_step in checkpoints:
    vals = {}
    for label, pts in early.items():
        closest = min(pts, key=lambda x: abs(x[0] - target_step))
        vals[label] = closest[1]
    row = f'  {target_step:>6}'
    for label in ['high_loss_2.79', 'high_loss_2.87', 'high_loss_3.02']:
        v = vals.get(label, float('nan'))
        row += f'  {v:>10.4f}'
    print(row)

print()

# Loss scale сравнение — важно за разбиране на fp16 stability
print('Loss scale @ ключови стъпки (fp16 stability indicator):')
print(f'  {"Step":>6}  {"2.79":>10}  {"2.87":>10}  {"3.02":>10}')
print('  ' + '-' * 42)
for target_step in [500, 1000, 2000, 3000]:
    vals = {}
    for label, pts in early.items():
        closest = min(pts, key=lambda x: abs(x[0] - target_step))
        vals[label] = closest[2] if len(closest) > 2 else '?'
    row = f'  {target_step:>6}'
    for label in ['high_loss_2.79', 'high_loss_2.87', 'high_loss_3.02']:
        v = vals.get(label, '?')
        row += f'  {str(v):>10}'
    print(row)

print()

# W-Twin върху ранната фаза — само стъпки 1-6000
print('W-Twin върху ранната фаза (warmup=50, стъпки 1-6000):')
print()
for label, pts in early.items():
    mon = WTwinMonitor(
        warmup_steps=50,
        alpha=2.0,
        n_consec=5,
        calibration_frac=0.15,
    )
    for step, loss, *_ in pts:
        mon.update(step, loss)

    first = mon.first_alert_step()
    W_vals = [s.W for s in mon.history if s.W == s.W and s.l_pred == s.l_pred]
    W_max = round(max(W_vals), 3) if W_vals else None
    W_min = round(min(W_vals), 3) if W_vals else None

    # Loss на различни точки
    loss_100  = next((l for s,l,*_ in pts if s >= 100),  None)
    loss_1000 = next((l for s,l,*_ in pts if s >= 1000), None)
    loss_3000 = next((l for s,l,*_ in pts if s >= 3000), None)
    loss_6000 = next((l for s,l,*_ in pts if s >= 5900), None)

    print(f'  {label}')
    print(f'    Alert: {first if first else "—"}')
    print(f'    W range: [{W_min}, {W_max}]')
    print(f'    Loss: {loss_100:.3f} (s100) → {loss_1000:.3f} (s1000) → {loss_3000:.3f} (s3000) → {loss_6000:.3f} (s6000)')

    # Скоростта на конвергенция в ранната фаза
    if loss_100 and loss_1000:
        delta_early = loss_100 - loss_1000
        print(f'    Delta s100→s1000: -{delta_early:.3f}  (по-голямо = по-бърза ранна конвергенция)')
    print()

print('=' * 65)
print('ИНТЕРПРЕТАЦИЯ:')
print()
print('high_loss_2.79 стартира от 3.84 (не 11.0) — различен модел/checkpoint!')
print('high_loss_2.87 и 3.02 стартират от ~11.0 — от нулата, като clean run.')
print()
print('W-Twin хваща 2.79 защото:')
print('  - Стартира от 3.84 (по-нисък), converge-ва по-бавно от очакваното')
print('  - Power-law baseline fit-нат на бавна ранна фаза')
print('  - Loss стагнира около 3.2 докато baseline очаква продължаващо намаляване')
print()
print('W-Twin не хваща 2.87 и 3.02 защото:')
print('  - Стартират от ~11.0, следват нормална power-law крива')
print('  - Просто converge-ват до по-висок финален loss')
print('  - Това не е trajectory deviation — това е различен capacity/regime')
print('=' * 65)
