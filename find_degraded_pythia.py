"""
find_degraded_pythia.py
=======================
Търси crashed/failed Pythia runs и сравнява W-Twin с CUSUM/threshold.
"""
import wandb
import json

api = wandb.Api()

print("Търсим crashed/failed runs в eleutherai/pythia...")
print()

# Търсим runs с различни states
for state in ['crashed', 'failed']:
    print(f"State: {state}")
    try:
        runs = api.runs(
            'eleutherai/pythia',
            filters={'state': state},
            per_page=10,
        )
        for r in runs:
            step = r.summary.get('_step', '?')
            loss = r.summary.get('train/lm_loss', '?')
            print(f"  {r.name:<40} step={step} loss={loss} id={r.id}")
    except Exception as e:
        print(f"  Грешка: {e}")
    print()
