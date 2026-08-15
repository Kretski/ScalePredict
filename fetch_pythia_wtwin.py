"""
fetch_pythia_wtwin.py
=====================
Изтегля реални Pythia training logs от WandB (EleutherAI)
и пуска W-Twin върху тях.

Pythia е публичен проект — не са нужни специални права.
Модели: pythia-14m (най-малък, най-бърз за тест)

Употреба:
    python fetch_pythia_wtwin.py
"""

import wandb
import numpy as np
import json
import sys

sys.path.insert(0, '.')  # за wtwin_optimizer ако е в същата папка

# ---------------------------------------------------------------------------
# Стъпка 1: Намираме публичните Pythia runs
# ---------------------------------------------------------------------------

def find_pythia_runs(model_size='14m', max_runs=3):
    """Търси публични Pythia runs в eleutherai проекта."""
    api = wandb.Api()
    print(f"Търсим Pythia-{model_size} runs...")

    try:
        runs = api.runs(
            "eleutherai/pythia",
            filters={"display_name": {"$regex": f"pythia-{model_size}"}},
            order="-created_at",
        )
        found = []
        for run in runs:
            if len(found) >= max_runs:
                break
            found.append(run)
            print(f"  Намерен: {run.name} | state={run.state} | steps={run.summary.get('_step', '?')}")
        return found
    except Exception as e:
        print(f"  Грешка при търсене: {e}")
        return []


def fetch_loss_history(run, max_points=5000):
    """Изтегля loss history от WandB run."""
    print(f"\nИзтегляме history от {run.name}...")
    try:
        history = run.scan_history(
            keys=["loss", "train/loss", "lm_loss", "_step"],
            page_size=1000,
        )
        points = []
        for row in history:
            step = row.get("_step")
            loss = row.get("loss") or row.get("train/loss") or row.get("lm_loss")
            if step is not None and loss is not None:
                points.append((int(step), float(loss)))
            if len(points) >= max_points:
                break

        points.sort(key=lambda x: x[0])
        print(f"  Точки: {len(points)}")
        if points:
            print(f"  Steps: {points[0][0]} → {points[-1][0]}")
            print(f"  Loss:  {points[0][1]:.4f} → {points[-1][1]:.4f}")
        return points

    except Exception as e:
        print(f"  Грешка: {e}")
        return []


# ---------------------------------------------------------------------------
# Стъпка 2: W-Twin върху реалните данни
# ---------------------------------------------------------------------------

def run_wtwin(points, run_name, warmup_steps=1000):
    """Пуска W-Twin върху реална loss крива."""
    try:
        from wtwin import WTwinMonitor
    except ImportError:
        print("  wtwin не е инсталиран — pip install git+https://github.com/Kretski/WTwin.git")
        return None

    if len(points) < 150:
        print(f"  Прескачаме — само {len(points)} точки (минимум 150)")
        return None

    monitor = WTwinMonitor(
        warmup_steps=warmup_steps,
        alpha=2.0,
        n_consec=5,
        calibration_frac=0.10,
    )

    W_vals = []
    for step, loss in points:
        state = monitor.update(step, loss)
        W_vals.append(state.W)

    first_alert = monitor.first_alert_step()
    fitted      = monitor.baseline.is_fitted

    print(f"\n  W-Twin резултат за {run_name}:")
    print(f"    Points:        {len(points)}")
    print(f"    Baseline fit:  {fitted}")
    print(f"    First alert:   {first_alert}")
    print(f"    W range:       [{min(W_vals):.2f}, {max(W_vals):.2f}]")

    if first_alert:
        alert_loss = next(l for s,l in points if s >= first_alert)
        print(f"    Alert @ step {first_alert}: loss={alert_loss:.4f}")
        print(f"    ⚠ Потенциална деградация открита!")
    else:
        print(f"    ✅ Чист run — без алерт")

    return {
        'run_name':    run_name,
        'n_points':    len(points),
        'fitted':      fitted,
        'first_alert': first_alert,
        'W_min':       round(min(W_vals), 3),
        'W_max':       round(max(W_vals), 3),
        'loss_start':  round(points[0][1], 4),
        'loss_end':    round(points[-1][1], 4),
        'step_start':  points[0][0],
        'step_end':    points[-1][0],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("="*60)
    print("Pythia WandB → W-Twin валидация")
    print("="*60)

    # Търсим runs
    runs = find_pythia_runs(model_size='14m', max_runs=3)

    if not runs:
        print("\nНе са намерени runs. Опитваме директно с известен run ID...")
        # Fallback: опитваме известни run имена
        api = wandb.Api()
        fallback_paths = [
            "eleutherai/pythia/pythia-14m",
            "eleutherai/pythia/pythia-70m",
        ]
        for path in fallback_paths:
            try:
                run = api.run(path)
                runs = [run]
                print(f"  Намерен: {path}")
                break
            except Exception:
                continue

    if not runs:
        print("\n❌ Не можем да намерим публични Pythia runs.")
        print("   Опции:")
        print("   1. Провери wandb.ai/eleutherai/pythia за публични runs")
        print("   2. Пробвай: wandb runs eleutherai/pythia")
        sys.exit(1)

    # Изтегляме и анализираме
    all_results = []
    for run in runs[:2]:  # максимум 2 за скорост
        points = fetch_loss_history(run, max_points=3000)
        if points:
            result = run_wtwin(points, run.name, warmup_steps=1000)
            if result:
                all_results.append(result)

    # Резюме
    if all_results:
        print("\n" + "="*60)
        print("РЕЗЮМЕ — реални Pythia данни")
        print("="*60)
        for r in all_results:
            alert_str = f"ALERT @ {r['first_alert']}" if r['first_alert'] else "чист ✅"
            print(f"  {r['run_name']}: {r['n_points']} точки | {alert_str}")
            print(f"    loss: {r['loss_start']} → {r['loss_end']}")
            print(f"    W:    [{r['W_min']}, {r['W_max']}]")

        with open('pythia_wtwin_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("\nResults saved → pythia_wtwin_results.json")
    else:
        print("\n❌ Няма резултати — провери connection-а")
