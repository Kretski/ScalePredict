# run_benchmark.py — ScalePredict Local Benchmark
# Пуска се на машината на клиента.
# Мери реална латентност, RAM, throughput.
# Записва резултат в scalepredict_profile.json
#
# pip install numpy psutil torch torchvision
# python run_benchmark.py

import os
import sys
import json
import time
import math
import platform
import datetime

import numpy as np
import psutil

# ─── ПРОВЕРКА НА ЗАВИСИМОСТИ ──────────────────────────────────────────────────
def check_deps():
    missing = []
    try:    import torch
    except: missing.append("torch")
    try:    import torchvision
    except: missing.append("torchvision")
    if missing:
        print(f"\n❌  Липсващи пакети: {', '.join(missing)}")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import torch
import torch.nn as nn
import torchvision.models as models

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BATCH_SIZES   = [1, 8, 32, 64, 128]
WARMUP_ROUNDS = 3      # пропускаме първите рундове (GPU/CPU warmup)
MEASURE_ROUNDS= 10     # измервания за всеки batch size
IMG_SIZE      = 224    # стандарт за ResNet
OUTPUT_FILE   = "scalepredict_profile.json"

# ─── BANNER ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("⚡ ScalePredict — Local Benchmark")
print("   Мери производителността на твоята машина")
print("   Резултатът се записва в scalepredict_profile.json")
print("=" * 60)

# ─── УСТРОЙСТВО ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
has_cuda = torch.cuda.is_available()

print(f"\n🖥️  Устройство:  {device}")
print(f"   OS:         {platform.system()} {platform.release()}")
print(f"   CPU:        {platform.processor()}")
print(f"   RAM:        {psutil.virtual_memory().total // (1024**2)} MB")
print(f"   Ядра:       {psutil.cpu_count(logical=False)} физически / "
      f"{psutil.cpu_count(logical=True)} логически")

if has_cuda:
    print(f"   GPU:        {torch.cuda.get_device_name(0)}")
    print(f"   VRAM:       {torch.cuda.get_device_properties(0).total_memory // (1024**2)} MB")

# ─── МОДЕЛ ────────────────────────────────────────────────────────────────────
print(f"\n📦 Зареждам модел (ResNet-18)...")
model = models.resnet18(weights=None)
model = model.to(device)
model.eval()

params = sum(p.numel() for p in model.parameters())
size_mb = params * 4 / (1024**2)
print(f"   Параметри:  {params:,}")
print(f"   Размер:     {size_mb:.2f} MB")

# ─── BENCHMARK ФУНКЦИЯ ────────────────────────────────────────────────────────
def benchmark_batch(batch_size, warmup=WARMUP_ROUNDS, rounds=MEASURE_ROUNDS):
    """
    Мери латентност за даден batch size.
    Връща: avg_ms, min_ms, max_ms, std_ms, throughput_img_per_s
    """
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)

    # Warmup — не измерваме
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if has_cuda:
                torch.cuda.synchronize()

    # Реални измервания
    times = []
    with torch.no_grad():
        for _ in range(rounds):
            if has_cuda:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                torch.cuda.synchronize()
            else:
                t0 = time.perf_counter()
                _ = model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # → ms

    avg_ms     = float(np.mean(times))
    min_ms     = float(np.min(times))
    max_ms     = float(np.max(times))
    std_ms     = float(np.std(times))
    throughput = batch_size / (avg_ms / 1000)  # img/s

    return {
        "avg_ms":     round(avg_ms, 3),
        "min_ms":     round(min_ms, 3),
        "max_ms":     round(max_ms, 3),
        "std_ms":     round(std_ms, 3),
        "throughput": round(throughput, 1),
    }

# ─── СИСТЕМНИ МЕТРИКИ ─────────────────────────────────────────────────────────
def get_system_metrics():
    cpu_pct  = psutil.cpu_percent(interval=1)
    ram      = psutil.virtual_memory()
    battery  = psutil.sensors_battery()
    temps    = {}
    try:
        raw = psutil.sensors_temperatures()
        for key, vals in raw.items():
            if vals:
                temps[key] = round(vals[0].current, 1)
    except: pass

    return {
        "cpu_pct":     cpu_pct,
        "ram_pct":     round(ram.percent, 1),
        "ram_used_mb": ram.used // (1024**2),
        "ram_total_mb":ram.total // (1024**2),
        "battery_pct": round(battery.percent, 1) if battery else None,
        "temperatures":temps,
    }

# ─── ГЛАВЕН BENCHMARK ─────────────────────────────────────────────────────────
print(f"\n🔬 Benchmark стартира...")
print(f"   Batch sizes: {BATCH_SIZES}")
print(f"   Warmup:      {WARMUP_ROUNDS} рунда (пропуснати)")
print(f"   Измервания:  {MEASURE_ROUNDS} рунда на batch\n")

results_by_batch = {}
bar_width = 40

for bs in BATCH_SIZES:
    sys.stdout.write(f"   batch={bs:3d}  [")
    sys.stdout.flush()

    res = benchmark_batch(bs)

    # Progress bar
    filled = int(bar_width * bs / max(BATCH_SIZES))
    sys.stdout.write("█" * filled + "░" * (bar_width - filled))
    sys.stdout.write(f"]  {res['avg_ms']:.1f}ms  {res['throughput']:.0f} img/s\n")
    sys.stdout.flush()

    results_by_batch[bs] = res

print(f"\n⏳ Пауза 3s — събирам системни метрики...")
time.sleep(3)
sys_metrics = get_system_metrics()

# ─── W SCORE (Quality/Density/Tension) ───────────────────────────────────────
# W = Q·D - T
# Q = качество (throughput нормализиран)
# D = достъпност (RAM свободна)
# T = напрежение (CPU + температура)
max_tput = max(r["throughput"] for r in results_by_batch.values())
avg_tput = sum(r["throughput"] for r in results_by_batch.values()) / len(results_by_batch)

Q = min(1.0, avg_tput / 200.0)           # нормализиран към 200 img/s
D = 1.0 - sys_metrics["ram_pct"] / 100.0 # свободна RAM
T = sys_metrics["cpu_pct"] / 100.0       # CPU напрежение
W = round(Q * D - T * 0.5, 4)

# ─── k(t,d) ПРОФИЛ ────────────────────────────────────────────────────────────
# Изчислява k за всеки batch size — ще се използва от app-а
k0, alpha, beta = 1e-4, 1e-4, 1e6
k_profile = {}
for bs, r in results_by_batch.items():
    t = float(bs)
    d = r["avg_ms"] * 1e3
    e = -alpha * t
    k = k0 * math.exp(e if e > -700 else -700) * (1.0 + beta / max(d, 1.0))
    k_profile[bs] = round(k, 8)

# ─── ПРОФИЛ ───────────────────────────────────────────────────────────────────
profile = {
    "meta": {
        "timestamp":    datetime.datetime.now().isoformat(),
        "scalepredict": "v0.1",
        "model":        "ResNet-18",
        "img_size":     IMG_SIZE,
        "warmup_rounds":WARMUP_ROUNDS,
        "measure_rounds":MEASURE_ROUNDS,
    },
    "hardware": {
        "os":       f"{platform.system()} {platform.release()}",
        "cpu":      platform.processor(),
        "ram_mb":   psutil.virtual_memory().total // (1024**2),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical":  psutil.cpu_count(logical=True),
        "has_cuda": has_cuda,
        "gpu":      torch.cuda.get_device_name(0) if has_cuda else None,
        "vram_mb":  torch.cuda.get_device_properties(0).total_memory // (1024**2)
                    if has_cuda else None,
    },
    "benchmark": {
        str(bs): r for bs, r in results_by_batch.items()
    },
    "system_metrics": sys_metrics,
    "w_score":   W,
    "k_profile": {str(k): v for k, v in k_profile.items()},
    "summary": {
        "avg_latency_ms":  round(
            sum(r["avg_ms"] for r in results_by_batch.values()) / len(results_by_batch), 2),
        "best_batch":      max(results_by_batch, key=lambda b: results_by_batch[b]["throughput"]),
        "max_throughput":  max_tput,
        "w_score":         W,
        "device":          str(device),
    }
}

# ─── ЗАПИС ────────────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w") as f:
    json.dump(profile, f, indent=2)

# ─── РЕЗУЛТАТ ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 РЕЗУЛТАТ")
print("=" * 60)
print(f"\n   Устройство:       {device}")
print(f"   Avg латентност:   {profile['summary']['avg_latency_ms']} ms")
print(f"   Най-добър batch:  {profile['summary']['best_batch']}")
print(f"   Max throughput:   {max_tput:.0f} img/s")
print(f"\n   W score:  {W:.4f}  ", end="")
if W > 0.3:   print("✅ Production ready")
elif W > 0.1: print("⚠️  Marginal — monitor resources")
else:         print("🔴 Resource constrained")

print(f"\n   CPU:   {sys_metrics['cpu_pct']}%")
print(f"   RAM:   {sys_metrics['ram_pct']}%  "
      f"({sys_metrics['ram_used_mb']}MB / {sys_metrics['ram_total_mb']}MB)")
if sys_metrics["temperatures"]:
    for key, temp in list(sys_metrics["temperatures"].items())[:2]:
        print(f"   Temp:  {temp}°C  ({key})")

print(f"\n✅ Профилът е записан в:  {OUTPUT_FILE}")
print(f"   Отвори ScalePredict app:")
print(f"   streamlit run scalepredict_app.py")
print("=" * 60)
