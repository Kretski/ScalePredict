# run_benchmark.py — ScalePredict Local Benchmark
# Пуска се на машината на клиента.
# Мери реална латентност, RAM, throughput.
# Записва резултат в scalepredict_profile.json
#
# pip install numpy psutil torch torchvision
# python run_benchmark.py

import os, sys, json, time, math, platform, datetime
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
BATCH_SIZES    = [1, 8, 32, 64, 128]
WARMUP_ROUNDS  = 3
MEASURE_ROUNDS = 10
IMG_SIZE       = 224
OUTPUT_FILE    = "scalepredict_profile.json"

# k(t,d) модел — централно дефинирани, използвани и тук и в app
K0    = 1e-4
ALPHA = 1e-4
BETA  = 1e6

# GPU speedup спрямо Lenovo batch=32 (78ms baseline)
# Източник: публично достъпни MLPerf benchmark резултати (inference, ResNet)
# https://mlcommons.org/benchmarks/inference-datacenter/
# Стойностите са приближения — реалното ускорение зависи от модела и batch
CLOUD_GPU_SPEEDUP = {
    "T4":   14.0,   # ~14x спрямо CPU baseline
    "V100": 25.0,   # ~25x
    "A100": 44.0,   # ~44x
    "A10G": 20.0,   # ~20x
}

print("=" * 60)
print("⚡ ScalePredict — Local Benchmark")
print("   Мери производителността на твоята машина")
print("   Резултатът се записва в scalepredict_profile.json")
print("=" * 60)

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

print(f"\n📦 Зареждам модел (ResNet-18)...")
model = models.resnet18(weights=None).to(device)
model.eval()
params  = sum(p.numel() for p in model.parameters())
size_mb = params * 4 / (1024**2)
print(f"   Параметри:  {params:,}")
print(f"   Размер:     {size_mb:.2f} MB")

# ─── BENCHMARK ────────────────────────────────────────────────────────────────
def benchmark_batch(batch_size, warmup=WARMUP_ROUNDS, rounds=MEASURE_ROUNDS):
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if has_cuda: torch.cuda.synchronize()
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
            times.append((time.perf_counter() - t0) * 1000)
    avg_ms = float(np.mean(times))
    return {
        "avg_ms":     round(avg_ms, 3),
        "min_ms":     round(float(np.min(times)), 3),
        "max_ms":     round(float(np.max(times)), 3),
        "std_ms":     round(float(np.std(times)), 3),
        "throughput": round(batch_size / (avg_ms / 1000), 1),
    }

def get_system_metrics():
    cpu_pct = psutil.cpu_percent(interval=1)
    ram     = psutil.virtual_memory()
    battery = psutil.sensors_battery()
    temps   = {}
    try:
        for key, vals in psutil.sensors_temperatures().items():
            if vals: temps[key] = round(vals[0].current, 1)
    except: pass
    return {
        "cpu_pct":     cpu_pct,
        "ram_pct":     round(ram.percent, 1),
        "ram_used_mb": ram.used // (1024**2),
        "ram_total_mb":ram.total // (1024**2),
        "battery_pct": round(battery.percent, 1) if battery else None,
        "temperatures":temps,
    }

print(f"\n🔬 Benchmark стартира...")
print(f"   Batch sizes: {BATCH_SIZES}")
print(f"   Warmup:      {WARMUP_ROUNDS} рунда (пропуснати)\n")

results_by_batch = {}
bar_width = 40
for bs in BATCH_SIZES:
    sys.stdout.write(f"   batch={bs:3d}  [")
    sys.stdout.flush()
    res    = benchmark_batch(bs)
    filled = int(bar_width * bs / max(BATCH_SIZES))
    sys.stdout.write("█" * filled + "░" * (bar_width - filled))
    sys.stdout.write(f"]  {res['avg_ms']:.1f}ms  {res['throughput']:.0f} img/s\n")
    sys.stdout.flush()
    results_by_batch[bs] = res

print(f"\n⏳ Пауза 3s — събирам системни метрики...")
time.sleep(3)
sys_metrics = get_system_metrics()

# ─── W SCORE: W = Q·D - T ─────────────────────────────────────────────────────
# FIX 1: Консистентна формула навсякъде — без коефициент на T
# Q = качество (throughput нормализиран към 200 img/s)
# D = достъпност (свободна RAM)
# T = напрежение (CPU %)
avg_tput = sum(r["throughput"] for r in results_by_batch.values()) / len(results_by_batch)
max_tput = max(r["throughput"] for r in results_by_batch.values())

Q = min(1.0, avg_tput / 200.0)
D = 1.0 - sys_metrics["ram_pct"] / 100.0
T = sys_metrics["cpu_pct"] / 100.0
W = round(Q * D - T, 4)   # W = Q·D - T (без коефициент)

# ─── k(t,d) ПРОФИЛ ────────────────────────────────────────────────────────────
k_profile = {}
for bs, r in results_by_batch.items():
    t = float(bs)
    d = r["avg_ms"] * 1e3
    e = -ALPHA * t
    k = K0 * math.exp(e if e > -700 else -700) * (1.0 + BETA / max(d, 1.0))
    k_profile[bs] = round(k, 8)

# ─── ПРОФИЛ ───────────────────────────────────────────────────────────────────
profile = {
    "meta": {
        "timestamp":      datetime.datetime.now().isoformat(),
        "scalepredict":   "v0.2",
        "model":          "ResNet-18",
        "img_size":       IMG_SIZE,
        "warmup_rounds":  WARMUP_ROUNDS,
        "measure_rounds": MEASURE_ROUNDS,
        "w_formula":      "W = Q*D - T  (Q=throughput/200, D=free_RAM, T=cpu_pct)",
        "speedup_source": "MLPerf inference datacenter benchmarks (approx)",
    },
    "hardware": {
        "os":             f"{platform.system()} {platform.release()}",
        "cpu":            platform.processor(),
        "ram_mb":         psutil.virtual_memory().total // (1024**2),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical":  psutil.cpu_count(logical=True),
        "has_cuda":       has_cuda,
        "gpu":            torch.cuda.get_device_name(0) if has_cuda else None,
        "vram_mb":        torch.cuda.get_device_properties(0).total_memory // (1024**2)
                          if has_cuda else None,
    },
    "benchmark":      {str(bs): r for bs, r in results_by_batch.items()},
    "system_metrics": sys_metrics,
    "w_score":        W,
    "w_components":   {"Q": round(Q,4), "D": round(D,4), "T": round(T,4)},
    "k_profile":      {str(k): v for k, v in k_profile.items()},
    "model_params":   {"k0": K0, "alpha": ALPHA, "beta": BETA},
    "summary": {
        "avg_latency_ms": round(
            sum(r["avg_ms"] for r in results_by_batch.values()) / len(results_by_batch), 2),
        "best_batch":     max(results_by_batch, key=lambda b: results_by_batch[b]["throughput"]),
        "max_throughput": max_tput,
        "w_score":        W,
        "device":         str(device),
    }
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(profile, f, indent=2)

print("\n" + "=" * 60)
print("📊 РЕЗУЛТАТ")
print("=" * 60)
print(f"\n   Устройство:       {device}")
print(f"   Avg латентност:   {profile['summary']['avg_latency_ms']} ms")
print(f"   Най-добър batch:  {profile['summary']['best_batch']}")
print(f"   Max throughput:   {max_tput:.0f} img/s")
print(f"\n   W = Q·D - T  =  {Q:.3f} × {D:.3f} - {T:.3f}  =  {W:.4f}", end="  ")
if W > 0.3:   print("✅ Production ready")
elif W > 0.1: print("⚠️  Marginal")
else:         print("🔴 Resource constrained")
print(f"\n   CPU:   {sys_metrics['cpu_pct']}%")
print(f"   RAM:   {sys_metrics['ram_pct']}%  "
      f"({sys_metrics['ram_used_mb']}MB / {sys_metrics['ram_total_mb']}MB)")
if sys_metrics["temperatures"]:
    for key, temp in list(sys_metrics["temperatures"].items())[:2]:
        print(f"   Temp:  {temp}°C  ({key})")
print(f"\n✅ Профилът е записан в:  {OUTPUT_FILE}")
print(f"   streamlit run scalepredict_app.py")
print("=" * 60)
