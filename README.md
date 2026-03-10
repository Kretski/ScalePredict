# ⚡ ScalePredict

> Run a 2-min local benchmark → predict how long your AI job will take on cloud GPU.

No guessing. No wasted money.

<p align="center">
  <a href="https://github.com/Kretski/ScalePredict">
    <img src="https://img.shields.io/github/stars/Kretski/ScalePredict?style=for-the-badge&color=yellow" alt="GitHub Stars">
  </a>
  &nbsp;
  <a href="https://scalepredict.streamlit.app">
    <img src="https://img.shields.io/badge/Live%20Demo-scalepredict.streamlit.app-00f5c4?style=for-the-badge" alt="Live Demo">
  </a>
  &nbsp;
  <a href="https://scalepredict.streamlit.app/calculator">
    <img src="https://img.shields.io/badge/Calculator-No%20Install%20Needed-orange?style=for-the-badge" alt="Calculator">
  </a>
</p>

<p align="center">
  <b>⭐ If this saved you from a wrong GPU choice — star the repo.</b>
</p>

---

## The Problem

You have 1 million images to process with AI.  
You open AWS and see:

```
T4   GPU  →  $0.52/hr
V100 GPU  →  $1.80/hr  
A100 GPU  →  $3.20/hr
```

You don't know which one to pick.  
You don't know how many hours you'll need.  
You guess. You pay. Sometimes you're wrong.

> *"I chose V100 for a job that turned out to be too easy —  
> could have done it on T4 for half the price."*  
> — Reddit user, r/learnmachinelearning

---

## The Solution

**Option A — Calculator (no install, 30 seconds):**

Open [scalepredict.streamlit.app/calculator](https://scalepredict.streamlit.app/calculator),
enter your data type, file count and model → see runtime instantly.

**Option B — Full benchmark (2 minutes, more accurate):**

```bash
python run_benchmark.py
```

Measures your actual machine. Then:

```
⚡ A100  →  0.4h   fastest
   V100  →  0.8h
   A10G  →  1.1h
   T4    →  2.3h
```

Look up the price yourself. Multiply. Done.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Step 1 — measure your machine (2 min)
python run_benchmark.py

# Step 2 — open dashboard
streamlit run scalepredict_app.py
```

Opens at `http://localhost:8501`

---

## Tested on Real Hardware

All three machines ran the same `run_benchmark.py` — no simulated data.

| Machine | CPU/GPU | Throughput | W Score | Ratio vs Lenovo |
|---------|---------|-----------|---------|-----------------|
| Lenovo L14 (Ryzen 7 Pro) | AMD CPU | 58 img/s | +0.054 | 1.0x baseline |
| Fujitsu H710 (Sandy Bridge) | Intel CPU | 14 img/s | -0.165 | 4.8x slower |
| Xeon + Quadro M4000 | Intel + GPU | 639 img/s | +0.730 | 7.6x faster |

### Cross-Machine Correlations

| Pair | Pearson r | Spearman ρ |
|------|-----------|------------|
| Lenovo ↔ Fujitsu | **0.9977** | **1.0000** |
| Lenovo ↔ Xeon+GPU | **0.9971** | **1.0000** |
| Fujitsu ↔ Xeon+GPU | **0.9998** | **1.0000** |

**Spearman ρ = 1.000 across all pairs** — measured, not theoretical.

---

## How It Works

```
run_benchmark.py
  → measures latency across batch sizes [1, 8, 32, 64, 128]
  → removes GPU warmup outliers automatically
  → computes W score = Q·D - T
  → saves scalepredict_profile.json

scalepredict_app.py
  → reads your profile
  → applies k(t,d) scaling model
  → predicts runtime on T4 / V100 / A100 / A10G
```

### The k(t,d) Model

```
k(t,d) = k₀ · e^(−αt) · (1 + β/d)

t  = batch size
d  = latency proxy (ms × 1000)
k₀ = architecture constant
```

Not a lookup table. Not a heuristic.  
Original formula — cross-architecture scaling model.

---

## Known Limitations

- Optimized for CNN inference (ResNet, YOLO, image classification)
- Transformer models with long context may show different memory access patterns — prediction less accurate for sequences > 512 tokens
- Prediction accuracy decreases for models with irregular memory access
- GPU warmup outliers are removed automatically (first batch excluded)

---

## Privacy

The `scalepredict_profile.json` contains:
- CPU model name
- RAM size
- Core count
- Benchmark results (latency, throughput)

**No usernames. No location. No personal data.**  
Open it in any text editor to verify before uploading.

---

## Files

```
ScalePredict/
├── run_benchmark.py      ← run this on your machine
├── scalepredict_app.py   ← Streamlit dashboard
├── calculator.py         ← simple calculator, no benchmark needed
├── requirements.txt      ← dependencies
└── README.md
```

---

## Roadmap

- [x] CPU benchmark (Lenovo L14)
- [x] CPU benchmark (Fujitsu H710)
- [x] GPU benchmark (Xeon + Quadro M4000)
- [x] Streamlit dashboard
- [x] Simple calculator (no install)
- [x] r > 0.997 on all 3 machine pairs
- [x] Known limitations documented
- [x] Privacy notice
- [ ] Transformer workload support
- [ ] GCP / Azure pricing links
- [ ] arXiv preprint
- [ ] pip package

---

## License

MIT — use freely.

---

*3 machines. 3 real benchmarks. Spearman ρ = 1.000.*  
*⭐ Star the repo if it helped you.*
