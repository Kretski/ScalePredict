# ⚡ ScalePredict
# ScalePredict
## Tested on Real Hardware

| Machine | Type | W Score | Best Batch |
|---------|------|---------|------------|
| Lenovo L14 | AMD Ryzen CPU | 0.054 | 128 |
| Fujitsu Server | Intel CPU | 0.073 | 128 |
| AMD64 (user) | AMD CPU | 0.054 | 1 |
| Xeon + Quadro M4000 | Intel + GPU | 0.730 | 128 |

CPU↔CPU correlation: r=0.9969
Note: CPU↔GPU correlation is negative by design —
GPU latency decreases with batch size, CPU increases.
<p align="center">
  <a href="https://github.com/Kretski/ScalePredict">
    <img src="https://img.shields.io/github/stars/Kretski/ScalePredict?style=social" alt="GitHub Stars">
  </a>
  <span> • </span>
  <strong>If you like this project, please give it a star! ⭐</strong>
</p>

<img width="1875" height="879" alt="lighshot_2026-03-06_07-09-58" src="https://github.com/user-attachments/assets/612de392-9124-4468-9f2a-b4aa8e046880" />

## The Problem
<img width="1887" height="900" alt="lighshot_2026-03-06_07-10-21" src="https://github.com/user-attachments/assets/5f7d2aa2-a1ea-437c-87c4-fca49ff3738f" />

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

---

## The Solution

```bash
python run_benchmark.py
```

2 minutes on your laptop. Then:

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

| Machine | Type | Max Throughput | W Score |
|---------|------|---------------|---------|
| Lenovo L14 | AMD Ryzen CPU | 58 img/s | 0.054 |
| Xeon + Quadro M4000 | Intel Xeon + GPU | 639 img/s | 0.730 |

**CPU↔CPU correlation: r = 0.9969** — measured, not theoretical.

---

## How It Works

```
run_benchmark.py
  → measures latency across batch sizes [1, 8, 32, 64, 128]
  → removes GPU warmup outliers automatically  
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

This is the original formula behind the cross-architecture prediction.  
Not a lookup table. Not a heuristic.

---

## Files

```
ScalePredict/
├── run_benchmark.py      ← run this on your machine
├── scalepredict_app.py   ← Streamlit dashboard  
├── requirements.txt      ← dependencies
└── README.md
```

---

## Requirements

```
Python 3.8+
torch >= 2.0
torchvision
psutil
streamlit
matplotlib
scipy
scikit-learn
numpy
```

---

## Results Example

Running `run_benchmark.py` on Xeon + Quadro M4000:

```
batch=  1  →   5.9ms   170 img/s
batch=  8  →  17.0ms   470 img/s
batch= 32  →  53.8ms   594 img/s
batch= 64  → 104.4ms   613 img/s
batch=128  → 200.2ms   639 img/s

W score: 0.7295  ✅ Production ready
```

---

## Roadmap

- [x] CPU benchmark (Lenovo L14)
- [x] GPU benchmark (Xeon + Quadro M4000)  
- [x] Streamlit dashboard
- [ ] Third machine validation
- [ ] arXiv preprint
- [ ] pip package

---

## License

MIT — use freely.

---

*Based on real measurements from 3 machines.  
CPU↔CPU correlation r=0.9969.*



## Calculator (no install needed)

Just open the browser and enter your numbers:
https://scalepredict.streamlit.app/calculator
