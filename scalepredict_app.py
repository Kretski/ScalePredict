# scalepredict_app.py — ScalePredict v0.2
# streamlit run scalepredict_app.py

import math, json, os, pathlib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

st.set_page_config(
    page_title="ScalePredict — How Long Will It Take?",
    page_icon="⚡", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { background:#080810; color:#e8e8f0; font-family:'Syne',sans-serif; }
.big-num { font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700; color:#00f5c4; }
.label   { font-family:'Space Mono',monospace; font-size:0.68rem; color:#5a5a7a; letter-spacing:.15em; text-transform:uppercase; }
.mbox    { background:#0d0d1a; border:1px solid #1e1e35; border-radius:4px; padding:18px 20px; margin-bottom:8px; }
.tag     { font-family:'Space Mono',monospace; font-size:0.7rem; color:#00f5c4; letter-spacing:.2em; text-transform:uppercase; margin-bottom:12px; }
.rcard   { background:#0d0d1a; border:1px solid #1e1e35; border-left:3px solid #00f5c4; border-radius:4px; padding:16px 20px; margin-bottom:10px; font-family:'Space Mono',monospace; font-size:0.82rem; }
.rbest   { border-left-color:#00f5c4; background:rgba(0,245,196,.04); }
.rmid    { border-left-color:#febc2e; }
.rslow   { border-left-color:#ff6b35; }
.formula { background:#050508; border:1px solid #1e1e35; border-radius:4px; padding:16px 20px; font-family:'Space Mono',monospace; font-size:0.8rem; color:#00f5c4; line-height:2.2; }
.disclaimer { background:#0d0d1a; border:1px solid #2a2a1a; border-radius:4px; padding:10px 16px; font-family:'Space Mono',monospace; font-size:0.7rem; color:#5a5a4a; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

MACHINES = {
    "Lenovo L14 (laptop)": {
        "color": "#2196F3", "has_cuda": False,
        "batch_lat":  {1:55.0, 8:62.0, 32:78.0, 64:95.0, 128:130.0},
        "throughput": {1:0.2,  8:1.3,  32:4.1,  64:6.7,  128:9.8},
    },
    "Fujitsu Server (CPU)": {
        "color": "#4CAF50", "has_cuda": False,
        "batch_lat":  {1:13.0, 8:14.5, 32:18.0, 64:24.0, 128:35.0},
        "throughput": {1:0.8,  8:5.5,  32:17.8, 64:26.7, 128:36.6},
    },
    "Xeon + Quadro M4000 (GPU)": {
        "color": "#FF9800", "has_cuda": True,
        "batch_lat":  {1:4.63, 8:4.84, 32:3.47, 64:3.63, 128:3.69},
        "throughput": {1:1.3,  8:10.7, 32:42.7, 64:85.3, 128:170.7},
    },
}

# FIX 3: GPU speedup с документиран източник
# Приближения от MLPerf inference datacenter (ResNet family)
# https://mlcommons.org/benchmarks/inference-datacenter/
CLOUD_GPUS = {
    "T4":   {"speedup": 14.0, "color": "#66BB6A", "link": "https://aws.amazon.com/ec2/instance-types/g4/"},
    "V100": {"speedup": 25.0, "color": "#FFB300", "link": "https://aws.amazon.com/ec2/instance-types/p3/"},
    "A100": {"speedup": 44.0, "color": "#EF5350", "link": "https://aws.amazon.com/ec2/instance-types/p4/"},
    "A10G": {"speedup": 20.0, "color": "#42A5F5", "link": "https://aws.amazon.com/ec2/instance-types/g5/"},
}

BATCH_SIZES   = [1, 8, 32, 64, 128]
DEFAULT_K0    = 1e-4
DEFAULT_ALPHA = 1e-4
DEFAULT_BETA  = 1e6

def dynamic_k(k0, alpha, beta, t, d):
    e = -alpha * t
    return 0.0 if e < -700 else k0 * math.exp(e) * (1.0 + beta / max(d, 1.0))

def predict_runtime(local_lat_ms, batch_size, total_samples, k0, alpha, beta):
    results = []
    for name, gpu in CLOUD_GPUS.items():
        k_corr      = 1.0 + dynamic_k(k0, alpha, beta, float(batch_size), local_lat_ms*1e3) * 0.5
        pred_lat_ms = (local_lat_ms / gpu["speedup"]) * k_corr
        total_s     = pred_lat_ms * math.ceil(total_samples / batch_size) / 1000
        results.append({
            "name": name, "lat_ms": round(pred_lat_ms, 3),
            "hours": round(total_s/3600, 2), "minutes": round(total_s/60, 1),
            "color": gpu["color"], "link": gpu["link"],
        })
    results.sort(key=lambda x: x["hours"])
    return results

def get_corr():
    names = list(MACHINES.keys()); out = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            m1, m2 = names[i], names[j]
            l1 = [MACHINES[m1]["batch_lat"][b] for b in BATCH_SIZES]
            l2 = [MACHINES[m2]["batch_lat"][b] for b in BATCH_SIZES]
            r, _ = pearsonr(l1, l2); rho, _ = spearmanr(l1, l2)
            out[f"{m1[:10]} ↔ {m2[:10]}"] = {"pearson": round(r,4), "spearman": round(rho,4)}
    return out

def plot_latency():
    fig, ax = plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor('#080810'); ax.set_facecolor('#0a0a1a')
    for name, m in MACHINES.items():
        ax.plot(BATCH_SIZES, [m["batch_lat"][b] for b in BATCH_SIZES],
                'o-', color=m["color"], lw=2, ms=6, label=name)
    ax.set_xlabel("Batch Size", color='white', fontsize=9)
    ax.set_ylabel("Latency (ms)", color='white', fontsize=9)
    ax.set_title("Latency vs Batch Size", color='white', fontsize=10)
    ax.set_yscale('log')
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e1e35')
    fig.tight_layout(); return fig

def plot_runtime(results):
    fig, ax = plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor('#080810'); ax.set_facecolor('#0a0a1a')
    names = [r["name"] for r in results]; hours = [r["hours"] for r in results]
    bars = ax.bar(range(len(names)), hours, color=[r["color"] for r in results], alpha=0.85)
    bars[0].set_edgecolor('white'); bars[0].set_linewidth(2.0)
    for i, (h, m) in enumerate(zip(hours, [r["minutes"] for r in results])):
        ax.text(i, h + max(hours)*0.03, f"{h:.1f}h" if h >= 1 else f"{m:.0f}min",
                ha='center', color='white', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"⚡ {names[0]}"] + names[1:], color='white', fontsize=9)
    ax.set_ylabel("Runtime (hours)", color='white', fontsize=9)
    ax.set_title("How Long Will It Take?", color='#00f5c4', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e1e35')
    fig.tight_layout(); return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="tag">// your workload</div>', unsafe_allow_html=True)
    batch_size    = st.select_slider("Batch size", options=BATCH_SIZES, value=32)
    total_samples = st.number_input("Total images", min_value=100,
                                    max_value=50_000_000, value=1_000_000, step=10_000)

    uploaded = st.file_uploader("📂 Upload your profile.json", type="json",
        help="Run run_benchmark.py → upload scalepredict_profile.json here")
    if uploaded:
        client_data = json.load(uploaded)
        st.session_state["client_profile"] = client_data
        # FIX 2: Зареждаме k параметрите от профила
        if "model_params" in client_data:
            st.session_state["k0"]    = client_data["model_params"]["k0"]
            st.session_state["alpha"] = client_data["model_params"]["alpha"]
            st.session_state["beta"]  = client_data["model_params"]["beta"]
        hw = client_data["hardware"]
        st.success(f"✅ {hw['cpu'][:30]}... {'GPU' if hw['has_cuda'] else 'CPU'}")
    elif "client_profile" not in st.session_state:
        st.session_state["client_profile"] = None

    st.markdown("---")
    st.markdown('<div class="tag">// demo machine</div>', unsafe_allow_html=True)
    local_machine = st.selectbox("Demo machine (replaced when you upload)",
                                 list(MACHINES.keys()), index=0)

    st.markdown("---")
    st.markdown('<div class="tag">// k(t,d) model</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula">k(t,d) = k₀·e^(−αt)·(1+β/d)<br>W = Q·D − T</div>',
                unsafe_allow_html=True)

    # FIX 2: Параметрите се използват реално — без override след това
    with st.expander("Advanced parameters"):
        k0    = st.number_input("k₀", value=st.session_state.get("k0",    DEFAULT_K0),    format="%.2e")
        alpha = st.number_input("α",  value=st.session_state.get("alpha", DEFAULT_ALPHA), format="%.2e")
        beta  = st.number_input("β",  value=st.session_state.get("beta",  DEFAULT_BETA),  format="%.2e")
    # k0, alpha, beta идват от widgets — не се презаписват

    st.markdown("---")
    # FIX 3: Disclaimer за speedup
    st.markdown("""<div class="disclaimer">
        ⚠️ GPU speedup = approximate estimates<br>
        Source: MLPerf inference datacenter<br>
        (ResNet family, batch=32 baseline)<br>
        Real results vary by model and load.
    </div>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Syne,sans-serif; font-size:2.8rem; font-weight:800;
     letter-spacing:-0.03em; margin-bottom:4px'>
  Scale<span style='color:#00f5c4'>Predict</span>
</div>
<div style='font-family:Space Mono,monospace; font-size:0.82rem;
     color:#5a5a7a; margin-bottom:24px'>
  2-минутен тест на твоя компютър → колко часа ще отнеме job-ът ти на cloud GPU
</div>
""", unsafe_allow_html=True)

# ── МЕТРИКИ ───────────────────────────────────────────────────────────────────
m       = MACHINES[local_machine]
lat     = m["batch_lat"][batch_size]
corrs   = get_corr()
results = predict_runtime(lat, batch_size, total_samples, k0, alpha, beta)
best    = results[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="mbox"><div class="big-num">{lat:.0f}ms</div>'
                f'<div class="label">латентност (batch={batch_size})</div></div>', unsafe_allow_html=True)
with c2:
    lbl = f"{best['hours']}h" if best['hours'] >= 1 else f"{best['minutes']}min"
    st.markdown(f'<div class="mbox"><div class="big-num">{lbl}</div>'
                f'<div class="label">най-бързо ({best["name"]})</div></div>', unsafe_allow_html=True)
with c3:
    r_val = list(corrs.values())[0]["pearson"]
    clr   = "#00f5c4" if abs(r_val) > 0.9 else "#FFB300"
    st.markdown(f'<div class="mbox"><div class="big-num" style="color:{clr}">r={r_val:.4f}</div>'
                f'<div class="label">CPU↔CPU корелация</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="mbox"><div class="big-num">{total_samples:,}</div>'
                f'<div class="label">изображения</div></div>', unsafe_allow_html=True)

st.markdown("---")

if not st.session_state.get("client_profile"):
    st.markdown("""
    <div style='background:rgba(0,245,196,0.06); border:1px solid rgba(0,245,196,0.4);
         border-radius:4px; padding:20px 24px; margin-bottom:8px; font-family:Space Mono,monospace;'>
      <div style='color:#00f5c4; font-size:0.85rem; font-weight:700; margin-bottom:8px'>
        ⚡ ДЕМО РЕЖИМ — виждаш примерни данни
      </div>
      <div style='color:#5a5a7a; font-size:0.8rem; line-height:1.9'>
        1. Свали <b style="color:#e8e8f0">run_benchmark.py</b> от GitHub<br>
        2. Пусни го локално — 2 минути<br>
        3. Качи <b style="color:#e8e8f0">scalepredict_profile.json</b> вляво
      </div>
    </div>""", unsafe_allow_html=True)

col_l, col_r = st.columns(2)
with col_l:
    st.markdown('<div class="tag">// latency профил</div>', unsafe_allow_html=True)
    st.pyplot(plot_latency())
    st.markdown('<div class="tag">// корелации</div>', unsafe_allow_html=True)
    for pair, vals in corrs.items():
        r    = vals["pearson"]
        icon = "🟢" if abs(r) > 0.9 else "🟡" if abs(r) > 0.5 else "🔴"
        st.markdown(f'<div class="rcard">{icon} <b>{pair}</b><br>'
                    f'Pearson r = {r:.4f} &nbsp;|&nbsp; Spearman ρ = {vals["spearman"]:.4f}</div>',
                    unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="tag">// колко часа ще отнеме?</div>', unsafe_allow_html=True)
    st.pyplot(plot_runtime(results))
    st.markdown('<div class="tag">// препоръка</div>', unsafe_allow_html=True)
    for i, r in enumerate(results):
        style    = "rbest" if i==0 else "rmid" if i==1 else "rslow"
        time_str = f"{r['hours']}h" if r['hours'] >= 1 else f"{r['minutes']}min"
        st.markdown(
            f'<div class="rcard {style}"><b>{r["name"]}</b> {"⚡ НАЙ-БЪРЗО" if i==0 else ""}<br>'
            f'Латентност: {r["lat_ms"]}ms &nbsp;|&nbsp; Време: <b>{time_str}</b><br>'
            f'<a href="{r["link"]}" target="_blank" style="color:#5a5a7a; font-size:0.75rem">'
            f'→ виж актуалната цена</a></div>', unsafe_allow_html=True)
    # FIX 3: Видим disclaimer
    st.markdown("""<div class="disclaimer">
        Speedup values are approximate (MLPerf inference benchmarks).<br>
        Run a small test on your target GPU before committing.
    </div>""", unsafe_allow_html=True)

# ── РЕАЛЕН ПРОФИЛ ─────────────────────────────────────────────────────────────
profile = None
if st.session_state.get("client_profile"):
    profile = st.session_state["client_profile"]
elif pathlib.Path("scalepredict_profile.json").exists():
    with open("scalepredict_profile.json") as f:
        profile = json.load(f)

if profile:
    st.markdown("---")
    st.markdown('<div class="tag">// твоят реален профил</div>', unsafe_allow_html=True)
    hw = profile["hardware"]

    # FIX 1: Показваме W = Q·D − T компонентите
    w_detail = ""
    if "w_components" in profile:
        wc = profile["w_components"]
        w_detail = f"&nbsp; (Q={wc['Q']} · D={wc['D']} − T={wc['T']})"

    st.markdown(
        f'<div class="rcard rbest">✅ <b>{hw["cpu"][:50]}</b><br>'
        f'RAM: {hw["ram_mb"]}MB | Ядра: {hw["cores_physical"]} | '
        f'CUDA: {"✅" if hw["has_cuda"] else "❌"} | '
        f'W = {profile["w_score"]}{w_detail}</div>', unsafe_allow_html=True)

    bench = profile["benchmark"]
    cols  = st.columns(len(bench))
    for i, (bs, data) in enumerate(bench.items()):
        with cols[i]:
            st.markdown(
                f'<div class="mbox"><div class="big-num" style="font-size:1.4rem">{data["avg_ms"]}ms</div>'
                f'<div class="label">batch={bs}<br>{data["throughput"]} img/s</div></div>',
                unsafe_allow_html=True)

    # Използваме k от профила (FIX 2)
    prof_k0    = profile.get("model_params", {}).get("k0",    k0)
    prof_alpha = profile.get("model_params", {}).get("alpha", alpha)
    prof_beta  = profile.get("model_params", {}).get("beta",  beta)
    best_bs    = int(profile["summary"]["best_batch"])
    real_lat   = float(bench[str(best_bs)]["avg_ms"])
    real_res   = predict_runtime(real_lat, best_bs, total_samples, prof_k0, prof_alpha, prof_beta)

    st.markdown(f'<div class="tag">// предсказание от твоите данни (batch={best_bs})</div>',
                unsafe_allow_html=True)
    rcols = st.columns(len(real_res))
    for i, r in enumerate(real_res):
        time_str = f"{r['hours']}h" if r['hours'] >= 1 else f"{r['minutes']}min"
        with rcols[i]:
            st.markdown(
                f'<div class="mbox"><div class="big-num" style="font-size:1.6rem; color:{r["color"]}">'
                f'{time_str}</div>'
                f'<div class="label">{r["name"]} {"⚡" if i==0 else ""}<br>'
                f'{r["lat_ms"]}ms</div></div>', unsafe_allow_html=True)

# ── EXPORT ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.download_button(
    "📥  Download JSON Report",
    data=json.dumps({
        "machine": local_machine, "batch_size": batch_size,
        "total_samples": total_samples,
        "model_params": {"k0": k0, "alpha": alpha, "beta": beta},
        "predictions": results, "correlations": corrs,
        "speedup_note": "GPU speedup approx. from MLPerf inference datacenter benchmarks",
    }, indent=2),
    file_name="scalepredict_report.json", mime="application/json")

st.markdown("""<div style='font-family:Space Mono,monospace; font-size:0.68rem;
    color:#3a3a5a; margin-top:12px; line-height:1.8'>
    W = Q·D − T &nbsp;|&nbsp; r=0.9969 CPU↔CPU (2 machines) &nbsp;|&nbsp;
    GPU speedup: MLPerf approx.
</div>""", unsafe_allow_html=True)
