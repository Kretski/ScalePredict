# scalepredict_app.py — ScalePredict v0.2
# Показва само ВРЕМЕТО — цената клиентът намира сам
# streamlit run scalepredict_app.py

import math, json, os, pathlib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

st.set_page_config(
    page_title="ScalePredict — How Long Will It Take?",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { background:#080810; color:#e8e8f0; font-family:'Syne',sans-serif; }
.big-num { font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700; color:#00f5c4; }
.label   { font-family:'Space Mono',monospace; font-size:0.68rem; color:#5a5a7a; letter-spacing:.15em; text-transform:uppercase; }
.mbox    { background:#0d0d1a; border:1px solid #1e1e35; border-radius:4px; padding:18px 20px; margin-bottom:8px; }
.tag     { font-family:'Space Mono',monospace; font-size:0.7rem; color:#00f5c4; letter-spacing:.2em; text-transform:uppercase; margin-bottom:12px; }
.rcard   { background:#0d0d1a; border:1px solid #1e1e35; border-left:3px solid #00f5c4;
           border-radius:4px; padding:16px 20px; margin-bottom:10px;
           font-family:'Space Mono',monospace; font-size:0.82rem; }
.rbest   { border-left-color:#00f5c4; background:rgba(0,245,196,.04); }
.rmid    { border-left-color:#febc2e; }
.rslow   { border-left-color:#ff6b35; }
.formula { background:#050508; border:1px solid #1e1e35; border-radius:4px;
           padding:16px 20px; font-family:'Space Mono',monospace;
           font-size:0.8rem; color:#00f5c4; line-height:2.2; }
</style>
""", unsafe_allow_html=True)

# ── ДАННИ ─────────────────────────────────────────────────────────────────────
MACHINES = {
    "Lenovo L14 (laptop)": {
        "color": "#2196F3",
        "has_cuda": False,
        "batch_lat": {1:55.0, 8:62.0, 32:78.0, 64:95.0, 128:130.0},
        "throughput": {1:0.2, 8:1.3, 32:4.1, 64:6.7, 128:9.8},
    },
    "Fujitsu Server (CPU)": {
        "color": "#4CAF50",
        "has_cuda": False,
        "batch_lat": {1:13.0, 8:14.5, 32:18.0, 64:24.0, 128:35.0},
        "throughput": {1:0.8, 8:5.5, 32:17.8, 64:26.7, 128:36.6},
    },
    "Xeon + Quadro M4000 (GPU)": {
        "color": "#FF9800",
        "has_cuda": True,
        "batch_lat": {1:4.63, 8:4.84, 32:3.47, 64:3.63, 128:3.69},
        "throughput": {1:1.3, 8:10.7, 32:42.7, 64:85.3, 128:170.7},
    },
}

# GPU speedup спрямо Lenovo batch=32 (78ms baseline)
CLOUD_GPUS = {
    "T4":   {"speedup": 14.0, "color": "#66BB6A",
             "link": "https://aws.amazon.com/ec2/instance-types/g4/"},
    "V100": {"speedup": 25.0, "color": "#FFB300",
             "link": "https://aws.amazon.com/ec2/instance-types/p3/"},
    "A100": {"speedup": 44.0, "color": "#EF5350",
             "link": "https://aws.amazon.com/ec2/instance-types/p4/"},
    "A10G": {"speedup": 20.0, "color": "#42A5F5",
             "link": "https://aws.amazon.com/ec2/instance-types/g5/"},
}

BATCH_SIZES = [1, 8, 32, 64, 128]

def dynamic_k(k0, alpha, beta, t, d):
    e = -alpha * t
    if e < -700: return 0.0
    return k0 * math.exp(e) * (1.0 + beta / max(d, 1.0))

def predict_runtime(local_lat_ms, batch_size, total_samples, k0, alpha, beta):
    results = []
    for name, gpu in CLOUD_GPUS.items():
        t      = float(batch_size)
        d      = local_lat_ms * 1e3
        k_corr = 1.0 + dynamic_k(k0, alpha, beta, t, d) * 0.5
        pred_lat_ms = (local_lat_ms / gpu["speedup"]) * k_corr
        batches     = math.ceil(total_samples / batch_size)
        total_s     = pred_lat_ms * batches / 1000
        total_h     = total_s / 3600
        total_min   = total_s / 60
        results.append({
            "name":     name,
            "lat_ms":   round(pred_lat_ms, 3),
            "hours":    round(total_h, 2),
            "minutes":  round(total_min, 1),
            "color":    gpu["color"],
            "link":     gpu["link"],
        })
    results.sort(key=lambda x: x["hours"])  # най-бързо първо
    return results

def get_corr():
    names = list(MACHINES.keys())
    out = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            m1, m2 = names[i], names[j]
            l1 = [MACHINES[m1]["batch_lat"][b] for b in BATCH_SIZES]
            l2 = [MACHINES[m2]["batch_lat"][b] for b in BATCH_SIZES]
            r, _   = pearsonr(l1, l2)
            rho, _ = spearmanr(l1, l2)
            out[f"{m1[:10]} ↔ {m2[:10]}"] = {
                "pearson": round(r,4), "spearman": round(rho,4)}
    return out

def plot_latency():
    fig, ax = plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor('#080810')
    ax.set_facecolor('#0a0a1a')
    for name, m in MACHINES.items():
        lats = [m["batch_lat"][b] for b in BATCH_SIZES]
        ax.plot(BATCH_SIZES, lats, 'o-', color=m["color"], lw=2, ms=6, label=name)
    ax.set_xlabel("Batch Size", color='white', fontsize=9)
    ax.set_ylabel("Latency (ms)", color='white', fontsize=9)
    ax.set_title("Latency vs Batch Size", color='white', fontsize=10)
    ax.set_yscale('log')
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e1e35')
    fig.tight_layout()
    return fig

def plot_runtime(results):
    fig, ax = plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor('#080810')
    ax.set_facecolor('#0a0a1a')
    names  = [r["name"]  for r in results]
    hours  = [r["hours"] for r in results]
    colors = [r["color"] for r in results]
    bars = ax.bar(range(len(names)), hours, color=colors, alpha=0.85)
    bars[0].set_edgecolor('white')
    bars[0].set_linewidth(2.0)
    for i, (h, m) in enumerate(zip(hours, [r["minutes"] for r in results])):
        label = f"{h:.1f}h" if h >= 1 else f"{m:.0f}min"
        ax.text(i, h + max(hours)*0.03, label,
                ha='center', color='white', fontsize=10, fontweight='bold')
    # Добавяме ⚡ на най-бързия
    xlabels = [f"⚡ {names[0]}"] + names[1:]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(xlabels, color='white', fontsize=9)
    ax.set_ylabel("Runtime (hours)", color='white', fontsize=9)
    ax.set_title("How Long Will It Take?  (най-бързо → най-бавно)",
                 color='#00f5c4', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e1e35')
    fig.tight_layout()
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="tag">// your workload</div>', unsafe_allow_html=True)

    batch_size = st.select_slider(
        "Batch size", options=BATCH_SIZES, value=32)

    total_samples = st.number_input(
        "Total images to process",
        min_value=100, max_value=50_000_000,
        value=1_000_000, step=10_000)

    local_machine = st.selectbox(
        "Your local machine",
        list(MACHINES.keys()), index=0)

    st.markdown("---")
    st.markdown('<div class="tag">// k(t,d) model</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula">k(t,d) = k₀·e^(−αt)·(1+β/d)</div>',
                unsafe_allow_html=True)

    with st.expander("Advanced parameters"):
        k0    = st.number_input("k₀", value=1e-4, format="%.2e")
        alpha = st.number_input("α",  value=1e-4, format="%.2e")
        beta  = st.number_input("β",  value=1e6,  format="%.2e")
    k0, alpha, beta = 1e-4, 1e-4, 1e6

    st.markdown("---")
    st.markdown("""<div style='font-family:Space Mono,monospace;
        font-size:0.68rem; color:#3a3a5a; line-height:1.8'>
        ScalePredict v0.2<br>
        Предсказва ВРЕМЕТО — не цената.<br>
        Цената намери сам на:<br>
        aws.amazon.com/ec2/pricing<br>
        cloud.google.com/compute/gpus<br><br>
        r=0.9969 измерена корелация<br>
        CPU↔CPU на реален хардуер
    </div>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Syne,sans-serif; font-size:2.8rem;
     font-weight:800; letter-spacing:-0.03em; margin-bottom:4px'>
  Scale<span style='color:#00f5c4'>Predict</span>
</div>
<div style='font-family:Space Mono,monospace; font-size:0.82rem;
     color:#5a5a7a; margin-bottom:24px'>
  Пусни 2-минутен тест на твоя компютър →
  разбери колко часа ще отнеме job-ът ти на cloud GPU
</div>
""", unsafe_allow_html=True)

# ── МЕТРИКИ ───────────────────────────────────────────────────────────────────
m      = MACHINES[local_machine]
lat    = m["batch_lat"][batch_size]
tput   = m["throughput"][batch_size]
corrs  = get_corr()
results= predict_runtime(lat, batch_size, total_samples, k0, alpha, beta)
best   = results[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f'<div class="mbox"><div class="big-num">{lat:.0f}ms</div>'
                f'<div class="label">твоята латентност (batch={batch_size})</div></div>',
                unsafe_allow_html=True)
with col2:
    label = f"{best['hours']}h" if best['hours'] >= 1 else f"{best['minutes']}min"
    st.markdown(f'<div class="mbox"><div class="big-num">{label}</div>'
                f'<div class="label">най-бързо ({best["name"]})</div></div>',
                unsafe_allow_html=True)
with col3:
    r_val = list(corrs.values())[0]["pearson"]
    color = "#00f5c4" if abs(r_val) > 0.9 else "#FFB300"
    st.markdown(f'<div class="mbox"><div class="big-num" style="color:{color}">'
                f'r={r_val:.4f}</div>'
                f'<div class="label">CPU↔CPU корелация</div></div>',
                unsafe_allow_html=True)
with col4:
    imgs_fmt = f"{total_samples:,}"
    st.markdown(f'<div class="mbox"><div class="big-num">{imgs_fmt}</div>'
                f'<div class="label">изображения за обработка</div></div>',
                unsafe_allow_html=True)

st.markdown("---")

# ── ГРАФИКИ ───────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown('<div class="tag">// latency профил на твоята машина</div>',
                unsafe_allow_html=True)
    st.pyplot(plot_latency())

    st.markdown('<div class="tag">// корелации между машините</div>',
                unsafe_allow_html=True)
    for pair, vals in corrs.items():
        r = vals["pearson"]
        icon = "🟢" if abs(r) > 0.9 else "🟡" if abs(r) > 0.5 else "🔴"
        st.markdown(f'<div class="rcard">{icon} <b>{pair}</b><br>'
                    f'Pearson r = {r:.4f} &nbsp;|&nbsp; '
                    f'Spearman ρ = {vals["spearman"]:.4f}</div>',
                    unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="tag">// колко часа ще отнеме?</div>',
                unsafe_allow_html=True)
    st.pyplot(plot_runtime(results))

    st.markdown('<div class="tag">// препоръка</div>', unsafe_allow_html=True)
    for i, r in enumerate(results):
        style = "rbest" if i == 0 else "rmid" if i == 1 else "rslow"
        badge = "⚡ НАЙ-БЪРЗО" if i == 0 else ""
        time_str = f"{r['hours']}h" if r['hours'] >= 1 else f"{r['minutes']}min"
        st.markdown(
            f'<div class="rcard {style}">'
            f'<b>{r["name"]}</b> {badge}<br>'
            f'Латентност: {r["lat_ms"]}ms &nbsp;|&nbsp; '
            f'Време: <b>{time_str}</b><br>'
            f'<a href="{r["link"]}" target="_blank" '
            f'style="color:#5a5a7a; font-size:0.75rem">'
            f'→ виж актуалната цена</a>'
            f'</div>',
            unsafe_allow_html=True)

# ── РЕАЛЕН ПРОФИЛ ─────────────────────────────────────────────────────────────
if pathlib.Path("scalepredict_profile.json").exists():
    with open("scalepredict_profile.json") as f:
        profile = json.load(f)

    st.markdown("---")
    st.markdown('<div class="tag">// твоят реален профил (от run_benchmark.py)</div>',
                unsafe_allow_html=True)

    hw = profile["hardware"]
    st.markdown(
        f'<div class="rcard rbest">✅ <b>{hw["cpu"][:50]}</b><br>'
        f'RAM: {hw["ram_mb"]}MB | '
        f'Ядра: {hw["cores_physical"]} физически | '
        f'CUDA: {"✅" if hw["has_cuda"] else "❌"} | '
        f'W score: <b>{profile["w_score"]}</b></div>',
        unsafe_allow_html=True)

    bench = profile["benchmark"]
    cols  = st.columns(len(bench))
    for i, (bs, data) in enumerate(bench.items()):
        with cols[i]:
            st.markdown(
                f'<div class="mbox">'
                f'<div class="big-num" style="font-size:1.4rem">{data["avg_ms"]}ms</div>'
                f'<div class="label">batch={bs}<br>{data["throughput"]} img/s</div>'
                f'</div>', unsafe_allow_html=True)

    best_bs  = int(profile["summary"]["best_batch"])
    real_lat = float(bench[str(best_bs)]["avg_ms"])
    real_res = predict_runtime(real_lat, best_bs, total_samples, k0, alpha, beta)

    st.markdown(f'<div class="tag">// предсказание от ТВОИТЕ реални данни (batch={best_bs})</div>',
                unsafe_allow_html=True)

    rcols = st.columns(len(real_res))
    for i, r in enumerate(real_res):
        time_str = f"{r['hours']}h" if r['hours'] >= 1 else f"{r['minutes']}min"
        badge = "⚡" if i == 0 else ""
        with rcols[i]:
            st.markdown(
                f'<div class="mbox">'
                f'<div class="big-num" style="font-size:1.6rem; '
                f'color:{r["color"]}">{time_str}</div>'
                f'<div class="label">{r["name"]} {badge}<br>'
                f'{r["lat_ms"]}ms латентност</div>'
                f'</div>', unsafe_allow_html=True)

# ── EXPORT ────────────────────────────────────────────────────────────────────
st.markdown("---")
report = {
    "machine": local_machine,
    "batch_size": batch_size,
    "total_samples": total_samples,
    "predictions": results,
    "correlations": {k: v for k, v in corrs.items()},
}
st.download_button(
    "📥  Download JSON Report",
    data=json.dumps(report, indent=2),
    file_name="scalepredict_report.json",
    mime="application/json")

st.markdown("""<div style='font-family:Space Mono,monospace; font-size:0.68rem;
    color:#3a3a5a; margin-top:12px; line-height:1.8'>
    ScalePredict предсказва времето — цената намери сам на сайта на доставчика.
    Базирано на реални измервания: r=0.9969 CPU↔CPU корелация.
</div>""", unsafe_allow_html=True)
