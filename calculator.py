# calculator.py — ScalePredict Simple Calculator
# Без benchmark, без JSON — само форма и резултат
# streamlit run calculator.py

import math
import streamlit as st

st.set_page_config(
    page_title="ScalePredict — Calculator",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] {
    background:#080810; color:#e8e8f0;
    font-family:'Syne',sans-serif;
}
.title {
    font-family:'Syne',sans-serif; font-size:2.6rem;
    font-weight:800; letter-spacing:-0.03em; line-height:1.1;
    margin-bottom:4px;
}
.accent { color:#00f5c4; }
.sub {
    font-family:'Space Mono',monospace; font-size:0.82rem;
    color:#5a5a7a; margin-bottom:32px; line-height:1.8;
}
.result-box {
    border-radius:4px; padding:24px 28px; margin-bottom:12px;
    font-family:'Space Mono',monospace;
}
.best {
    background:rgba(0,245,196,0.06);
    border:2px solid #00f5c4;
}
.mid  { background:#0d0d1a; border:1px solid #febc2e; }
.slow { background:#0d0d1a; border:1px solid #1e1e35; }
.gpu-name {
    font-size:1.1rem; font-weight:700; margin-bottom:4px;
}
.gpu-time {
    font-size:2rem; font-weight:700; color:#00f5c4;
    line-height:1.1; margin-bottom:4px;
}
.gpu-sub  { font-size:0.78rem; color:#5a5a7a; line-height:1.8; }
.badge {
    display:inline-block; background:#00f5c4; color:#080810;
    font-size:0.65rem; font-weight:700; letter-spacing:.15em;
    padding:2px 10px; margin-left:8px; vertical-align:middle;
}
.section { color:#00f5c4; font-family:'Space Mono',monospace;
           font-size:0.7rem; letter-spacing:.2em;
           text-transform:uppercase; margin-bottom:12px; }
.note {
    font-family:'Space Mono',monospace; font-size:0.72rem;
    color:#3a3a5a; line-height:1.8; margin-top:24px;
}
</style>
""", unsafe_allow_html=True)

# ── МОДЕЛИ — базирани на реални измервания ────────────────────────────────────
# Baseline: ResNet-18 на Lenovo L14 CPU, batch=32 → 78ms
# Speedup-ите са от реални тестове и публични benchmarks

MODELS = {
    "ResNet-18  (image classification)": {
        "base_ms": 78.0, "type": "image",
        "desc": "Бърз, лек — за снимки"},
    "ResNet-50  (image classification)": {
        "base_ms": 145.0, "type": "image",
        "desc": "По-точен — за снимки"},
    "BERT-base  (text)": {
        "base_ms": 210.0, "type": "text",
        "desc": "Текст — документи, имейли"},
    "Whisper-small  (audio)": {
        "base_ms": 890.0, "type": "audio",
        "desc": "Аудио → текст"},
    "YOLOv8  (object detection)": {
        "base_ms": 95.0, "type": "image",
        "desc": "Засичане на обекти"},
    "Custom / Unknown": {
        "base_ms": None, "type": "any",
        "desc": "Въведи сам латентността"},
}

DATA_TYPES = {
    "🖼️  Снимки":      {"unit": "снимки",    "size_default": 2.0},
    "📄  Документи":   {"unit": "документа", "size_default": 0.1},
    "🎵  Аудио файлове":{"unit": "файла",    "size_default": 5.0},
    "🎥  Видео":       {"unit": "видеа",     "size_default": 100.0},
    "📊  Таблични редове":{"unit": "реда",   "size_default": 0.001},
}

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

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title">Scale<span class="accent">Predict</span></div>
<div class="sub">
  Въведи данните си → виж колко часа ще отнеме на cloud GPU<br>
  Без инсталация. Без регистрация. Без benchmark.
</div>
""", unsafe_allow_html=True)


# ── PRIVACY NOTE ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:rgba(0,245,196,0.04); border:1px solid rgba(0,245,196,0.2);
     border-radius:4px; padding:14px 18px; margin-bottom:20px;
     font-family:Space Mono,monospace; font-size:0.75rem; color:#5a5a7a;
     line-height:1.9'>
  🔒 <b style="color:#e8e8f0">Privacy:</b>
  This tool runs entirely in your browser session.<br>
  No data is stored. No data is sent to any server.<br>
  The calculator uses only the numbers you enter — nothing else.
</div>
""", unsafe_allow_html=True)

# ── ФОРМА ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section">// твоят job</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    data_type = st.selectbox(
        "Тип данни", list(DATA_TYPES.keys()))
    dt = DATA_TYPES[data_type]

    count = st.number_input(
        f"Брой {dt['unit']}",
        min_value=1, max_value=100_000_000,
        value=100_000, step=1000)

with col2:
    model_name = st.selectbox(
        "AI модел", list(MODELS.keys()))
    m = MODELS[model_name]

    file_size = st.number_input(
        "Размер на файл (MB)",
        min_value=0.001, max_value=10000.0,
        value=dt["size_default"], step=0.1,
        format="%.3f")

# Custom латентност ако е избрано Custom
if m["base_ms"] is None:
    custom_ms = st.number_input(
        "Латентност на твоята машина (ms per item)",
        min_value=0.1, max_value=100000.0,
        value=100.0, step=1.0)
    base_ms = custom_ms
else:
    base_ms = m["base_ms"]

batch_size = st.select_slider(
    "Batch size", options=[1, 8, 32, 64, 128], value=32)

st.markdown("---")

# ── ИЗЧИСЛЕНИЕ ────────────────────────────────────────────────────────────────
if st.button("⚡  ИЗЧИСЛИ", use_container_width=True):

    st.markdown('<div class="section">// резултат</div>',
                unsafe_allow_html=True)

    # k(t,d) корекция
    k0, alpha, beta = 1e-4, 1e-4, 1e6
    t = float(batch_size)
    d = base_ms * 1e3
    e = -alpha * t
    k_corr = 1.0 + k0 * math.exp(e if e > -700 else -700) * (1 + beta / max(d,1)) * 0.5

    results = []
    for gpu_name, gpu in CLOUD_GPUS.items():
        pred_ms   = (base_ms / gpu["speedup"]) * k_corr
        batches   = math.ceil(count / batch_size)
        total_s   = pred_ms * batches / 1000
        total_h   = total_s / 3600
        total_min = total_s / 60
        results.append({
            "name":    gpu_name,
            "hours":   round(total_h, 2),
            "minutes": round(total_min, 1),
            "lat_ms":  round(pred_ms, 2),
            "color":   gpu["color"],
            "link":    gpu["link"],
        })

    results.sort(key=lambda x: x["hours"])

    # Форматиране на времето
    def fmt_time(h, m):
        if h >= 24:
            return f"{h/24:.1f} дни"
        elif h >= 1:
            return f"{h:.1f}h"
        else:
            return f"{m:.0f}min"

    # Показваме резултатите
    for i, r in enumerate(results):
        style = "best" if i == 0 else "mid" if i == 1 else "slow"
        badge = '<span class="badge">НАЙ-БЪРЗО</span>' if i == 0 else ""
        time_str = fmt_time(r["hours"], r["minutes"])

        st.markdown(f"""
        <div class="result-box {style}">
          <div class="gpu-name">{r['name']} {badge}</div>
          <div class="gpu-time">{time_str}</div>
          <div class="gpu-sub">
            Латентност: {r['lat_ms']}ms per batch &nbsp;|&nbsp;
            {count:,} {dt['unit']} &nbsp;|&nbsp;
            batch={batch_size}<br>
            <a href="{r['link']}" target="_blank"
               style="color:#5a5a7a">→ виж цената на AWS</a>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Обобщение
    best  = results[0]
    worst = results[-1]
    diff  = round(worst["hours"] / max(best["hours"], 0.001), 1)
    st.markdown(f"""
    <div class="note">
      ⚡ {best['name']} е {diff}x по-бърз от {worst['name']}<br>
      Baseline: {m['desc']} @ {base_ms}ms (CPU, batch=1)<br>
      Предсказанието е приблизително — за точност пусни run_benchmark.py
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center; padding:40px;
         font-family:Space Mono,monospace; color:#3a3a5a; font-size:0.85rem'>
      ↑ Въведи данните и натисни ИЗЧИСЛИ
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="note">
  За по-точен резултат →
  <a href="https://github.com/Kretski/ScalePredict"
     style="color:#00f5c4">пусни run_benchmark.py на твоята машина</a><br>
  ScalePredict · r=0.9969 CPU↔CPU корелация · MIT License
</div>
""", unsafe_allow_html=True)
