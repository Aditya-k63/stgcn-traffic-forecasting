import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Forecast · STGCN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS  — clean light theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f4f6f9;
    color: #1a1f2e;
}

.stApp {
    background-color: #f4f6f9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e6ed !important;
}
[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #1a1f2e !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e6ed;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em;
    color: #6b7280 !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #1a1f2e !important;
    font-size: 1.55rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    color: #6b7280 !important;
}

/* ── Button ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 0 !important;
    width: 100% !important;
    transition: background 0.18s ease !important;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background-color: #1d4ed8 !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] label p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e6ed !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: #e2e6ed !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f4f6f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj, dtype=torch.float32))
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = torch.einsum("ij,btjf->btif", self.A, x)
        return self.linear(x)


class STGCN(nn.Module):
    def __init__(self, adj, num_features):
        super().__init__()
        self.temporal1 = nn.Conv2d(num_features, 64, (3, 1), padding=(1, 0))
        self.graph1    = GraphConv(64, 64, adj)
        self.temporal2 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
        self.temporal3 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
        self.graph2    = GraphConv(64, 64, adj)
        self.temporal4 = nn.Conv2d(64, 64, (3, 1), padding=(1, 0))
        self.fc        = nn.Linear(64, 6)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.temporal1(x))
        x = x.permute(0, 2, 3, 1);  x = self.graph1(x)
        x = x.permute(0, 3, 1, 2);  x = torch.relu(self.temporal2(x))
        x = torch.relu(self.temporal3(x))
        x = x.permute(0, 2, 3, 1);  x = self.graph2(x)
        x = x.permute(0, 3, 1, 2);  x = torch.relu(self.temporal4(x))
        x = x[:, :, -1, :]
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x.permute(0, 2, 1)


# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # stgcn_final/

@st.cache_resource
def load_model():
    adj = np.load(os.path.join(BASE_DIR, "artifacts", "adjacency_full.npy"))
    mdl = STGCN(adj, 6)
    mdl.load_state_dict(torch.load(os.path.join(BASE_DIR, "artifacts", "best_stgcn_model.pth"), map_location="cpu"))
    mdl.eval()
    return mdl

@st.cache_resource
def load_data():
    return np.load(os.path.join(BASE_DIR, "artifacts", "X_test.npy"))


@st.cache_resource
def load_scaler():
    return pickle.load(open(os.path.join(BASE_DIR, "artifacts", "scaler.pkl"), "rb"))


model  = load_model()
X_test = load_data()
scaler = load_scaler()

def to_real_speed(arr):
    if isinstance(scaler, dict):
        if "mean" in scaler:
            return arr * scaler["std"] + scaler["mean"]
        elif "min" in scaler:
            return arr * (scaler["max"] - scaler["min"]) + scaler["min"]
    try:
        return scaler.inverse_transform(arr)
    except Exception:
        return arr


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='padding: 4px 0 20px'>
            <div style='font-size:0.7rem; font-weight:600; letter-spacing:0.1em;
                        color:#6b7280; text-transform:uppercase; margin-bottom:2px;'>
                STGCN · Los Angeles
            </div>
            <div style='font-size:1.3rem; font-weight:700; color:#1a1f2e;'>
                Traffic Forecast
            </div>
            <div style='height:2px; background:linear-gradient(90deg,#2563eb,#93c5fd);
                        border-radius:2px; margin-top:10px;'></div>
        </div>
    """, unsafe_allow_html=True)

    sensor_id    = st.slider("Sensor ID", 0, 206, 50)
    sample_index = st.slider("Test Sample", 0, len(X_test) - 1, 0)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Predict Traffic")

    st.markdown("""
        <div style='margin-top:28px; padding:14px 16px;
                    background:#f8fafc; border:1px solid #e2e6ed;
                    border-radius:8px; font-size:0.8rem; color:#6b7280; line-height:1.9;'>
            <b style='color:#374151'>Dataset</b> &nbsp;·&nbsp; METR-LA<br>
            <b style='color:#374151'>Sensors</b> &nbsp;·&nbsp; 207 road sensors<br>
            <b style='color:#374151'>History</b> &nbsp;·&nbsp; 60 min input<br>
            <b style='color:#374151'>Horizon</b> &nbsp;·&nbsp; 30 min forecast
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
    <div style='padding: 6px 0 18px'>
        <p style='font-size:0.72rem; font-weight:600; letter-spacing:0.1em;
                  color:#6b7280; text-transform:uppercase; margin:0 0 4px;'>
            Spatio-Temporal Graph Convolutional Network
        </p>
        <h1 style='font-size:1.9rem; font-weight:700; color:#1a1f2e; margin:0;'>
            Traffic Forecast Dashboard
        </h1>
        <div style='height:2px; background:linear-gradient(90deg,#2563eb,#bfdbfe,transparent);
                    border-radius:2px; margin-top:12px;'></div>
    </div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# IDLE STATE
# ─────────────────────────────────────────────
if not run:
    st.markdown("""
        <div style='margin: 56px auto; max-width: 460px; text-align:center;
                    padding: 44px 32px; background:#ffffff;
                    border: 1px solid #e2e6ed; border-radius: 14px;'>
            <div style='font-size:2.6rem; margin-bottom:14px;'>🚦</div>
            <div style='font-size:1.15rem; font-weight:600;
                        color:#1a1f2e; margin-bottom:10px;'>
                Ready to Predict
            </div>
            <div style='font-size:0.92rem; color:#6b7280; line-height:1.65;'>
                Choose a <b style="color:#2563eb">Sensor</b> and
                <b style="color:#2563eb">Test Sample</b> from the sidebar,
                then click <b style="color:#2563eb">Predict Traffic</b>.
            </div>
            <div style='margin-top:22px; padding:10px 16px;
                        background:#eff6ff; border-radius:6px;
                        font-size:0.82rem; color:#3b82f6; font-weight:500;'>
                207 sensors · 60-min history → 30-min forecast
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# RUN PREDICTION
# ─────────────────────────────────────────────
sample = torch.tensor(
    X_test[sample_index : sample_index + 1], dtype=torch.float32
)
with torch.no_grad():
    pred = model(sample)

pred_np      = pred.numpy()[0]
real_speed   = to_real_speed(pred_np)
sensor_speed = real_speed[:, sensor_id]

avg_speed = float(sensor_speed.mean())
max_speed = float(sensor_speed.max())
min_speed = float(sensor_speed.min())
delta     = sensor_speed[-1] - sensor_speed[0]
trend     = f"↑ +{delta:.1f} mph" if delta >= 0 else f"↓ {delta:.1f} mph"

# Traffic status
if avg_speed < 25:
    status_label  = "🔴  Heavy Traffic"
    status_color  = "#ef4444"
    status_bg     = "#fef2f2"
    status_border = "#fecaca"
elif avg_speed < 45:
    status_label  = "🟡  Moderate Traffic"
    status_color  = "#f59e0b"
    status_bg     = "#fffbeb"
    status_border = "#fde68a"
else:
    status_label  = "🟢  Free Flow Traffic"
    status_color  = "#22c55e"
    status_bg     = "#f0fdf4"
    status_border = "#bbf7d0"


# ─────────────────────────────────────────────
# STAT CARDS
# ─────────────────────────────────────────────
st.markdown(f"""
    <p style='font-size:0.72rem; font-weight:600; letter-spacing:0.08em;
              color:#6b7280; text-transform:uppercase; margin:0 0 10px;'>
        Sensor {sensor_id} · Summary
    </p>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Avg Speed",   f"{avg_speed:.1f} mph")
with c2:
    st.metric("Peak Speed",  f"{max_speed:.1f} mph")
with c3:
    st.metric("Min Speed",   f"{min_speed:.1f} mph")
with c4:
    st.metric("Speed Trend", trend)

# Status badge
st.markdown(f"""
    <div style='margin: 14px 0 22px; padding: 12px 18px;
                background:{status_bg}; border:1px solid {status_border};
                border-left: 4px solid {status_color};
                border-radius: 8px; font-size:0.95rem; font-weight:600;
                color:{status_color};'>
        {status_label}
        <span style='font-weight:400; color:#9ca3af; margin-left:10px;'>
            Sensor {sensor_id} · Sample {sample_index}
        </span>
    </div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SHARED LAYOUT DEFAULTS
# ─────────────────────────────────────────────
STEPS = [f"t+{(i+1)*5}min" for i in range(6)]

BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="DM Sans", color="#374151", size=12),
    margin=dict(l=8, r=8, t=44, b=8),
    xaxis=dict(
        showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e6ed",
        tickfont=dict(size=11),
        title_font=dict(color="#6b7280", size=11)
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e6ed",
        tickfont=dict(size=11),
        title_font=dict(color="#6b7280", size=11)
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#e2e6ed", borderwidth=1,
        font=dict(size=11)
    )
)


# ─────────────────────────────────────────────
# SPEED FORECAST CHART
# ─────────────────────────────────────────────
st.markdown("""
    <p style='font-size:0.72rem; font-weight:600; letter-spacing:0.08em;
              color:#6b7280; text-transform:uppercase; margin:0 0 8px;'>
        Speed Forecast
    </p>
""", unsafe_allow_html=True)

fig1 = go.Figure()

# shaded fill
fig1.add_trace(go.Scatter(
    x=STEPS, y=sensor_speed,
    fill='tozeroy', fillcolor='rgba(37,99,235,0.06)',
    line=dict(width=0), showlegend=False, hoverinfo='skip'
))

# main line
fig1.add_trace(go.Scatter(
    x=STEPS, y=sensor_speed,
    mode='lines+markers',
    name=f'Sensor {sensor_id}',
    line=dict(color='#2563eb', width=2.5),
    marker=dict(size=8, color='#2563eb',
                line=dict(color='#ffffff', width=2)),
    hovertemplate="<b>%{x}</b><br>%{y:.1f} mph<extra></extra>"
))

# threshold reference lines
fig1.add_hline(y=25, line_dash="dot", line_color="#ef4444", line_width=1,
               annotation_text="Heavy / Moderate",
               annotation_font=dict(size=10, color="#ef4444"),
               annotation_position="bottom right")
fig1.add_hline(y=45, line_dash="dot", line_color="#f59e0b", line_width=1,
               annotation_text="Moderate / Free",
               annotation_font=dict(size=10, color="#f59e0b"),
               annotation_position="bottom right")

fig1.update_layout(
    **BASE,
    title=dict(text=f"Predicted Speed · Sensor {sensor_id}",
               font=dict(size=13, color="#1a1f2e"), x=0),
    xaxis_title="Future Time Step",
    yaxis_title="Speed (mph)",
    height=310
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# NEIGHBORING SENSORS CHART
# ─────────────────────────────────────────────
st.markdown("""
    <p style='font-size:0.72rem; font-weight:600; letter-spacing:0.08em;
              color:#6b7280; text-transform:uppercase; margin:0 0 8px;'>
        Neighboring Sensors Comparison
    </p>
""", unsafe_allow_html=True)

raw_neighbors = [
    max(0,   sensor_id - 2),
    max(0,   sensor_id - 1),
    sensor_id,
    min(206, sensor_id + 1),
    min(206, sensor_id + 2),
]
seen = set()
neighbors = [s for s in raw_neighbors if not (s in seen or seen.add(s))]

PALETTE = ["#94a3b8", "#64748b", "#2563eb", "#60a5fa", "#bfdbfe"]

fig2 = go.Figure()
for idx, s in enumerate(neighbors):
    selected = (s == sensor_id)
    fig2.add_trace(go.Scatter(
        x=STEPS,
        y=real_speed[:, s],
        mode='lines+markers',
        name=f"Sensor {s}" + ("  ◀ selected" if selected else ""),
        line=dict(
            color=PALETTE[idx],
            width=2.8 if selected else 1.5,
            dash="solid" if selected else "dot"
        ),
        marker=dict(
            size=7 if selected else 4,
            color=PALETTE[idx],
            line=dict(color='#ffffff', width=1.5 if selected else 0)
        ),
        hovertemplate=f"<b>Sensor {s}</b><br>%{{x}}: %{{y:.1f}} mph<extra></extra>"
    ))

fig2.update_layout(
    **BASE,
    title=dict(text="Speed Forecast · Selected + 4 Neighbours",
               font=dict(size=13, color="#1a1f2e"), x=0),
    xaxis_title="Future Time Step",
    yaxis_title="Speed (mph)",
    height=310
)
st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# RAW TABLE  (collapsible)
# ─────────────────────────────────────────────
with st.expander("View Prediction Table"):
    disp = list(range(max(0, sensor_id - 5), min(207, sensor_id + 6)))
    df = pd.DataFrame(
        real_speed[:, disp].round(2),
        columns=[f"S-{i}" for i in disp],
        index=[f"t+{(i+1)*5}min" for i in range(6)]
    )
    st.dataframe(df, use_container_width=True)
