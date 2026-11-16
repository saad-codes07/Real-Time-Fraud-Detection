"""
app.py - "Fraud Radar": Streamlit dashboard for the fraud detection API.

Run from the project root:
    streamlit run frontend/app.py
"""

import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# --- Settings (Docker overrides these through environment variables) ----------
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
DEMO_CSV = os.getenv("DEMO_CSV", "data/demo_transactions.csv")
MLFLOW_LINK = os.getenv("MLFLOW_PUBLIC_URL", "http://localhost:5000")
API_DOCS_LINK = os.getenv("API_PUBLIC_URL", "http://localhost:8000") + "/docs"

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_BATCH_ROWS = 1000
MAX_EVENTS = 300  # how many recent live events we keep in memory

# --- Colours -----------------------------------------------------------------
CYAN = "#22d3ee"
RED = "#ef4444"
GREEN = "#22c55e"
AMBER = "#f59e0b"
MUTED = "#8ea0bf"

st.set_page_config(page_title="Fraud Radar", page_icon="🛡️", layout="wide")

# Our own CSS classes (hero banner, KPI cards, live feed...). The dark base
# colours come from .streamlit/config.toml.
CSS = """
.block-container { padding-top: 1.5rem; max-width: 1250px; }
footer { visibility: hidden; }
.hero { display: flex; justify-content: space-between; align-items: center;
  padding: 22px 26px; border-radius: 16px; margin-bottom: 18px;
  background: linear-gradient(120deg, #0f1b34 0%, #12294d 55%, #0e3a4a 100%);
  border: 1px solid #1f2b45; }
.eyebrow { letter-spacing: .18em; font-size: .72rem; color: #22d3ee; font-weight: 600; }
.hero h1 { margin: .1rem 0 .2rem 0; padding: 0; font-size: 2rem; }
.hero p { margin: 0; color: #8ea0bf; }
.pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
  border-radius: 999px; font-size: .85rem; font-weight: 600; }
.pill.ok { background: rgba(34,197,94,.12); color: #4ade80; border: 1px solid rgba(34,197,94,.35); }
.pill.down { background: rgba(239,68,68,.12); color: #f87171; border: 1px solid rgba(239,68,68,.35); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.kpi { background: #111a2e; border: 1px solid #1f2b45; border-radius: 12px;
  padding: 14px 18px; margin-bottom: 10px; }
.kpi-label { color: #8ea0bf; font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; }
.kpi-value { font-size: 1.9rem; font-weight: 700; line-height: 1.2; margin-top: 4px; }
.kpi-sub { color: #8ea0bf; font-size: .8rem; }
.feed { background: #111a2e; border: 1px solid #1f2b45; border-radius: 12px; padding: 12px 14px; }
.feed-title { color: #8ea0bf; font-size: .75rem; text-transform: uppercase;
  letter-spacing: .08em; margin-bottom: 6px; }
.feed-row { display: flex; align-items: center; gap: 10px; padding: 7px 6px;
  border-bottom: 1px solid #1a2540; font-size: .9rem; border-radius: 6px; }
.feed-row:last-child { border-bottom: none; }
.feed-row.bad { background: rgba(239,68,68,.07); }
.feed-main { flex: 1; }
.feed-prob { color: #8ea0bf; font-variant-numeric: tabular-nums; }
.badge { font-size: .7rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; letter-spacing: .06em; }
.badge.ok { background: rgba(34,197,94,.15); color: #4ade80; }
.badge.bad { background: rgba(239,68,68,.18); color: #f87171; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 10px 0 18px 0; }
.step { background: #111a2e; border: 1px solid #1f2b45; border-radius: 12px; padding: 16px; }
.step-num { color: #22d3ee; font-weight: 700; font-size: .8rem; letter-spacing: .12em; }
.step h4 { margin: 4px 0 6px 0; padding: 0; }
.step p { margin: 0; color: #8ea0bf; font-size: .88rem; }
"""


# --- Small helpers -----------------------------------------------------------
def kpi(label, value, sub="", accent=CYAN):
    """HTML for one KPI card."""
    return (
        f'<div class="kpi" style="border-top: 3px solid {accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def show_kpis(cards):
    """Lay KPI cards out side by side."""
    for col, card in zip(st.columns(len(cards)), cards):
        col.markdown(card, unsafe_allow_html=True)


def style_fig(fig, height=340):
    """Make a Plotly chart match the dark theme."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def api_is_ready(api_url):
    """True if the API answers /health and has the model loaded."""
    try:
        reply = requests.get(f"{api_url}/health", timeout=3)
        reply.raise_for_status()
        return bool(reply.json().get("model_loaded"))
    except requests.RequestException:
        return False


def to_payload(row):
    """Turn one CSV row into the JSON the API expects."""
    payload = {name: float(row[name]) for name in FEATURES}
    # Same conversion as training: seconds since start -> hour of day.
    payload["Hour"] = int((row["Time"] // 3600) % 24)
    return payload


def score_one(session, api_url, row):
    """Score one transaction. Returns (API reply, latency in milliseconds)."""
    start = time.perf_counter()
    reply = session.post(f"{api_url}/predict", json=to_payload(row), timeout=10)
    reply.raise_for_status()
    latency_ms = (time.perf_counter() - start) * 1000
    return reply.json(), latency_ms


@st.cache_data
def load_demo(path):
    """Load the demo transactions (shuffled), or None if the file is missing."""
    if not Path(path).exists():
        return None
    return pd.read_csv(path).sample(frac=1, random_state=7).reset_index(drop=True)


# --- Live stream state -------------------------------------------------------
def reset_stream():
    st.session_state.update(
        events=[], pos=0, processed=0, flagged=0,
        latency_sum=0.0, threshold=None, stream_error=None,
    )


def init_state():
    if "events" not in st.session_state:
        reset_stream()


def run_tick(demo, api_url, per_tick=3):
    """Score the next few transactions of the replayed stream."""
    ss = st.session_state
    ss["stream_error"] = None
    try:
        with requests.Session() as session:
            for _ in range(per_tick):
                row = demo.iloc[ss["pos"] % len(demo)]
                ss["pos"] += 1
                result, latency = score_one(session, api_url, row)
                ss["threshold"] = result["threshold"]
                ss["processed"] += 1
                ss["flagged"] += int(result["is_fraud"])
                ss["latency_sum"] += latency
                ss["events"].append({
                    "n": ss["processed"],
                    "time": time.strftime("%H:%M:%S"),
                    "Amount": float(row["Amount"]),
                    "Hour": int((row["Time"] // 3600) % 24),
                    "prob": result["fraud_probability"],
                    "fraud": bool(result["is_fraud"]),
                })
    except requests.RequestException as err:
        ss["stream_error"] = str(err)
    del ss["events"][:-MAX_EVENTS]  # keep only the most recent events


def feed_html(df):
    """The 'latest transactions' list, newest first."""
    rows = []
    for r in df.itertuples():
        kind = "bad" if r.fraud else "ok"
        verdict = "FRAUD" if r.fraud else "OK"
        rows.append(
            f'<div class="feed-row {kind}"><span class="badge {kind}">{verdict}</span>'
            f'<span class="feed-main">Amount {r.Amount:,.2f} &middot; {r.Hour:02d}:00</span>'
            f'<span class="feed-prob">{r.prob:.1%}</span></div>'
        )
    return '<div class="feed"><div class="feed-title">Latest transactions</div>' + "".join(rows) + "</div>"


def draw_live():
    """Draw the KPI cards, the probability chart and the live feed."""
    ss = st.session_state
    processed, flagged = ss["processed"], ss["flagged"]
    avg_latency = ss["latency_sum"] / processed if processed else 0.0
    share = flagged / processed if processed else 0.0
    threshold = ss["threshold"]

    show_kpis([
        kpi("Transactions processed", f"{processed:,}", "since the stream started", CYAN),
        kpi("Fraud alerts", f"{flagged:,}", f"{share:.1%} of traffic", RED),
        kpi("Avg. latency", f"{avg_latency:.0f} ms", "per API call", GREEN),
        kpi("Alert threshold", f"{threshold:.2f}" if threshold is not None else "-", "chosen during training", AMBER),
    ])

    if ss["stream_error"]:
        st.warning(f"Could not reach the API: {ss['stream_error']}")
    if not ss["events"]:
        st.info("Switch on **Start stream** to begin replaying transactions.")
        return

    df = pd.DataFrame(ss["events"])
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        legit, fraud = df[~df["fraud"]], df[df["fraud"]]
        fig.add_trace(go.Scatter(
            x=legit["n"], y=legit["prob"], mode="markers", name="Legitimate",
            marker=dict(color=GREEN, size=6, opacity=0.7)))
        fig.add_trace(go.Scatter(
            x=fraud["n"], y=fraud["prob"], mode="markers", name="Flagged",
            marker=dict(color=RED, size=12, symbol="diamond")))
        if threshold is not None:
            fig.add_hline(y=threshold, line_dash="dash", line_color=AMBER,
                          annotation_text="threshold")
        fig.update_yaxes(range=[-0.03, 1.03], title="Fraud probability")
        fig.update_xaxes(title="Transaction number")
        st.plotly_chart(style_fig(fig))
    with right:
        st.markdown(feed_html(df.tail(8).iloc[::-1]), unsafe_allow_html=True)


def live_tab(api_url, demo):
    if demo is None:
        st.warning(
            f"Demo file not found at `{DEMO_CSV}`. "
            "Create it with `python model_training/make_demo_csv.py`."
        )
        return

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        running = st.toggle("Start stream", key="running")
    with c2:
        speed = st.select_slider("Speed", options=["Slow", "Normal", "Fast"], value="Normal")
    with c3:
        if st.button("Reset"):
            reset_stream()
    st.caption(
        "Replays the demo file one transaction at a time. The demo is about 5% fraud, "
        "far more than real life, so alerts appear often."
    )
    interval = {"Slow": 3.0, "Normal": 1.5, "Fast": 0.8}[speed]

    # A fragment re-runs by itself on a timer, so the page updates live
    # without re-running the whole script.
    @st.fragment(run_every=interval if running else None)
    def panel():
        if running:
            run_tick(demo, api_url)
        draw_live()

    panel()


# --- Batch analysis ----------------------------------------------------------
def score_batch(df, api_url):
    """Score every row of a table. Returns (results table, threshold)."""
    replies, latencies = [], []
    bar = st.progress(0.0, text="Scoring transactions...")
    with requests.Session() as session:
        for i, (_, row) in enumerate(df.iterrows()):
            reply, latency = score_one(session, api_url, row)
            replies.append(reply)
            latencies.append(latency)
            bar.progress((i + 1) / len(df), text=f"Scoring... {i + 1}/{len(df)}")
    bar.empty()

    out = df.reset_index(drop=True).copy()
    out["Hour"] = ((out["Time"] // 3600) % 24).astype(int)
    out["fraud_probability"] = [r["fraud_probability"] for r in replies]
    out["is_fraud"] = [bool(r["is_fraud"]) for r in replies]
    out["latency_ms"] = latencies
    out.insert(0, "row", range(1, len(out) + 1))
    return out, replies[0]["threshold"]


def show_batch_results(results, threshold):
    flagged = results[results["is_fraud"]]
    show_kpis([
        kpi("Scored", f"{len(results):,}", "transactions", CYAN),
        kpi("Flagged as fraud", f"{len(flagged):,}", f"{len(flagged) / len(results):.1%} of the file", RED),
        kpi("Avg. latency", f"{results['latency_ms'].mean():.0f} ms", "per API call", GREEN),
    ])

    # The demo file includes the true labels, so we can check the model.
    if "Class" in results.columns:
        actual = results["Class"] == 1
        caught = int((results["is_fraud"] & actual).sum())
        missed = int((~results["is_fraud"] & actual).sum())
        false_alarms = int((results["is_fraud"] & ~actual).sum())
        show_kpis([
            kpi("Frauds caught", f"{caught} of {caught + missed}", "against the known labels", GREEN),
            kpi("Missed frauds", f"{missed}", "real fraud we did not flag", AMBER),
            kpi("False alarms", f"{false_alarms}", "legitimate but flagged", AMBER),
        ])

    st.markdown("##### How confident is the model?")
    hist = px.histogram(
        results, x="fraud_probability", nbins=40, log_y=True,
        labels={"fraud_probability": "Fraud probability"},
        color_discrete_sequence=[CYAN],
    )
    hist.add_vline(x=threshold, line_dash="dash", line_color=AMBER, annotation_text="threshold")
    st.plotly_chart(style_fig(hist, height=300))

    st.markdown("##### Flagged transactions")
    if flagged.empty:
        st.info("Nothing was flagged.")
    else:
        cols = ["row", "Amount", "Hour", "fraud_probability"]
        if "Class" in results.columns:
            cols.append("Class")
        st.dataframe(flagged.sort_values("fraud_probability", ascending=False)[cols])

    st.markdown("##### Inspect one transaction")
    ranked = results.sort_values("fraud_probability", ascending=False)
    chosen = st.selectbox("Row (most suspicious first)", ranked["row"].tolist()[:100])
    picked = results.loc[results["row"] == chosen].iloc[0]
    is_fraud = bool(picked["is_fraud"])
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(picked["fraud_probability"]) * 100,
        number={"suffix": "%", "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": RED if is_fraud else GREEN},
            "threshold": {"line": {"color": AMBER, "width": 4}, "value": threshold * 100},
        },
    ))
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(style_fig(gauge, height=260))
    with right:
        st.markdown(f"**Amount:** {picked['Amount']:,.2f}  \n**Hour of day:** {int(picked['Hour'])}")
        if is_fraud:
            st.error("Flagged as FRAUD")
        else:
            st.success("Looks legitimate")


def batch_tab(api_url, demo):
    st.markdown("##### Score a file of transactions")
    uploaded = st.file_uploader("Upload a CSV in the Kaggle credit-card format", type="csv")
    if uploaded is not None:
        data, source = pd.read_csv(uploaded), uploaded.name
    elif demo is not None:
        st.caption("No file uploaded, so the built-in demo file is used (500 transactions).")
        data, source = demo, "built-in demo"
    else:
        st.info("Upload a CSV to begin.")
        return

    missing = [c for c in FEATURES + ["Time"] if c not in data.columns]
    if missing:
        st.error(f"The CSV is missing these columns: {missing}")
        return
    if len(data) > MAX_BATCH_ROWS:
        st.warning(f"Only the first {MAX_BATCH_ROWS} rows will be scored.")
        data = data.head(MAX_BATCH_ROWS)

    # A different file means the old results no longer apply.
    if st.session_state.get("batch_source") != source:
        st.session_state.pop("batch_results", None)
        st.session_state["batch_source"] = source

    if st.button("Score transactions", type="primary"):
        try:
            results, threshold = score_batch(data, api_url)
            # Streamlit re-runs the whole script on every click, so we keep
            # the results in session_state to stop them vanishing.
            st.session_state["batch_results"] = results
            st.session_state["batch_threshold"] = threshold
        except requests.RequestException as err:
            st.error(f"Could not reach the API: {err}")

    results = st.session_state.get("batch_results")
    if results is not None:
        show_batch_results(results, st.session_state["batch_threshold"])


# --- About -------------------------------------------------------------------
def about_tab():
    st.markdown(
        '<div class="steps">'
        '<div class="step"><div class="step-num">01</div><h4>Data</h4>'
        "<p>Kaggle credit-card transactions: 284,807 rows, only 0.17% are fraud.</p></div>"
        '<div class="step"><div class="step-num">02</div><h4>Training</h4>'
        "<p>Stratified splits, optional SMOTE, and XGBoost tuned for AUPRC.</p></div>"
        '<div class="step"><div class="step-num">03</div><h4>Tracking</h4>'
        "<p>Every run's metrics, plots and model are logged to MLflow.</p></div>"
        '<div class="step"><div class="step-num">04</div><h4>Serving</h4>'
        "<p>FastAPI loads the model once and scores each request in milliseconds.</p></div>"
        '<div class="step"><div class="step-num">05</div><h4>Dashboard</h4>'
        "<p>This app calls the API for every transaction you see here.</p></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Why AUPRC?** Fraud is rare, so a model that always says *not fraud* is 99.8% "
        "accurate and useless. AUPRC measures how well the model finds fraud without "
        "burying analysts in false alarms. The held-out test score is about 0.87."
    )
    c1, c2, _ = st.columns([1, 1, 2])
    c1.link_button("Open MLflow experiments", MLFLOW_LINK)
    c2.link_button("Open API docs", API_DOCS_LINK)


# --- Page --------------------------------------------------------------------
init_state()
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API URL", DEFAULT_API_URL)

ready = api_is_ready(api_url)
pill = (
    '<div class="pill ok"><span class="dot"></span>API online</div>'
    if ready
    else '<div class="pill down"><span class="dot"></span>API unreachable</div>'
)
st.markdown(
    '<div class="hero"><div><div class="eyebrow">REAL-TIME FRAUD DETECTION</div>'
    "<h1>Fraud Radar</h1>"
    "<p>Every transaction is scored by an XGBoost model served through FastAPI.</p></div>"
    + pill + "</div>",
    unsafe_allow_html=True,
)

demo_data = load_demo(DEMO_CSV)
tab_live, tab_batch, tab_about = st.tabs(["Live monitor", "Batch analysis", "How it works"])
with tab_live:
    live_tab(api_url, demo_data)
with tab_batch:
    batch_tab(api_url, demo_data)
with tab_about:
    about_tab()