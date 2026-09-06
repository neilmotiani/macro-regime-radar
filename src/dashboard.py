"""
Stage 5: Macro Regime Radar dashboard.

Run from inside src/:

    ../venv/bin/python -m streamlit run dashboard.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from regime_labels import MIN_REGIME_RUN, smooth_regimes

st.set_page_config(page_title="Macro Regime Radar", layout="wide")
st.title("Macro Regime Radar")

# --- Regime display styles -------------------------------------------------
#
# Keyed by the regime number Stage 4 assigned (0 = calmest ... 4 = crisis).
#   "band"   - low-opacity RGBA, drawn as a background band BEHIND the lines.
#              Crisis gets more alpha (0.25 vs 0.15) so the eye catches it.
#   "swatch" - the same hue at full opacity, for legend squares / the banner
#              (a 15%-opacity swatch is nearly invisible on the page).
REGIME_STYLE = {
    0: {"name": "Calm / Low-Rate", "band": "rgba(76, 175, 80, 0.15)",   "swatch": "#4CAF50"},
    1: {"name": "High-Yield",      "band": "rgba(66, 133, 244, 0.15)",  "swatch": "#4285F4"},
    2: {"name": "Commodity-Macro", "band": "rgba(240, 200, 80, 0.15)",  "swatch": "#F0C850"},
    3: {"name": "Safe-Haven",      "band": "rgba(156, 100, 210, 0.15)", "swatch": "#9C64D2"},
    4: {"name": "Crisis",          "band": "rgba(229, 57, 53, 0.25)",   "swatch": "#E53935"},
}


# --- Data loading --------------------------------------------------------


@st.cache_data
def load_data():
    """Read the three pipeline outputs and merge regimes onto prices.

    @st.cache_data memoises this: Streamlit re-runs the whole script top to
    bottom on every widget interaction (each slider nudge), and without the
    cache we'd re-read three CSVs from disk every time. With it, the files
    are read once and the same DataFrames are handed back on later runs.
    """
    prices = pd.read_csv("../data/clean_prices.csv", index_col=0, parse_dates=True)
    regimes = pd.read_csv("../data/regimes_gmm.csv", index_col=0, parse_dates=True)
    features = pd.read_csv("../data/features.csv", index_col=0, parse_dates=True)

    # Smooth away the sub-two-week flicker. Keep the raw label around too.
    regimes["regime_raw"] = regimes["regime"]
    regimes["regime"] = smooth_regimes(regimes["regime"], MIN_REGIME_RUN)

    # how="left" keeps every price row; regimes_gmm.csv is ~20 rows shorter
    # because Stage 3 dropped the first 20 days as rolling-window warm-up.
    # An inner join would silently delete those early price rows.
    prices = prices.join(regimes[["regime", "confidence"]], how="left")
    return prices, regimes, features


prices, regimes, features = load_data()


# --- Shared helper: regime background shading --------------------------


def add_regime_shading(fig, frame, row=None, col=1, opaque=False):
    """Draw one background band per contiguous regime episode in `frame`.

    The day-by-day `regime` column is collapsed into "blocks" first:
      .ne(.shift())  -> True on the first day of each new episode
      .cumsum()      -> a running id that's constant within an episode
    so grouping by it yields one sub-frame per episode, and we draw one
    rectangle spanning each episode's date range instead of one per day.

    row/col route the band to a specific subplot when `fig` is a
    make_subplots grid. `opaque=True` uses the solid swatch colour (for
    the timeline ribbon, which is nothing but shading).
    """
    regime_series = frame["regime"]
    block_id = regime_series.ne(regime_series.shift()).cumsum()

    for _, block in frame.groupby(block_id):
        regime_value = block["regime"].iloc[0]
        if pd.isna(regime_value):  # warm-up rows with no regime yet
            continue

        style = REGIME_STYLE[int(regime_value)]
        target = {} if row is None else {"row": row, "col": col}
        fig.add_vrect(
            x0=block.index[0],
            x1=block.index[-1],
            fillcolor=style["swatch"] if opaque else style["band"],
            opacity=0.85 if opaque else 1.0,
            line_width=0,
            layer="below",
            **target,
        )


# --- Section 1: current-regime banner --------------------------------

# The most recent row that actually has a regime label.
labelled = prices.dropna(subset=["regime"])
latest_date = labelled.index[-1]
current_regime = int(labelled["regime"].iloc[-1])
latest_confidence = float(labelled["confidence"].iloc[-1])

# How long has this regime been in force? Walk backwards from the end
# while the label stays the same.
reg_values = labelled["regime"].to_numpy()
run_length = 1
while run_length < len(reg_values) and reg_values[-1 - run_length] == current_regime:
    run_length += 1
active_since = labelled.index[-run_length]

style = REGIME_STYLE[current_regime]
st.markdown(
    f"""
    <div style="border-left:6px solid {style['swatch']};background:{style['band']};
         padding:12px 18px;border-radius:6px;margin-bottom:8px;">
      <div style="font-size:0.78rem;letter-spacing:0.06em;opacity:0.7;">
        CURRENT REGIME &nbsp;·&nbsp; as of {latest_date:%Y-%m-%d}
      </div>
      <div style="font-size:1.5rem;font-weight:600;margin:2px 0;">
        {style['name']}
      </div>
      <div style="font-size:0.9rem;opacity:0.85;">
        Active since {active_since:%Y-%m-%d}
        &nbsp;·&nbsp; {run_length} trading days
        &nbsp;·&nbsp; GMM confidence {latest_confidence:.0%}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Window selector -------------------------------------------------

days_back = st.slider(
    "Show last N trading days", min_value=30, max_value=len(prices), value=90
)
recent = prices.tail(days_back)


def regime_legend():
    """A horizontal 'swatch + name' key, rendered below a chart."""
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin:2px 18px 2px 0;'
        f'white-space:nowrap;">'
        f'<span style="display:inline-block;width:13px;height:13px;border-radius:3px;'
        f'background:{s["swatch"]};margin-right:6px;"></span>{s["name"]}</span>'
        for s in REGIME_STYLE.values()
    )
    st.markdown(
        '<div style="font-size:0.8rem;opacity:0.7;margin-bottom:2px;">Regime shading</div>'
        f'<div style="display:flex;flex-wrap:wrap;font-size:0.85rem;">{items}</div>',
        unsafe_allow_html=True,
    )


# --- Section 2: Gold vs Oil ----------------------------------------

st.subheader("Gold vs. Oil")

# Normalise to 100 at the START OF THE SELECTED WINDOW, not the start of all
# history - otherwise a short window is just two flat lines near 100.
normalized = recent[["gold", "oil"]] / recent[["gold", "oil"]].iloc[0] * 100

fig = go.Figure()
add_regime_shading(fig, recent)
fig.add_trace(go.Scatter(x=normalized.index, y=normalized["gold"], name="Gold", mode="lines"))
fig.add_trace(go.Scatter(x=normalized.index, y=normalized["oil"], name="Oil", mode="lines"))
fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",  # one tooltip with both values at the cursor's date
    yaxis_title="Indexed to 100 at window start",
    xaxis_title=None,
    margin=dict(t=30, b=20),
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, width="stretch")
regime_legend()

col1, col2 = st.columns(2)
with col1:
    gold_change = (recent["gold"].iloc[-1] / recent["gold"].iloc[0] - 1) * 100
    st.metric("Gold", f"${recent['gold'].iloc[-1]:,.2f}", f"{gold_change:+.1f}%")
with col2:
    oil_change = (recent["oil"].iloc[-1] / recent["oil"].iloc[0] - 1) * 100
    st.metric("Oil", f"${recent['oil'].iloc[-1]:,.2f}", f"{oil_change:+.1f}%")

with st.expander("Show data table for this window"):
    st.dataframe(recent.tail(15).sort_index(ascending=False))


# --- Section 3: macro drivers ------------------------------------

st.subheader("Macro drivers")
st.caption(
    "The forces the regimes are built from. Same window and same regime "
    "shading as the chart above - shown as raw levels, not indexed."
)

DRIVERS = [
    ("dollar_index", "US Dollar Index (DXY)"),
    ("yield_10y", "10-Year Treasury Yield (%)"),
    ("vix", "VIX"),
]

drivers_fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=[label for _, label in DRIVERS],
)
for i, (column, _label) in enumerate(DRIVERS, start=1):
    # Trace FIRST, then shading. plotly's add_vrect(row=, col=) skips any
    # subplot it considers "empty" (no traces yet), so shading a subplot
    # before it has a line silently draws nothing.
    drivers_fig.add_trace(
        go.Scatter(x=recent.index, y=recent[column], mode="lines", name=column,
                   line=dict(color="#9ecbff")),
        row=i, col=1,
    )
    add_regime_shading(drivers_fig, recent, row=i, col=1)
drivers_fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    showlegend=False,
    height=620,
    margin=dict(t=40, b=20),
)
st.plotly_chart(drivers_fig, width="stretch")


# --- Section 4: full-history regime ribbon ---------------------

st.subheader("Regime history")
st.caption(
    "Every trading day since 2004, coloured by regime - independent of the "
    f"slider above. Episodes shorter than {MIN_REGIME_RUN} trading days have "
    "been merged into their neighbour to strip out day-to-day flicker."
)

# Built as a Gantt-style timeline: one bar per regime episode. We first
# collapse the full-history regime column into episodes (same block trick
# as the shading helper), then hand plotly a table of
# start / end / regime-name rows. px.timeline draws each as a horizontal
# bar and gives us hover (regime name + dates) for free.
labelled_full = prices.dropna(subset=["regime"])
full_block_id = labelled_full["regime"].ne(labelled_full["regime"].shift()).cumsum()

episodes = []
for _, block in labelled_full.groupby(full_block_id):
    regime_value = int(block["regime"].iloc[0])
    episodes.append(
        {
            "start": block.index[0],
            # +1 day so a single-day episode still has visible width
            "end": block.index[-1] + pd.Timedelta(days=1),
            "Regime": REGIME_STYLE[regime_value]["name"],
            "lane": "regime",
        }
    )
episodes_df = pd.DataFrame(episodes)

ribbon = px.timeline(
    episodes_df,
    x_start="start",
    x_end="end",
    y="lane",
    color="Regime",
    color_discrete_map={s["name"]: s["swatch"] for s in REGIME_STYLE.values()},
)
ribbon.update_yaxes(visible=False)
ribbon.update_layout(
    template="plotly_dark",
    height=130,
    margin=dict(l=0, r=0, t=8, b=24),
    showlegend=False,
    xaxis_title=None,
)
st.plotly_chart(ribbon, width="stretch")
regime_legend()


# --- Section 5: per-regime statistics --------------------------

st.subheader("What each regime looks like")
st.caption(
    "Averages over the full history (2004-present), one row per regime. "
    "Returns and correlations come from the Stage 3 features; regimes are "
    f"smoothed to a {MIN_REGIME_RUN}-day minimum episode length."
)

# features.csv and regimes_gmm.csv share the same dates (regimes were built
# from features), so an inner join lines them up 1:1.
feat_reg = features.join(regimes[["regime", "confidence"]], how="inner")

summary = feat_reg.groupby("regime").agg(
    days=("regime", "size"),
    gold_ret=("gold_log_return", "mean"),
    oil_ret=("oil_log_return", "mean"),
    gold_vol=("gold_vol_20d", "mean"),
    oil_vol=("oil_vol_20d", "mean"),
    gold_oil_corr=("gold_oil_corr_20d", "mean"),
    yield_10y=("yield_10y", "mean"),
    confidence=("confidence", "mean"),
)
summary["share"] = summary["days"] / summary["days"].sum()

# Readable row labels (regime name) and column headers for display.
summary.index = [REGIME_STYLE[r]["name"] for r in summary.index]
summary.index.name = "Regime"
summary = summary[
    ["days", "share", "gold_ret", "oil_ret", "gold_vol", "oil_vol",
     "gold_oil_corr", "yield_10y", "confidence"]
]
summary.columns = [
    "Days", "Share of days",
    "Avg gold daily return", "Avg oil daily return",
    "Gold vol (20d)", "Oil vol (20d)", "Gold–oil corr (20d)",
    "Avg 10Y yield (%)", "Avg GMM confidence",
]

st.dataframe(
    summary.style.format(
        {
            "Share of days": "{:.1%}",
            "Avg gold daily return": "{:+.3%}",
            "Avg oil daily return": "{:+.3%}",
            "Gold vol (20d)": "{:.3f}",
            "Oil vol (20d)": "{:.3f}",
            "Gold–oil corr (20d)": "{:+.2f}",
            "Avg 10Y yield (%)": "{:.2f}",
            "Avg GMM confidence": "{:.0%}",
        }
    ),
    width="stretch",
)
