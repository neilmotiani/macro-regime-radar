import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Macro Regime Radar")

# --- Load data built in previous stages -------------------------------------

prices = pd.read_csv("../data/clean_prices.csv", index_col=0, parse_dates=True)

# The Stage 4 GMM output: one row per trading day with a `regime` label
# (0-4) and a `confidence` (the GMM's max posterior probability that day).
regimes = pd.read_csv("../data/regimes_gmm.csv", index_col=0, parse_dates=True)

# Merge the regime + confidence columns onto `prices` by date.
#
# how="left" keeps EVERY row of `prices` and just fills regime/confidence
# with NaN where regimes_gmm.csv has no matching date. That matters here:
# regimes_gmm.csv is ~20 rows shorter than clean_prices.csv because Stage 3
# threw away the first 20 days as rolling-window warm-up (a 20-day rolling
# feature has nothing to compute until day 21). An inner join would
# silently drop those early price rows; a left join keeps the price
# history whole and simply has "no regime yet" at the very start.
prices = prices.join(regimes[["regime", "confidence"]], how="left")

# --- Regime display styles -------------------------------------------------

# Keyed by the regime number Stage 4 assigned (0 = calmest ... 4 = crisis).
# Each entry has two colours:
#   "band"   - the low-opacity RGBA fill drawn as a background band BEHIND
#              the price lines. It should tint the chart, not dominate it.
#              Crisis gets a higher alpha (0.25 vs 0.15) so the eye catches
#              it: it's the rarest regime (~4-6% of days) and the one you
#              most want to notice.
#   "swatch" - the same hue at full opacity, used for the little squares in
#              the legend below the chart (a 15%-opacity swatch would be
#              almost invisible against the page background).
REGIME_STYLE = {
    0: {"name": "Calm / Low-Rate", "band": "rgba(76, 175, 80, 0.15)",   "swatch": "#4CAF50"},
    1: {"name": "High-Yield",      "band": "rgba(66, 133, 244, 0.15)",  "swatch": "#4285F4"},
    2: {"name": "Commodity-Macro", "band": "rgba(240, 200, 80, 0.15)",  "swatch": "#F0C850"},
    3: {"name": "Safe-Haven",      "band": "rgba(156, 100, 210, 0.15)", "swatch": "#9C64D2"},
    4: {"name": "Crisis",          "band": "rgba(229, 57, 53, 0.25)",   "swatch": "#E53935"},
}

st.subheader("Gold vs. Oil")

# Streamlit's built-in line chart wants the columns you want plotted
days_back = st.slider("Show last N trading days", min_value=30, max_value=len(prices), value=90)

recent = prices.tail(days_back)

st.write(f"Data for the selected window ({days_back} trading days):")
st.dataframe(recent.tail(10).sort_index(ascending=False))

# Normalize to 100 at the start of the SELECTED window, not the start of
# all history - otherwise a short recent window would just look like two
# flat lines near 100, since nothing's had time to drift from day one.
normalized = recent[["gold", "oil"]] / recent[["gold", "oil"]].iloc[0] * 100

# --- Build the regime-shaded Plotly chart ---------------------------------

fig = go.Figure()

# Step 1: collapse the day-by-day regime column into consecutive "blocks".
#
# `recent["regime"]` is one label per day, e.g. 0,0,0,1,1,4,4,4,1,1. We
# want to shade one rectangle per *episode* (0-run, 1-run, 4-run, 1-run),
# not one tiny rectangle per day.
#
#   .shift()           moves every value down one row, so row i now sees
#                      "what was the regime yesterday?"
#   .ne(...)           True on any row whose regime differs from the day
#                      before - i.e. the first day of a new episode. Row 0
#                      is always True (its shifted value is NaN).
#   .cumsum()          running total of those True flags. It ticks up by 1
#                      at the start of each new episode and stays flat
#                      within an episode, so every day in the same episode
#                      gets the same integer id.
#
# For 0,0,0,1,1,4,4,4,1,1  ->  flags 1,0,0,1,0,1,0,0,1,0
#                          ->  block ids 1,1,1,2,2,3,3,3,4,4
regime_series = recent["regime"]
block_id = regime_series.ne(regime_series.shift()).cumsum()

# Step 2: one shaded rectangle (vrect) per block.
for _, block in recent.groupby(block_id):
    regime_value = block["regime"].iloc[0]

    # Skip the warm-up stretch at the very start of history, where the
    # left join left regime as NaN. (NaN != NaN in pandas, so each such
    # day actually lands in its own block - we just skip them all.)
    if pd.isna(regime_value):
        continue

    style = REGIME_STYLE[int(regime_value)]
    fig.add_vrect(
        x0=block.index[0],
        x1=block.index[-1],
        fillcolor=style["band"],
        line_width=0,
        layer="below",              # keep the band behind the price lines
    )
    # No inline label: when several short episodes sit close together the
    # per-rectangle annotations overlap into an unreadable smear. The
    # colour key lives in a separate legend below the chart instead.

# Step 3: the price lines themselves, drawn on top of the shading, using
# the already-normalized (start = 100) values.
fig.add_trace(
    go.Scatter(x=normalized.index, y=normalized["gold"], name="Gold", mode="lines")
)
fig.add_trace(
    go.Scatter(x=normalized.index, y=normalized["oil"], name="Oil", mode="lines")
)

# Step 4: styling. "x unified" hover shows gold AND oil in one tooltip at
# whatever date the cursor is over, instead of two separate hover boxes.
fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    yaxis_title="Indexed to 100 at window start",
    xaxis_title=None,
    margin=dict(t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

st.plotly_chart(fig, width="stretch")

# --- Regime colour legend (below the chart, not crammed onto it) ----------
#
# One flex row of "swatch + name" items. Built as a single HTML string and
# handed to st.markdown with unsafe_allow_html=True (Streamlit blocks raw
# HTML by default). flex-wrap lets it spill onto a second line on a narrow
# screen instead of overflowing.
legend_items = "".join(
    f'<span style="display:inline-flex;align-items:center;margin:2px 18px 2px 0;'
    f'white-space:nowrap;">'
    f'<span style="display:inline-block;width:13px;height:13px;border-radius:3px;'
    f'background:{style["swatch"]};margin-right:6px;"></span>{style["name"]}</span>'
    for style in REGIME_STYLE.values()
)
st.markdown(
    '<div style="font-size:0.8rem;opacity:0.7;margin-bottom:2px;">Regime shading</div>'
    f'<div style="display:flex;flex-wrap:wrap;font-size:0.85rem;">{legend_items}</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    gold_change = (recent["gold"].iloc[-1] / recent["gold"].iloc[0] - 1) * 100
    st.metric("Gold", f"${recent['gold'].iloc[-1]:,.2f}", f"{gold_change:+.1f}%")
with col2:
    oil_change = (recent["oil"].iloc[-1] / recent["oil"].iloc[0] - 1) * 100
    st.metric("Oil", f"${recent['oil'].iloc[-1]:,.2f}", f"{oil_change:+.1f}%")
