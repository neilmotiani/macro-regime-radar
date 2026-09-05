"""
Stage 3: Feature engineering.

Reads data/clean_prices.csv (the gap-free wide table from clean_data.py)
and writes data/features.csv - the table the regime-detection clustering
in Stage 4 will actually consume.

Everything here is computed on a 20-trading-day rolling window (~1 calendar
month). The features, and why each one is shaped the way it is:

  1. Daily LOG returns for gold, oil, dollar_index, vix.

       log_return_t = ln(P_t) - ln(P_{t-1})

     Why log and not percent change? Log returns add up across time
     (summing 20 daily log returns gives the exact 20-day return), and
     they're more symmetric around zero, which clustering likes.

     yield_10y is deliberately NOT turned into a return. It is already a
     rate (4.37 means 4.37% per year), not a price level, so "% change of
     a %" is not a meaningful quantity. We carry the yield level through
     unchanged so Stage 4 still has the macro-rates signal to cluster on.
     (If we later want its *movement*, the right transform is a simple
     difference in percentage points, not a return - a note for Stage 4.)

  2. 20-day rolling VOLATILITY for gold and oil: the standard deviation of
     that asset's daily log returns over the trailing 20 days. This is the
     raw (not annualised) daily-return std - multiply by sqrt(252) if you
     ever want an annualised number.

  3. 20-day rolling CORRELATIONS between pairs of daily log-return series:
       - gold vs oil       (do the two commodities move together?)
       - gold vs dollar     (a strongly negative value = "dollar-driven"
                             gold: gold is just the inverse of the USD)
       - gold vs vix        (a positive value = "safe-haven" gold: gold
                             rises when fear rises)
       - oil vs dollar      (oil's own dollar sensitivity)
     These are the headline regime signals. Which macro force is in
     charge - the dollar, global risk appetite, or something
     asset-specific like an oil supply shock - shows up as a shift in
     which of these correlations is large.

  4. 20-day rolling MOMENTUM for gold and oil: the cumulative return over
     the trailing 20 days. Because we already have daily log returns,
     this is just their rolling sum:

       sum of ln(P_t/P_{t-1}) over 20 days  =  ln(P_t / P_{t-20})

     i.e. the 20-day log return. Positive = the asset drifted up over the
     last month, negative = drifted down.

We drop the first 20 rows: every rolling window needs 20 prior
observations before it produces a real number, so those early rows are
all NaN and carry no information.

Run standalone from inside src/:

    cd src && python3 build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --- Configuration -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "clean_prices.csv"
OUTPUT_PATH = BASE_DIR / "data" / "features.csv"

# Length of every rolling window, in trading days (~1 calendar month).
WINDOW = 20

# Columns that are genuine price levels -> get log returns.
RETURN_COLS = ["gold", "oil", "dollar_index", "vix"]

# Columns we compute rolling vol / momentum for.
ROLLING_ASSET_COLS = ["gold", "oil"]

# Pairs we compute a rolling return-correlation for. Each entry is
# (name_a, name_b) and produces a column "{a}_{b}_corr_{WINDOW}d".
CORRELATION_PAIRS = [
    ("gold", "oil"),
    ("gold", "dollar_index"),
    ("gold", "vix"),
    ("oil", "dollar_index"),
]


# --- Feature construction ----------------------------------------------


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Turn the cleaned price table into the Stage 4 feature table."""
    feats = pd.DataFrame(index=prices.index)

    # (1) Daily log returns. np.log turns the price column into log-prices;
    #     .diff() subtracts the previous row, giving ln(P_t) - ln(P_{t-1}).
    #     Row 0 of each series is NaN (no previous day to difference against).
    log_returns = {}
    for col in RETURN_COLS:
        r = np.log(prices[col]).diff()
        log_returns[col] = r
        feats[f"{col}_log_return"] = r

    # Carry the 10-year yield through as a level (see module docstring).
    feats["yield_10y"] = prices["yield_10y"]

    # (2) 20-day rolling volatility = std of daily log returns over the
    #     trailing WINDOW days. pandas uses sample std (ddof=1) by default.
    for col in ROLLING_ASSET_COLS:
        feats[f"{col}_vol_{WINDOW}d"] = (
            log_returns[col].rolling(window=WINDOW).std()
        )

    # (3) 20-day rolling correlations between pairs of daily-log-return
    #     series. Series.rolling(...).corr(other) lines the two series up by
    #     date and computes Pearson correlation within each trailing window.
    for a, b in CORRELATION_PAIRS:
        feats[f"{a}_{b}_corr_{WINDOW}d"] = (
            log_returns[a].rolling(window=WINDOW).corr(log_returns[b])
        )

    # (4) 20-day rolling momentum = cumulative return over the window.
    #     Sum of daily log returns over WINDOW days == ln(P_t / P_{t-WINDOW}),
    #     the 20-day log return.
    for col in ROLLING_ASSET_COLS:
        feats[f"{col}_momentum_{WINDOW}d"] = (
            log_returns[col].rolling(window=WINDOW).sum()
        )

    # Drop the warm-up rows: the first WINDOW rows can't have a full
    # rolling window behind them, so they're NaN and useless to Stage 4.
    feats = feats.iloc[WINDOW:]

    return feats


# --- Entry point ------------------------------------------------------


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run clean_data.py first."
        )

    prices = pd.read_csv(INPUT_PATH, index_col="date", parse_dates=["date"])
    print(f"Loaded {len(prices)} cleaned price rows from {INPUT_PATH}")

    feats = build_features(prices)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUTPUT_PATH)

    print(f"\nWrote features -> {OUTPUT_PATH}")
    print(f"Shape: {feats.shape[0]} rows x {feats.shape[1]} columns")
    print(f"Columns: {list(feats.columns)}")
    print(f"Date range: {feats.index.min().date()} -> {feats.index.max().date()}")

    # Sanity check: after dropping the warm-up rows nothing should be NaN.
    n_missing = int(feats.isna().sum().sum())
    print(f"Remaining missing values: {n_missing}")

    # Show enough columns/width to actually read the output.
    with pd.option_context(
        "display.max_columns", None, "display.width", 200, "display.float_format", "{:.5f}".format
    ):
        print("\nFirst 5 rows:")
        print(feats.head())
        print("\nLast 5 rows:")
        print(feats.tail())


if __name__ == "__main__":
    main()
