"""
Stage 1: Multi-asset data pipeline.

Fetches daily price/level history for the two assets we care about
(gold and crude oil) plus three macro drivers (US Dollar Index, 10-year
Treasury yield, VIX), then aligns them into a single wide table indexed
by date.

Design choices (deliberate, for a learning project):

  * We fetch each ticker on its OWN request rather than one bulk
    yf.download([...]) call. It's a little slower, but the code is easier
    to read, errors are per-ticker instead of one opaque failure, and we
    sidestep yfinance's confusing multi-level column layout.

  * We keep the "Close" column only. For indices (^VIX, ^TNX, DX-Y.NYB)
    there are no splits or dividends, so Adjusted Close == Close. For the
    futures (GC=F, CL=F) yfinance also does no adjustment. So "Close" is
    the honest raw number in every case.

  * We OUTER-join on date. Each instrument trades on a slightly different
    holiday calendar (e.g. the Treasury-yield index and the commodity
    futures don't observe exactly the same days off), so an outer join
    keeps every date on which *anything* traded and leaves gaps as NaN.
    We do NOT fill or drop those gaps here - deciding how to handle
    missingness is Stage 2, and it should be informed by actually looking
    at the gaps (which this script prints a report on).

Run it standalone from inside src/:

    cd src && python3 fetch_data.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

# --- Configuration -----------------------------------------------------------

# How far back to pull. yfinance's `start` is inclusive.
START_DATE = "2004-01-01"

# Map each yfinance ticker to a short, readable column name.
#
#   GC=F        COMEX gold front-month futures (USD / troy ounce)
#   CL=F        NYMEX WTI crude oil front-month futures (USD / barrel)
#   DX-Y.NYB    ICE US Dollar Index (DXY) - USD vs a basket of 6 currencies
#   ^TNX        CBOE index tracking the 10-year US Treasury note yield (%)
#   ^VIX        CBOE Volatility Index - 30-day implied vol of the S&P 500
TICKERS = {
    "GC=F": "gold",
    "CL=F": "oil",
    "DX-Y.NYB": "dollar_index",
    "^TNX": "yield_10y",
    "^VIX": "vix",
}

# Where the aligned raw table gets written. Resolved relative to this file
# so the script works no matter what directory you launch it from.
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_prices.csv"


# --- Fetching --------------------------------------------------------------


def fetch_one(ticker: str, column_name: str) -> pd.Series:
    """Download one ticker's daily closing price as a named Series.

    The Series is indexed by date and named `column_name` so that a later
    pd.concat lines the columns up correctly.
    """
    print(f"  fetching {ticker:<10} -> {column_name}")

    # auto_adjust=False: we want the plain "Close", not a back-adjusted one.
    # progress=False: silence yfinance's per-download progress bar.
    raw = yf.download(
        ticker,
        start=START_DATE,
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no rows for {ticker!r}. "
            "The ticker symbol may have changed, or the network/API is down."
        )

    # With a single ticker, recent yfinance still hands back a 2-level
    # column index like ('Close', 'GC=F'). Flatten by grabbing the
    # top-level 'Close' and, if that leaves a 1-column frame, squeeze it.
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.copy()
    close.name = column_name
    close.index.name = "date"
    return close


def fetch_all() -> pd.DataFrame:
    """Fetch every configured ticker and outer-join them on date."""
    series_list = []
    for ticker, column_name in TICKERS.items():
        series_list.append(fetch_one(ticker, column_name))

    # axis=1 -> put each Series in its own column.
    # join="outer" -> union of all dates; missing cells become NaN.
    # sort=False -> don't let concat order the rows; we sort the index
    # ourselves on the next line so the intent is explicit.
    combined = pd.concat(series_list, axis=1, join="outer", sort=False)
    combined = combined.sort_index()
    return combined


# --- Reporting ------------------------------------------------------------


def print_missingness_report(df: pd.DataFrame) -> None:
    """Describe the gaps so Stage 2 (cleaning) can be evidence-based."""
    print("\n--- Missingness report -------------------------------------")
    print(f"Date range : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Total rows : {len(df)}")

    print("\nMissing values per column:")
    n = len(df)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        first_valid = df[col].first_valid_index()
        pct = 100 * missing / n if n else 0.0
        first_str = first_valid.date() if first_valid is not None else "never"
        print(
            f"  {col:<13} {missing:>5} missing "
            f"({pct:4.1f}%)   first real value: {first_str}"
        )

    # Rows where at least one series is missing but not all - these are the
    # calendar-mismatch days that Stage 2 has to make a call on.
    any_missing = df.isna().any(axis=1)
    all_missing = df.isna().all(axis=1)
    partial = any_missing & ~all_missing
    print(
        f"\nRows with a partial gap (some assets traded, others didn't): "
        f"{int(partial.sum())}"
    )
    print("-----------------------------------------------------------\n")


# --- Entry point ---------------------------------------------------------


def main() -> None:
    print(f"Fetching {len(TICKERS)} tickers from {START_DATE} to today...")
    prices = fetch_all()

    print_missingness_report(prices)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT_PATH)
    print(f"Wrote aligned raw table -> {OUTPUT_PATH}")
    print(f"Shape: {prices.shape[0]} rows x {prices.shape[1]} columns")
    print("\nHead:")
    print(prices.head())
    print("\nTail:")
    print(prices.tail())


if __name__ == "__main__":
    main()
