"""
Stage 2: Data cleaning.

Turns data/raw_prices.csv (the outer-joined raw table from fetch_data.py)
into data/clean_prices.csv by applying these rules, IN THIS ORDER:

  0. Drop any row where a price column (gold, oil, dollar_index, vix) is
     <= 0. A price level can't be zero or negative. The one real case is
     2020-04-20, when front-month WTI crude "settled" at -$37.63 because
     holders had nowhere to store physical barrels at contract expiry -
     a futures-plumbing artifact, not a macro signal about oil, and it
     would break the log-return math in Stage 3. (yield_10y is exempt: a
     yield legitimately can be zero or negative.)

  1. Forward-fill `yield_10y` ONLY on rows where it is the single missing
     column - i.e. every other asset traded that day, so it was a normal
     market day and the bond market just took one of its extra half-days
     off (Columbus Day, Veterans Day). Carrying yesterday's yield forward
     one day is a small, honest approximation on those days.

  2. Forward-fill `gold` ONLY on rows where it is the single missing
     column. These are not holidays (oil, the dollar, yields and the VIX
     all printed) - they're isolated gaps in yfinance's GC=F series.
     Same reasoning: one day of carry-forward.

  3. After those two targeted fills, DROP every row that still has any
     missing value. What's left missing at this point is genuine
     market-wide holidays (Thanksgiving, July 4th, MLK Day, ...) where
     nothing meaningful traded, so there is no real data to keep.

Why "only this column missing" instead of a blanket ffill?
A blanket `df["yield_10y"].ffill()` would also invent a yield for
Thanksgiving, when the bond market was genuinely shut and we'd rather
just drop the row. Restricting the fill to rows where *only* that one
series is absent keeps us from fabricating data on true holidays.

Run standalone from inside src/:

    cd src && python3 clean_data.py
"""

from pathlib import Path

import pandas as pd

# --- Configuration ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "raw_prices.csv"
OUTPUT_PATH = BASE_DIR / "data" / "clean_prices.csv"


# --- Cleaning -------------------------------------------------------------


def _targeted_ffill(df: pd.DataFrame, column: str) -> pd.Index:
    """Forward-fill `column` only on rows where it is the ONLY missing one.

    Returns the DatetimeIndex of the rows that were actually filled, so the
    caller can count and report them.

    How the "only at these rows" part works:
      * `df[column].ffill()` builds a fully carried-forward copy of the
        series (every gap filled with the last valid value).
      * We then write that copy back into `df` ONLY at the masked rows.
        Rows not in the mask are left exactly as they were - still NaN if
        they were NaN.
    """
    other_columns = [c for c in df.columns if c != column]

    # This column is missing AND every other column has a value.
    only_this_missing = df[column].isna() & df[other_columns].notna().all(axis=1)
    target_rows = df.index[only_this_missing]

    carried_forward = df[column].ffill()
    df.loc[target_rows, column] = carried_forward.loc[target_rows]

    return target_rows


# Price columns that must be strictly positive (yield_10y is deliberately
# not in this list - see Rule 0 in the module docstring).
PRICE_COLUMNS = ["gold", "oil", "dollar_index", "vix"]


def clean_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning rules in order. Returns (clean_df, report_dict)."""
    df = raw.copy()

    # Rule 0 - drop rows with an impossible (<= 0) price level.
    nonpositive_mask = (df[PRICE_COLUMNS] <= 0).any(axis=1)
    nonpositive_rows = df.index[nonpositive_mask]
    df = df[~nonpositive_mask]

    # Rule 1 - yield_10y, then Rule 2 - gold. Order matters only in that we
    # re-check missingness on the already-updated frame; in practice the two
    # target-row sets are disjoint (Rule 1 needs gold present, Rule 2 needs
    # yield present), so neither fill affects the other's mask.
    yield_filled = _targeted_ffill(df, "yield_10y")
    gold_filled = _targeted_ffill(df, "gold")

    # Rule 3 - drop whatever still has a hole. how="any" is the default but
    # we state it to make the intent unmissable.
    before_drop = len(df)
    dropped_rows = df.index[df.isna().any(axis=1)]
    df = df.dropna(how="any")
    n_dropped = before_drop - len(df)

    report = {
        "nonpositive_rows": nonpositive_rows,
        "yield_filled": yield_filled,
        "gold_filled": gold_filled,
        "dropped_rows": dropped_rows,
        "n_filled": len(yield_filled) + len(gold_filled),
        "n_dropped": n_dropped,
        "rows_in": len(raw),
        "rows_out": len(df),
    }
    return df, report


# --- Reporting -----------------------------------------------------------


def print_report(report: dict) -> None:
    print("\n--- Cleaning report -------------------------------------")
    print(f"Rows in  (raw)   : {report['rows_in']}")
    print(f"Rows out (clean) : {report['rows_out']}")

    print(
        f"\nDropped (bad price, <= 0)  : {len(report['nonpositive_rows'])} rows"
    )
    for d in report["nonpositive_rows"]:
        print(f"    drop             {d.date()}")

    print(
        f"\nForward-filled    : {report['n_filled']} rows "
        f"({len(report['yield_filled'])} yield_10y, "
        f"{len(report['gold_filled'])} gold)"
    )
    for d in report["yield_filled"]:
        print(f"    ffill yield_10y  {d.date()}")
    for d in report["gold_filled"]:
        print(f"    ffill gold       {d.date()}")

    print(f"\nDropped           : {report['n_dropped']} rows (true holidays)")
    for d in report["dropped_rows"]:
        print(f"    drop             {d.date()}")
    print("--------------------------------------------------------\n")


# --- Entry point --------------------------------------------------------


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run fetch_data.py first."
        )

    raw = pd.read_csv(INPUT_PATH, index_col="date", parse_dates=["date"])
    print(f"Loaded {len(raw)} rows from {INPUT_PATH}")

    clean, report = clean_prices(raw)
    print_report(report)

    clean.to_csv(OUTPUT_PATH)
    print(f"Wrote cleaned table -> {OUTPUT_PATH}")
    print(f"Shape: {clean.shape[0]} rows x {clean.shape[1]} columns")
    print(f"Remaining missing values: {int(clean.isna().sum().sum())}")


if __name__ == "__main__":
    main()
