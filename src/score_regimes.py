"""
Stage 7: production regime scorer.

Reads data/features.csv, assigns every day to a regime with the FROZEN
model in models/regime_gmm.joblib, and writes data/regimes_gmm.csv - the
file the dashboard and the event layer consume.

This is what the daily GitHub Actions job runs (step 4 of the pipeline).
It never re-fits: the regime numbering and cluster boundaries are fixed at
baseline time, so yesterday's labels can't change just because today's row
arrived.

Usage:

    cd src && python3 score_regimes.py          # score with the frozen model
    cd src && python3 score_regimes.py --fit     # re-baseline, then score
"""

import argparse
from pathlib import Path

import pandas as pd

from regime_model import MODEL_PATH, fit_and_save, load_bundle, score

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "features.csv"
OUTPUT_PATH = BASE_DIR / "data" / "regimes_gmm.csv"

# A regime label may shift this many days back from the latest date without
# it being suspicious - Yahoo revises recent closes and the 20-day rolling
# features spread that revision backwards.
STALE_GRACE_DAYS = 35

REGIME_NAMES = {
    0: "Calm / Low-Rate",
    1: "High-Yield",
    2: "Commodity-Macro",
    3: "Safe-Haven",
    4: "Crisis",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Re-fit the model on the full current feature history and "
        "overwrite models/regime_gmm.joblib before scoring.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if a previously-labelled day changes regime "
        "without --fit (used by the daily job to catch model drift).",
    )
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run build_features.py first."
        )
    features = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)

    # Note whether this is a genuine data refresh (for the daily-job log).
    previous = None
    if OUTPUT_PATH.exists():
        previous = pd.read_csv(OUTPUT_PATH, index_col=0, parse_dates=True)

    if args.fit:
        print(f"Re-fitting the regime model on {len(features)} feature rows...")
        bundle = fit_and_save(features)
        print(f"  saved -> {MODEL_PATH}  (fitted through {bundle['fitted_through']})")
    else:
        bundle = load_bundle()
        print(
            f"Loaded frozen model ({MODEL_PATH.name}, "
            f"fitted through {bundle['fitted_through']} on {bundle['n_fit_rows']} rows)."
        )

    scored = score(features, bundle)

    # Compare to the previous run BEFORE writing, so a --strict failure
    # leaves the committed file untouched.
    if previous is not None:
        new_rows = scored.index.difference(previous.index)
        common = scored.index.intersection(previous.index)
        diff = scored.loc[common, "regime"].to_numpy() != previous.loc[common, "regime"].to_numpy()
        changed_dates = common[diff]

        # Recent labels legitimately move: Yahoo Finance revises the last
        # few sessions' closes, and a 20-day rolling feature carries that
        # revision ~20 rows back. Only a change OLDER than this is a red
        # flag (the model is frozen; settled data doesn't change).
        stale_cutoff = scored.index[-1] - pd.Timedelta(days=STALE_GRACE_DAYS)
        stale_changes = changed_dates[changed_dates < stale_cutoff]

        print(
            f"vs previous run: {len(new_rows)} new dated rows; "
            f"{len(changed_dates)} existing labels changed "
            f"({len(stale_changes)} older than {STALE_GRACE_DAYS} days)."
        )
        if len(stale_changes) and not args.fit:
            print(f"  settled-history labels changed: {list(stale_changes.date)}")
            if args.strict:
                raise SystemExit(
                    f"--strict: {len(stale_changes)} settled labels changed "
                    "without --fit - the frozen model or the data is off."
                )

    scored.to_csv(OUTPUT_PATH)

    latest_date = scored.index[-1]
    latest_regime = int(scored["regime"].iloc[-1])
    print(f"\nWrote {len(scored)} rows -> {OUTPUT_PATH}")
    print(
        f"Latest: {latest_date.date()}  regime {latest_regime} "
        f"({REGIME_NAMES[latest_regime]}), confidence {scored['confidence'].iloc[-1]:.0%}"
    )


if __name__ == "__main__":
    main()
