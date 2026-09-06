"""
Shared regime-label helpers.

Both the dashboard (Stage 5) and the event-annotation layer (Stage 6) need
to take the raw GMM labels in data/regimes_gmm.csv, smooth out the
day-to-day flicker, and then reason about the regime *episodes* and the
*shifts* between them. That logic lives here so the two don't drift apart.
"""

import numpy as np
import pandas as pd

# Regime number -> human name. The numbers are Stage 4's, ordered by
# volatility (0 = calmest, 4 = crisis).
REGIME_NAMES = {
    0: "Calm / Low-Rate",
    1: "High-Yield",
    2: "Commodity-Macro",
    3: "Safe-Haven",
    4: "Crisis",
}

# The GMM labels each day independently, so the regime can flip for a day
# or two around a transition and then flip straight back. Episodes shorter
# than this many trading days are treated as flicker, not real regime
# changes, and merged into a neighbour before anything looks at the labels.
MIN_REGIME_RUN = 10  # ~two trading weeks


def smooth_regimes(regime: pd.Series, min_run: int = MIN_REGIME_RUN) -> pd.Series:
    """Absorb regime episodes shorter than `min_run` days into a neighbour.

    Run-length-encode the label sequence; repeatedly take the shortest run
    that is still below the threshold and overwrite it with whichever
    adjacent run is *longer*. Merging shortest-first, into the longer side,
    keeps the result stable and stops it from snowballing toward one
    regime. Each pass strictly reduces the number of runs, so it
    terminates. NaN warm-up rows are left untouched.
    """
    valid = regime.dropna()
    vals = valid.to_numpy().astype(int)

    while True:
        change = np.flatnonzero(np.diff(vals)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(vals)]])
        lengths = ends - starts

        shortest = next(
            (i for i in np.argsort(lengths) if lengths[i] < min_run), None
        )
        if shortest is None:
            break

        left_len = lengths[shortest - 1] if shortest > 0 else -1
        right_len = lengths[shortest + 1] if shortest < len(lengths) - 1 else -1
        source = shortest - 1 if left_len >= right_len else shortest + 1
        vals[starts[shortest]:ends[shortest]] = vals[starts[source]]

    out = regime.copy()
    out.loc[valid.index] = vals
    return out


def regime_episodes(regime: pd.Series) -> pd.DataFrame:
    """One row per contiguous run of the same label.

    Columns: start, end, regime, name, length (in trading days).
    NaN rows (warm-up) are dropped first.
    """
    valid = regime.dropna()
    block_id = valid.ne(valid.shift()).cumsum()

    rows = []
    for _, block in valid.groupby(block_id):
        label = int(block.iloc[0])
        rows.append(
            {
                "start": block.index[0],
                "end": block.index[-1],
                "regime": label,
                "name": REGIME_NAMES[label],
                "length": len(block),
            }
        )
    return pd.DataFrame(rows)


def regime_shifts(regime: pd.Series) -> pd.DataFrame:
    """One row per transition between episodes.

    Columns: date (first day of the new regime), from_regime, to_regime,
    from_name, to_name, new_regime_days (length of the episode being
    entered).
    """
    episodes = regime_episodes(regime)

    rows = []
    for prev, curr in zip(episodes.itertuples(), episodes.iloc[1:].itertuples()):
        rows.append(
            {
                "date": curr.start,
                "from_regime": prev.regime,
                "to_regime": curr.regime,
                "from_name": prev.name,
                "to_name": curr.name,
                "new_regime_days": curr.length,
            }
        )
    return pd.DataFrame(rows)
