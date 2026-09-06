"""
Stage 4: Regime detection via k-means clustering.

The idea: each trading day is a point in "feature space" - described by
the Stage 3 rolling features (volatilities, cross-asset correlations,
momentum, the yield level). k-means groups the days into K clusters so
that days in the same cluster look similar. We then read those clusters
as macro "regimes" and try to name them (dollar-driven, safe-haven,
risk-off, calm, ...).

Key modelling choices and their caveats:

  * STANDARDISE first. k-means measures distance with a plain sum of
    squares, so a feature measured in big numbers (yield_10y ~ 0..5)
    would completely swamp one measured in small numbers (gold_vol_20d
    ~ 0.01). StandardScaler turns every feature into "number of standard
    deviations from its mean", putting them on equal footing.

  * We cluster on the ROLLING features only, not the raw daily log
    returns. Daily returns are almost pure noise day to day; the rolling
    features are smooth and persistent, which is what a "regime" is.

  * PICKING K. There's no ground truth, so we scan K = 2..10 and look at
    two diagnostics (written to figures/regime_k_selection.png):
      - inertia (within-cluster sum of squares): always falls as K grows;
        we look for the "elbow" where extra clusters stop helping much.
      - silhouette score: how well-separated the clusters are, higher is
        better, range -1..1.
    CHOSEN_K below is set from eyeballing that plot.

  * EACH DAY IS CLUSTERED INDEPENDENTLY. k-means has no notion of time,
    so the regime label can flicker between two similar clusters around a
    transition. That's expected. A later step (or Stage 5) can smooth the
    labels or move to a Hidden Markov Model, which builds in persistence.

  * We fit on the WHOLE history, so the regime definitions implicitly
    "know" about the future. That's fine for this project's goal -
    explaining past regimes - but it would be look-ahead bias if you
    tried to use these labels as a live trading signal.

Output: data/regimes.csv  (date, regime) plus a printed regime profile.

Run standalone from inside src/:

    cd src && python3 detect_regimes.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no interactive window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# The features k-means clusters on (the smooth, persistent ones). Defined
# in regime_model.py so the exploratory scripts here and the frozen
# production model can never disagree about the feature set.
from regime_model import CLUSTER_FEATURES

# --- Configuration ------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "features.csv"
OUTPUT_PATH = BASE_DIR / "data" / "regimes.csv"
FIGURE_PATH = BASE_DIR / "figures" / "regime_k_selection.png"

# Range of K to scan for the diagnostic plot.
K_RANGE = range(2, 11)

# Number of regimes to actually use.
#
# The K-selection diagnostics are INCONCLUSIVE here: the elbow is smooth
# (no sharp bend) and the silhouette barely moves (0.129..0.139 across
# K=2..10). That's normal for markets - the days form a continuum, not
# tidy separated blobs. So K is a judgement call, not a number the data
# hands us. K=5 is chosen because the resulting regimes are interpretable
# and a distinct ~4%-of-days "crisis" regime falls out that cleanly
# isolates the 2008 and 2020 crashes.
CHOSEN_K = 5

# Fixed seed so re-runs give identical clusters.
RANDOM_STATE = 42


# --- Steps -------------------------------------------------------------


def load_features() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run build_features.py first."
        )
    df = pd.read_csv(INPUT_PATH, index_col="date", parse_dates=["date"])
    print(f"Loaded {len(df)} feature rows from {INPUT_PATH}")
    return df


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Z-score the clustering features. Returns (X_scaled, fitted_scaler)."""
    X_raw = df[CLUSTER_FEATURES].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, scaler


def scan_k(X: np.ndarray) -> pd.DataFrame:
    """Fit k-means for every K in K_RANGE, collect inertia + silhouette."""
    rows = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, labels),
            }
        )
        print(
            f"  k={k:2d}  inertia={model.inertia_:10.1f}  "
            f"silhouette={rows[-1]['silhouette']:.3f}"
        )
    return pd.DataFrame(rows).set_index("k")


def plot_diagnostics(scan: pd.DataFrame) -> None:
    """Save the elbow (inertia) and silhouette curves side by side."""
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4))

    ax_left.plot(scan.index, scan["inertia"], marker="o")
    ax_left.set_title("Elbow: within-cluster sum of squares")
    ax_left.set_xlabel("K (number of regimes)")
    ax_left.set_ylabel("inertia")

    ax_right.plot(scan.index, scan["silhouette"], marker="o", color="tab:green")
    ax_right.axvline(CHOSEN_K, color="grey", linestyle="--", label=f"CHOSEN_K={CHOSEN_K}")
    ax_right.set_title("Silhouette score (higher = better separated)")
    ax_right.set_xlabel("K (number of regimes)")
    ax_right.set_ylabel("silhouette")
    ax_right.legend()

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=120)
    plt.close(fig)
    print(f"Wrote K-selection plot -> {FIGURE_PATH}")


def fit_regimes(X: np.ndarray) -> np.ndarray:
    """Final k-means with CHOSEN_K. Returns raw cluster labels (0..K-1)."""
    model = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=RANDOM_STATE)
    return model.fit_predict(X)


def stress_remap(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """Map raw cluster id -> stress rank (0 = calmest, K-1 = most volatile).

    Clustering hands back arbitrary integer labels. We sort the clusters
    by their average total volatility (gold_vol + oil_vol), so the regime
    numbers carry a consistent meaning across runs and across algorithms.
    Returned as a dict so callers can also reorder e.g. GMM probability
    columns, not just the label vector.
    """
    stress = (
        pd.Series(df["gold_vol_20d"].to_numpy() + df["oil_vol_20d"].to_numpy())
        .groupby(labels)
        .mean()
        .sort_values()
    )
    return {old: new for new, old in enumerate(stress.index)}


def order_labels_by_stress(df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Renumber clusters so 0 = calmest, K-1 = most volatile."""
    remap = stress_remap(df, labels)
    return np.array([remap[old] for old in labels])


def profile_regimes(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean of each feature per regime, on the original (unscaled) scale."""
    profile_cols = CLUSTER_FEATURES + [
        "gold_log_return",
        "oil_log_return",
        "dollar_index_log_return",
        "vix_log_return",
    ]
    tmp = df[profile_cols].copy()
    tmp["regime"] = labels

    means = tmp.groupby("regime").mean().T
    counts = pd.Series(labels).value_counts().sort_index()
    means.loc["n_days"] = counts.values
    means.loc["pct_of_days"] = (100 * counts / counts.sum()).values
    return means


def main() -> None:
    df = load_features()
    X, _ = scale_features(df)

    print(f"\nScanning K = {K_RANGE.start}..{K_RANGE.stop - 1}:")
    scan = scan_k(X)
    plot_diagnostics(scan)

    print(f"\nFitting final k-means with CHOSEN_K = {CHOSEN_K}")
    raw_labels = fit_regimes(X)
    labels = order_labels_by_stress(df, raw_labels)

    out = pd.DataFrame({"regime": labels}, index=df.index)
    out.to_csv(OUTPUT_PATH)
    print(f"Wrote regime labels -> {OUTPUT_PATH}")

    profile = profile_regimes(df, labels)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print("\n--- Regime profile (mean feature value per regime) ------")
        print(profile)
        print("\n--- How often each regime is active, by year -----------")
        by_year = (
            out.assign(year=out.index.year)
            .groupby(["year", "regime"])
            .size()
            .unstack(fill_value=0)
        )
        print(by_year)


if __name__ == "__main__":
    main()
