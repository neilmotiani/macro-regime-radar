"""
Stage 4 (follow-up): k-means vs. Gaussian Mixture Model.

detect_regimes.py uses k-means. This script fits a Gaussian Mixture Model
(GMM) on the same standardised features and compares the two.

Why bother with GMM after k-means?

  * SOFT assignments. k-means gives each day one hard label. A GMM gives
    a probability for every regime, e.g. "70% risk-off, 25% calm, 5%
    ...". The days where no regime is above ~60% are exactly the
    transition periods - the interesting bit that a hard label hides.

  * SHAPE. k-means implicitly assumes every cluster is a round blob of
    equal size. A GMM with `covariance_type="full"` lets each regime be
    its own stretched, tilted ellipse - a better fit when (say) the
    crisis regime is spread out along the volatility axis but tight on
    the others.

  * PRINCIPLED-LOOKING model selection. A GMM has a real likelihood, so
    we can compare K with BIC (Bayesian Information Criterion) instead of
    the eyeballed elbow. BIC = -2*loglik + (n_params)*ln(n_obs): it
    rewards fit and penalises complexity, and you take the K that
    minimises it. Spoiler for this dataset: BIC never bottoms out - it
    just keeps falling as K grows. That happens when the data isn't
    actually a mix of a few Gaussians (asset returns have fat tails), and
    it's a useful lesson: BIC only rescues you when the model family is
    roughly right. K stays a judgement call here too.

What the script reports:

  1. BIC / AIC vs K (written to figures/gmm_bic.png).
  2. How much the k-means and GMM hard labels agree - a cross-tab plus
     the Adjusted Rand Index (1.0 = identical partitions, 0.0 = agreement
     no better than chance).
  3. Whether GMM's labels are more persistent (fewer regime episodes).
  4. The GMM "confidence" distribution and where the low-confidence days
     sit.
  5. The GMM regime profile, next to k-means'.

Output: data/regimes_gmm.csv  (date, regime, prob_0..prob_{K-1}, confidence)

Run standalone from inside src/:

    cd src && python3 compare_regimes.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from detect_regimes import (
    BASE_DIR,
    CHOSEN_K,
    RANDOM_STATE,
    load_features,
    order_labels_by_stress,
    profile_regimes,
    scale_features,
    stress_remap,
)

GMM_OUTPUT_PATH = BASE_DIR / "data" / "regimes_gmm.csv"
BIC_FIGURE_PATH = BASE_DIR / "figures" / "gmm_bic.png"

# GMM model-selection scan. Wider than the k-means scan because a GMM can
# keep "improving" by adding components to paper over non-Gaussian shape,
# and we want the plot to show whether BIC ever actually bottoms out.
GMM_K_RANGE = range(2, 16)

# covariance_type="full": each regime gets its own full covariance matrix
# (most flexible). "diag" or "tied" are cheaper fallbacks if "full" ever
# looks unstable on this much data.
COVARIANCE_TYPE = "full"


def count_episodes(labels: np.ndarray) -> int:
    """Number of contiguous same-label runs - a crude persistence measure."""
    return int(1 + np.sum(np.diff(labels) != 0))


def scan_gmm_bic(X: np.ndarray) -> pd.DataFrame:
    """Fit a GMM for every K in GMM_K_RANGE, record BIC and AIC."""
    rows = []
    for k in GMM_K_RANGE:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=COVARIANCE_TYPE,
            random_state=RANDOM_STATE,
            n_init=5,
        )
        gmm.fit(X)
        rows.append({"k": k, "bic": gmm.bic(X), "aic": gmm.aic(X)})
        print(f"  k={k:2d}  BIC={rows[-1]['bic']:12.1f}  AIC={rows[-1]['aic']:12.1f}")
    return pd.DataFrame(rows).set_index("k")


def plot_bic(scan: pd.DataFrame) -> None:
    BIC_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scan.index, scan["bic"], marker="o", label="BIC")
    ax.plot(scan.index, scan["aic"], marker="o", label="AIC", alpha=0.6)
    best_k = int(scan["bic"].idxmin())
    ax.axvline(best_k, color="grey", linestyle="--", label=f"min BIC at K={best_k}")
    ax.axvline(CHOSEN_K, color="tab:red", linestyle=":", label=f"k-means CHOSEN_K={CHOSEN_K}")
    ax.set_xlabel("K (number of regimes)")
    ax.set_ylabel("information criterion (lower = better)")
    ax.set_title("GMM model selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BIC_FIGURE_PATH, dpi=120)
    plt.close(fig)
    print(f"Wrote GMM BIC plot -> {BIC_FIGURE_PATH}")


def fit_gmm(df: pd.DataFrame, X: np.ndarray, k: int):
    """Fit the GMM and return (hard_labels, proba_df) both stress-ordered."""
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=COVARIANCE_TYPE,
        random_state=RANDOM_STATE,
        n_init=5,
    )
    raw_labels = gmm.fit_predict(X)
    raw_proba = gmm.predict_proba(X)

    # Reorder components by stress so GMM regime i means the same thing as
    # k-means regime i. Apply the same remap to the probability columns.
    remap = stress_remap(df, raw_labels)
    order = [old for old, _ in sorted(remap.items(), key=lambda kv: kv[1])]

    labels = np.array([remap[old] for old in raw_labels])
    proba = raw_proba[:, order]

    proba_df = pd.DataFrame(
        proba, index=df.index, columns=[f"prob_{i}" for i in range(k)]
    )
    return labels, proba_df


def main() -> None:
    df = load_features()
    X, _ = scale_features(df)

    # --- k-means baseline (same settings as detect_regimes.py) ---
    km = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=RANDOM_STATE)
    km_labels = order_labels_by_stress(df, km.fit_predict(X))

    # --- GMM: model selection, then fit at CHOSEN_K ---
    print(f"\nScanning GMM K = {GMM_K_RANGE.start}..{GMM_K_RANGE.stop - 1}:")
    bic_scan = scan_gmm_bic(X)
    plot_bic(bic_scan)
    best_k = int(bic_scan["bic"].idxmin())
    if best_k == GMM_K_RANGE.stop - 1:
        print(
            f"\nBIC is still falling at K={best_k} (the edge of the scan) - it "
            "never bottoms out. That's the tell that these features are NOT a "
            "mixture of a few clean Gaussians (returns have fat tails), so BIC "
            "won't hand us a K either. Staying with CHOSEN_K=5 on interpretability."
        )
    else:
        print(f"\nBIC is minimised at K={best_k}.")
    print(f"Fitting GMM at CHOSEN_K={CHOSEN_K} to compare like with like.")
    gmm_labels, proba_df = fit_gmm(df, X, CHOSEN_K)

    # --- 1. agreement between the two hard labelings ---
    ari = adjusted_rand_score(km_labels, gmm_labels)
    crosstab = pd.crosstab(
        pd.Series(km_labels, name="kmeans"),
        pd.Series(gmm_labels, name="gmm"),
    )
    same = float(np.mean(km_labels == gmm_labels))
    print("\n--- k-means vs GMM agreement ---------------------------")
    print(f"Adjusted Rand Index : {ari:.3f}   (1 = identical, 0 = chance)")
    print(f"Same label on        : {same:.1%} of days")
    print("\nCross-tab (rows = k-means regime, cols = GMM regime):")
    print(crosstab)

    # --- 2. persistence ---
    print("\n--- Persistence (contiguous regime episodes) -----------")
    print(f"k-means : {count_episodes(km_labels):4d} episodes")
    print(f"GMM     : {count_episodes(gmm_labels):4d} episodes")

    # --- 3. GMM confidence ---
    confidence = proba_df.max(axis=1)
    ambiguous = (confidence < 0.60).to_numpy()
    print("\n--- GMM confidence (max posterior probability) ---------")
    print(confidence.describe().round(3).to_string())
    print(f"\nDays below 0.60 confidence: {int(ambiguous.sum())} "
          f"({ambiguous.mean():.1%})")

    # Do the low-confidence days sit near regime changes? Mark every day
    # within 3 trading days of a GMM label switch, then see what share of
    # ambiguous days fall in that band vs. the base rate.
    switch = np.diff(gmm_labels) != 0
    near_switch = np.zeros(len(gmm_labels), dtype=bool)
    switch_idx = np.where(switch)[0]
    for i in switch_idx:
        near_switch[max(0, i - 2): i + 4] = True
    print(
        f"Share of ALL days within 3 days of a regime change : "
        f"{near_switch.mean():.1%}"
    )
    print(
        f"Share of AMBIGUOUS days within 3 days of a change   : "
        f"{near_switch[ambiguous].mean():.1%}   "
        "(higher = low confidence really is a transition signal)"
    )
    amb_by_year = (
        pd.Series(ambiguous, index=df.index)
        .groupby(df.index.year)
        .sum()
    )
    print("\nAmbiguous days by year (top 8):")
    print(amb_by_year.sort_values(ascending=False).head(8).to_string())

    # --- 4. GMM regime profile ---
    with pd.option_context(
        "display.max_columns", None, "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print("\n--- GMM regime profile (mean feature value per regime) -")
        print(profile_regimes(df, gmm_labels))

    # --- 5. save ---
    out = proba_df.copy()
    out.insert(0, "regime", gmm_labels)
    out["confidence"] = confidence
    out.to_csv(GMM_OUTPUT_PATH)
    print(f"\nWrote GMM regimes -> {GMM_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
