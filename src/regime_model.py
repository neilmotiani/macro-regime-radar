"""
Frozen regime model - fit once, score forever.

Stage 4 (detect_regimes.py / compare_regimes.py) fits clustering models on
the whole history for exploration. That is the wrong thing to do on a
daily schedule: re-fitting when one new day arrives can nudge the cluster
boundaries and silently relabel years-old dates, so every automated commit
would carry a huge noisy regime diff.

So the daily pipeline uses this module instead. We fit the Gaussian
Mixture (plus its scaler and its regime numbering) ONCE, save the whole
bundle to models/regime_gmm.joblib, and from then on only *score* new days
against those frozen parameters. History stays put; only genuinely new
rows get a label.

Re-baseline deliberately (not on a schedule) with:

    cd src && python3 score_regimes.py --fit
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "regime_gmm.joblib"

# The 9 rolling features the model clusters on (mirrors detect_regimes.py -
# that script imports this list so the two never drift apart).
CLUSTER_FEATURES = [
    "gold_vol_20d",
    "oil_vol_20d",
    "gold_oil_corr_20d",
    "gold_dollar_index_corr_20d",
    "gold_vix_corr_20d",
    "oil_dollar_index_corr_20d",
    "gold_momentum_20d",
    "oil_momentum_20d",
    "yield_10y",
]

K = 5                    # number of regimes (see detect_regimes.py for why)
COVARIANCE_TYPE = "full"
RANDOM_STATE = 42


def _stress_order(features: pd.DataFrame, raw_labels: np.ndarray) -> list[int]:
    """Raw GMM component ids sorted calmest -> most volatile.

    Position i in the returned list is the raw component that becomes
    regime i. Sorting by average total volatility keeps "regime 4 = crisis"
    stable across re-fits and consistent with detect_regimes.py.
    """
    stress = (
        (features["gold_vol_20d"] + features["oil_vol_20d"])
        .groupby(raw_labels)
        .mean()
        .sort_values()
    )
    return list(stress.index)


def fit_and_save(features: pd.DataFrame) -> dict:
    """Fit scaler + GMM on the given features, freeze the regime numbering,
    and write the bundle to MODEL_PATH."""
    frame = features[CLUSTER_FEATURES].dropna()
    x_raw = frame.to_numpy()

    scaler = StandardScaler().fit(x_raw)
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=COVARIANCE_TYPE,
        random_state=RANDOM_STATE,
        n_init=5,
    ).fit(scaler.transform(x_raw))

    raw_labels = gmm.predict(scaler.transform(x_raw))
    component_order = _stress_order(frame, raw_labels)

    bundle = {
        "scaler": scaler,
        "gmm": gmm,
        "component_order": component_order,  # regime i  <-  component_order[i]
        "features": CLUSTER_FEATURES,
        "k": K,
        "fitted_through": str(frame.index.max().date()),
        "n_fit_rows": len(frame),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, MODEL_PATH)
    return bundle


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Baseline the model first:\n"
            "    cd src && python3 score_regimes.py --fit"
        )
    return load(MODEL_PATH)


def score(features: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Assign every row of `features` to a regime using the frozen model.

    Returns a DataFrame indexed by date with columns:
    regime, prob_0..prob_{k-1}, confidence.
    """
    frame = features[bundle["features"]].dropna()
    x = bundle["scaler"].transform(frame.to_numpy())

    raw_proba = bundle["gmm"].predict_proba(x)
    # Reorder the probability columns into regime order (calmest first),
    # so column j is P(regime j) and argmax is the regime label directly.
    proba = raw_proba[:, bundle["component_order"]]

    out = pd.DataFrame(
        proba,
        index=frame.index,
        columns=[f"prob_{i}" for i in range(bundle["k"])],
    )
    out.insert(0, "regime", proba.argmax(axis=1))
    out["confidence"] = proba.max(axis=1)
    return out
