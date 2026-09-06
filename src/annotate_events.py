"""
Stage 6: Event annotation layer.

Ties the hand-curated macro events in data/events.csv to the regime shifts
the model found (Stage 4, smoothed as in Stage 5).

The point is interpretability, not prediction. Regime detection is
unsupervised - it never sees a headline. This script checks, after the
fact, whether each regime shift lines up with a real-world event, and
prints:

  * every regime shift, with any events in its neighbourhood
  * a shortlist of the "major" shifts (into or out of Crisis, or into a
    long new episode) with their likely trigger
  * events that DON'T sit near a shift - useful context that happened
    mid-regime and didn't move the needle enough to reclassify the days

Matching window: an event counts as "near" a shift if it falls from
LOOKBACK_DAYS before the shift date to LOOKAHEAD_DAYS after. The window is
lopsided on purpose - the regime label lags its cause, because the 20-day
rolling features need time to move and the episode-length smoothing pushes
the detected change later still.

Output: data/regime_shifts.csv

Run standalone from inside src/:

    cd src && python3 annotate_events.py
"""

from pathlib import Path

import pandas as pd

from regime_labels import (
    MIN_REGIME_RUN,
    regime_episodes,
    regime_shifts,
    smooth_regimes,
)

BASE_DIR = Path(__file__).resolve().parent.parent
EVENTS_PATH = BASE_DIR / "data" / "events.csv"
REGIMES_PATH = BASE_DIR / "data" / "regimes_gmm.csv"
OUTPUT_PATH = BASE_DIR / "data" / "regime_shifts.csv"

# How far on either side of a shift an event still counts as "near" it.
LOOKBACK_DAYS = 45   # the regime label lags its cause
LOOKAHEAD_DAYS = 10


def load_shifts_and_episodes():
    regime = pd.read_csv(REGIMES_PATH, index_col=0, parse_dates=True)["regime"]
    regime = smooth_regimes(regime, MIN_REGIME_RUN)
    return regime_shifts(regime), regime_episodes(regime)


def load_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_PATH, parse_dates=["date"])
    return events.sort_values("date").reset_index(drop=True)


def events_near(shift_date: pd.Timestamp, events: pd.DataFrame) -> pd.DataFrame:
    """Events falling within the lopsided window around a shift date."""
    lo = shift_date - pd.Timedelta(days=LOOKBACK_DAYS)
    hi = shift_date + pd.Timedelta(days=LOOKAHEAD_DAYS)
    window = events[(events["date"] >= lo) & (events["date"] <= hi)].copy()
    window["days_from_shift"] = (window["date"] - shift_date).dt.days
    return window


def episode_containing(date: pd.Timestamp, episodes: pd.DataFrame):
    """The regime episode a given date sits inside (or None)."""
    hit = episodes[(episodes["start"] <= date) & (episodes["end"] >= date)]
    return hit.iloc[0] if len(hit) else None


def is_major(shift) -> bool:
    """A shift worth calling out: touches Crisis, or opens a long episode."""
    return (
        shift.to_regime == 4
        or shift.from_regime == 4
        or shift.new_regime_days >= 90
    )


def build_shift_table(shifts: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per regime shift, with the events sitting near it.

    Reused by both this script's report and the dashboard, so it returns a
    plain DataFrame (dates as Timestamps) and does no printing.
    """
    rows = []
    for shift in shifts.itertuples():
        near = events_near(shift.date, events)

        # the closest event that is not *after* the shift - the likely lead
        leads = near[near["days_from_shift"] <= 0]
        lead = leads.iloc[-1] if len(leads) else None

        rows.append(
            {
                "date": shift.date,
                "regime_from": shift.from_name,
                "regime_to": shift.to_name,
                "new_regime_days": shift.new_regime_days,
                "major": is_major(shift),
                "n_events": len(near),
                "lead_event": None if lead is None else lead["event"],
                "lead_event_date": None if lead is None else lead["date"],
                "lead_gap_days": None if lead is None else int(lead["days_from_shift"]),
                "nearby_events": " ; ".join(
                    f"{r.date.date()} {r.event}" for r in near.itertuples()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    shifts, episodes = load_shifts_and_episodes()
    events = load_events()

    shift_table = build_shift_table(shifts, events)
    shift_table.to_csv(OUTPUT_PATH, index=False)

    # An event is "matched" if it lands in the window of at least one shift.
    all_shift_dates = shifts["date"]
    matched_mask = events["date"].apply(
        lambda d: bool(
            (
                (all_shift_dates - d >= pd.Timedelta(days=-LOOKAHEAD_DAYS))
                & (all_shift_dates - d <= pd.Timedelta(days=LOOKBACK_DAYS))
            ).any()
        )
    )

    # --- report ---
    n_shifts = len(shift_table)
    n_with_event = int((shift_table["n_events"] > 0).sum())
    print(
        f"\n{n_shifts} regime shifts (smoothed, min {MIN_REGIME_RUN}d episodes). "
        f"{n_with_event} have an event within "
        f"-{LOOKBACK_DAYS}/+{LOOKAHEAD_DAYS} days.\n"
    )

    def describe_match(row) -> None:
        if pd.notna(row.lead_gap_days):
            gap = int(row.lead_gap_days)
            # A tight gap reads as causal; a loose one is just "around then".
            label = "likely trigger" if gap >= -21 else "preceded by"
            print(f'     {label}: {row.lead_event_date.date()}  "{row.lead_event}"  ({gap:+d}d)')
        elif row.nearby_events:
            print(f"     nearby: {row.nearby_events}")
        else:
            print("     no curated event nearby")

    print("=== Major shifts (Crisis in/out, or a 90+ day new regime) =========")
    for row in shift_table[shift_table["major"]].itertuples():
        print(f"\n  {row.date.date()}  {row.regime_from}  ->  {row.regime_to}   (new episode {row.new_regime_days}d)")
        describe_match(row)

    print("\n\n=== Other regime shifts that line up with an event ===============")
    other = shift_table[(~shift_table["major"]) & (shift_table["n_events"] > 0)]
    for row in other.itertuples():
        print(f"\n  {row.date.date()}  {row.regime_from}  ->  {row.regime_to}   (new episode {row.new_regime_days}d)")
        describe_match(row)

    # --- events that didn't land near any shift ---
    unmatched = events[~matched_mask]
    print("\n\n=== Events with no regime shift nearby (mid-regime context) =======")
    for ev in unmatched.itertuples():
        ep = episode_containing(ev.date, episodes)
        where = f"during {ep['name']}" if ep is not None else "outside the labelled range"
        print(f"  {ev.date.date()}  {ev.event:<45}  ({where})")

    print(f"\n\nWrote shift-by-shift table -> {OUTPUT_PATH}")
    print(f"  {n_shifts} shifts, {n_with_event} event-matched, "
          f"{len(events) - len(unmatched)} of {len(events)} events used.")


if __name__ == "__main__":
    main()
