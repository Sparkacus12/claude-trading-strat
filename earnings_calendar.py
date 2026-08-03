"""
Earnings-calendar utilities for event-driven rebalancing.

WHY THIS EXISTS
NOWCAST = earnings_beta x business_cycle. A stock's beta ONLY changes when
that company files. Between filings its NOWCAST score is stale by
construction. Rebalancing the whole book monthly therefore trades 11 months
out of 12 on information that hasn't changed for that name - pure turnover
cost for no new signal.

Momentum and trend DO update continuously, so the answer isn't "rebalance
everything quarterly" either. The right design runs the sleeves on
DIFFERENT CLOCKS:
  - NOWCAST      -> each stock's own earnings calendar (staggered, ~quarterly)
  - momentum/trend -> continuous (monthly checkpoints)

DATA LIMITATION (important)
EDGAR gives us PAST filing dates. There is no free, reliable forward earnings
calendar. So "expected next announcement" is ESTIMATED from each company's
own historical filing cadence (median gap between filings). That estimate is
usually good to within a few days for large caps with regular reporting, but
it IS an estimate - a company that shifts its reporting date will be mistimed.
For a pre-announcement entry strategy that timing error matters, so treat the
event-window mode as more approximate than the staggered-refresh mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_earnings_calendar(earnings: pd.DataFrame) -> dict:
    """
    Per-ticker filing history and cadence.

    Returns {ticker: {"dates": sorted DatetimeIndex,
                      "median_gap_days": float,
                      "last": Timestamp}}
    """
    if earnings is None or earnings.empty:
        return {}

    df = earnings.copy()
    df["announcement_date"] = pd.to_datetime(df["announcement_date"])
    cal = {}
    for tk, g in df.groupby("ticker"):
        d = pd.DatetimeIndex(sorted(g["announcement_date"].dropna().unique()))
        if len(d) < 2:
            continue
        gaps = np.diff(d.values).astype("timedelta64[D]").astype(float)
        # keep plausible quarterly gaps; drop restatement/amendment artefacts
        gaps = gaps[(gaps > 45) & (gaps < 200)]
        med = float(np.median(gaps)) if len(gaps) else 91.0
        cal[tk] = {"dates": d, "median_gap_days": med, "last": d[-1]}
    return cal


def filed_between(cal: dict, ticker: str, start, end) -> bool:
    """Did this ticker file in (start, end]? -> its NOWCAST is refreshed."""
    info = cal.get(ticker)
    if not info:
        return False
    d = info["dates"]
    return bool(((d > pd.Timestamp(start)) & (d <= pd.Timestamp(end))).any())


def expected_next_announcement(cal: dict, ticker: str, as_of) -> pd.Timestamp:
    """
    Estimate the next filing date from the company's own cadence.
    ESTIMATE ONLY - see the data-limitation note at the top of this file.
    """
    info = cal.get(ticker)
    if not info:
        return pd.NaT
    as_of = pd.Timestamp(as_of)
    prior = info["dates"][info["dates"] <= as_of]
    if len(prior) == 0:
        return pd.NaT
    nxt = prior[-1] + pd.Timedelta(days=info["median_gap_days"])
    # roll forward if our estimate is already in the past
    while nxt <= as_of:
        nxt = nxt + pd.Timedelta(days=info["median_gap_days"])
    return nxt


def days_to_next_announcement(cal: dict, ticker: str, as_of) -> float:
    nxt = expected_next_announcement(cal, ticker, as_of)
    if pd.isna(nxt):
        return np.nan
    return float((nxt - pd.Timestamp(as_of)).days)


def recently_filed(cal: dict, ticker: str, as_of, window_days: int = 45) -> bool:
    """Filed within the last `window_days` -> NOWCAST information is fresh."""
    info = cal.get(ticker)
    if not info:
        return False
    as_of = pd.Timestamp(as_of)
    d = info["dates"]
    recent = d[(d <= as_of) & (d > as_of - pd.Timedelta(days=window_days))]
    return len(recent) > 0

