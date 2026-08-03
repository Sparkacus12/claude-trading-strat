"""
Three separate backtests, each testing a different thing.

  1. backtest_nowcast_faithful()  - the paper's ACTUAL strategy
  2. backtest_momentum()          - canonical 12-1 momentum long-short
  3. backtest_combined()          - the long-only blended book (as before)

WHY SEPARATE: they are structurally different animals and comparing them on
one chart is misleading. (1) and (2) are market-NEUTRAL long-short spreads -
their return is a spread, not an equity return, and the right benchmark is
zero. (3) is a long-only equity book whose right benchmark is the index.

WHAT WE'D BEEN GETTING WRONG (from re-reading Carabias 2018, Table 9):

  * The paper's strategy is LONG-SHORT: "buys (short-sells) the stocks in the
    top (bottom) decile of NOWCAST". We had built long-only. A concentrated
    long-only equity book is dominated by market beta, which is why it tracked
    the index. The paper's portfolio has a market loading of -0.14 (t=-0.94):
    essentially market-neutral.

  * The HOLD IS EVENT-ANCHORED AND SHORT: enter before the announcement, exit
    "the day after the earnings announcement". Not a fixed monthly hold.

  * The alpha is MOMENTUM-ORTHOGONAL: MOM loading -0.04 (t=-0.09). Blending
    momentum into the nowcast book mixes in the very factor the documented
    alpha is independent of - so momentum belongs in its OWN sleeve, not
    stirred into this one.

  * Paper's sample filters: calendar fiscal quarters only (Mar/Jun/Sep/Dec),
    drop stocks under $1, winsorise the top/bottom 0.5% each quarter.

HONESTY: survivorship bias (current index membership) still inflates
everything here, betas are still estimated on ~30 noisy quarters, and a
long-short spread on free data ignores borrow cost and shorting frictions
entirely. Read the split-sample table before the headline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as e
import enhancements as x
import earnings_calendar as ec


# ----------------------------------------------------------------------
# Paper's sample filters
# ----------------------------------------------------------------------

def apply_paper_filters(earnings: pd.DataFrame, prices: pd.DataFrame,
                        calendar_quarters_only: bool = True,
                        min_price: float = 1.0,
                        winsorise: float = 0.005) -> pd.DataFrame:
    """Filters from the paper's sample construction."""
    df = earnings.copy()
    if calendar_quarters_only:
        m = pd.to_datetime(df["fiscal_period_end"]).dt.month
        df = df[m.isin([3, 6, 9, 12])]
    if min_price and "price_at_period_end" in df.columns:
        df = df[df["price_at_period_end"] >= min_price]
    return df


def _winsorise(s: pd.Series, p: float = 0.005) -> pd.Series:
    if s.dropna().empty or p <= 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def _stats(r: pd.Series, ppy: int = 12, label: str = "Strategy") -> dict:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return {}
    eq = (1 + r).cumprod()
    return {
        "Total": f"{eq.iloc[-1] - 1:.1%}",
        "Annualised": f"{eq.iloc[-1] ** (ppy / len(r)) - 1:.1%}",
        "Vol": f"{r.std() * np.sqrt(ppy):.1%}",
        "Sharpe": f"{r.mean() / r.std() * np.sqrt(ppy):.2f}",
        "Max DD": f"{(eq / eq.cummax() - 1).min():.1%}",
        "Hit rate": f"{(r > 0).mean():.0%}",
        "N": len(r),
    }


def _split(r: pd.Series, ppy: int = 12) -> pd.DataFrame:
    r = r.dropna()
    if len(r) < 12:
        return pd.DataFrame({"note": ["Too few periods to split."]})
    mid = len(r) // 2
    out = {}
    for label, seg in [("First half", r.iloc[:mid]),
                       ("Second half (out-of-sample)", r.iloc[mid:])]:
        if len(seg) < 2 or seg.std() == 0:
            continue
        eq = (1 + seg).cumprod()
        out[label] = {
            "Annualised": f"{eq.iloc[-1] ** (ppy / len(seg)) - 1:.1%}",
            "Sharpe": f"{seg.mean() / seg.std() * np.sqrt(ppy):.2f}",
            "Hit rate": f"{(seg > 0).mean():.0%}",
            "N": len(seg),
        }
    return pd.DataFrame(out).T


# ======================================================================
# 1. FAITHFUL NOWCAST LONG-SHORT
# ======================================================================

def backtest_nowcast_faithful(prices, earnings, macro,
                              decile_frac: float = 0.10,
                              event_window_days: int = 45,
                              exit_days_after: int = 1,
                              cost_bps: float = 10.0,
                              beta_shrink: float = 0.4,
                              sectors: pd.Series = None,
                              use_kalman: bool = True,
                              start_after_months: int = 24,
                              apply_filters: bool = True,
                              min_names: int = 20,
                              min_upcoming: int = 10) -> dict:
    """
    The paper's strategy, as literally as free data allows.

      Each month: rank the cross-section by NOWCAST.
      Among names whose expected announcement falls in the next
      `event_window_days`, go LONG the top decile and SHORT the bottom decile.
      Hold each position from formation until `exit_days_after` days after
      that name's announcement, then exit.
      Dollar-neutral: long leg and short leg equally sized.

    Return each period is the LONG-SHORT SPREAD, so the benchmark is ZERO,
    not the index. Market exposure should be near zero by construction.
    """
    sectors = sectors if sectors is not None else pd.Series(dtype=str)
    if apply_filters:
        earnings = apply_paper_filters(earnings, prices)

    bc = e.build_business_cycle_factor(macro)
    bc_q = e.factor_quarterly(bc)
    cal = ec.build_earnings_calendar(earnings)
    if not cal:
        return {"error": "No earnings calendar could be built."}

    monthly = prices.resample("ME").last()
    grid = monthly.index[start_after_months:]
    if len(grid) < 12:
        return {"error": "Not enough history."}

    rows, detail = [], []
    skips = {"short_price_history": 0, "no_earnings_yet": 0, "no_betas": 0,
             "too_few_nowcast": 0, "too_few_upcoming": 0, "leg_return_nan": 0,
             "ok": 0}
    live_counts = []

    for dt in grid[:-1]:
        px_slice = prices.loc[:dt]
        if px_slice.shape[0] < 260:
            skips["short_price_history"] += 1
            continue

        e_slice = earnings[pd.to_datetime(earnings["announcement_date"]) <= dt]
        if e_slice.empty:
            skips["no_earnings_yet"] += 1
            continue
        bcq_slice = bc_q.loc[:dt]

        sue = e.compute_sue(e_slice)
        betas = (x.estimate_earnings_betas_kalman(sue, bcq_slice)
                 if use_kalman else e.estimate_earnings_betas(sue, bcq_slice))
        if betas.empty:
            skips["no_betas"] += 1
            continue
        nc = e.compute_nowcast(betas, bcq_slice, dt)
        if nc.empty or len(nc) < min_names:
            skips["too_few_nowcast"] += 1
            continue
        if beta_shrink > 0:
            nc = x.shrink_betas(nc, sectors, shrink=beta_shrink)

        # winsorise the signal tails (paper drops top/bottom 0.5%)
        nc["nowcast"] = _winsorise(nc["nowcast"], 0.005)

        # only names with an announcement coming up in the window
        nc["days_to_ann"] = nc["ticker"].apply(
            lambda t: ec.days_to_next_announcement(cal, t, dt))
        live = nc[nc["days_to_ann"].between(0, event_window_days)].copy()
        live_counts.append(len(live))
        if len(live) < min_upcoming:
            skips["too_few_upcoming"] += 1
            continue

        live = live.sort_values("nowcast", ascending=False)
        k = max(1, int(round(len(live) * decile_frac)))
        longs = live.head(k)
        shorts = live.tail(k)

        def leg_return(sub, sign):
            rets = []
            for _, row in sub.iterrows():
                t = row["ticker"]
                if t not in prices.columns:
                    continue
                ann = ec.expected_next_announcement(cal, t, dt)
                if pd.isna(ann):
                    continue
                exit_d = ann + pd.Timedelta(days=exit_days_after)
                seg = prices[t].loc[dt:exit_d].dropna()
                if len(seg) < 2:
                    continue
                rets.append(sign * (seg.iloc[-1] / seg.iloc[0] - 1))
            return float(np.mean(rets)) if rets else np.nan

        lr = leg_return(longs, 1.0)
        sr = leg_return(shorts, -1.0)
        if np.isnan(lr) or np.isnan(sr):
            skips["leg_return_nan"] += 1
            continue
        skips["ok"] += 1

        gross = 0.5 * lr + 0.5 * sr          # dollar-neutral
        # both legs traded in and out each period
        cost = 2 * (2 * cost_bps / 10000)
        rows.append({"date": dt, "spread_gross": gross,
                     "spread_net": gross - cost,
                     "long_leg": lr, "short_leg": sr,
                     "n_long": len(longs), "n_short": len(shorts)})
        detail.append({"date": dt.date(),
                       "longs": ", ".join(longs["ticker"].head(10)),
                       "shorts": ", ".join(shorts["ticker"].head(10))})

    diag = {"periods_attempted": len(grid) - 1, **skips,
            "median_upcoming_names": (int(np.median(live_counts))
                                      if live_counts else 0),
            "tickers_in_calendar": len(cal)}

    if len(rows) < 6:
        return {"error": f"Only {len(rows)} usable period(s) — not enough for a "
                         "backtest.", "diagnostics": diag}

    bt = pd.DataFrame(rows).set_index("date")
    bt["equity_net"] = (1 + bt["spread_net"]).cumprod()
    bt["equity_gross"] = (1 + bt["spread_gross"]).cumprod()

    stats = pd.DataFrame({
        "Long-short (net)": _stats(bt["spread_net"]),
        "Long-short (gross)": _stats(bt["spread_gross"]),
        "Long leg only": _stats(bt["long_leg"]),
        "Short leg only": _stats(bt["short_leg"]),
    }).T

    return {"table": bt, "stats": stats,
            "split": _split(bt["spread_net"]),
            "detail": pd.DataFrame(detail),
            "diagnostics": diag,
            "note": "Market-NEUTRAL long-short spread. Benchmark is ZERO, not "
                    "the index. Positive Sharpe here means genuine spread "
                    "capture, independent of market direction."}


# ======================================================================
# 2. MOMENTUM STANDALONE
# ======================================================================

def backtest_momentum(prices, decile_frac: float = 0.10,
                      cost_bps: float = 10.0,
                      hold_months: int = 1,
                      start_after_months: int = 14,
                      long_only: bool = False) -> dict:
    """
    Canonical 12-1 cross-sectional momentum, long-short decile spread
    (or long-only top decile if long_only=True), monthly rebalance.

    Kept SEPARATE from nowcast because the paper's alpha is momentum-
    orthogonal (MOM loading -0.04) - blending them obscures both.
    """
    monthly = prices.resample("ME").last()
    grid = monthly.index[start_after_months::hold_months]
    if len(grid) < 12:
        return {"error": "Not enough history."}

    mret = monthly.pct_change()
    rows, detail = [], []
    prev_long, prev_short = set(), set()

    for i, dt in enumerate(grid[:-1]):
        nxt = grid[i + 1]
        mom = e.momentum_score(prices.loc[:dt], dt)
        if mom.empty or len(mom) < 20:
            continue
        mom = mom.sort_values("mom_score", ascending=False)
        k = max(1, int(round(len(mom) * decile_frac)))
        longs = [t for t in mom.head(k)["ticker"] if t in mret.columns]
        shorts = [t for t in mom.tail(k)["ticker"] if t in mret.columns]
        if not longs:
            continue

        win = mret.loc[dt:nxt].iloc[1:]
        if win.empty:
            continue
        lr = float((1 + win[longs].mean(axis=1)).prod() - 1)
        sr = float((1 + win[shorts].mean(axis=1)).prod() - 1) if shorts else 0.0

        turn = (len(set(longs) ^ prev_long) + len(set(shorts) ^ prev_short)) / \
               max(2 * (len(longs) + len(shorts)), 1)
        cost = turn * 2 * cost_bps / 10000
        prev_long, prev_short = set(longs), set(shorts)

        gross = lr if long_only else 0.5 * lr - 0.5 * sr
        bench = float((1 + win.mean(axis=1)).prod() - 1)

        rows.append({"date": nxt, "gross": gross, "net": gross - cost,
                     "long_leg": lr, "short_leg": sr, "benchmark": bench})
        detail.append({"date": dt.date(), "longs": ", ".join(longs[:10]),
                       "shorts": ", ".join(shorts[:10])})

    if not rows:
        return {"error": "No periods produced."}

    bt = pd.DataFrame(rows).set_index("date")
    bt["equity_net"] = (1 + bt["net"]).cumprod()
    bt["equity_bench"] = (1 + bt["benchmark"]).cumprod()

    s = {"Momentum (net)": _stats(bt["net"]),
         "Momentum (gross)": _stats(bt["gross"]),
         "Long leg": _stats(bt["long_leg"])}
    if not long_only:
        s["Short leg"] = _stats(bt["short_leg"])
    s["Benchmark (EW)"] = _stats(bt["benchmark"])

    return {"table": bt, "stats": pd.DataFrame(s).T,
            "split": _split(bt["net"]), "detail": pd.DataFrame(detail),
            "note": ("Long-only top decile — benchmark is the index."
                     if long_only else
                     "Long-short decile spread — benchmark is ZERO, not the index.")}


# ======================================================================
# 3. COMBINED (long-only blended book) - thin wrapper over strategy.py
# ======================================================================

def backtest_combined(prices, earnings, macro, **kwargs) -> dict:
    """
    The long-only blended book (nowcast + momentum + trend), unchanged from
    before. Benchmark IS the index here, because this is a long equity book.

    Kept for comparison, but note: mixing momentum into the nowcast book
    blends in the factor the paper's alpha is explicitly orthogonal to, so
    a null result here doesn't tell you the nowcast signal is dead - test (1).
    """
    import strategy as S
    return S.run_backtest(prices, earnings, macro, **kwargs)
