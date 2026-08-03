"""
Signal combiner + backtest (v3).

NEW IN v3: REBALANCE MODES. The sleeves run on different clocks.

  "monthly"   - whole book reconsidered every month (v2 behaviour).
                Suits momentum/trend; OVER-TRADES nowcast, whose betas only
                change when a company files.

  "quarterly" - whole book reconsidered every 3 months. Blunt but cheap;
                roughly matches the earnings clock in aggregate.

  "earnings"  - STAGGERED, per-stock earnings clock (recommended).
                A holding is only re-evaluated on nowcast grounds when THAT
                company has actually filed. Momentum/trend still act every
                month as a risk overlay (trend-break exits). Turnover
                naturally staggers across the universe instead of the whole
                book churning on one date.

Why "earnings" is the honest design: NOWCAST = beta x cycle, and beta is
constant between filings. Monthly rebalancing therefore pays turnover 11
months out of 12 for no new nowcast information - while momentum and trend
genuinely do update continuously, so a pure quarterly clock would leave
those stale. Different clocks for different sleeves.

HONESTY NOTES (unchanged and still important):
  - SURVIVORSHIP BIAS: current index membership only. Results inflated.
  - Earnings side IS point-in-time (real SEC filing dates).
  - Betas from ~30 noisy quarterly observations; shrinkage helps, not cures.
  - Flat bps cost, no borrow, no impact.
  - READ THE SPLIT-SAMPLE TABLE FIRST. If the second half is much weaker,
    the edge is fitted, not predictive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as e
import enhancements as x
import earnings_calendar as ec


# ----------------------------------------------------------------------
# COMBINER (unchanged from v2)
# ----------------------------------------------------------------------

def combine_signals(nowcast, momentum, trend,
                    w_nowcast=0.4, w_momentum=0.4, w_trend=0.2,
                    n_buy=10, n_sell=10, require_trend_for_buy=True,
                    sectors=None, sector_neutral=False,
                    w_sue_lag=0.0) -> pd.DataFrame:
    """
    w_sue_lag: weight on the NEGATIVE lagged-SUE leg (paper Table 8 col 3:
    CAR = +0.56*NOWCAST - 0.17*SUE_lag1). Investors over-weight last
    quarter's realised earnings and under-react to the macro nowcast, so the
    profitable trade is long nowcast AND SHORT lagged SUE. Paper's implied
    ratio is roughly 0.17/0.56 ~= 0.30 of the nowcast weight.
    """
    sectors = sectors if sectors is not None else pd.Series(dtype=str)
    frames = []

    if nowcast is not None and not nowcast.empty:
        nc = nowcast[["ticker", "nowcast"]].copy()
        nc["nowcast_pct"] = (x.sector_neutral_rank(nc, "nowcast", sectors)
                             if sector_neutral else nc["nowcast"].rank(pct=True))
        frames.append(nc[["ticker", "nowcast", "nowcast_pct"]])

    if momentum is not None and not momentum.empty:
        mo = momentum[["ticker", "mom_score"]].copy()
        mo["momentum_pct"] = (x.sector_neutral_rank(mo, "mom_score", sectors)
                              if sector_neutral else mo["mom_score"].rank(pct=True))
        frames.append(mo[["ticker", "mom_score", "momentum_pct"]])

    if trend is not None and not trend.empty:
        tr = trend[["ticker", "trend_score", "clean_trend"]].copy()
        tr["trend_pct"] = (x.sector_neutral_rank(tr, "trend_score", sectors)
                           if sector_neutral else tr["trend_score"].rank(pct=True))
        frames.append(tr[["ticker", "trend_score", "trend_pct", "clean_trend"]])

    if not frames:
        return pd.DataFrame()

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="ticker", how="outer")

    for col in ["nowcast_pct", "momentum_pct", "trend_pct"]:
        if col not in df.columns:
            df[col] = 0.5
        df[col] = df[col].fillna(0.5)
    if "clean_trend" not in df.columns:
        df["clean_trend"] = True
    df["clean_trend"] = df["clean_trend"].fillna(False)

    # Paper Table 8: SHORT the lagged-SUE component (investors over-weight it)
    if w_sue_lag > 0 and nowcast is not None and not nowcast.empty \
            and "sue_lag1_pct" in nowcast.columns:
        sl = nowcast[["ticker", "sue_lag1_pct"]]
        df = df.merge(sl, on="ticker", how="left")
        df["sue_lag1_pct"] = df["sue_lag1_pct"].fillna(0.5)
    else:
        df["sue_lag1_pct"] = 0.5

    tot = w_nowcast + w_momentum + w_trend + w_sue_lag
    df["combined_score"] = (w_nowcast * df["nowcast_pct"]
                            + w_momentum * df["momentum_pct"]
                            + w_trend * df["trend_pct"]
                            + w_sue_lag * (1.0 - df["sue_lag1_pct"])) / tot
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    df["action"] = "HOLD"
    pool = df[df["clean_trend"]] if require_trend_for_buy else df
    buys = pool.head(n_buy)["ticker"].tolist()
    sells = df.tail(n_sell)["ticker"].tolist() if n_sell > 0 else []
    df.loc[df["ticker"].isin(buys), "action"] = "BUY"
    df.loc[df["ticker"].isin(sells), "action"] = "SELL"

    if len(sectors):
        df["sector"] = df["ticker"].map(sectors)

    agree = {}
    for a, b in [("nowcast_pct", "momentum_pct"), ("nowcast_pct", "trend_pct"),
                 ("momentum_pct", "trend_pct")]:
        try:
            c = df[a].corr(df[b])
            agree[f"{a[:-4]} vs {b[:-4]}"] = round(float(c), 3) if pd.notna(c) else None
        except Exception:
            pass
    df.attrs["sleeve_agreement"] = agree
    return df


# ----------------------------------------------------------------------
# BACKTEST
# ----------------------------------------------------------------------

def run_backtest(prices, earnings, macro, sectors=None,
                 n_hold=10, cost_bps=10.0,
                 w_nowcast=0.4, w_momentum=0.4, w_trend=0.2,
                 start_after_months=24, use_vol_weights=True,
                 sector_neutral=False, beta_shrink=0.4,
                 rebalance_mode="monthly",
                 use_event_overlay=True,
                 event_window_days=45,
                 w_event=0.25,
                 use_kalman=True,
                 kalman_signal_noise=0.05,
                 w_sue_lag=0.0) -> dict:
    """
    rebalance_mode: "monthly" | "quarterly"  (cadence of the CORE book)
    use_event_overlay: layer the earnings-event tilt on top (recommended)

    The design: the core book rebalances monthly, so momentum and trend -
    which genuinely update continuously - stay fresh. On top of that, the
    event overlay tilts toward high-nowcast names whose announcement is
    imminent, which is where Carabias' edge is actually realised.
    """
    sectors = sectors if sectors is not None else pd.Series(dtype=str)
    bc = e.build_business_cycle_factor(macro)
    bc_q = e.factor_quarterly(bc)
    cal = ec.build_earnings_calendar(earnings)

    monthly = prices.resample("ME").last()
    step = 3 if rebalance_mode == "quarterly" else 1
    grid = monthly.index[start_after_months::step]
    if len(grid) < 6:
        return {"error": "Not enough history. Increase the history window."}

    daily_ret = prices.pct_change()
    rows, holdings_log = [], []
    prev_w = pd.Series(dtype=float)
    held: list[str] = []
    last_seen = pd.Timestamp(grid[0])

    for i, dt in enumerate(grid[:-1]):
        nxt = grid[i + 1]
        px_slice = prices.loc[:dt]
        if px_slice.shape[0] < 260:
            continue

        e_slice = earnings[pd.to_datetime(earnings["announcement_date"]) <= dt]
        bcq_slice = bc_q.loc[:dt]

        nc = pd.DataFrame()
        if not e_slice.empty:
            sue = e.compute_sue(e_slice)
            # IMPROVEMENT A: Kalman filter is the paper's actual estimator
            if use_kalman:
                betas = x.estimate_earnings_betas_kalman(
                    sue, bcq_slice, signal_noise=kalman_signal_noise)
            else:
                betas = e.estimate_earnings_betas(sue, bcq_slice)
            if not betas.empty:
                nc = e.compute_nowcast(betas, bcq_slice, dt)
                if not nc.empty and beta_shrink > 0:
                    nc = x.shrink_betas(nc, sectors, shrink=beta_shrink)
                # IMPROVEMENT B: attach lagged SUE for the short leg
                if not nc.empty and w_sue_lag > 0:
                    nc = x.attach_lagged_sue(nc, sue, dt)

        mom = e.momentum_score(px_slice, dt)
        trd = e.trend_quality(px_slice, dt)

        combo = combine_signals(nc, mom, trd, w_nowcast, w_momentum, w_trend,
                                n_buy=n_hold, n_sell=0, sectors=sectors,
                                sector_neutral=sector_neutral,
                                w_sue_lag=w_sue_lag)
        if combo.empty:
            continue

        # EVENT OVERLAY: monthly core book, tilted toward high-nowcast names
        # whose announcement is imminent (the paper's actual trade).
        if use_event_overlay and cal:
            combo = apply_event_overlay(combo, cal, dt,
                                        event_window_days=event_window_days,
                                        w_event=w_event)

        picks = combo[combo["action"] == "BUY"]["ticker"].tolist()
        if not picks:
            picks = combo.head(n_hold)["ticker"].tolist()

        last_seen = dt
        picks = [p for p in picks if p in daily_ret.columns]
        if not picks:
            continue
        held = picks

        if use_vol_weights:
            w = x.inverse_vol_weights(px_slice, picks, as_of=dt)
        else:
            w = pd.Series(1.0 / len(picks), index=picks)
        w = w[w.index.isin(daily_ret.columns)]
        if w.empty:
            continue
        w = w / w.sum()

        window = daily_ret.loc[dt:nxt, list(w.index)].iloc[1:]
        if window.empty:
            continue
        period_ret = float((1 + (window * w).sum(axis=1)).prod() - 1)

        all_t = w.index.union(prev_w.index)
        turnover = float((w.reindex(all_t).fillna(0)
                          - prev_w.reindex(all_t).fillna(0)).abs().sum() / 2)
        cost = turnover * 2 * cost_bps / 10000
        prev_w = w

        bw = daily_ret.loc[dt:nxt].iloc[1:]
        bench = float((1 + bw.mean(axis=1)).prod() - 1) if not bw.empty else 0.0

        rows.append({"date": nxt, "strategy_gross": period_ret,
                     "strategy_net": period_ret - cost, "benchmark_ew": bench,
                     "cost": cost, "turnover": turnover, "n_holdings": len(w)})
        holdings_log.append({"date": dt.date(), "holdings": ", ".join(w.index)})

    if not rows:
        return {"error": "Backtest produced no periods."}

    bt = pd.DataFrame(rows).set_index("date")
    for a, b in [("strategy_net", "equity_net"), ("strategy_gross", "equity_gross"),
                 ("benchmark_ew", "equity_bench")]:
        bt[b] = (1 + bt[a]).cumprod()

    ppy = 12 / step
    return {"table": bt, "stats": _stats(bt, ppy), "split": _split(bt, ppy),
            "holdings": pd.DataFrame(holdings_log),
            "mode": rebalance_mode,
            "avg_turnover": round(float(bt["turnover"].mean()), 3),
            "annual_turnover": round(float(bt["turnover"].mean() * ppy), 2)}


def apply_event_overlay(combo, cal, dt, event_window_days=45, w_event=0.25):
    """
    MONTHLY CORE + EVENT OVERLAY (the design that actually matches the paper).

    Why not rebalance nowcast on the earnings clock? Because NOWCAST =
    beta_i x BC_now, and BC_now is a scalar common to every stock. Between
    filings the beta is fixed, so the cross-sectional RANKING barely moves -
    re-ranking nowcast monthly adds almost nothing either way.

    What DOES matter is Carabias' actual finding: the edge is realised AROUND
    THE ANNOUNCEMENT, when the market reprices a surprise it hadn't fully
    anticipated. So the right use of the earnings calendar is not a
    rebalancing cadence but a TILT: among names the core model already likes,
    prefer the ones whose announcement is imminent.

    Implementation: a fourth score component, `event_pct`, which equals the
    name's nowcast percentile if its expected announcement falls inside
    `event_window_days`, and a neutral 0.5 otherwise. So a high-nowcast name
    approaching earnings is pushed up; a high-nowcast name far from earnings
    is left alone; a low-nowcast name approaching earnings is pushed DOWN
    (its expected surprise is negative).

    TIMING CAVEAT: expected announcement dates are ESTIMATED from each
    company's own historical filing cadence (no free forward calendar exists).
    Companies that shift their reporting date will be mistimed by days.
    """
    if combo is None or combo.empty or not cal:
        return combo

    df = combo.copy()
    days = df["ticker"].apply(lambda t: ec.days_to_next_announcement(cal, t, dt))
    df["days_to_earnings"] = days
    approaching = days.between(0, event_window_days, inclusive="both")

    df["event_pct"] = np.where(approaching, df["nowcast_pct"], 0.5)
    df["approaching_earnings"] = approaching

    # blend the overlay into the existing combined score
    df["combined_score"] = ((1 - w_event) * df["combined_score"]
                            + w_event * df["event_pct"])

    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # re-derive actions on the overlaid score, keeping the trend veto
    n_buy = int((combo["action"] == "BUY").sum()) or 10
    n_sell = int((combo["action"] == "SELL").sum())
    df["action"] = "HOLD"
    pool = df[df["clean_trend"]] if "clean_trend" in df.columns else df
    df.loc[df["ticker"].isin(pool.head(n_buy)["ticker"]), "action"] = "BUY"
    if n_sell:
        df.loc[df["ticker"].isin(df.tail(n_sell)["ticker"]), "action"] = "SELL"

    df.attrs["sleeve_agreement"] = combo.attrs.get("sleeve_agreement", {})
    return df


def _stats(bt, ppy=12):
    out = {}
    for label, col, eq in [("Strategy (net)", "strategy_net", "equity_net"),
                           ("Strategy (gross)", "strategy_gross", "equity_gross"),
                           ("Benchmark (EW universe)", "benchmark_ew", "equity_bench")]:
        r = bt[col].dropna()
        if len(r) < 2 or r.std() == 0:
            continue
        eqs, n = bt[eq], len(r)
        out[label] = {"Total": f"{eqs.iloc[-1]-1:.1%}",
                      "Annualised": f"{eqs.iloc[-1]**(ppy/n)-1:.1%}",
                      "Vol": f"{r.std()*np.sqrt(ppy):.1%}",
                      "Sharpe": f"{r.mean()/r.std()*np.sqrt(ppy):.2f}",
                      "Max DD": f"{(eqs/eqs.cummax()-1).min():.1%}", "N": n}
    return pd.DataFrame(out).T


def _split(bt, ppy=12):
    n = len(bt)
    if n < 12:
        return pd.DataFrame({"note": ["Too few periods to split."]})
    mid = n // 2
    out = {}
    for label, seg in [("First half", bt.iloc[:mid]),
                       ("Second half (out-of-sample)", bt.iloc[mid:])]:
        r, b = seg["strategy_net"].dropna(), seg["benchmark_ew"].dropna()
        if len(r) < 2 or r.std() == 0:
            continue
        eq, beq = (1 + r).cumprod(), (1 + b).cumprod()
        out[label] = {"Strategy ann.": f"{eq.iloc[-1]**(ppy/len(r))-1:.1%}",
                      "Sharpe": f"{r.mean()/r.std()*np.sqrt(ppy):.2f}",
                      "Benchmark ann.": f"{beq.iloc[-1]**(ppy/len(b))-1:.1%}",
                      "Excess": f"{eq.iloc[-1]**(ppy/len(r)) - beq.iloc[-1]**(ppy/len(b)):.1%}",
                      "N": len(r)}
    return pd.DataFrame(out).T
