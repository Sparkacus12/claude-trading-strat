"""
Signal combiner + backtest (v2).

NEW IN v2:
  - combiner can SECTOR-NEUTRALISE the sleeve ranks
  - backtest uses INVERSE-VOL weights instead of equal weight
  - backtest reports an OUT-OF-SAMPLE SPLIT (first half vs second half)
  - optional regime scaling of gross exposure

HONESTY NOTES - read before trusting any number here:
  - SURVIVORSHIP BIAS: universe is CURRENT S&P 500 membership. Names that
    dropped out of the index (usually after doing badly) are invisible. This
    inflates results, potentially a lot.
  - The earnings side IS point-in-time (real SEC filing dates). Good.
  - Betas come from ~30 noisy quarterly observations. Shrinkage helps but
    doesn't eliminate the problem.
  - Costs are a flat bps haircut on turnover. No borrow, no market impact.
  - THE SPLIT-SAMPLE NUMBER IS THE ONE THAT MATTERS. If the second half is
    much weaker than the first, the strategy is fitted to the past, not
    predictive. Read that before the headline Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as e
import enhancements as x


# ----------------------------------------------------------------------
# COMBINER
# ----------------------------------------------------------------------

def combine_signals(
    nowcast: pd.DataFrame,
    momentum: pd.DataFrame,
    trend: pd.DataFrame,
    w_nowcast: float = 0.4,
    w_momentum: float = 0.4,
    w_trend: float = 0.2,
    n_buy: int = 10,
    n_sell: int = 10,
    require_trend_for_buy: bool = True,
    sectors: pd.Series = None,
    sector_neutral: bool = False,
) -> pd.DataFrame:
    """
    Merge the sleeves into ONE ranked list with a BUY/SELL/HOLD action.

    Sleeves are converted to PERCENTILE RANKS before weighting, because raw
    scores are on wildly different scales (nowcast ~0.001, momentum ~2.5,
    trend t-stat ~3) and would otherwise let one sleeve dominate through
    units alone.

    sector_neutral=True ranks within GICS sector, removing the unintended
    sector bet that a cyclicality signal naturally creates.
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
        df[col] = df[col].fillna(0.5)     # missing sleeve = neutral, not penalised

    if "clean_trend" not in df.columns:
        df["clean_trend"] = True
    df["clean_trend"] = df["clean_trend"].fillna(False)

    tot = w_nowcast + w_momentum + w_trend
    df["combined_score"] = (w_nowcast * df["nowcast_pct"]
                            + w_momentum * df["momentum_pct"]
                            + w_trend * df["trend_pct"]) / tot

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

    # Sleeve agreement: if these are all ~0.8+, you're making one bet three
    # times and should drop a sleeve.
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

def run_backtest(
    prices: pd.DataFrame,
    earnings: pd.DataFrame,
    macro: pd.DataFrame,
    sectors: pd.Series = None,
    n_hold: int = 10,
    rebalance_months: int = 1,
    cost_bps: float = 10.0,
    w_nowcast: float = 0.4,
    w_momentum: float = 0.4,
    w_trend: float = 0.2,
    start_after_months: int = 24,
    use_vol_weights: bool = True,
    sector_neutral: bool = False,
    beta_shrink: float = 0.4,
) -> dict:
    """
    Walk-forward, point-in-time backtest of the combined signal.

    At each rebalance we rebuild every signal using ONLY data available at
    that date: prices sliced, earnings filtered by actual SEC filing date,
    macro sliced. No look-ahead.
    """
    sectors = sectors if sectors is not None else pd.Series(dtype=str)

    bc = e.build_business_cycle_factor(macro)
    bc_q = e.factor_quarterly(bc)

    monthly = prices.resample("ME").last()
    rebal = monthly.index[start_after_months::rebalance_months]
    if len(rebal) < 6:
        return {"error": "Not enough history for a backtest. Increase the "
                         "history window or reduce start_after_months."}

    daily_ret = prices.pct_change()
    rows, holdings_log = [], []
    prev_w = pd.Series(dtype=float)

    for i, dt in enumerate(rebal[:-1]):
        nxt = rebal[i + 1]
        px_slice = prices.loc[:dt]
        if px_slice.shape[0] < 260:
            continue

        e_slice = earnings[pd.to_datetime(earnings["announcement_date"]) <= dt]
        bcq_slice = bc_q.loc[:dt]

        nc = pd.DataFrame()
        if not e_slice.empty:
            sue = e.compute_sue(e_slice)
            betas = e.estimate_earnings_betas(sue, bcq_slice)
            if not betas.empty:
                nc = e.compute_nowcast(betas, bcq_slice, dt)
                if not nc.empty and beta_shrink > 0:
                    nc = x.shrink_betas(nc, sectors, shrink=beta_shrink)

        mom = e.momentum_score(px_slice, dt)
        trd = e.trend_quality(px_slice, dt)

        combo = combine_signals(nc, mom, trd, w_nowcast, w_momentum, w_trend,
                                n_buy=n_hold, n_sell=0, sectors=sectors,
                                sector_neutral=sector_neutral)
        if combo.empty:
            continue

        picks = combo[combo["action"] == "BUY"]["ticker"].tolist()
        if not picks:
            picks = combo.head(n_hold)["ticker"].tolist()
        picks = [p for p in picks if p in daily_ret.columns]
        if not picks:
            continue

        # weights
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

        # turnover cost on weight changes
        all_t = w.index.union(prev_w.index)
        turnover = float((w.reindex(all_t).fillna(0)
                          - prev_w.reindex(all_t).fillna(0)).abs().sum() / 2)
        cost = turnover * 2 * cost_bps / 10000
        prev_w = w

        bench_w = daily_ret.loc[dt:nxt].iloc[1:]
        bench_ret = float((1 + bench_w.mean(axis=1)).prod() - 1) if not bench_w.empty else 0.0

        rows.append({"date": nxt, "strategy_gross": period_ret,
                     "strategy_net": period_ret - cost, "benchmark_ew": bench_ret,
                     "cost": cost, "turnover": turnover, "n_holdings": len(w)})
        holdings_log.append({"date": dt.date(), "holdings": ", ".join(w.index)})

    if not rows:
        return {"error": "Backtest produced no periods. Try a longer history "
                         "window or a larger universe."}

    bt = pd.DataFrame(rows).set_index("date")
    for a, b in [("strategy_net", "equity_net"), ("strategy_gross", "equity_gross"),
                 ("benchmark_ew", "equity_bench")]:
        bt[b] = (1 + bt[a]).cumprod()

    return {"table": bt, "stats": _stats(bt),
            "split": _split_sample(bt),
            "holdings": pd.DataFrame(holdings_log)}


def _stats(bt: pd.DataFrame, ppy: int = 12) -> pd.DataFrame:
    out = {}
    for label, col, eq in [("Strategy (net)", "strategy_net", "equity_net"),
                           ("Strategy (gross)", "strategy_gross", "equity_gross"),
                           ("Benchmark (EW universe)", "benchmark_ew", "equity_bench")]:
        r = bt[col].dropna()
        if len(r) < 2 or r.std() == 0:
            continue
        eqs, n = bt[eq], len(r)
        out[label] = {
            "Total": f"{eqs.iloc[-1] - 1:.1%}",
            "Annualised": f"{eqs.iloc[-1] ** (ppy / n) - 1:.1%}",
            "Vol": f"{r.std() * np.sqrt(ppy):.1%}",
            "Sharpe": f"{r.mean() / r.std() * np.sqrt(ppy):.2f}",
            "Max DD": f"{(eqs / eqs.cummax() - 1).min():.1%}",
            "N": n,
        }
    return pd.DataFrame(out).T


def _split_sample(bt: pd.DataFrame, ppy: int = 12) -> pd.DataFrame:
    """
    THE TEST THAT MATTERS. Split the history in half and report each half
    separately. If the second half is much weaker, the apparent edge is
    fitted to the past rather than predictive.
    """
    n = len(bt)
    if n < 12:
        return pd.DataFrame({"note": ["Too few periods to split meaningfully."]})
    mid = n // 2
    out = {}
    for label, seg in [("First half (in-sample-ish)", bt.iloc[:mid]),
                       ("Second half (out-of-sample)", bt.iloc[mid:])]:
        r = seg["strategy_net"].dropna()
        b = seg["benchmark_ew"].dropna()
        if len(r) < 2 or r.std() == 0:
            continue
        eq = (1 + r).cumprod()
        beq = (1 + b).cumprod()
        out[label] = {
            "Strategy ann.": f"{eq.iloc[-1] ** (ppy / len(r)) - 1:.1%}",
            "Strategy Sharpe": f"{r.mean() / r.std() * np.sqrt(ppy):.2f}",
            "Benchmark ann.": f"{beq.iloc[-1] ** (ppy / len(b)) - 1:.1%}",
            "Excess": f"{(eq.iloc[-1] ** (ppy / len(r))) - (beq.iloc[-1] ** (ppy / len(b))):.1%}",
            "N": len(r),
        }
    return pd.DataFrame(out).T
