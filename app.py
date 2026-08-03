"""
Streamlit front-end for the free-data strategy (v3).

NEW: combined BUY/SELL list, backtest with split-sample test, dispersion
regime (VIXEQ/VIX via yfinance), sector-neutral option, vol-weighted book.
"""
import numpy as np
import pandas as pd
import streamlit as st

from free_data_adapter import FreeDataAdapter
import engine as e
import enhancements as x
import strategy as S

st.set_page_config(page_title="NOWCAST free-data strategy", layout="wide")
st.title("NOWCAST earnings-revision strategy (free data)")
st.caption("Macro-nowcast × earnings-beta (Carabias 2018) + momentum + trend, "
           "combined into one list. EDGAR + FRED + yfinance. "
           "Research only — not investment advice.")

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.header("Data")
universe_cap = st.sidebar.slider("Universe size", 20, 400, 150, 10)
hist_years = st.sidebar.slider("History (years)", 4, 12, 8, 1)

st.sidebar.header("Signal weights")
w_nc = st.sidebar.slider("NOWCAST", 0.0, 1.0, 0.4, 0.1)
w_mo = st.sidebar.slider("Momentum", 0.0, 1.0, 0.4, 0.1)
w_tr = st.sidebar.slider("Trend", 0.0, 1.0, 0.2, 0.1)

st.sidebar.header("Improvements")
sector_neutral = st.sidebar.checkbox("Sector-neutral ranks", value=True)
beta_shrink = st.sidebar.slider("Beta shrinkage", 0.0, 0.8, 0.4, 0.1,
                                help="Pulls noisy betas toward the sector mean.")
use_vol_w = st.sidebar.checkbox("Inverse-vol weights", value=True)
use_regime = st.sidebar.checkbox("Regime exposure scaling", value=False,
                                 help="EXPERIMENTAL. Scales gross exposure by the "
                                      "dispersion regime. Highest overfitting risk "
                                      "in the app — validate out-of-sample first.")

st.sidebar.header("Portfolio")
n_buy = st.sidebar.slider("Number of BUYs", 5, 30, 10, 5)
n_sell = st.sidebar.slider("Number of SELLs", 0, 30, 10, 5)

run = st.sidebar.button("Run / refresh")


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_data(universe_cap: int, hist_years: int):
    a = FreeDataAdapter()
    end = pd.Timestamp.today()
    start_px = end - pd.DateOffset(years=hist_years + 1)
    start_hist = end - pd.DateOffset(years=hist_years)

    tickers = a.get_universe()[:universe_cap]
    prices = a.get_prices(tickers, start_px, end)
    macro = a.get_macro_panel(start_hist, end)
    earnings = a.get_earnings(tickers, start_hist, end, prices=prices)
    sectors = x.get_sectors()
    disp = x.get_dispersion_series(start=str((end - pd.DateOffset(years=hist_years)).date()))
    return tickers, prices, earnings, macro, sectors, disp, dict(a.diagnostics)


if not run:
    st.info("Set options in the sidebar and click **Run / refresh**.")
    st.stop()

with st.spinner("Loading data and computing signals…"):
    tickers, prices, earnings, macro, sectors, disp, diags = load_data(universe_cap, hist_years)

# ---- diagnostics ----
with st.expander("Data diagnostics", expanded=False):
    for k in ["universe", "prices", "macro", "earnings"]:
        v = diags.get(k, "not run")
        st.write(f"{'✅' if v.startswith('OK') else '⚠️'} **{k}** — {v}")
    st.write(f"• sectors: {len(sectors)} | dispersion rows: {len(disp)}")

if macro.empty or prices.empty:
    st.error("Core data missing — check diagnostics above.")
    st.stop()

# ---- signals ----
bc = e.build_business_cycle_factor(macro)
bc_q = e.factor_quarterly(bc)
as_of = prices.index.max()

nowcast = pd.DataFrame()
if not earnings.empty:
    sue = e.compute_sue(earnings)
    betas = e.estimate_earnings_betas(sue, bc_q)
    if not betas.empty:
        nowcast = e.compute_nowcast(betas, bc_q, as_of)
        if not nowcast.empty and beta_shrink > 0:
            nowcast = x.shrink_betas(nowcast, sectors, shrink=beta_shrink)

momentum = e.momentum_score(prices, as_of)
trend = e.trend_quality(prices, as_of)

disp_regime = x.dispersion_regime_score(disp)

# ---- headline ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Business-cycle factor", f"{bc.dropna().iloc[-1]:.2f}" if len(bc.dropna()) else "n/a")
c2.metric("Dispersion regime", disp_regime.get("regime", "n/a"))
c3.metric("Risk-on score", f"{disp_regime.get('risk_on_score', float('nan')):.2f}")
c4.metric("Names with betas", 0 if nowcast.empty else nowcast["ticker"].nunique())

st.caption(f"Dispersion source: {disp_regime.get('source','n/a')} "
           f"(level {disp_regime.get('level','n/a')}, "
           f"{disp_regime.get('percentile', float('nan')):.0%} of its own history). "
           "Low dispersion = high implied correlation = risk-on.")

# ======================================================================
# THE COMBINED LIST
# ======================================================================
st.header("Final buy / sell list")

combo = S.combine_signals(nowcast, momentum, trend, w_nc, w_mo, w_tr,
                          n_buy=n_buy, n_sell=n_sell,
                          sectors=sectors, sector_neutral=sector_neutral)

if combo.empty:
    st.warning("No combined signal — check diagnostics.")
else:
    buys = combo[combo["action"] == "BUY"]
    sells = combo[combo["action"] == "SELL"]

    if use_vol_w and not buys.empty:
        w = x.inverse_vol_weights(prices, buys["ticker"].tolist(), as_of=as_of)
        gross = disp_regime.get("risk_on_score", 1.0) if use_regime else 1.0
        buys = buys.copy()
        buys["weight"] = buys["ticker"].map(w).fillna(0) * gross
        if use_regime:
            st.caption(f"Regime scaling ON: gross exposure {gross:.0%} "
                       "(experimental — validate out-of-sample).")

    cols = ["ticker", "combined_score", "nowcast_pct", "momentum_pct", "trend_pct"]
    if "sector" in combo.columns:
        cols.insert(1, "sector")
    if "weight" in buys.columns:
        cols.append("weight")

    cA, cB = st.columns(2)
    cA.subheader(f"BUY ({len(buys)})")
    cA.dataframe(buys[[c for c in cols if c in buys.columns]], use_container_width=True)
    cB.subheader(f"SELL / avoid ({len(sells)})")
    cB.dataframe(sells[[c for c in cols if c in sells.columns and c != "weight"]],
                 use_container_width=True)

    agree = combo.attrs.get("sleeve_agreement", {})
    if agree:
        st.caption("**Sleeve agreement** (rank correlation): "
                   + " | ".join(f"{k}: {v}" for k, v in agree.items() if v is not None)
                   + " — if these approach 0.8+, the sleeves are redundant and "
                     "you're making one bet three times.")

    st.download_button("Download list as CSV",
                       combo.to_csv(index=False).encode(),
                       "signals.csv", "text/csv")

# ======================================================================
# BACKTEST
# ======================================================================
st.header("Backtest")
st.caption("Walk-forward and point-in-time: at each rebalance, signals are "
           "rebuilt using only data available then (earnings filtered by real "
           "SEC filing date). **Universe is current index membership, so results "
           "are survivorship-inflated.**")

n_hold = st.number_input("Holdings in backtest", 5, 30, 10, 5)
cost_bps = st.number_input("Cost per side (bps)", 0, 50, 10, 5)

if st.button("Run backtest"):
    with st.spinner("Running walk-forward backtest… this takes a while."):
        res = S.run_backtest(prices, earnings, macro, sectors=sectors,
                             n_hold=int(n_hold), cost_bps=float(cost_bps),
                             w_nowcast=w_nc, w_momentum=w_mo, w_trend=w_tr,
                             use_vol_weights=use_vol_w,
                             sector_neutral=sector_neutral,
                             beta_shrink=beta_shrink)
    if "error" in res:
        st.error(res["error"])
    else:
        st.subheader("Performance")
        st.dataframe(res["stats"], use_container_width=True)

        st.subheader("Split-sample test — read this first")
        st.caption("If the second half is much weaker than the first, the edge "
                   "is fitted to the past rather than predictive.")
        st.dataframe(res["split"], use_container_width=True)

        st.subheader("Equity curves")
        st.line_chart(res["table"][["equity_net", "equity_bench"]]
                      .rename(columns={"equity_net": "Strategy (net)",
                                      "equity_bench": "Benchmark (EW)"}))

        with st.expander("Holdings history"):
            st.dataframe(res["holdings"], use_container_width=True)

st.markdown("---")
st.caption("Free data: EDGAR earnings (no analyst consensus), FRED macro, "
           "yfinance prices + VIXEQ. Survivorship-approximate. "
           "Model output, not advice — a backtest is a hypothesis, not evidence.")
