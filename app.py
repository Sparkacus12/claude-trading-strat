"""
Streamlit front-end for the free-data strategy (v4).

NEW: rebalance mode switch (periodic vs event-driven overlay), Kalman beta
estimator, and the paper's lagged-SUE short leg.
"""
import numpy as np
import pandas as pd
import streamlit as st

from free_data_adapter import FreeDataAdapter
import engine as e
import enhancements as x
import strategy as S
import earnings_calendar as ec

st.set_page_config(page_title="NOWCAST free-data strategy", layout="wide")
st.title("NOWCAST earnings-revision strategy (free data)")
st.caption("Macro-nowcast × earnings-beta (Carabias 2018) + momentum + trend. "
           "EDGAR + FRED + yfinance. Research only — not investment advice.")

# ---------------- sidebar ----------------
st.sidebar.header("Data")
universe_cap = st.sidebar.slider("Universe size", 20, 400, 150, 10)
hist_years = st.sidebar.slider("History (years)", 4, 12, 8, 1)

st.sidebar.header("Signal weights")
w_nc = st.sidebar.slider("NOWCAST", 0.0, 1.0, 0.4, 0.05)
w_mo = st.sidebar.slider("Momentum", 0.0, 1.0, 0.4, 0.05)
w_tr = st.sidebar.slider("Trend", 0.0, 1.0, 0.2, 0.05)

st.sidebar.header("Paper improvements")
use_kalman = st.sidebar.checkbox("Kalman beta (paper's estimator)", value=True,
    help="Carabias models beta as a random-walk latent state estimated by "
         "Kalman filter. Theoretically correct; in testing it performed about "
         "the same as the simpler rolling regression at this sample size.")
kalman_sn = st.sidebar.slider("Kalman signal/noise", 0.01, 0.20, 0.05, 0.01,
    help="Higher = beta adapts faster.")
w_sue = st.sidebar.slider("Lagged-SUE short leg", 0.0, 0.4, 0.0, 0.05,
    help="Paper Table 8: CAR = +0.56*NOWCAST - 0.17*SUE_lag1. Investors "
         "over-weight last quarter's earnings, so SHORT that component. "
         "Paper's implied ratio ≈ 0.30 of the nowcast weight. Untestable on "
         "synthetic data — evaluate this one on real data.")

st.sidebar.header("Rebalancing")
mode = st.sidebar.radio("Core cadence", ["monthly", "quarterly"], index=0,
    help="Momentum and trend update continuously, so a monthly core keeps "
         "them fresh. Quarterly halves turnover.")
use_event = st.sidebar.checkbox("Event-driven earnings overlay", value=False,
    help="Tilts toward high-nowcast names whose announcement is imminent — "
         "where the paper's edge is actually realised. Announcement dates are "
         "ESTIMATED from each firm's filing cadence.")
w_event = st.sidebar.slider("Event tilt weight", 0.0, 0.4, 0.10, 0.05,
                            disabled=not use_event)
event_win = st.sidebar.slider("Event window (days)", 15, 90, 45, 15,
                              disabled=not use_event)

st.sidebar.header("Improvements")
sector_neutral = st.sidebar.checkbox("Sector-neutral ranks", value=True)
beta_shrink = st.sidebar.slider("Beta shrinkage", 0.0, 0.8, 0.4, 0.1)
use_vol_w = st.sidebar.checkbox("Inverse-vol weights", value=True)

st.sidebar.header("Portfolio")
n_buy = st.sidebar.slider("Number of BUYs", 5, 30, 10, 5)
n_sell = st.sidebar.slider("Number of SELLs", 0, 30, 10, 5)

run = st.sidebar.button("Run / refresh")
if run:
    st.session_state["has_run"] = True


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_data(universe_cap: int, hist_years: int):
    a = FreeDataAdapter()
    end = pd.Timestamp.today()
    tickers = a.get_universe()[:universe_cap]
    prices = a.get_prices(tickers, end - pd.DateOffset(years=hist_years + 1), end)
    macro = a.get_macro_panel(end - pd.DateOffset(years=hist_years), end)
    earnings = a.get_earnings(tickers, end - pd.DateOffset(years=hist_years), end,
                              prices=prices)
    sectors = x.get_sectors()
    disp = x.get_dispersion_series(
        start=str((end - pd.DateOffset(years=hist_years)).date()))
    return tickers, prices, earnings, macro, sectors, disp, dict(a.diagnostics)


if not run:
    st.info("Set options in the sidebar and click **Run / refresh**.")
    st.stop()

with st.spinner("Loading data…"):
    tickers, prices, earnings, macro, sectors, disp, diags = load_data(universe_cap, hist_years)

with st.expander("Data diagnostics", expanded=False):
    for k in ["universe", "prices", "macro", "earnings"]:
        v = diags.get(k, "not run")
        st.write(f"{'✅' if v.startswith('OK') else '⚠️'} **{k}** — {v}")
    st.write(f"• sectors: {len(sectors)} | dispersion rows: {len(disp)}")

if macro.empty or prices.empty:
    st.error("Core data missing — check diagnostics.")
    st.stop()

# ---------------- signals ----------------
bc = e.build_business_cycle_factor(macro)
bc_q = e.factor_quarterly(bc)
as_of = prices.index.max()
cal = ec.build_earnings_calendar(earnings)

nowcast = pd.DataFrame()
if not earnings.empty:
    sue = e.compute_sue(earnings)
    betas = (x.estimate_earnings_betas_kalman(sue, bc_q, signal_noise=kalman_sn)
             if use_kalman else e.estimate_earnings_betas(sue, bc_q))
    if not betas.empty:
        nowcast = e.compute_nowcast(betas, bc_q, as_of)
        if not nowcast.empty and beta_shrink > 0:
            nowcast = x.shrink_betas(nowcast, sectors, shrink=beta_shrink)
        if not nowcast.empty and w_sue > 0:
            nowcast = x.attach_lagged_sue(nowcast, sue, as_of)

momentum = e.momentum_score(prices, as_of)
trend = e.trend_quality(prices, as_of)
disp_regime = x.dispersion_regime_score(disp)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Business-cycle factor", f"{bc.dropna().iloc[-1]:.2f}" if len(bc.dropna()) else "n/a")
c2.metric("Dispersion regime", disp_regime.get("regime", "n/a"))
c3.metric("Risk-on score", f"{disp_regime.get('risk_on_score', float('nan')):.2f}")
c4.metric("Names with betas", 0 if nowcast.empty else nowcast["ticker"].nunique())
st.caption(f"Beta estimator: {'Kalman filter' if use_kalman else 'EW rolling regression'}. "
           f"Dispersion source: {disp_regime.get('source','n/a')}.")

# ---------------- combined list ----------------
st.header("Final buy / sell list")
combo = S.combine_signals(nowcast, momentum, trend, w_nc, w_mo, w_tr,
                          n_buy=n_buy, n_sell=n_sell, sectors=sectors,
                          sector_neutral=sector_neutral, w_sue_lag=w_sue)

if use_event and not combo.empty and cal:
    combo = S.apply_event_overlay(combo, cal, as_of,
                                  event_window_days=event_win, w_event=w_event)

if combo.empty:
    st.warning("No combined signal — check diagnostics.")
else:
    buys = combo[combo["action"] == "BUY"].copy()
    sells = combo[combo["action"] == "SELL"]
    if use_vol_w and not buys.empty:
        w = x.inverse_vol_weights(prices, buys["ticker"].tolist(), as_of=as_of)
        buys["weight"] = buys["ticker"].map(w).fillna(0)

    cols = ["ticker", "combined_score", "nowcast_pct", "momentum_pct", "trend_pct"]
    if "sector" in combo.columns:
        cols.insert(1, "sector")
    if use_event and "days_to_earnings" in combo.columns:
        cols.append("days_to_earnings")
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
        st.caption("**Sleeve agreement**: "
                   + " | ".join(f"{k}: {v}" for k, v in agree.items() if v is not None)
                   + " — approaching 0.8+ means the sleeves are redundant.")
    st.download_button("Download CSV", combo.to_csv(index=False).encode(),
                       "signals.csv", "text/csv")

# ---------------- backtest ----------------
st.header("Backtest")
st.caption("Walk-forward, point-in-time (earnings filtered by real SEC filing "
           "date). **Universe is current index membership — results are "
           "survivorship-inflated.**")
n_hold = st.number_input("Holdings", 5, 30, 10, 5)
cost_bps = st.number_input("Cost per side (bps)", 0, 50, 10, 5)

if st.button("Run backtest"):
    with st.spinner("Running walk-forward backtest… this takes a few minutes."):
        res = S.run_backtest(prices, earnings, macro, sectors=sectors,
                             n_hold=int(n_hold), cost_bps=float(cost_bps),
                             w_nowcast=w_nc, w_momentum=w_mo, w_trend=w_tr,
                             use_vol_weights=use_vol_w,
                             sector_neutral=sector_neutral,
                             beta_shrink=beta_shrink,
                             rebalance_mode=mode,
                             use_event_overlay=use_event,
                             event_window_days=event_win, w_event=w_event,
                             use_kalman=use_kalman,
                             kalman_signal_noise=kalman_sn,
                             w_sue_lag=w_sue)
    if "error" in res:
        st.error(res["error"])
    else:
        st.caption(f"Mode: {res['mode']} | annual turnover: {res['annual_turnover']}")
        st.subheader("Performance")
        st.dataframe(res["stats"], use_container_width=True)
        st.subheader("Split-sample test — read this first")
        st.caption("If the second half is much weaker, the edge is fitted, "
                   "not predictive.")
        st.dataframe(res["split"], use_container_width=True)
        st.subheader("Equity curves")
        st.line_chart(res["table"][["equity_net", "equity_bench"]]
                      .rename(columns={"equity_net": "Strategy (net)",
                                      "equity_bench": "Benchmark (EW)"}))
        with st.expander("Holdings history"):
            st.dataframe(res["holdings"], use_container_width=True)

st.markdown("---")
st.caption("Free data. Survivorship-approximate. A backtest is a hypothesis, "
           "not evidence — compare against the benchmark column before "
           "concluding anything.")
