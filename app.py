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
import backtests as B

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
    names = x.get_company_names()
    disp = x.get_dispersion_series(
        start=str((end - pd.DateOffset(years=hist_years)).date()))
    return tickers, prices, earnings, macro, sectors, names, disp, dict(a.diagnostics)


if not st.session_state.get("has_run", False):
    st.info("Set options in the sidebar and click **Run / refresh**.")
    st.stop()

with st.spinner("Loading data…"):
    tickers, prices, earnings, macro, sectors, names, disp, diags = load_data(universe_cap, hist_years)

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

    # readable company name next to the ticker
    combo["name"] = combo["ticker"].map(names)
    buys["name"] = buys["ticker"].map(names)
    sells = sells.copy()
    sells["name"] = sells["ticker"].map(names)

    cols = ["ticker", "name", "combined_score", "nowcast_pct", "momentum_pct", "trend_pct"]
    if "sector" in combo.columns:
        cols.insert(2, "sector")
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

# ---------------- backtests: three separate ----------------
st.header("Backtests")
st.caption("Three different strategies, three separate tests. **They are not "
           "comparable to each other**: the long-short spreads are market-neutral "
           "(benchmark = zero), the combined book is long-only (benchmark = index).")

bt_cost = st.number_input("Cost per side (bps)", 0, 50, 10, 5, key="btcost")

tab1, tab2, tab3 = st.tabs(["1. NOWCAST (faithful long-short)",
                            "2. Momentum", "3. Combined (long-only)"])

with tab1:
    st.markdown("**The paper's actual strategy.** Long top decile / short bottom "
                "decile of NOWCAST among names with an announcement coming up; "
                "hold each position until just after that name reports. "
                "Dollar-neutral, so the benchmark is **zero**, not the index. "
                "Momentum is deliberately excluded — the paper's alpha is "
                "momentum-orthogonal (MOM loading −0.04).")
    ew = st.slider("Event window (days ahead)", 15, 90, 45, 15, key="nw_win")
    dec = st.slider("Decile fraction", 0.05, 0.30, 0.10, 0.05, key="nw_dec")
    c1_, c2_ = st.columns(2)
    min_nm = c1_.number_input("Min names with a signal", 5, 60, 20, 5, key="nw_mn")
    min_up = c2_.number_input("Min names announcing", 4, 40, 10, 2, key="nw_mu")
    if st.button("Run NOWCAST backtest"):
        with st.spinner("Running…"):
            r = B.backtest_nowcast_faithful(
                prices, earnings, macro, sectors=sectors,
                decile_frac=dec, event_window_days=ew,
                cost_bps=float(bt_cost), beta_shrink=beta_shrink,
                use_kalman=use_kalman, min_names=int(min_nm),
                min_upcoming=int(min_up))
        if r.get("diagnostics"):
            with st.expander("Why periods were skipped", expanded="error" in r):
                d = r["diagnostics"]
                st.write(f"Periods attempted: **{d.get('periods_attempted')}**, "
                         f"usable: **{d.get('ok')}**")
                st.write(f"Tickers in earnings calendar: {d.get('tickers_in_calendar')} | "
                         f"median names with an upcoming announcement: "
                         f"**{d.get('median_upcoming_names')}**")
                st.write("Skip reasons:")
                for k in ["short_price_history", "no_earnings_yet", "no_betas",
                          "too_few_nowcast", "too_few_upcoming", "leg_return_nan"]:
                    if d.get(k):
                        st.write(f"  • {k}: {d[k]}")
        if "error" in r:
            st.error(r["error"])
        else:
            st.info(r["note"])
            st.dataframe(r["stats"], use_container_width=True)
            st.subheader("Split-sample test")
            st.dataframe(r["split"], use_container_width=True)
            st.line_chart(r["table"][["equity_net"]]
                          .rename(columns={"equity_net": "Long-short (net)"}))
            with st.expander("Positions history"):
                st.dataframe(r["detail"], use_container_width=True)

with tab2:
    st.markdown("**Canonical 12-1 momentum**, kept separate so its contribution "
                "is visible on its own.")
    m_ls = st.radio("Construction", ["Long-short decile", "Long-only top decile"],
                    key="m_ls")
    m_dec = st.slider("Decile fraction", 0.05, 0.30, 0.10, 0.05, key="m_dec")
    if st.button("Run Momentum backtest"):
        with st.spinner("Running…"):
            r = B.backtest_momentum(prices, decile_frac=m_dec,
                                    cost_bps=float(bt_cost),
                                    long_only=(m_ls == "Long-only top decile"))
        if "error" in r:
            st.error(r["error"])
        else:
            st.info(r["note"])
            st.dataframe(r["stats"], use_container_width=True)
            st.subheader("Split-sample test")
            st.dataframe(r["split"], use_container_width=True)
            cols = ["equity_net"] + (["equity_bench"] if m_ls.startswith("Long-only") else [])
            st.line_chart(r["table"][cols])
            with st.expander("Positions history"):
                st.dataframe(r["detail"], use_container_width=True)

with tab3:
    st.markdown("**The blended long-only book** (nowcast + momentum + trend), "
                "using the sidebar weights. Benchmark is the equal-weight index. "
                "Note: blending momentum in mixes the factor the paper's alpha is "
                "orthogonal to, so a null here doesn't condemn the nowcast signal — "
                "test that in tab 1.")
    c_hold = st.number_input("Holdings", 5, 30, 10, 5, key="c_hold")
    if st.button("Run Combined backtest"):
        with st.spinner("Running…"):
            r = B.backtest_combined(
                prices, earnings, macro, sectors=sectors,
                n_hold=int(c_hold), cost_bps=float(bt_cost),
                w_nowcast=w_nc, w_momentum=w_mo, w_trend=w_tr,
                use_vol_weights=use_vol_w, sector_neutral=sector_neutral,
                beta_shrink=beta_shrink, rebalance_mode=mode,
                use_event_overlay=use_event, event_window_days=event_win,
                w_event=w_event, use_kalman=use_kalman,
                kalman_signal_noise=kalman_sn, w_sue_lag=w_sue)
        if "error" in r:
            st.error(r["error"])
        else:
            st.caption(f"Mode: {r['mode']} | annual turnover: {r['annual_turnover']}")
            st.dataframe(r["stats"], use_container_width=True)
            st.subheader("Split-sample test")
            st.dataframe(r["split"], use_container_width=True)
            st.line_chart(r["table"][["equity_net", "equity_bench"]]
                          .rename(columns={"equity_net": "Strategy (net)",
                                          "equity_bench": "Benchmark (EW)"}))
            with st.expander("Holdings history"):
                h = r["holdings"].copy()
                if not h.empty and "holdings" in h.columns:
                    def _lab(s):
                        return "; ".join(f"{t} ({names.get(t)})" if isinstance(names.get(t), str) else t
                                         for t in str(s).split(", "))
                    h["holdings"] = h["holdings"].apply(_lab)
                st.dataframe(h, use_container_width=True)

st.markdown("---")
st.caption("Free data. Survivorship-approximate. A backtest is a hypothesis, "
           "not evidence — compare against the benchmark column before "
           "concluding anything.")
