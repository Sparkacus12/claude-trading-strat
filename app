"""
Streamlit front-end for the free-data strategy.

Deploy on Streamlit Community Cloud: point it at this repo, set the main file
to app.py. The heavy data pull is cached (st.cache_data) so the universe is
fetched occasionally, not on every page load - this keeps it inside free-tier
limits. Open the resulting URL on your phone.

NOTE: free EDGAR/FRED/stooq pulls are rate-limited and slow across the full
S&P 500. The sidebar lets you cap the universe size; start small (e.g. 40
names) to confirm it works, then raise it.
"""
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from free_data_adapter import FreeDataAdapter
import engine as e

st.set_page_config(page_title="NOWCAST free-data strategy", layout="wide")
st.title("NOWCAST earnings-revision strategy (free data)")
st.caption("Macro-nowcast × earnings-beta signal (Carabias 2018), plus momentum, "
           "trend and a macro regime read. EDGAR + FRED + stooq. Research only — "
           "not investment advice.")

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Settings")
universe_cap = st.sidebar.slider("Universe size (names)", 20, 500, 60, 20,
                                 help="Start small; free data pulls are slow.")
hist_years = st.sidebar.slider("Earnings history (years)", 4, 12, 8, 1)
top_n = st.sidebar.slider("Show top N signals", 5, 30, 10, 5)
run = st.sidebar.button("Run / refresh data")

st.sidebar.markdown("---")
st.sidebar.caption("First run pulls data and caches it (can take a few minutes "
                   "for large universes). Later loads use the cache.")


# ----------------------------------------------------------------------
# Cached data layer (the expensive part)
# ----------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_all(universe_cap: int, hist_years: int):
    a = FreeDataAdapter()
    end = pd.Timestamp.today()
    start_px = end - pd.DateOffset(years=2)
    start_e = end - pd.DateOffset(years=hist_years)
    start_m = end - pd.DateOffset(years=hist_years)

    tickers = a.get_universe()[:universe_cap]
    prices = a.get_prices(tickers, start_px, end)
    earnings = a.get_earnings(tickers, start_e, end)
    macro = a.get_macro_panel(start_m, end)
    return tickers, prices, earnings, macro


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def compute_signals(universe_cap: int, hist_years: int):
    tickers, prices, earnings, macro = load_all(universe_cap, hist_years)
    res = {"prices_shape": prices.shape, "n_earnings": len(earnings),
           "macro_cols": list(macro.columns)}

    if macro.empty or prices.empty:
        res["error"] = "Data pull returned empty. Check internet / EDGAR_UA email."
        return res

    bc = e.build_business_cycle_factor(macro)
    bc_q = e.factor_quarterly(bc)
    res["bc_latest"] = float(bc.dropna().iloc[-1]) if len(bc.dropna()) else np.nan
    res["regime"] = e.regime_score(macro)

    as_of = prices.index.max()

    # NOWCAST
    if not earnings.empty:
        sue = e.compute_sue(earnings)
        betas = e.estimate_earnings_betas(sue, bc_q)
        nc = e.compute_nowcast(betas, bc_q, as_of) if not betas.empty else pd.DataFrame()
        res["nowcast"] = nc
    else:
        res["nowcast"] = pd.DataFrame()

    # Momentum + trend
    res["momentum"] = e.momentum_score(prices, as_of)
    res["trend"] = e.trend_quality(prices, as_of)
    res["bc_series"] = bc
    return res


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if not run:
    st.info("Set the universe size in the sidebar and click **Run / refresh data**. "
            "Start with a small universe (~40–60) to confirm everything works.")
    st.stop()

with st.spinner("Pulling free data and computing signals (cached after first run)…"):
    res = compute_signals(universe_cap, hist_years)

if res.get("error"):
    st.error(res["error"])
    st.stop()

# Regime banner
reg = res.get("regime", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Business-cycle factor", f"{res.get('bc_latest', float('nan')):.2f}")
c2.metric("Regime", reg.get("regime", "n/a"))
c3.metric("Risk-on score", f"{reg.get('risk_on_score', float('nan')):.2f}")
c4.metric("Universe / earnings", f"{res['prices_shape'][1]} / {res['n_earnings']}")

st.markdown("### Business-cycle factor")
if "bc_series" in res and len(res["bc_series"].dropna()):
    st.line_chart(res["bc_series"].dropna())

# NOWCAST
st.markdown("### NOWCAST signal (macro-nowcast × earnings-beta)")
nc = res.get("nowcast", pd.DataFrame())
if nc.empty:
    st.warning("No NOWCAST signals — earnings history too thin for this universe. "
               "Try more history years or a different universe slice.")
else:
    bc_now = nc["bc_now"].iloc[0]
    st.caption(f"Current cycle nowcast bc_now = {bc_now:+.2f}. "
               f"{'Positive: high earnings-beta names rank top.' if bc_now > 0 else 'Negative: defensive (low/negative-beta) names rank top — the model favours low cyclicality in a soft cycle.'}")
    cL, cR = st.columns(2)
    cL.markdown("**Top (long candidates)**")
    cL.dataframe(nc.head(top_n)[["ticker", "beta_hat", "nowcast", "decile"]],
                 use_container_width=True)
    cR.markdown("**Bottom (avoid / short candidates)**")
    cR.dataframe(nc.tail(top_n)[["ticker", "beta_hat", "nowcast", "decile"]]
                 .iloc[::-1], use_container_width=True)

# Momentum
st.markdown("### 12-1 Momentum")
mom = res.get("momentum", pd.DataFrame())
if not mom.empty:
    st.dataframe(mom.head(top_n)[["ticker", "mom_score", "mom_decile"]],
                 use_container_width=True)

# Trend
st.markdown("### Clean uptrends (trend-quality filter)")
tr = res.get("trend", pd.DataFrame())
if not tr.empty:
    clean = tr[tr["clean_trend"]].sort_values("trend_score", ascending=False)
    st.caption(f"{len(clean)} names in statistically clean uptrends.")
    st.dataframe(clean.head(top_n)[["ticker", "trend_t", "trend_score"]],
                 use_container_width=True)

st.markdown("---")
st.caption("Free-data version: EDGAR earnings (~years of history, no analyst "
           "consensus), FRED macro, stooq prices. Survivorship-approximate "
           "(current S&P 500 membership). Signals are model output, not advice.")
