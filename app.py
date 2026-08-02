"""
Streamlit front-end for the free-data strategy.

v2: adds a DIAGNOSTICS panel that reports what each data source returned,
so a failure tells you WHICH source failed instead of a generic message.
"""
import numpy as np
import pandas as pd
import streamlit as st

from free_data_adapter import FreeDataAdapter
import engine as e

st.set_page_config(page_title="NOWCAST free-data strategy", layout="wide")
st.title("NOWCAST earnings-revision strategy (free data)")
st.caption("Macro-nowcast × earnings-beta signal (Carabias 2018), plus momentum, "
           "trend and a macro regime read. EDGAR + FRED + stooq/yfinance. "
           "Research only — not investment advice.")

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.header("Settings")
universe_cap = st.sidebar.slider("Universe size (names)", 10, 200, 30, 10,
                                 help="Start small; free data pulls are slow.")
hist_years = st.sidebar.slider("Earnings history (years)", 4, 12, 8, 1)
top_n = st.sidebar.slider("Show top N", 5, 30, 10, 5)
run = st.sidebar.button("Run / refresh data")
st.sidebar.markdown("---")
st.sidebar.caption("First run pulls and caches data (slow). Later loads use cache.")


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_and_compute(universe_cap: int, hist_years: int):
    a = FreeDataAdapter()
    end = pd.Timestamp.today()
    start_px = end - pd.DateOffset(years=2)
    start_hist = end - pd.DateOffset(years=hist_years)

    res = {}
    tickers = a.get_universe()[:universe_cap]
    prices = a.get_prices(tickers, start_px, end)
    macro = a.get_macro_panel(start_hist, end)
    earnings = a.get_earnings(tickers, start_hist, end, prices=prices)

    res["diagnostics"] = dict(a.diagnostics)
    res["n_tickers"] = len(tickers)
    res["prices_shape"] = tuple(prices.shape)
    res["n_earnings"] = len(earnings)

    if macro.empty:
        res["fatal"] = "FRED macro pull returned nothing — the business-cycle factor can't be built."
        return res
    if prices.empty:
        res["fatal"] = "Price pull returned nothing — momentum/trend can't be built."
        return res

    bc = e.build_business_cycle_factor(macro)
    bc_q = e.factor_quarterly(bc)
    res["bc_series"] = bc
    res["bc_latest"] = float(bc.dropna().iloc[-1]) if len(bc.dropna()) else np.nan
    res["regime"] = e.regime_score(macro)

    as_of = prices.index.max()
    if not earnings.empty:
        sue = e.compute_sue(earnings)
        betas = e.estimate_earnings_betas(sue, bc_q)
        res["n_betas"] = 0 if betas.empty else betas["ticker"].nunique()
        res["nowcast"] = e.compute_nowcast(betas, bc_q, as_of) if not betas.empty else pd.DataFrame()
    else:
        res["n_betas"] = 0
        res["nowcast"] = pd.DataFrame()

    res["momentum"] = e.momentum_score(prices, as_of)
    res["trend"] = e.trend_quality(prices, as_of)
    return res


if not run:
    st.info("Set the universe size in the sidebar and click **Run / refresh data**. "
            "Start small (~30 names) to confirm it works.")
    st.stop()

with st.spinner("Pulling free data and computing (cached after first run)…"):
    res = load_and_compute(universe_cap, hist_years)

# ---- diagnostics (always visible) ------------------------------------
with st.expander("Data diagnostics — open this if something looks wrong", expanded=bool(res.get("fatal"))):
    d = res.get("diagnostics", {})
    for k in ["universe", "prices", "macro", "earnings"]:
        v = d.get(k, "not run")
        icon = "✅" if v.startswith("OK") else "⚠️"
        st.write(f"{icon} **{k}** — {v}")
    st.write(f"• tickers requested: {res.get('n_tickers')} | "
             f"price frame: {res.get('prices_shape')} | "
             f"earnings rows: {res.get('n_earnings')} | "
             f"names with betas: {res.get('n_betas', 'n/a')}")

if res.get("fatal"):
    st.error(res["fatal"])
    st.stop()

# ---- headline --------------------------------------------------------
reg = res.get("regime", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Business-cycle factor", f"{res.get('bc_latest', float('nan')):.2f}")
c2.metric("Regime", reg.get("regime", "n/a"))
c3.metric("Risk-on score", f"{reg.get('risk_on_score', float('nan')):.2f}")
c4.metric("Names / earnings rows", f"{res['prices_shape'][1]} / {res['n_earnings']}")

st.markdown("### Business-cycle factor")
if "bc_series" in res and len(res["bc_series"].dropna()):
    st.line_chart(res["bc_series"].dropna())

# ---- NOWCAST ---------------------------------------------------------
st.markdown("### NOWCAST signal (macro-nowcast × earnings-beta)")
nc = res.get("nowcast", pd.DataFrame())
if nc.empty:
    st.warning("No NOWCAST signals yet — not enough earnings history for these names. "
               "Raise 'Earnings history (years)' or the universe size.")
else:
    bc_now = nc["bc_now"].iloc[0]
    st.caption(
        f"Current cycle nowcast = {bc_now:+.2f}. "
        + ("Positive: high earnings-beta (cyclical) names rank top."
           if bc_now > 0 else
           "Negative: low/defensive-beta names rank top — the model favours "
           "low cyclicality into a softening cycle. This is expected behaviour, not a bug.")
    )
    cL, cR = st.columns(2)
    cL.markdown("**Top (long candidates)**")
    cL.dataframe(nc.head(top_n)[["ticker", "beta_hat", "nowcast", "decile"]],
                 use_container_width=True)
    cR.markdown("**Bottom (avoid / short candidates)**")
    cR.dataframe(nc.tail(top_n)[["ticker", "beta_hat", "nowcast", "decile"]].iloc[::-1],
                 use_container_width=True)

# ---- momentum --------------------------------------------------------
st.markdown("### 12-1 Momentum")
mom = res.get("momentum", pd.DataFrame())
if mom.empty:
    st.info("Momentum needs ~13 months of prices; not enough history yet.")
else:
    st.dataframe(mom.head(top_n)[["ticker", "mom_score", "mom_decile"]],
                 use_container_width=True)

# ---- trend -----------------------------------------------------------
st.markdown("### Clean uptrends (trend-quality filter)")
tr = res.get("trend", pd.DataFrame())
if not tr.empty:
    clean = tr[tr["clean_trend"]].sort_values("trend_score", ascending=False)
    st.caption(f"{len(clean)} of {len(tr)} names in statistically clean uptrends.")
    st.dataframe(clean.head(top_n)[["ticker", "trend_t", "trend_score"]],
                 use_container_width=True)

st.markdown("---")
st.caption("Free-data version: EDGAR earnings (no analyst consensus), FRED macro, "
           "stooq/yfinance prices. Survivorship-approximate. Model output, not advice.")