"""
Strategy engine (free-data version), consolidated into one module for clean
Streamlit deployment.

Contains, all data-source-agnostic:
  - business-cycle factor  E(BC)         (PCA over the macro panel; CFNAI proxy)
  - seasonally-adjusted earnings  SUE    (YoY EPS change / price)
  - earnings betas  beta_hat             (EW rolling regression of SUE on BC)
  - NOWCAST = beta_hat x E(BC)           (the faithful Carabias signal)
  - 12-1 momentum                        (cross-sectional)
  - trend-quality filter                 (slope t-stat + low vol)
  - regime risk-on score                 (from macro: growth/inflation/credit)

Faithful to the rebuild: NOWCAST uses an EARNINGS beta (SUE on the cycle), not
a price beta. It never used analyst expectations, so it runs fully on free data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
# Business-cycle factor
# ----------------------------------------------------------------------
ALREADY_STATIONARY = {"cfnai", "fin_conditions", "baa_spread", "hy_spread", "unemployment"}


def build_business_cycle_factor(macro_panel: pd.DataFrame) -> pd.Series:
    panel = macro_panel.copy()
    transformed = {}
    for col in panel.columns:
        s = panel[col].astype(float)
        transformed[col] = s if col in ALREADY_STATIONARY else s.diff()
    X = pd.DataFrame(transformed).ffill().dropna()
    if X.shape[0] < 24 or X.shape[1] < 2:
        if "cfnai" in panel.columns:
            f = panel["cfnai"].astype(float)
        else:
            f = panel.apply(lambda c: (c - c.mean()) / c.std()).mean(axis=1)
        return ((f - f.mean()) / f.std()).rename("BC")
    Z = StandardScaler().fit_transform(X.values)
    pc1 = PCA(n_components=1).fit_transform(Z).ravel()
    factor = pd.Series(pc1, index=X.index, name="BC")
    # sign-align to a pro-cyclical anchor
    for cand in ["cfnai", "payrolls", "indpro"]:
        if cand in panel.columns:
            anchor = panel[cand].reindex(factor.index).diff()
            if factor.corr(anchor) < 0:
                factor = -factor
            break
    return ((factor - factor.mean()) / factor.std()).rename("BC")


def factor_quarterly(bc_monthly: pd.Series) -> pd.Series:
    return bc_monthly.resample("QE").mean().rename("BC_q")


# ----------------------------------------------------------------------
# SUE and earnings betas
# ----------------------------------------------------------------------
def compute_sue(earnings: pd.DataFrame) -> pd.DataFrame:
    df = earnings.sort_values(["ticker", "fiscal_period_end"]).copy()
    df["eps_lag4"] = df.groupby("ticker")["eps_actual"].shift(4)
    df["sue"] = (df["eps_actual"] - df["eps_lag4"]) / df["price_at_period_end"]
    df["sue"] = df["sue"].replace([np.inf, -np.inf], np.nan)
    return df


def _nearest_q(bc_q, d):
    q_end = pd.Timestamp(d) + pd.offsets.QuarterEnd(0)
    if q_end in bc_q.index:
        return bc_q.loc[q_end]
    prior = bc_q.loc[:q_end]
    return float(prior.iloc[-1]) if len(prior) else np.nan


def _wslope(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    var = np.average((x - mx) ** 2, weights=w)
    return float(cov / var) if var > 0 else np.nan


def estimate_earnings_betas(sue_panel, bc_q, min_obs=8, halflife=8):
    out = []
    bc_q = bc_q.dropna()
    for ticker, g in sue_panel.groupby("ticker"):
        g = g.dropna(subset=["sue"]).sort_values("fiscal_period_end")
        if len(g) < min_obs:
            continue
        m = g.copy()
        m["bc"] = m["fiscal_period_end"].map(lambda d: _nearest_q(bc_q, d))
        m = m.dropna(subset=["bc", "sue"])
        if len(m) < min_obs:
            continue
        y, x, dates = m["sue"].values, m["bc"].values, m["fiscal_period_end"].values
        for t in range(min_obs - 1, len(m)):
            yi, xi = y[:t + 1], x[:t + 1]
            n = len(yi)
            w = 0.5 ** ((n - 1 - np.arange(n)) / halflife)
            out.append({"ticker": ticker, "fiscal_period_end": pd.Timestamp(dates[t]),
                        "beta_hat": _wslope(xi, yi, w)})
    return pd.DataFrame(out)


def compute_nowcast(betas, bc_q, as_of):
    cur_q = pd.Timestamp(as_of) + pd.offsets.QuarterEnd(0)
    bc_now = _nearest_q(bc_q.loc[:as_of], cur_q)
    rows = []
    for ticker, g in betas.groupby("ticker"):
        past = g[g["fiscal_period_end"] <= as_of].sort_values("fiscal_period_end")
        if past.empty:
            continue
        b = past["beta_hat"].iloc[-1]
        if pd.isna(b) or pd.isna(bc_now):
            continue
        rows.append({"ticker": ticker, "beta_hat": b, "bc_now": bc_now,
                     "nowcast": b * bc_now})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["nowcast_rank"] = df["nowcast"].rank(pct=True)
    df["decile"] = np.ceil(df["nowcast_rank"] * 10).clip(1, 10).astype(int)
    return df.sort_values("nowcast", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Momentum (12-1) and trend quality
# ----------------------------------------------------------------------
def momentum_score(prices, as_of, lookback_m=12, skip_m=1):
    px = prices.loc[:as_of]
    if len(px) < (lookback_m + skip_m) * 21 + 5:
        return pd.DataFrame()
    p_recent = px.shift(skip_m * 21).iloc[-1]
    p_old = px.shift(lookback_m * 21).iloc[-1]
    score = (p_recent / p_old - 1)
    df = pd.DataFrame({"ticker": score.index, "mom_score": score.values}).dropna()
    if df.empty:
        return df
    df["mom_rank"] = df["mom_score"].rank(pct=True)
    df["mom_decile"] = np.ceil(df["mom_rank"] * 10).clip(1, 10).astype(int)
    return df.sort_values("mom_score", ascending=False).reset_index(drop=True)


def trend_quality(prices, as_of, lookback=63, t_threshold=2.0):
    px = prices.loc[:as_of]
    rows = []
    for tk in px.columns:
        s = px[tk].dropna()
        if len(s) < lookback + 1:
            continue
        w = np.log(s.iloc[-lookback:].values)
        x = np.arange(len(w))
        reg = linregress(x, w)
        tstat = reg.slope / reg.stderr if reg.stderr > 0 else 0.0
        vol = np.std(np.diff(s.iloc[-lookback:].values) / s.iloc[-lookback:-1].values)
        ret = s.iloc[-1] / s.iloc[-lookback] - 1
        rows.append({"ticker": tk, "trend_t": tstat,
                     "clean_trend": bool(reg.slope > 0 and tstat >= t_threshold and ret > 0),
                     "trend_score": float(tstat / (vol + 1e-6))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Regime risk-on score (macro-based; free-data subset)
# ----------------------------------------------------------------------
def regime_score(macro_panel: pd.DataFrame) -> dict:
    """
    Simple, transparent risk-on read from free macro series. Returns latest
    score in [0,1] plus components. (The dispersion/VIXEQ input lives in the
    full repo; here we use growth + credit + curve from FRED.)
    """
    p = macro_panel.copy()
    out = {}

    def z_last(s, win=36):
        s = s.dropna()
        if len(s) < 12:
            return 0.0
        w = s.iloc[-win:]
        return float((w.iloc[-1] - w.mean()) / w.std()) if w.std() else 0.0

    growth = np.nanmean([
        z_last(p["cfnai"]) if "cfnai" in p else np.nan,
        z_last(p["indpro"].diff()) if "indpro" in p else np.nan,
    ])
    credit = -z_last(p["baa_spread"]) if "baa_spread" in p else 0.0   # tight spread = risk-on
    curve = z_last((p["ten_year"] - p["two_year"])) if {"ten_year", "two_year"} <= set(p.columns) else 0.0

    raw = np.nanmean([growth, credit, 0.5 * curve])
    score = float(1 / (1 + np.exp(-raw)))  # squash to 0..1
    out = {"risk_on_score": round(score, 3),
           "growth_z": round(float(np.nan_to_num(growth)), 2),
           "credit_z": round(float(credit), 2),
           "curve_z": round(float(curve), 2),
           "regime": "risk_on" if score >= 0.6 else "risk_off" if score <= 0.4 else "neutral"}
    return out
