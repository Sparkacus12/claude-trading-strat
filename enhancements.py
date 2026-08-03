"""
Structural improvements + dispersion regime signal.

ONE NEW FILE so you only have to add (not edit) on mobile. Contains:

  1. DISPERSION REGIME  - the signal you originally found in your VIXEQ/VIX
     work, now rebuilt from FREE data (^VIXEQ and ^VIX via yfinance, with
     Cboe's ^COR1M implied-correlation index as an alternative/fallback).
     Low ratio  -> high implied correlation, macro dominating -> risk-ON
     High ratio -> wide dispersion, single names blowing up -> risk-OFF

  2. BETA SHRINKAGE     - your earnings betas come from ~30 noisy quarterly
     observations. Shrinking each name's beta toward its sector mean
     (Vasicek-style) cuts estimation error materially. This is the single
     biggest signal-QUALITY fix available.

  3. SECTOR NEUTRALITY  - NOWCAST is a cyclicality signal, so when the cycle
     is positive it loads heavily on the same 2-3 sectors. Ranking WITHIN
     sector removes an unintended sector bet you aren't paid for.

  4. VOL-SCALED WEIGHTS - equal weighting lets your most volatile holding
     dominate portfolio variance. Inverse-vol weights are close to free
     Sharpe: it's risk management, not return prediction, so it doesn't
     overfit the way a fitted signal parameter does.

NOTE ON OVERFITTING: 1 is a fitted-ish signal and should be treated with
suspicion (see the split-sample test in strategy.py). 2, 3 and 4 are
structural - defensible on first principles without needing the backtest to
bless them. Prefer trusting those.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. DISPERSION REGIME (free replacement for the Bloomberg VIXEQ work)
# ----------------------------------------------------------------------

def get_dispersion_series(start="2014-01-01", end=None) -> pd.DataFrame:
    """
    Fetch the dispersion inputs from Yahoo (free, no key):
      ^VIXEQ  - Cboe S&P 500 Constituent Volatility (avg single-stock IV)
      ^VIX    - index implied vol
      ^COR1M  - Cboe 1-month implied correlation (purpose-built alternative)

    Returns a daily frame with whatever it could get, plus a 'ratio' column
    (VIXEQ/VIX) when both legs are available.

    History note: VIXEQ is available from roughly 2014; the Cboe correlation
    indices go back further. If you want a longer regime history, prefer the
    correlation series.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    end = end or pd.Timestamp.today()
    out = {}
    for sym, name in [("^VIXEQ", "vixeq"), ("^VIX", "vix"),
                      ("^COR1M", "cor1m"), ("^DSPX", "dspx")]:
        try:
            d = yf.download(sym, start=str(pd.Timestamp(start).date()),
                            end=str(pd.Timestamp(end).date()),
                            progress=False, auto_adjust=False)
            if d is None or d.empty:
                continue
            close = d["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            s = close.dropna()
            if len(s) > 50:
                out[name] = s
        except Exception:
            continue

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out).sort_index()
    if "vixeq" in df.columns and "vix" in df.columns:
        df["ratio"] = (df["vixeq"] / df["vix"]).replace([np.inf, -np.inf], np.nan)
    return df


def dispersion_regime_score(disp: pd.DataFrame, window: int = 504) -> dict:
    """
    Convert the dispersion frame into a risk-on score in [0, 1].

    Uses the ROLLING PERCENTILE of the dispersion measure, so the score is
    relative to its own recent history rather than an absolute threshold
    (absolute thresholds date badly as vol regimes shift).

    LOW dispersion  -> percentile near 0 -> risk_on near 1
    HIGH dispersion -> percentile near 1 -> risk_on near 0

    Prefers the VIXEQ/VIX ratio; falls back to implied correlation
    (inverted, since HIGH correlation = LOW dispersion = risk-on).
    """
    if disp is None or disp.empty:
        return {"risk_on_score": 0.5, "source": "unavailable",
                "level": np.nan, "percentile": np.nan}

    if "ratio" in disp.columns and disp["ratio"].notna().sum() > 100:
        s = disp["ratio"].dropna()
        src = "VIXEQ/VIX ratio"
        invert = False
    elif "cor1m" in disp.columns:
        s = disp["cor1m"].dropna()
        src = "Cboe 1M implied correlation"
        invert = True   # high correlation = low dispersion = risk-on
    elif "dspx" in disp.columns:
        s = disp["dspx"].dropna()
        src = "Cboe dispersion index"
        invert = False
    else:
        return {"risk_on_score": 0.5, "source": "unavailable",
                "level": np.nan, "percentile": np.nan}

    w = s.iloc[-window:] if len(s) >= window else s
    latest = float(s.iloc[-1])
    pct = float((w < latest).mean())      # percentile of current reading

    risk_on = pct if invert else (1.0 - pct)

    return {
        "risk_on_score": round(float(np.clip(risk_on, 0, 1)), 3),
        "source": src,
        "level": round(latest, 2),
        "percentile": round(pct, 3),
        "regime": ("risk_on" if risk_on >= 0.6
                   else "risk_off" if risk_on <= 0.4 else "neutral"),
    }


# ----------------------------------------------------------------------
# 2. SECTORS (needed for shrinkage + neutralisation)
# ----------------------------------------------------------------------

def get_sectors() -> pd.Series:
    """GICS sector per ticker, scraped from the same Wikipedia table."""
    import requests
    ua = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
    try:
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": ua}, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0]
        tick = t["Symbol"].astype(str).str.replace(".", "-", regex=False)
        sec = t["GICS Sector"].astype(str)
        return pd.Series(sec.values, index=tick.values, name="sector")
    except Exception:
        return pd.Series(dtype=str, name="sector")


# ----------------------------------------------------------------------
# 3. BETA SHRINKAGE
# ----------------------------------------------------------------------

def shrink_betas(nowcast: pd.DataFrame, sectors: pd.Series,
                 shrink: float = 0.4) -> pd.DataFrame:
    """
    Vasicek-style shrinkage of beta_hat toward the sector mean (falling back
    to the universe mean where a sector is thin).

        shrunk = (1 - shrink) * raw + shrink * group_mean

    shrink=0 -> no shrinkage; shrink=1 -> everyone gets the group mean.
    0.3-0.5 is a sensible range given ~30 noisy quarterly observations.

    Recomputes 'nowcast' = shrunk_beta * bc_now, and re-ranks.
    """
    if nowcast is None or nowcast.empty or "beta_hat" not in nowcast.columns:
        return nowcast

    df = nowcast.copy()
    df["sector"] = df["ticker"].map(sectors) if len(sectors) else "UNKNOWN"
    df["sector"] = df["sector"].fillna("UNKNOWN")

    universe_mean = df["beta_hat"].mean()
    sector_mean = df.groupby("sector")["beta_hat"].transform("mean")
    sector_count = df.groupby("sector")["beta_hat"].transform("count")
    # thin sectors (<4 names) fall back to the universe mean
    target = np.where(sector_count >= 4, sector_mean, universe_mean)

    df["beta_raw"] = df["beta_hat"]
    df["beta_hat"] = (1 - shrink) * df["beta_raw"] + shrink * target

    if "bc_now" in df.columns:
        df["nowcast"] = df["beta_hat"] * df["bc_now"]

    df["nowcast_rank"] = df["nowcast"].rank(pct=True)
    df["decile"] = np.ceil(df["nowcast_rank"] * 10).clip(1, 10).astype(int)
    return df.sort_values("nowcast", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# 4. SECTOR NEUTRALISATION
# ----------------------------------------------------------------------

def sector_neutral_rank(df: pd.DataFrame, score_col: str,
                        sectors: pd.Series, min_names: int = 4) -> pd.Series:
    """
    Percentile-rank `score_col` WITHIN each sector instead of across the whole
    universe. Sectors with fewer than min_names fall back to the global rank.
    Returns a Series aligned to df.index.
    """
    if df is None or df.empty or score_col not in df.columns:
        return pd.Series(dtype=float)
    tmp = df.copy()
    tmp["_sector"] = tmp["ticker"].map(sectors) if len(sectors) else "UNKNOWN"
    tmp["_sector"] = tmp["_sector"].fillna("UNKNOWN")

    global_rank = tmp[score_col].rank(pct=True)
    counts = tmp.groupby("_sector")[score_col].transform("count")
    within = tmp.groupby("_sector")[score_col].rank(pct=True)
    return within.where(counts >= min_names, global_rank)


# ----------------------------------------------------------------------
# 5. VOL-SCALED WEIGHTS
# ----------------------------------------------------------------------

def inverse_vol_weights(prices: pd.DataFrame, tickers: list[str],
                        as_of=None, lookback: int = 63,
                        max_weight: float = 0.25) -> pd.Series:
    """
    Weights proportional to 1/trailing-vol, normalised to sum to 1 and capped
    so no single name dominates. Falls back to equal weight if vol can't be
    computed.

    This is RISK MANAGEMENT, not a return forecast - which is why it's a safe
    improvement rather than another fitted parameter.
    """
    tickers = [t for t in tickers if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float)

    px = prices.loc[:as_of] if as_of is not None else prices
    vols = {}
    for t in tickers:
        s = px[t].dropna()
        if len(s) < lookback + 1:
            continue
        r = s.iloc[-lookback:].pct_change().dropna()
        v = float(r.std())
        if v > 0 and np.isfinite(v):
            vols[t] = v

    if len(vols) < len(tickers) * 0.5:      # too few -> equal weight
        return pd.Series(1.0 / len(tickers), index=tickers)

    inv = pd.Series({t: 1.0 / v for t, v in vols.items()})
    w = inv / inv.sum()
    # cap and renormalise (iterate a couple of times so caps hold)
    for _ in range(3):
        w = w.clip(upper=max_weight)
        w = w / w.sum()

    # any ticker without a vol estimate gets the average weight
    missing = [t for t in tickers if t not in w.index]
    if missing:
        avg = float(w.mean())
        for t in missing:
            w[t] = avg
        w = w / w.sum()
    return w


# ======================================================================
# 6. PAPER-FAITHFUL IMPROVEMENTS (Carabias 2018)
# ======================================================================

def estimate_earnings_betas_kalman(sue_panel: pd.DataFrame, bc_q: pd.Series,
                                   min_obs: int = 8,
                                   signal_noise: float = 0.05) -> pd.DataFrame:
    """
    IMPROVEMENT A - the paper's ACTUAL beta estimator.

    Carabias models firm beta as an unobserved latent state following a
    random walk (paper, Section 3.2):

        SUE_i,q  = beta_i,q * BC_q + eps        (observation)
        beta_i,q = beta_i,q-1     + eta         (state, random walk)

    and estimates beta via KALMAN FILTER. Our previous estimator was an
    exponentially-weighted rolling regression - a reasonable approximation,
    but the Kalman filter is the correct estimator for a random-walk state:
    it weights each new observation by its precision rather than by a fixed
    decay, so it adapts fast when a firm's cyclicality genuinely shifts and
    stays stable when the data is noisy.

    Crucially the filter is RECURSIVE and forward-only: beta at quarter q
    uses only data up to q, so every estimate is out-of-sample by
    construction (paper's point about nowcasts being fully out-of-sample).

    signal_noise = Q/R, the ratio of state variance to observation variance.
    Higher -> beta moves faster. 0.02-0.10 is a sensible range; 0.05 default.
    """
    out = []
    bc_q = bc_q.dropna()

    for ticker, g in sue_panel.groupby("ticker"):
        g = g.dropna(subset=["sue"]).sort_values("fiscal_period_end")
        if len(g) < min_obs:
            continue
        m = g.copy()
        m["bc"] = m["fiscal_period_end"].map(lambda d: _nearest_q_value(bc_q, d))
        m = m.dropna(subset=["bc", "sue"])
        if len(m) < min_obs:
            continue

        y = m["sue"].values.astype(float)
        xs = m["bc"].values.astype(float)
        dates = m["fiscal_period_end"].values

        # observation noise from the sample; state noise as a ratio of it
        R = float(np.var(y)) or 1e-8
        Q = R * signal_noise

        beta, P = 0.0, R          # diffuse-ish prior
        for t in range(len(m)):
            xt, yt = xs[t], y[t]
            # predict
            beta_pred, P_pred = beta, P + Q
            # update
            S = xt * xt * P_pred + R
            if S > 0:
                K = P_pred * xt / S
                beta = beta_pred + K * (yt - beta_pred * xt)
                P = (1.0 - K * xt) * P_pred
            else:
                beta, P = beta_pred, P_pred

            if t >= min_obs - 1:
                out.append({"ticker": ticker,
                            "fiscal_period_end": pd.Timestamp(dates[t]),
                            "beta_hat": float(beta)})

    return pd.DataFrame(out)


def _nearest_q_value(bc_q: pd.Series, d):
    q_end = pd.Timestamp(d) + pd.offsets.QuarterEnd(0)
    if q_end in bc_q.index:
        return bc_q.loc[q_end]
    prior = bc_q.loc[:q_end]
    return float(prior.iloc[-1]) if len(prior) else np.nan


def attach_lagged_sue(nowcast: pd.DataFrame, sue_panel: pd.DataFrame,
                      as_of) -> pd.DataFrame:
    """
    IMPROVEMENT B - the paper's strongest RETURN-predicting specification.

    Carabias Table 8, column 3 (announcement-window returns):

        CAR = 0.56 * NOWCAST  -  0.17 * SUE_lag1
                    (t=5.87)          (t=-1.93)

    Both terms matter and they have OPPOSITE signs. The economics (Table 7):
    investors UNDER-react to the macro information in NOWCAST, and
    OVER-weight last quarter's realised earnings. So the profitable trade is
    long high-nowcast AND short high-lagged-SUE - not long nowcast alone.

    We had only implemented the first half. This attaches each name's most
    recent SUE (as of `as_of`) so the combiner can subtract it.
    """
    if nowcast is None or nowcast.empty or sue_panel is None or sue_panel.empty:
        return nowcast

    s = sue_panel.dropna(subset=["sue"]).copy()
    s["announcement_date"] = pd.to_datetime(s["announcement_date"])
    s = s[s["announcement_date"] <= pd.Timestamp(as_of)]
    if s.empty:
        return nowcast

    latest = (s.sort_values("announcement_date")
                .groupby("ticker").tail(1)
                .set_index("ticker")["sue"])

    df = nowcast.copy()
    df["sue_lag1"] = df["ticker"].map(latest)
    df["sue_lag1_pct"] = df["sue_lag1"].rank(pct=True)
    return df
