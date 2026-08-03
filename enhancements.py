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
