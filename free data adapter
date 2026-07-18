"""
Free-data adapter: EDGAR (earnings) + FRED (macro) + stooq (prices).
No API keys, no licensed data. Runs anywhere with internet - including
Streamlit Community Cloud.

SET YOUR EMAIL in EDGAR_UA below: SEC's fair-access policy requires a real
contact in the User-Agent or they may block requests.
"""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import requests

EDGAR_UA = "nowcast-research your.email@example.com"  # <-- PUT YOUR EMAIL

FRED_SERIES = {
    "cfnai": "CFNAI", "indpro": "INDPRO", "payrolls": "PAYEMS",
    "unemployment": "UNRATE", "retail_sales": "RSAFS", "core_cpi": "CPILFESL",
    "ten_year": "DGS10", "two_year": "DGS2", "baa_spread": "BAA10Y",
    "hy_spread": "BAMLH0A0HYM2", "fin_conditions": "NFCI",
}


class FreeDataAdapter:
    def __init__(self, ua: str = EDGAR_UA, polite_sleep: float = 0.12):
        self._sleep = polite_sleep
        self._sess = requests.Session()
        self._sess.headers.update({"User-Agent": ua})
        self._cik_map = None

    # ---- universe -------------------------------------------------------
    def get_universe(self, as_of=None) -> list[str]:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = self._sess.get(url, timeout=30).text
        table = pd.read_html(io.StringIO(html))[0]
        return table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()

    # ---- prices (stooq) -------------------------------------------------
    def get_prices(self, tickers, start, end) -> pd.DataFrame:
        out = {}
        for t in tickers:
            s = self._stooq_one(t, start, end)
            if s is not None and len(s):
                out[t] = s
            time.sleep(self._sleep)
        if not out:
            return pd.DataFrame()
        px = pd.DataFrame(out).sort_index()
        return px.loc[str(start.date()):str(end.date())].ffill()

    def _stooq_one(self, ticker, start, end):
        sym = ticker.lower() + ".us"
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        try:
            r = self._sess.get(url, timeout=30)
            if r.status_code != 200 or "Date" not in r.text[:50]:
                return None
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
            return df["Close"].rename(ticker)
        except Exception:
            return None

    # ---- earnings (EDGAR) ----------------------------------------------
    def _load_cik_map(self):
        if self._cik_map is not None:
            return
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._sess.get(url, timeout=30).json()
        self._cik_map = {
            v["ticker"].upper().replace(".", "-"): str(v["cik_str"]).zfill(10)
            for v in data.values()
        }

    def get_earnings(self, tickers, start, end) -> pd.DataFrame:
        self._load_cik_map()
        prices = self.get_prices(tickers, start, end)
        rows = []
        for t in tickers:
            cik = self._cik_map.get(t.upper())
            if not cik:
                continue
            facts = self._companyfacts(cik)
            if facts is None:
                continue
            for rec in self._extract_eps(facts):
                if not (start <= rec["end"] <= end):
                    continue
                rows.append({
                    "ticker": t,
                    "fiscal_period_end": rec["end"],
                    "announcement_date": rec["filed"],
                    "eps_actual": rec["val"],
                    "eps_consensus": np.nan,
                    "price_at_period_end": self._price_at(prices, t, rec["end"]),
                })
            time.sleep(self._sleep)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return (df.sort_values("announcement_date")
                  .groupby(["ticker", "fiscal_period_end"]).tail(1)
                  .sort_values(["ticker", "fiscal_period_end"]).reset_index(drop=True))

    def _companyfacts(self, cik):
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            r = self._sess.get(url, timeout=30)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    @staticmethod
    def _extract_eps(facts):
        out = []
        try:
            gaap = facts["facts"]["us-gaap"]
        except KeyError:
            return out
        for tag in ["EarningsPerShareDiluted", "EarningsPerShareBasic"]:
            if tag not in gaap:
                continue
            for unit, recs in gaap[tag]["units"].items():
                for rec in recs:
                    if rec.get("form") not in ("10-Q", "10-K"):
                        continue
                    if not all(k in rec for k in ("end", "filed", "val")):
                        continue
                    try:
                        end_d = pd.Timestamp(rec["end"])
                        start_d = pd.Timestamp(rec["start"]) if "start" in rec else None
                    except Exception:
                        continue
                    if start_d is not None and (end_d - start_d).days > 100:
                        continue  # drop annual/ytd aggregates
                    out.append({"end": end_d, "filed": pd.Timestamp(rec["filed"]),
                                "val": float(rec["val"])})
            if out:
                break
        seen = {}
        for r in sorted(out, key=lambda x: x["filed"]):
            seen[r["end"]] = r
        return list(seen.values())

    @staticmethod
    def _price_at(prices, ticker, dt):
        if ticker not in prices.columns:
            return np.nan
        s = prices[ticker].loc[:dt].dropna()
        return float(s.iloc[-1]) if len(s) else np.nan

    # ---- macro (FRED) ---------------------------------------------------
    def get_macro_panel(self, start, end) -> pd.DataFrame:
        out = {}
        for name, code in FRED_SERIES.items():
            s = self._fred_one(code, start, end)
            if s is not None and len(s):
                out[name] = s
            time.sleep(self._sleep)
        if not out:
            return pd.DataFrame()
        return pd.DataFrame(out).sort_index().resample("ME").last().ffill()

    def _fred_one(self, code, start, end):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
        try:
            r = self._sess.get(url, timeout=30)
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "val"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["val"] = pd.to_numeric(df["val"].replace(".", np.nan), errors="coerce")
            df = df.dropna().set_index("date")["val"]
            return df.loc[str(start.date()):str(end.date())]
        except Exception:
            return None
