"""
Free-data adapter: EDGAR (earnings) + FRED (macro) + stooq/yfinance (prices).

v2 fixes for cloud hosting (Streamlit Community Cloud):
  - SEPARATE User-Agents. The SEC contact string is sent ONLY to sec.gov.
    Wikipedia/stooq get a normal browser UA (they reject odd UAs).
  - HARDCODED ticker fallback if the Wikipedia scrape is blocked.
  - yfinance FALLBACK for prices if stooq refuses (stooq commonly blocks
    datacenter IPs, which is what a cloud host looks like).
  - self.diagnostics records what each source returned, so failures are
    visible instead of a generic "empty" message.

SET YOUR EMAIL in EDGAR_UA below (SEC fair-access policy requires a real
contact). It is used for sec.gov requests only.
"""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import requests

EDGAR_UA = "nowcast-research your.email@example.com"  # <-- PUT YOUR EMAIL

# Ordinary browser UA for non-SEC public sources.
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

FRED_SERIES = {
    "cfnai": "CFNAI", "indpro": "INDPRO", "payrolls": "PAYEMS",
    "unemployment": "UNRATE", "retail_sales": "RSAFS", "core_cpi": "CPILFESL",
    "ten_year": "DGS10", "two_year": "DGS2", "baa_spread": "BAA10Y",
    "hy_spread": "BAMLH0A0HYM2", "fin_conditions": "NFCI",
}

# Fallback universe if the Wikipedia scrape is blocked. Large, liquid, sector-spread.
FALLBACK_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY","AVGO","JPM",
    "TSLA","XOM","UNH","V","PG","MA","JNJ","COST","HD","MRK",
    "ABBV","CVX","PEP","ADBE","KO","WMT","CRM","BAC","TMO","MCD",
    "CSCO","ACN","ABT","LIN","NFLX","AMD","CMCSA","PFE","DIS","WFC",
    "TXN","DHR","VZ","INTC","PM","INTU","COP","AMGN","NOW","UNP",
    "IBM","CAT","GE","QCOM","SPGI","HON","NEE","RTX","BA","LOW",
    "UPS","ELV","DE","BKNG","SBUX","MDT","GS","BLK","PLD","LMT",
    "MS","ADI","AXP","MDLZ","GILD","ADP","TJX","MMC","CVS","VRTX",
]


class FreeDataAdapter:
    def __init__(self, ua: str = EDGAR_UA, polite_sleep: float = 0.12):
        self._sleep = polite_sleep
        self._edgar_ua = ua
        # session for public (non-SEC) sources
        self._pub = requests.Session()
        self._pub.headers.update({"User-Agent": BROWSER_UA})
        # session for SEC only
        self._sec = requests.Session()
        self._sec.headers.update({"User-Agent": ua})
        self._cik_map = None
        self.diagnostics: dict[str, str] = {}

    # ---- universe -------------------------------------------------------
    def get_universe(self, as_of=None) -> list[str]:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            html = self._pub.get(url, timeout=30).text
            table = pd.read_html(io.StringIO(html))[0]
            tks = table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            if len(tks) > 100:
                self.diagnostics["universe"] = f"OK: {len(tks)} from Wikipedia"
                return tks
            raise ValueError("table too small")
        except Exception as ex:
            self.diagnostics["universe"] = (
                f"Wikipedia blocked ({type(ex).__name__}); using "
                f"{len(FALLBACK_TICKERS)}-name fallback list")
            return list(FALLBACK_TICKERS)

    # ---- prices ---------------------------------------------------------
    def get_prices(self, tickers, start, end) -> pd.DataFrame:
        out, ok_stooq = {}, 0
        for t in tickers:
            s = self._stooq_one(t)
            if s is not None and len(s):
                out[t] = s
                ok_stooq += 1
            time.sleep(self._sleep)

        if ok_stooq >= max(3, len(tickers) // 4):
            self.diagnostics["prices"] = f"OK: stooq returned {ok_stooq}/{len(tickers)}"
        else:
            # stooq likely blocked (common from cloud IPs) -> try yfinance
            yf_df = self._yfinance_bulk(tickers, start, end)
            if yf_df is not None and not yf_df.empty:
                self.diagnostics["prices"] = (
                    f"stooq blocked ({ok_stooq}/{len(tickers)}); "
                    f"yfinance fallback returned {yf_df.shape[1]}")
                return yf_df.loc[str(start.date()):str(end.date())].ffill()
            self.diagnostics["prices"] = (
                f"FAIL: stooq {ok_stooq}/{len(tickers)}, yfinance unavailable")

        if not out:
            return pd.DataFrame()
        px = pd.DataFrame(out).sort_index()
        return px.loc[str(start.date()):str(end.date())].ffill()

    def _stooq_one(self, ticker):
        sym = ticker.lower() + ".us"
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        try:
            r = self._pub.get(url, timeout=30)
            if r.status_code != 200 or "Date" not in r.text[:60]:
                return None
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
            return df["Close"].rename(ticker)
        except Exception:
            return None

    @staticmethod
    def _yfinance_bulk(tickers, start, end):
        try:
            import yfinance as yf
        except ImportError:
            return None
        try:
            data = yf.download(list(tickers), start=str(start.date()),
                               end=str(end.date()), auto_adjust=True,
                               progress=False, threads=True)
            if data is None or data.empty:
                return None
            close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
            return close.dropna(axis=1, how="all")
        except Exception:
            return None

    # ---- earnings (EDGAR) ----------------------------------------------
    def _load_cik_map(self):
        if self._cik_map is not None:
            return
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._sec.get(url, timeout=30).json()
        self._cik_map = {
            v["ticker"].upper().replace(".", "-"): str(v["cik_str"]).zfill(10)
            for v in data.values()
        }

    def get_earnings(self, tickers, start, end, prices=None) -> pd.DataFrame:
        try:
            self._load_cik_map()
        except Exception as ex:
            self.diagnostics["earnings"] = f"FAIL: SEC ticker map ({type(ex).__name__})"
            return pd.DataFrame()

        if prices is None:
            prices = self.get_prices(tickers, start, end)

        rows, hit = [], 0
        for t in tickers:
            cik = self._cik_map.get(t.upper())
            if not cik:
                continue
            facts = self._companyfacts(cik)
            if facts is None:
                continue
            recs = self._extract_eps(facts)
            if recs:
                hit += 1
            for rec in recs:
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
        self.diagnostics["earnings"] = (
            f"OK: {len(df)} firm-quarters from {hit}/{len(tickers)} names"
            if len(df) else f"FAIL: no EPS records ({hit} names had facts)")
        if df.empty:
            return df
        return (df.sort_values("announcement_date")
                  .groupby(["ticker", "fiscal_period_end"]).tail(1)
                  .sort_values(["ticker", "fiscal_period_end"]).reset_index(drop=True))

    def _companyfacts(self, cik):
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            r = self._sec.get(url, timeout=30)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    @staticmethod
    def _extract_eps(facts):
        out = []
        try:
            gaap = facts["facts"]["us-gaap"]
        except (KeyError, TypeError):
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
                        continue  # drop annual / ytd aggregates
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
        if prices is None or ticker not in getattr(prices, "columns", []):
            return np.nan
        s = prices[ticker].loc[:dt].dropna()
        return float(s.iloc[-1]) if len(s) else np.nan

    # ---- macro (FRED) ---------------------------------------------------
    def get_macro_panel(self, start, end) -> pd.DataFrame:
        out, failed = {}, []
        for name, code in FRED_SERIES.items():
            s = self._fred_one(code, start, end)
            if s is not None and len(s):
                out[name] = s
            else:
                failed.append(code)
            time.sleep(self._sleep)
        if not out:
            self.diagnostics["macro"] = f"FAIL: all {len(FRED_SERIES)} FRED series empty"
            return pd.DataFrame()
        self.diagnostics["macro"] = (
            f"OK: {len(out)}/{len(FRED_SERIES)} series"
            + (f" (missing: {', '.join(failed)})" if failed else ""))
        return pd.DataFrame(out).sort_index().resample("ME").last().ffill()

    def _fred_one(self, code, start, end):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
        try:
            r = self._pub.get(url, timeout=30)
            if r.status_code != 200:
                return None
            df = pd.read_csv(io.StringIO(r.text))
            if df.shape[1] < 2:
                return None
            df = df.iloc[:, :2]
            df.columns = ["date", "val"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["val"] = pd.to_numeric(df["val"].replace(".", np.nan), errors="coerce")
            df = df.dropna().set_index("date")["val"]
            return df.loc[str(start.date()):str(end.date())]
        except Exception:
            return None