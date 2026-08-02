"""
Generate macro_cache.csv for the repo.

Run this ONCE on your Mac (where FRED isn't blocked), then commit the
resulting macro_cache.csv to the repo. The app will read it instantly
instead of making ~11 slow/blocked FRED calls on every cold start.

    pip install pandas requests
    python3 refresh_macro_cache.py

Re-run occasionally (monthly is plenty — this is monthly macro data) and
re-commit to refresh.
"""
import io

import numpy as np
import pandas as pd
import requests

FRED_SERIES = {
    "cfnai": "CFNAI", "indpro": "INDPRO", "payrolls": "PAYEMS",
    "unemployment": "UNRATE", "retail_sales": "RSAFS", "core_cpi": "CPILFESL",
    "ten_year": "DGS10", "two_year": "DGS2", "baa_spread": "BAA10Y",
    "hy_spread": "BAMLH0A0HYM2", "fin_conditions": "NFCI",
}

START = "1990-01-01"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def fetch(code):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text)).iloc[:, :2]
    df.columns = ["date", "val"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"].replace(".", np.nan), errors="coerce")
    return df.dropna().set_index("date")["val"]


def main():
    out, failed = {}, []
    for name, code in FRED_SERIES.items():
        try:
            s = fetch(code)
            out[name] = s
            print(f"  {name:16s} ({code:12s}) {len(s):6d} obs")
        except Exception as ex:
            failed.append(code)
            print(f"  {name:16s} ({code:12s}) FAILED: {type(ex).__name__}")

    if not out:
        print("\nNothing fetched — check your internet connection.")
        return

    panel = (pd.DataFrame(out).sort_index().loc[START:]
             .resample("ME").last().ffill())
    panel.index.name = "date"
    panel.to_csv("macro_cache.csv")

    print(f"\nWrote macro_cache.csv: {panel.shape[0]} months x {panel.shape[1]} series")
    print(f"Range: {panel.index.min().date()} to {panel.index.max().date()}")
    if failed:
        print(f"Missing series (app will cope): {', '.join(failed)}")
    print("\nNow commit macro_cache.csv to your repo.")


if __name__ == "__main__":
    main()

