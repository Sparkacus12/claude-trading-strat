"""
Generate macro_cache.csv for the repo - v3, using the REAL FRED API.

Why this version exists: the previous "fredgraph.csv" endpoint is a
browser-export tool, not a proper API - and FRED blocks that endpoint from
well-known cloud/CI IP ranges (including GitHub Actions runners) as an
anti-scraping measure. No amount of User-Agent tweaking fixes an IP block.

The FIX: use FRED's official Observations API (api.stlouisfed.org), which
requires a free API key. Key-authenticated traffic is treated as legitimate
programmatic access and isn't subject to the same IP-range blocking.

Get a free key: fred.stlouisfed.org -> create account -> My Account ->
API Keys (instant, free).

The key is read from the FRED_API_KEY environment variable - NOT hardcoded
here. In GitHub Actions, set it as a repo secret (Settings > Secrets and
variables > Actions > New repository secret, name FRED_API_KEY) and the
workflow passes it in as an env var.

Run via the GitHub Actions workflow (Run Workflow button), or locally:
    export FRED_API_KEY=your_key_here
    pip install pandas requests
    python3 refresh_macro_cache.py
"""
import os
import sys

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
API_KEY = os.environ.get("FRED_API_KEY", "").strip()


def fetch(code, session):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": code,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": START,
    }
    r = session.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    obs = data.get("observations", [])
    if not obs:
        return None
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    return df.dropna().set_index("date")["value"]


def main():
    if not API_KEY:
        print("ERROR: FRED_API_KEY environment variable is not set.")
        print("Get a free key at fred.stlouisfed.org (My Account > API Keys),")
        print("then add it as a GitHub Actions secret named FRED_API_KEY,")
        print("or `export FRED_API_KEY=...` before running locally.")
        sys.exit(1)

    session = requests.Session()
    out, failed = {}, []
    for name, code in FRED_SERIES.items():
        try:
            s = fetch(code, session)
            if s is None or len(s) == 0:
                raise ValueError("empty response")
            out[name] = s
            print(f"  {name:16s} ({code:12s}) {len(s):6d} obs")
        except Exception as ex:
            failed.append(code)
            print(f"  {name:16s} ({code:12s}) FAILED: {type(ex).__name__}: {ex}")

    if not out:
        print("\nNothing fetched. Check the API key is correct and active "
              "(new keys can take a minute to activate).")
        sys.exit(1)

    panel = (pd.DataFrame(out).sort_index().loc[START:]
             .resample("ME").last().ffill())
    panel.index.name = "date"
    panel.to_csv("macro_cache.csv")

    print(f"\nWrote macro_cache.csv: {panel.shape[0]} months x {panel.shape[1]} series")
    print(f"Range: {panel.index.min().date()} to {panel.index.max().date()}")
    if failed:
        print(f"Missing series (app will cope): {', '.join(failed)}")
    print("\nCommit macro_cache.csv (the workflow does this automatically).")


if __name__ == "__main__":
    main()
