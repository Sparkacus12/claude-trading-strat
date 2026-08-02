"""
Generate macro_cache.csv for the repo.

v2 fix: FRED requests need a normal BROWSER User-Agent, or they hang/timeout
from cloud IPs (GitHub Actions runners included) - this is the same fix
already applied to free_data_adapter.py. Timeout is also short (8s) so a
genuine failure fails fast instead of the whole run taking 5+ minutes.

Run via the GitHub Actions workflow (Run Workflow button), or locally:
    pip install pandas requests
    python3 refresh_macro_cache.py
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

# Normal browser UA - FRED (and most public data sites) will hang or refuse
# requests from generic/script-like User-Agents, especially from cloud IPs.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def fetch(code, session):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    r = session.get(url, timeout=8)  # short timeout: fail fast, not fail slow
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text)).iloc[:, :2]
    df.columns = ["date", "val"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"].replace(".", np.nan), errors="coerce")
    return df.dropna().set_index("date")["val"]


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    out, failed = {}, []
    for name, code in FRED_SERIES.items():
        try:
            s = fetch(code, session)
            out[name] = s
            print(f"  {name:16s} ({code:12s}) {len(s):6d} obs")
        except Exception as ex:
            failed.append(code)
            print(f"  {name:16s} ({code:12s}) FAILED: {type(ex).__name__}")

    if not out:
        print("\nNothing fetched. If every series failed with ReadTimeout/"
              "403, FRED may be temporarily blocking this runner's IP - "
              "try again in a few minutes, or run this locally on your Mac.")
        raise SystemExit(1)  # non-zero exit -> Actions shows red X, not a silent green pass

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
