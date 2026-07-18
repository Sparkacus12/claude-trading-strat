# NOWCAST free-data strategy

A free-data implementation of the NOWCAST earnings-revision strategy
(macro-nowcast × earnings-beta, after Carabias 2018), with momentum, a
trend-quality filter, and a macro regime read. Runs anywhere with internet —
**no Bloomberg, no API keys, no licensed data** — and deploys to Streamlit
Community Cloud so you can open it on your phone.

## What it does

- **NOWCAST** — the core alpha. Estimates each firm’s *earnings* sensitivity
  to the business cycle (SUE regressed on a macro factor) and multiplies by a
  current-quarter cycle nowcast. Ranks the cross-section. Never used analyst
  expectations, so it runs fully on free data.
- **12-1 momentum** — canonical cross-sectional momentum.
- **Trend quality** — names in statistically clean uptrends (slope t-stat + low vol).
- **Regime read** — a 0–1 risk-on score from free macro (growth, credit, curve).

## Data sources (all free, no key)

- **SEC EDGAR** companyfacts API → quarterly EPS (uses the real filing date as
  the announcement date).
- **FRED** → macro panel incl. CFNAI for the business-cycle factor.
- **stooq** → daily prices.

## Deploy to Streamlit Community Cloud (phone-viewable)

1. Push this repo to GitHub (private is fine).
1. **Edit `free_data_adapter.py`** → put your email in `EDGAR_UA`
   (SEC requires a real contact or it may block requests).
1. Go to share.streamlit.io, sign in with GitHub, **New app**, pick this repo,
   set main file = `app.py`, deploy.
1. Open the URL it gives you — on your laptop or phone. Share the URL with whoever.

## Files

|File                                                        |Role                                                                            |
|------------------------------------------------------------|--------------------------------------------------------------------------------|
|`app.py`                                                    |Streamlit front-end. Caches the heavy data pull; sidebar caps universe size.    |
|`engine.py`                                                 |All signal logic: factor, SUE, earnings betas, NOWCAST, momentum, trend, regime.|
|`free_data_adapter.py`                                      |EDGAR + FRED + stooq. Set your email in `EDGAR_UA`.                             |
|`requirements.txt` / `.gitignore` / `.streamlit/config.toml`|Deploy housekeeping.                                                            |

## Honest limitations

- **Free data is shallow and slow.** EDGAR EPS is ~10y; pulls are rate-limited.
  Start with a small universe (40–60 names) in the sidebar; raise it once it works.
- **No analyst consensus.** PEAD-style surprise would use a YoY proxy, not
  actual-vs-consensus; the “analyst revisions” signal isn’t possible on free data.
- **Survivorship-approximate.** Universe is *current* S&P 500 membership, not
  point-in-time — results are optimistic vs a clean backtest.
- **NOWCAST sign follows the cycle.** When the current cycle nowcast is negative,
  the model ranks low/defensive-beta names top — that’s correct behaviour
  (you don’t want high cyclical beta into a softening cycle), not a bug.
- **This is research tooling, not investment advice.** A signal table is not a
  recommendation, and a clean backtest is not evidence of live profitability.

## Run the data test locally first (optional)

Before deploying, you can sanity-check the data layer on your Mac:

```
pip install -r requirements.txt
python3 -c "from free_data_adapter import FreeDataAdapter; import pandas as pd; \
a=FreeDataAdapter(); print(a.get_macro_panel(pd.Timestamp('2018-01-01'), pd.Timestamp.today()).tail())"
```

If macro, then prices, then earnings each return sensible data, you’re good to deploy.