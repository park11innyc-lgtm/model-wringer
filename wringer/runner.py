"""Walk-forward runner: adapter + data -> tidy results CSV.

Alignment contract (audited in the Kronos eval to 4e-7 across 620 rows):
history = rows [i-lookback, i); forecast origin close = row i-1; realized
act_ret = close[i+horizon-1]/close[i-1] - 1. Adapters see history only.
"""
import numpy as np, pandas as pd
from .data import load_daily


def run(adapter, tickers, lookback=90, horizon=5, n_samples=50, ntest=252,
        step=2, start="2018-01-01", end=None, seed=0, verbose=True, data=None):
    """data: optional {ticker: DataFrame} to bypass yfinance (offline tests)."""
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 (density scoring divides by n*(n-1))")
    rows = []
    rng = np.random.default_rng(seed)
    for tkr in tickers:
        df = data[tkr] if data is not None else load_daily(tkr, start=start, end=end)
        n = len(df)
        for i in range(max(lookback, n - ntest), n - horizon, step):
            hist = df.iloc[i - lookback:i]
            if len(hist) < lookback: continue
            last = float(hist["close"].iloc[-1])
            act = float(df["close"].iloc[i + horizon - 1] / last - 1.0)
            fut_kw = dict(future_ret=act) if adapter.wants_future else {}
            if getattr(adapter, "wants_future_stamps", False):
                # The instrument's own future bar dates (a public calendar fact, no prices),
                # so weekend-trading and holiday-closed instruments are both conditioned correctly.
                fut_kw["future_stamps"] = df["timestamps"].iloc[i:i + horizon].reset_index(drop=True)
            s = np.asarray(adapter.sample_terminal_returns(
                hist, horizon, n_samples, rng, **fut_kw), dtype=float)
            if s.shape != (n_samples,):
                raise ValueError(f"adapter returned {s.shape}, want ({n_samples},)")
            if not np.isfinite(s).all():
                raise ValueError(f"adapter returned non-finite samples at {tkr} row i={i}")
            if i - 1 - horizon < 0: continue   # hist_ret undefined -> row unusable
            hist_ret = float(last / df["close"].iloc[i - 1 - horizon] - 1.0)
            rc = np.concatenate([[last], df["close"].iloc[i:i + horizon].values])
            rlr = np.diff(np.log(rc))
            rv = float(np.std(rlr, ddof=1)) if len(rlr) >= 2 else float("nan")
            row = dict(ticker=tkr, date=str(df["timestamps"].iloc[i + horizon - 1].date()),
                       i=i, last=last, act_ret=act, hist_ret=hist_ret,
                       real_vol=rv, horizon=horizon, model=adapter.name,
                       p_up=float((s > 0).mean()), exp_ret=float(s.mean()),
                       med_ret=float(np.median(s)),
                       spread=float(np.percentile(s, 90) - np.percentile(s, 10)))
            for k, v in enumerate(np.sort(s)): row[f"s{k}"] = float(v)
            rows.append(row)
            if verbose and len(rows) % 50 == 0: print(f"{len(rows)} preds", flush=True)
    return pd.DataFrame(rows)
