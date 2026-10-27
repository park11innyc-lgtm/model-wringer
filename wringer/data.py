import pandas as pd

PRICE = ["open", "high", "low", "close"]

def load_daily(ticker, start="2018-01-01", end=None, interval="1d", period=None):
    """yfinance OHLCV, adjusted, tidy columns, tz-naive timestamps."""
    import yfinance as yf
    kw = dict(interval=interval, auto_adjust=True, progress=False)
    if period: kw["period"] = period
    else: kw["start"], kw["end"] = start, end
    d = yf.download(ticker, **kw)
    d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
    d = d.reset_index()
    tscol = [c for c in d.columns if c.lower() in ("date", "datetime", "index")][0]
    d = d.rename(columns={tscol: "timestamps"})
    d["timestamps"] = pd.to_datetime(d["timestamps"], utc=True).dt.tz_localize(None)
    return d.dropna(subset=PRICE).reset_index(drop=True)
