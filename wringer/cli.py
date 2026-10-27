"""CLI: python -m wringer.cli run|score ..."""
import argparse, sys
import pandas as pd


def main():
    ap = argparse.ArgumentParser(prog="wringer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="walk-forward run of an adapter")
    r.add_argument("--adapter", required=True,
                   help="noise|oracle|reversal-clone|kronos or module.path:ClassName")
    r.add_argument("--tickers", default="AAPL,MSFT,SPY,JPM,XOM")
    r.add_argument("--lookback", type=int, default=90)
    r.add_argument("--horizon", type=int, default=5)
    r.add_argument("--samples", type=int, default=50)
    r.add_argument("--ntest", type=int, default=252)
    r.add_argument("--step", type=int, default=2)
    r.add_argument("--start", default="2018-01-01")
    r.add_argument("--end", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    s = sub.add_parser("score", help="score a run CSV")
    s.add_argument("--csv", required=True)
    s.add_argument("--density", action="store_true")
    s.add_argument("--horizon", type=int, default=None,
                   help="default: read from the run CSV")
    s.add_argument("--start", default="2018-01-01")
    s.add_argument("--end", default=None)
    a = ap.parse_args()

    if a.cmd == "run":
        from .adapters import REGISTRY
        from .runner import run
        if a.adapter in REGISTRY:
            adapter = REGISTRY[a.adapter]()
        elif ":" not in a.adapter:
            sys.exit(f"unknown adapter '{a.adapter}'. Choose one of "
                     f"{sorted(REGISTRY)} or module.path:ClassName")
        else:
            mod, cls = a.adapter.rsplit(":", 1)
            import importlib
            adapter = getattr(importlib.import_module(mod), cls)()
        df = run(adapter, a.tickers.split(","), lookback=a.lookback, horizon=a.horizon,
                 n_samples=a.samples, ntest=a.ntest, step=a.step,
                 start=a.start, end=a.end, seed=a.seed)
        df.to_csv(a.out, index=False)
        print(f"wrote {a.out}  {len(df)} rows")
    else:
        from . import directional, report
        df = pd.read_csv(a.csv)
        print("=== DIRECTIONAL ===")
        print(report.directional_verdict(directional.score(df)))
        if a.density:
            from . import density
            from .data import load_daily
            data = {t: load_daily(t, start=a.start, end=a.end) for t in df.ticker.unique()}
            print("\n=== DENSITY ===")
            res = density.score(df, data, horizon=a.horizon)
            print(report.density_verdict(res))


if __name__ == "__main__":
    main()
