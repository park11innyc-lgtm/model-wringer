"""Density battery. Estimator choices are the audited ones (see the README case study):
fair (Ferro 2008) CRPS divisor 2n(n-1); weibull quantiles for sample coverage;
PIT midrank convention; baselines fit strictly on pre-origin data; bootstrap
keeps duplicate blocks (a with-replacement resample, matching the audit)."""
import numpy as np, pandas as pd
from scipy.stats import norm

Z80 = 1.281552
MIN_PRE_ORIGIN = 600   # audited guard: >=600 pre-origin bars to fit baselines


def crps_samples(x, y):
    x = np.asarray(x, float); n = len(x)
    t1 = np.abs(x - y).mean()
    t2 = np.sum(np.abs(x[:, None] - x[None, :])) / (2 * n * (n - 1))
    return t1 - t2


def crps_normal(mu, sig, y):
    z = (y - mu) / sig
    return sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def pit_samples(x, y):
    x = np.asarray(x, float)
    return (np.sum(x < y) + 0.5 * np.sum(x == y) + 0.5) / (len(x) + 1)


def cov80_samples(x, y):
    lo = np.quantile(x, .1, method="weibull"); hi = np.quantile(x, .9, method="weibull")
    return int(lo <= y <= hi)


def ewma_sigma(returns, lam=0.94, horizon=5):
    """RiskMetrics: seed on first 250 obs, recurse over the remainder only."""
    v = float(np.var(returns[:250]))
    for r in returns[250:]:
        v = lam * v + (1 - lam) * r * r
    return float(np.sqrt(v * horizon))


def fit_garch(returns_pct):
    """GARCH(1,1), constant mean, on 100x simple returns (rescale disabled so the
    /1e4 variance and /100 mu unit conversions in score() stay valid)."""
    from arch import arch_model
    am = arch_model(returns_pct * 100, mean="Constant", vol="GARCH", p=1, q=1,
                    rescale=False)
    return am.fit(disp="off", show_warning=False)


def score(df, tickers_data, horizon=None, with_garch=True):
    """df: runner output. tickers_data: {ticker: full daily df}.
    Verifies per-row alignment against the stored origin close; raises on drift."""
    if horizon is None:
        horizon = int(df["horizon"].iloc[0]) if "horizon" in df.columns else 5
    scols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
    out = []; skipped = 0; garch_fail = 0
    for _, row in df.iterrows():
        y = row.act_ret
        ks = row[scols].values.astype(float)
        full = tickers_data[row.ticker]
        i = int(row.i)
        if i - 1 >= len(full) or abs(full["close"].iloc[i - 1] / row.last - 1) > 1e-6:
            raise ValueError(
                f"alignment failure at {row.ticker} row i={i}: score-time data does not "
                f"match the run (close[i-1]={float(full['close'].iloc[i-1]) if i-1 < len(full) else 'OOB'} "
                f"vs stored last={row.last}). Use the same --start/--end as the run; "
                f"provider history may also have been revised — re-run if so.")
        if i - 1 < MIN_PRE_ORIGIN:
            skipped += 1; continue
        r = pd.Series(full["close"].values[:i]).pct_change().dropna().values
        sig5 = ewma_sigma(r, horizon=horizon)
        ksc = ks - ks.mean()
        kae = ksc * (sig5 / ks.std(ddof=1))          # shape-only, anchored to EWMA vol
        rec = dict(ticker=row.ticker, date=row.date,
                   crps_model=crps_samples(ks, y),
                   crps_model_ctr=crps_samples(ksc, y),
                   crps_model_anch=crps_samples(kae, y),
                   crps_ewma=crps_normal(0.0, sig5, y),
                   pit_model=pit_samples(ks, y), cov_model=cov80_samples(ks, y),
                   pit_model_anch=pit_samples(kae, y), cov_model_anch=cov80_samples(kae, y),
                   pit_ewma=float(norm.cdf(y / sig5)),
                   cov_ewma=int(-Z80 * sig5 <= y <= Z80 * sig5))
        hs = pd.Series(full["close"].values[:i]).pct_change(horizon).dropna().values[-500:]
        if len(hs) >= 100:
            rec["crps_hist"] = crps_samples(hs, y)
            rec["pit_hist"] = pit_samples(hs, y)
            rec["cov_hist"] = cov80_samples(hs, y)
        if with_garch:
            try:
                res = fit_garch(r[-1000:])
                f = res.forecast(horizon=horizon, reindex=False)
                gvar = float(f.variance.values[0].sum()) / 1e4
                gmu = float(res.params.get("mu", 0.0)) / 100 * horizon
                gsig = float(np.sqrt(gvar))
                rec["crps_garch"] = crps_normal(gmu, gsig, y)
                rec["pit_garch"] = float(norm.cdf((y - gmu) / gsig))
                rec["cov_garch"] = int(gmu - Z80 * gsig <= y <= gmu + Z80 * gsig)
            except Exception:
                garch_fail += 1
        out.append(rec)
    sc = pd.DataFrame(out)

    def ks_stat(p):
        p = np.sort(np.asarray(p, float))
        n = len(p)
        if n == 0: return float("nan")
        grid = np.arange(1, n + 1) / n
        return float(np.max(np.abs(p - grid + 0.5 / n)) + 0.5 / n)

    def block_dcrps(a, b, n=800, block=10, seed=0):
        d = sc.dropna(subset=[a, b])
        if not len(d): return None
        rng = np.random.default_rng(seed)
        dates = sorted(d.date.unique())
        bydate = {k: g for k, g in d.groupby("date")}
        nb = max(1, len(dates) // block); reps = []
        for _ in range(n):
            picks = rng.integers(0, max(1, len(dates) - block + 1), size=nb)
            sel = [d2 for p_ in picks for d2 in dates[p_:p_ + block]]  # keep duplicates
            bdf = pd.concat([bydate[d2] for d2 in sel])
            reps.append((bdf[a] - bdf[b]).mean())
        return float((d[a] - d[b]).mean()), tuple(np.percentile(reps, [2.5, 97.5]))

    summary = {}
    for m in ["model", "model_ctr", "model_anch", "ewma", "garch", "hist"]:
        c = f"crps_{m}"
        if c in sc.columns and sc[c].notna().any():
            summary[m] = dict(crps_bps=float(sc[c].mean() * 1e4))
            pcol, ccol = f"pit_{m}", f"cov_{m}"
            if pcol in sc.columns:
                summary[m]["pit_ks"] = ks_stat(sc[pcol].dropna())
                summary[m]["cov80"] = float(sc[ccol].mean())
    deltas = {}
    for base in ["ewma", "garch", "hist"]:
        if f"crps_{base}" in sc.columns and sc[f"crps_{base}"].notna().any():
            for tag, col in [("raw", "crps_model"), ("ctr", "crps_model_ctr"),
                             ("anch", "crps_model_anch")]:
                d_ = block_dcrps(col, f"crps_{base}")
                if d_: deltas[f"{tag}_vs_{base}"] = d_
    return dict(per_row=sc, summary=summary, deltas=deltas,
                scored=len(sc), skipped=skipped, garch_failures=garch_fail)
