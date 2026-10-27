"""Directional battery. All statistics validated in the Kronos audit:
pooled AUC confounds ticker level offsets -> rank within ticker first;
iid bootstrap understates overlap -> circular-shift permutation + date blocks."""
import numpy as np, pandas as pd
from scipy import stats


def auc(score, label):
    label = np.asarray(label).astype(int); score = np.asarray(score, float)
    if label.sum() in (0, len(label)): return np.nan
    r = pd.Series(score).rank().values
    n1 = label.sum(); n0 = len(label) - n1
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_auc(df, col="p_up"):
    d = df.copy()
    d["_r"] = d.groupby("ticker")[col].rank(pct=True)
    return auc(d["_r"], (d.act_ret > 0).astype(int))


def perm_test(df, col="p_up", shifts=60, stride=3):
    """Circular-shift null preserving autocorrelation. Returns obs, pct-in-null, null mu/sd."""
    obs = within_auc(df, col); null = []
    for k in range(1, shifts + 1):
        parts = []
        for _, g in df.groupby("ticker"):
            gg = g.copy(); gg[col] = np.roll(g[col].values, k * stride)
            parts.append(gg)
        null.append(within_auc(pd.concat(parts), col))
    null = np.array(null)
    return obs, float((null <= obs).mean()), float(null.mean()), float(null.std())


def block_ci(df, fn, n=800, block=10, seed=0):
    rng = np.random.default_rng(seed); dates = sorted(df.date.unique()); out = []
    nb = max(1, len(dates) // block)
    for _ in range(n):
        picks = rng.integers(0, max(1, len(dates) - block + 1), size=nb)
        sel = [d for p in picks for d in dates[p:p + block]]
        out.append(fn(pd.concat([df[df.date == d] for d in sel])))
    out = [v for v in out if not (isinstance(v, float) and np.isnan(v))]
    if len(out) < n // 2:
        return (float('nan'), float('nan'))
    return np.percentile(out, [2.5, 97.5])


def incremental_beyond_reversal(df):
    """Does the model add signal beyond -hist_ret? Residual AUC + block CI."""
    d = df.copy()
    d["up"] = (d.act_ret > 0).astype(int)
    d["rp"] = d.groupby("ticker").p_up.rank(pct=True)
    d["rrev"] = d.groupby("ticker").hist_ret.rank(pct=True) * -1
    beta = np.polyfit(d.rrev, d.rp, 1)
    d["resid"] = d.rp - np.polyval(beta, d.rrev)
    lo, hi = block_ci(d, lambda x: auc(x.resid, x.up))
    return dict(resid_auc=auc(d.resid, d.up), ci=(float(lo), float(hi)),
                corr_with_reversal=float(stats.spearmanr(d.rp, d.rrev).statistic))


def score(df):
    d = df.copy(); d["up"] = (d.act_ret > 0).astype(int)
    obs, pct, nmu, nsd = perm_test(d)
    per = {t: auc(g.p_up, g.up) for t, g in d.groupby("ticker")}
    inc = incremental_beyond_reversal(d)
    return dict(n=len(d), base_up=float(d.up.mean()),
                within_auc=obs, null_mu=nmu, null_sd=nsd, null_pct=pct,
                per_ticker=per,
                reversal_auc=within_auc(d.assign(p_up=-d.hist_ret)),
                momentum_auc=within_auc(d.assign(p_up=d.hist_ret)),
                incremental=inc,
                bias_bps=float((d.exp_ret.mean() - d.act_ret.mean()) * 1e4))
