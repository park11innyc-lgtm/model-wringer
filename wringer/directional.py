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


def ticker_rank(d, col):
    """Rank within ticker, centered on 0.5 at any history length so tickers pool fairly."""
    g = d.groupby("ticker")[col]
    return (g.rank() - 0.5) / g.transform("size")


def within_auc(df, col="p_up"):
    d = df.copy()
    d["_r"] = ticker_rank(d, col)
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


MIN_BLOCKS = 2   # block bootstrap needs at least this many distinct date blocks


def block_ci(df, fn, n=800, block=10, seed=0):
    rng = np.random.default_rng(seed); dates = sorted(df.date.unique()); out = []
    if len(dates) < MIN_BLOCKS * block:   # too few blocks: every replicate is the same sample
        return (float('nan'), float('nan'))
    nb = max(1, len(dates) // block)
    bydate = {k: g for k, g in df.groupby("date")}
    for _ in range(n):
        picks = rng.integers(0, max(1, len(dates) - block + 1), size=nb)
        sel = [d for p in picks for d in dates[p:p + block]]
        out.append(fn(pd.concat([bydate[d] for d in sel])))
    out = [v for v in out if not (isinstance(v, float) and np.isnan(v))]
    if len(out) < n // 2:
        return (float('nan'), float('nan'))
    return np.percentile(out, [2.5, 97.5])


BLOCK_GRID = (5, 10, 20)   # the gate must hold at every block length, not one lucky choice


def _paired_edge_ci(rp, rrev, up, dates, block, n=800, seed=0):
    """Date-block CI of AUC(model) - AUC(better of reversal, momentum). NaN if too few dates."""
    rng = np.random.default_rng(seed); uniq = sorted(set(dates))
    if len(uniq) < MIN_BLOCKS * block:
        return (float('nan'), float('nan'))
    idx = {k: np.flatnonzero(dates == k) for k in uniq}
    nb = len(uniq) // block; out = []
    for _ in range(n):
        picks = rng.integers(0, len(uniq) - block + 1, size=nb)
        b = np.concatenate([idx[d] for p in picks for d in uniq[p:p + block]])
        a_m = auc(rp[b], up[b]); a_r = auc(rrev[b], up[b])
        if not (np.isnan(a_m) or np.isnan(a_r)):
            out.append(a_m - max(a_r, 1 - a_r))
    if len(out) < n // 2:
        return (float('nan'), float('nan'))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return (float(lo), float(hi))


def incremental_beyond_reversal(df):
    """Does the model beat the free rule outright? Paired AUC(model) - AUC(free rule), where the
    free rule is reversal or momentum on hist_ret, whichever did better (they are mirror images).
    A scale-free AUC of an OLS residual is NOT used: it ranks up tiny leftovers of a pure function
    of reversal and passed a one-line rule. Conservative: a model with real but smaller signal than
    the free rule stays ATTRIBUTED. ci spans the widest bound across BLOCK_GRID."""
    d = df.copy()
    up = (d.act_ret > 0).astype(int).values
    rp = ticker_rank(d, "p_up").values
    rrev = (ticker_rank(d, "hist_ret") * -1).values
    dates = d.date.astype(str).values
    a_m = auc(rp, up); a_r = auc(rrev, up)
    by_block = {b: _paired_edge_ci(rp, rrev, up, dates, b) for b in BLOCK_GRID}
    los, his = zip(*by_block.values())
    return dict(model_auc=float(a_m), free_rule_auc=float(max(a_r, 1 - a_r)),
                edge=float(a_m - max(a_r, 1 - a_r)),
                ci=(float(np.min(los)), float(np.max(his))), ci_by_block=by_block,
                corr_with_reversal=float(stats.spearmanr(rp, rrev).statistic))


def score(df):
    if "horizon" in df.columns and df["horizon"].nunique() > 1:
        raise ValueError(f"run CSV mixes horizons {sorted(df['horizon'].unique())}; "
                         "score each horizon separately")
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
