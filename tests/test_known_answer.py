"""Offline known-answer suite: synthetic data, no network.
Run: .venv/bin/python tests/test_known_answer.py
The wringer MUST certify an oracle, never PASS noise across seeds, and attribute
reversal/momentum rules in a world where those rules work. Every case asserts on
report.directional_verdict, the string a person acts on."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import norm
from wringer.adapters import ModelAdapter, OracleAdapter, NoiseAdapter, ReversalCloneAdapter
from wringer.runner import run
from wringer import directional, report


def synth(ticker_seed, n=1500):
    rng = np.random.default_rng(ticker_seed)
    lr = rng.normal(0.0003, 0.015, n)
    close = 100 * np.exp(np.cumsum(lr))
    ts = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame(dict(timestamps=ts, open=close, high=close * 1.01,
                             low=close * 0.99, close=close, volume=1e6))


def feedback_world(seed, k, nt=5, n=1500, vol=0.015, win=5):
    """k>0: returns revert the last `win` days (reversal works). k<0: they trend (momentum works)."""
    g = np.random.default_rng(seed); data = {}
    for t in range(nt):
        lr = np.zeros(n); e = g.normal(0, vol, n)
        for i in range(n):
            lr[i] = e[i] - k * lr[max(0, i - win):i].sum()
        c = 100 * np.exp(np.cumsum(lr))
        data[f"T{t}"] = pd.DataFrame(dict(timestamps=pd.bdate_range("2019-01-02", periods=n),
                                          open=c, high=c, low=c, close=c, volume=1e6))
    return data


class RuleAdapter(ModelAdapter):
    """Deterministic one-line rule: fan centered on sign * recent return, no other information."""
    name = "rule"
    def __init__(self, sign, sigma=0.03): self.sign, self.sigma = sign, sigma
    def sample_terminal_returns(self, hist, horizon, n_samples, rng, future_ret=None):
        c = hist["close"].values
        past = c[-1] / c[-1 - horizon] - 1
        return self.sign * 0.5 * past + self.sigma * norm.ppf((np.arange(n_samples) + 0.5) / n_samples)


DATA = {f"SYN{k}": synth(100 + k) for k in range(3)}
KW = dict(tickers=list(DATA), data=DATA, ntest=300, step=2, horizon=5,
          n_samples=30, verbose=False)


def battery(adapter, seed=0):
    df = run(adapter, seed=seed, **KW)
    return directional.score(df)


def world_battery(adapter, data, seed=0, n_samples=50):
    df = run(adapter, list(data), data=data, ntest=252, step=2, horizon=5,
             n_samples=n_samples, seed=seed, verbose=False)
    return directional.score(df)


def verdict(s):
    """First word of the rendered verdict."""
    line = report.directional_verdict(s).split("\n")[-1]
    assert line.startswith("VERDICT: "), line
    return line[len("VERDICT: "):].split(":")[0]


def stub(within, inc_lo, null_mu=0.5, null_sd=0.02):
    return dict(n=100, base_up=0.5, within_auc=within, null_mu=null_mu, null_sd=null_sd, null_pct=0.5,
                reversal_auc=0.5, momentum_auc=0.5, bias_bps=0.0, per_ticker={"T": within},
                incremental=dict(edge=0.0, ci=(inc_lo, inc_lo + 0.1), corr_with_reversal=0.0))


def test_verdict_branches():
    assert verdict(stub(0.60, 0.01)) == "PASS"
    assert verdict(stub(0.60, -0.01)) == "ATTRIBUTED"
    assert verdict(stub(0.60, 0.0)) == "ATTRIBUTED"
    assert verdict(stub(0.60, float("nan"))) == "ATTRIBUTED"   # unmeasurable edge never passes
    assert verdict(stub(0.54, 0.01)) == "FAIL"                 # gate 1 is mu + 2 sd = 0.54
    assert verdict(stub(0.45, 0.01)) == "FAIL"


def test_estimators():
    assert directional.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert directional.auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert directional.auc([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5
    assert np.isnan(directional.auc([1, 2], [1, 1]))
    # within-ticker AUC ignores a level offset between tickers that pooled AUC would pick up
    rows = []
    for t, off in (("A", 0.0), ("B", 100.0)):
        for k in range(4):
            rows.append(dict(ticker=t, p_up=off + k, act_ret=1.0 if k >= 2 else -1.0))
    assert directional.within_auc(pd.DataFrame(rows)) == 1.0
    # identical scores carry no information, whatever the tickers' history lengths
    flat = pd.DataFrame(dict(ticker=["A"] * 40 + ["B"] * 400, p_up=0.5, hist_ret=0.0,
                             act_ret=[1.0] * 40 + [-1.0] * 400, date=[str(k) for k in range(440)]))
    assert directional.within_auc(flat) == 0.5, directional.within_auc(flat)
    assert directional.incremental_beyond_reversal(flat)["model_auc"] == 0.5

    # 95% block CI of an iid mean has width about 2*1.96*sd/sqrt(n) (an IQR would be ~0.34x this)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(dict(date=np.arange(300), x=rng.normal(0, 1, 300)))
    lo, hi = directional.block_ci(frame, lambda f: f.x.mean(), n=400, block=5)
    assert 0.16 < hi - lo < 0.30, (lo, hi)
    assert lo < frame.x.mean() < hi

    # circular-shift null sd of iid scores matches the Mann-Whitney sd
    sds = []; theory = []
    for s in range(6):
        g = np.random.default_rng(10 + s); n = 400
        f = pd.DataFrame(dict(ticker="T", p_up=g.normal(size=n), act_ret=g.normal(size=n)))
        sds.append(directional.perm_test(f)[3])
        n1 = (f.act_ret > 0).sum(); n0 = n - n1
        theory.append(np.sqrt((n + 1) / (12 * n1 * n0)))
    assert 0.75 < np.mean(sds) / np.mean(theory) < 1.25, (np.mean(sds), np.mean(theory))


def test_edge_gate():
    """A rank-identical copy of the reversal rule has edge exactly 0, on data where reversal works."""
    g = np.random.default_rng(3); n = 600
    hist = g.normal(size=n); act = -0.5 * hist + g.normal(size=n)
    f = pd.DataFrame(dict(ticker="T", date=[str(k // 2) for k in range(n)],
                          p_up=-3 * hist, hist_ret=hist, act_ret=act))
    inc = directional.incremental_beyond_reversal(f)
    assert abs(inc["edge"]) < 1e-12, inc
    assert inc["free_rule_auc"] > 0.6, inc
    assert inc["ci"][0] <= 0, inc
    assert inc["corr_with_reversal"] > 0.999, inc
    # the mirror image: a momentum copy in a momentum frame has edge 0 too
    fm = f.assign(p_up=3 * hist, act_ret=0.5 * hist + g.normal(size=n))
    im = directional.incremental_beyond_reversal(fm)
    assert abs(im["edge"]) < 1e-12 and im["free_rule_auc"] > 0.6, im
    # the same frame with a future-informed score must clear the gate at every block length
    f2 = f.assign(p_up=act + g.normal(0, 0.3, n))
    inc2 = directional.incremental_beyond_reversal(f2)
    assert all(lo > 0 for lo, _ in inc2["ci_by_block"].values()), inc2
    los, his = zip(*inc2["ci_by_block"].values())
    assert set(inc2["ci_by_block"]) == set(directional.BLOCK_GRID), inc2
    assert len(set(los)) > 1, inc2                               # the block lengths really differ here
    assert inc2["ci"] == (min(los), max(his)), inc2              # worst case, not one block length


def main():
    test_verdict_branches()
    test_estimators()
    test_edge_gate()
    print("unit      verdict branches, estimators and edge gate OK")

    for seed, edge in ((0, 1.0), (1, 1.0), (2, 0.5)):
        o = battery(OracleAdapter(edge=edge, sigma=0.015), seed=seed)
        assert verdict(o) == "PASS", report.directional_verdict(o)
        print(f"oracle    PASS  edge={edge} auc={o['within_auc']:.3f}  dAUC_lo={o['incremental']['ci'][0]:+.3f}")
    rw = feedback_world(7, k=0.12)
    o = world_battery(OracleAdapter(edge=0.5, sigma=0.015), rw)
    assert verdict(o) == "PASS", report.directional_verdict(o)
    print(f"oracle    PASS  in reversal world auc={o['within_auc']:.3f} rev={o['reversal_auc']:.3f}")

    seen = []
    for seed in range(1, 11):
        n = battery(NoiseAdapter(sigma=0.03), seed=seed)
        v = verdict(n); seen.append(v)
        assert v != "PASS", report.directional_verdict(n)
        z = abs(n["within_auc"] - n["null_mu"]) / n["null_sd"]
        assert z < 4, f"noise {z:.1f} sd from its null: {n['within_auc']} vs {n['null_mu']}"
    assert seen.count("FAIL") >= 8, seen   # gate 1 alone is a 2.5% test
    print(f"noise     no PASS across 10 seeds: {seen.count('FAIL')} FAIL, {seen.count('ATTRIBUTED')} ATTRIBUTED")

    # reversal really works in this world, so these clear the null and must be attributed
    for seed, ns in ((11, 500), (12, 500), (13, 50)):
        rw = feedback_world(seed, k=0.12)
        r = world_battery(RuleAdapter(sign=-1), rw, n_samples=ns)
        assert r["reversal_auc"] > 0.55, r["reversal_auc"]
        assert r["incremental"]["corr_with_reversal"] > 0.9, r["incremental"]
        assert verdict(r) == "ATTRIBUTED", report.directional_verdict(r)
        print(f"rule      ATTRIBUTED samples={ns} auc={r['within_auc']:.3f} rev={r['reversal_auc']:.3f} "
              f"dAUC={r['incremental']['edge']:+.3f}")
    c = world_battery(ReversalCloneAdapter(k=0.5, sigma=0.01), feedback_world(14, k=0.12), n_samples=100)
    assert verdict(c) == "ATTRIBUTED", report.directional_verdict(c)
    print(f"clone     ATTRIBUTED auc={c['within_auc']:.3f} rev={c['reversal_auc']:.3f} dAUC={c['incremental']['edge']:+.3f}")

    # the mirror case: a momentum rule where momentum works
    m = world_battery(RuleAdapter(sign=1), feedback_world(15, k=-0.08), n_samples=500)
    assert m["momentum_auc"] > 0.55, m["momentum_auc"]
    assert verdict(m) == "ATTRIBUTED", report.directional_verdict(m)
    print(f"momentum  ATTRIBUTED auc={m['within_auc']:.3f} mom={m['momentum_auc']:.3f} dAUC={m['incremental']['edge']:+.3f}")

    # no real reversal: the rule is noise here, so it must not pass
    g = battery(RuleAdapter(sign=-1), seed=16)
    assert verdict(g) != "PASS", report.directional_verdict(g)
    print(f"rule/GBM  {verdict(g)} auc={g['within_auc']:.3f}")
    print("ALL KNOWN-ANSWER TESTS PASSED")


if __name__ == "__main__":
    main()
