"""Offline guard tests: nonfinite samples, short-window bootstraps, horizon mismatch,
density estimators and density verdicts.
Run: .venv/bin/python tests/test_guards.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from wringer.adapters import NoiseAdapter
from wringer.runner import run
from wringer import density, directional, report
from test_known_answer import DATA, KW, synth


class NanAdapter(NoiseAdapter):
    name = "nan"
    def sample_terminal_returns(self, hist, horizon, n_samples, rng, future_ret=None):
        return np.full(n_samples, np.nan)


def raises(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), e
        return
    raise AssertionError(f"expected ValueError containing {needle!r}")


def close(a, b, tol=1e-6):
    assert abs(a - b) < tol, (a, b)


def density_verdict_of(deltas, anch_hi=None):
    """Last line of the density verdict, on a hand-built result."""
    d = dict(summary={"model_ctr": {"crps_bps": 1.0}}, deltas=deltas, scored=100,
             anch_hi={} if anch_hi is None else anch_hi)
    return report.density_verdict(d).split("\n")[-1]


def test_density_estimators():
    close(density.crps_samples([0.0, 1.0], 0.0), 0.0)        # fair divisor: 0.5 - 2/(2*2*1)
    close(density.crps_samples([0.0, 1.0], 1.0), 0.0)
    close(density.crps_samples([0.0, 2.0, 4.0], 0.0), 2.0 / 3.0)    # 2 - 16/12
    close(density.crps_normal(0.0, 1.0, 0.0), 0.233695, 1e-5)
    close(density.crps_normal(0.0, 2.0, 0.0), 0.467390, 1e-5)
    close(density.pit_samples([1.0, 2.0, 3.0], 2.0), (1 + 0.5 + 0.5) / 4)
    close(density.pit_samples([1.0, 2.0, 3.0], 9.0), 3.5 / 4)
    x = np.arange(1.0, 100.0)
    assert density.cov80_samples(x, 50.0) == 1 and density.cov80_samples(x, 99.0) == 0
    close(density.ewma_sigma(np.array([0.01, -0.01] * 200), horizon=4), 0.02, 1e-9)


def test_density_verdicts():
    wide = (0.0, (-1e-4, 1e-4))
    lose = (1e-3, (5e-4, 2e-3))
    win = (-1e-3, (-2e-3, -5e-4))
    assert "VERDICT: FAIL" in density_verdict_of({"ctr_vs_ewma": lose})
    assert "VERDICT: density is competitive" in density_verdict_of({"ctr_vs_ewma": wide})
    # a shape win must hold at every block length; L=10 alone, as Kronos had it, is no win
    v = density_verdict_of({"ctr_vs_ewma": lose, "anch_vs_ewma": win}, {"anch_vs_ewma": -1e-5})
    assert "VERDICT: FAIL" in v and "Shape-only" in v and "anch_vs_ewma" in v, v
    v = density_verdict_of({"ctr_vs_ewma": lose, "anch_vs_ewma": win}, {"anch_vs_ewma": 2e-5})
    assert "VERDICT: FAIL" in v and "Shape-only" not in v, v
    v = density_verdict_of({"ctr_vs_ewma": lose, "anch_vs_ewma": win}, {"anch_vs_ewma": float("nan")})
    assert "Shape-only" not in v, v
    assert "NO VERDICT" in density_verdict_of({"ctr_vs_ewma": (0.0, (float("nan"), float("nan")))})


def main():
    test_density_estimators()
    test_density_verdicts()

    # 1. nonfinite samples rejected at run and at score; verdict needs valid comparisons
    raises(lambda: run(NanAdapter(), seed=0, **KW), "non-finite")
    df = run(NoiseAdapter(), seed=0, **KW)
    bad = df.copy(); bad.loc[bad.index[0], "s0"] = np.nan
    raises(lambda: density.score(bad, DATA, with_garch=False), "nonfinite")
    v = report.density_verdict(dict(summary={"ewma": {"crps_bps": 1.0}}, deltas={}, scored=5))
    assert "NO VERDICT" in v and "competitive" not in v, v

    # 2. short windows: no collapsed interval
    few = df[df.date.isin(sorted(df.date.unique())[:10])]
    lo, hi = directional.block_ci(few, lambda x: x.act_ret.mean())
    assert np.isnan(lo) and np.isnan(hi), (lo, hi)
    res = density.score(few, DATA, with_garch=False)
    assert res["deltas"], "short window produced no deltas to check"
    assert all(np.isnan(lo) for _, (lo, _) in res["deltas"].values()), res["deltas"]
    assert "NO VERDICT" in report.density_verdict(res)
    # enough dates for L=5 and 10 but not 20: a block length it can't measure is no win
    mid = df[df.date.isin(sorted(df.date.unique())[:25])]
    res = density.score(mid, DATA, with_garch=False)
    assert res["anch_hi"] and all(np.isnan(v) for v in res["anch_hi"].values()), res["anch_hi"]

    # the last origin whose realized close exists is scored: 95 bars, lookback 90, horizon 5 is one row
    one = run(NoiseAdapter(), ["X"], data={"X": synth(1, n=95)}, lookback=90, ntest=95, step=1,
              horizon=5, n_samples=30, seed=0, verbose=False)
    assert len(one) == 1, len(one)

    # 3. horizon mismatch and mixed horizons rejected
    raises(lambda: density.score(df, DATA, horizon=1, with_garch=False), "does not match")
    mixed = df.copy(); mixed.loc[mixed.index[0], "horizon"] = 1
    raises(lambda: density.score(mixed, DATA, with_garch=False), "mixes horizons")

    # 4. a full run reaches a real verdict with real numbers
    ok = density.score(df, DATA, horizon=5, with_garch=False)
    assert ok["scored"] == len(df), (ok["scored"], len(df))
    assert set(ok["deltas"]) >= {"ctr_vs_ewma", "anch_vs_ewma", "ctr_vs_hist"}, ok["deltas"].keys()
    assert all(np.isfinite(lo) and lo < hi for _, (lo, hi) in ok["deltas"].values()), ok["deltas"]
    v = report.density_verdict(ok)
    assert "VERDICT: " in v and "NO VERDICT" not in v, v
    # noise fans are recentered onto the baselines' mean with a constant width: a valid density,
    # but a tenfold-too-narrow fan on the same data must lose to the free baselines
    narrow = density.score(run(NoiseAdapter(sigma=0.003), seed=0, **KW), DATA, with_garch=False)
    assert narrow["summary"]["model_ctr"]["crps_bps"] > narrow["summary"]["ewma"]["crps_bps"], narrow["summary"]
    assert "VERDICT: FAIL" in report.density_verdict(narrow), report.density_verdict(narrow)
    print("guards PASS")


if __name__ == "__main__":
    main()
