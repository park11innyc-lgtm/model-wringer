"""Offline known-answer suite: synthetic GBM data, no network.
Run: .venv/bin/python tests/test_known_answer.py
The wringer MUST certify an oracle, fail noise, and attribute a reversal clone."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from wringer.adapters import OracleAdapter, NoiseAdapter, ReversalCloneAdapter
from wringer.runner import run
from wringer import directional


def synth(ticker_seed, n=1500):
    rng = np.random.default_rng(ticker_seed)
    lr = rng.normal(0.0003, 0.015, n)
    close = 100 * np.exp(np.cumsum(lr))
    ts = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame(dict(timestamps=ts, open=close, high=close * 1.01,
                             low=close * 0.99, close=close, volume=1e6))


DATA = {f"SYN{k}": synth(100 + k) for k in range(3)}
KW = dict(tickers=list(DATA), data=DATA, ntest=300, step=2, horizon=5,
          n_samples=30, verbose=False)


def battery(adapter, seed=0):
    df = run(adapter, seed=seed, **KW)
    return directional.score(df)


def main():
    o = battery(OracleAdapter(edge=1.0, sigma=0.015))
    assert o["within_auc"] > o["null_mu"] + 2 * o["null_sd"], f"oracle not above null: {o['within_auc']}"
    assert o["incremental"]["ci"][0] > 0.5, f"oracle incremental CI low: {o['incremental']}"
    print(f"oracle    PASS-certified  auc={o['within_auc']:.3f}  inc_lo={o['incremental']['ci'][0]:.3f}")

    n = battery(NoiseAdapter(sigma=0.03), seed=1)
    band = (n["null_mu"] - 3 * n["null_sd"], n["null_mu"] + 3 * n["null_sd"])
    assert band[0] < n["within_auc"] < band[1], f"noise outside 3sd band: {n['within_auc']} vs {band}"
    assert n["incremental"]["ci"][0] < 0.5 < n["incremental"]["ci"][1] or \
           abs(n["incremental"]["resid_auc"] - 0.5) < 0.06, f"noise incremental off: {n['incremental']}"
    print(f"noise     correctly-failed auc={n['within_auc']:.3f}  null={n['null_mu']:.3f}±{n['null_sd']:.3f}")

    c = battery(ReversalCloneAdapter(k=0.5, sigma=0.01), seed=2)
    assert c["incremental"]["corr_with_reversal"] > 0.7, f"clone corr low: {c['incremental']}"
    assert c["incremental"]["ci"][0] < 0.5 < c["incremental"]["ci"][1] or \
           abs(c["incremental"]["resid_auc"] - 0.5) < 0.06, f"clone incremental not ~0.5: {c['incremental']}"
    not_pass = not (c["within_auc"] > c["null_mu"] + 2 * c["null_sd"]
                    and c["incremental"]["ci"][0] > 0.5)
    assert not_pass, "reversal clone must never PASS"
    print(f"clone     attributed       corr={c['incremental']['corr_with_reversal']:+.2f}  resid={c['incremental']['resid_auc']:.3f}")
    print("ALL KNOWN-ANSWER TESTS PASSED")


if __name__ == "__main__":
    main()
