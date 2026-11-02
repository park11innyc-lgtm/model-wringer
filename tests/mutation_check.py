"""Proves the suites can fail: breaks the package one way at a time in a temp copy and
expects tests/test_known_answer.py or tests/test_guards.py to exit nonzero.
Run: .venv/bin/python tests/mutation_check.py [jobs]
Exits 1 if any mutant survives or a mutation no longer applies to the source."""
import os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITES = ("tests/test_known_answer.py", "tests/test_guards.py")

_D = "wringer/directional.py"; _R = "wringer/report.py"; _N = "wringer/density.py"
HARD_PASS = '    lines.append("VERDICT: " + v)\n    return "\\n".join(lines)\n\n\ndef density_verdict'
MUTANTS = [
    # hardcoded verdicts
    ("directional_always_PASS", _R, HARD_PASS,
     HARD_PASS.replace('"VERDICT: " + v', '"VERDICT: PASS: hardcoded."')),
    ("directional_always_ATTRIBUTED", _R, HARD_PASS,
     HARD_PASS.replace('"VERDICT: " + v', '"VERDICT: ATTRIBUTED: hardcoded."')),
    ("directional_always_FAIL", _R, HARD_PASS,
     HARD_PASS.replace('"VERDICT: " + v', '"VERDICT: FAIL: hardcoded."')),
    ("density_always_NO_VERDICT", _R, '    def fmt(v):\n',
     '    return "VERDICT: NO VERDICT: hardcoded."\n    def fmt(v):\n'),
    ("density_always_competitive", _R, '    def fmt(v):\n',
     '    return "VERDICT: density is competitive with free baselines after centering."\n    def fmt(v):\n'),
    ("density_always_FAIL", _R, '    def fmt(v):\n',
     '    return "VERDICT: FAIL: centered density loses to free baselines (ctr_vs_ewma)."\n    def fmt(v):\n'),
    # gates
    ("pass_ignores_edge", _R, "if up > hi_null and inc_lo > 0:", "if up > hi_null:"),
    ("pass_ignores_null", _R, "if up > hi_null and inc_lo > 0:", "if inc_lo > 0:"),
    ("edge_gate_at_half", _R, "and inc_lo > 0:", "and inc_lo > -0.5:"),
    ("edge_ignores_momentum", _D, "edge=float(a_m - max(a_r, 1 - a_r))", "edge=float(a_m - a_r)"),
    ("edge_ci_ignores_momentum", _D, "out.append(a_m - max(a_r, 1 - a_r))", "out.append(a_m - a_r)"),
    ("gate_at_one_block_length", _D, "ci=(float(np.min(los)), float(np.max(his)))", "ci=by_block[10]"),
    ("edge_ci_ignores_free_rule", _D, "out.append(a_m - max(a_r, 1 - a_r))", "out.append(a_m - 0.5)"),
    ("density_lo_sign_flipped", _R, 'k.startswith("ctr") and lo > 0', 'k.startswith("ctr") and lo < 0'),
    ("density_anch_gate_ignored", _R, "if hi < 0]", "if hi < 1]"),
    # estimators
    ("ci_is_iqr", _D, "np.percentile(out, [2.5, 97.5])\n\n\nBLOCK_GRID", "np.percentile(out, [25, 75])\n\n\nBLOCK_GRID"),
    ("perm_null_sd_halved", _D, "float(null.mean()), float(null.std())", "float(null.mean()), float(null.std()) / 2"),
    ("pooled_not_within_ticker", _D, 'd["_r"] = d.groupby("ticker")[col].rank(pct=True)', 'd["_r"] = d[col]'),
    ("crps_zero", _N, "    return t1 - t2\n", "    return 0.0\n"),
    ("crps_unfair_divisor", _N, "(2 * n * (n - 1))", "(2 * n * n)"),
    ("crps_normal_wrong", _N, "2 * norm.pdf(z) - 1 / np.sqrt(np.pi)", "2 * norm.pdf(z)"),
    ("pit_wrong", _N, "np.sum(x == y) + 0.5) / (len(x) + 1)", "np.sum(x == y)) / len(x)"),
    ("anch_hi_skips_unmeasured", _N, 'float("nan") if np.isnan(his).any() else max(his)', "max(his)"),
    ("runner_drops_last_origin", "wringer/runner.py", "n - horizon + 1, step", "n - horizon, step"),
    ("zero_variance_scored", _N, "if not ks.max() > ks.min():", "if False:"),
    ("garch_unconverged_kept", _N, "if res.convergence_flag != 0:", "if False:"),
    ("directional_pools_horizons", _D, 'df["horizon"].nunique() > 1', 'df["horizon"].nunique() > 99'),
    ("runner_drops_short_ticker", "wringer/runner.py", "if n < lookback + horizon:", "if False:"),
    ("deltas_empty", _N, "    return dict(per_row=sc, summary=summary, deltas=deltas,",
     "    deltas = {}\n    return dict(per_row=sc, summary=summary, deltas=deltas,"),
]


def check(mutant):
    name, path, old, new = mutant
    src = open(os.path.join(ROOT, path)).read()
    if old not in src:
        return name, "STALE (pattern gone)"
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("wringer", "tests"):
            shutil.copytree(os.path.join(ROOT, sub), os.path.join(tmp, sub),
                            ignore=shutil.ignore_patterns("__pycache__"))
        with open(os.path.join(tmp, path), "w") as f:
            f.write(src.replace(old, new))
        for suite in SUITES:
            r = subprocess.run([sys.executable, suite], cwd=tmp, capture_output=True, text=True)
            if r.returncode != 0:
                last = r.stderr.strip().split("\n")[-1][:90]
                return name, f"killed by {os.path.basename(suite)}: {last}"
    return name, "SURVIVED"


def main():
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(check, MUTANTS))
    bad = 0
    for name, outcome in results:
        print(f"{name:32s} {outcome}", flush=True)
        bad += outcome.startswith(("SURVIVED", "STALE"))
    print(f"{len(results) - bad}/{len(results)} mutants killed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
