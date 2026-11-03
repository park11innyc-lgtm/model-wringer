"""Terse verdict rendering."""

def directional_verdict(s):
    lines = [f"n={s['n']}  base_up={s['base_up']:.3f}",
             f"within-ticker AUC   {s['within_auc']:.3f}   null {s['null_mu']:.3f}±{s['null_sd']:.3f}   pct {s['null_pct']:.2f}",
             f"baselines           reversal {s['reversal_auc']:.3f}   momentum {s['momentum_auc']:.3f}",
             f"edge over free rule ΔAUC {s['incremental']['edge']:+.3f}  CI [{s['incremental']['ci'][0]:+.3f},{s['incremental']['ci'][1]:+.3f}]  corr(rev) {s['incremental']['corr_with_reversal']:+.2f}",
             f"calibration bias    {s['bias_bps']:+.0f} bps (pred − realized mean)",
             "per-ticker          " + "  ".join(f"{k}:{v:.3f}" for k, v in s["per_ticker"].items())]
    up = s["within_auc"]; hi_null = s["null_mu"] + 2 * s["null_sd"]
    inc_lo = s["incremental"]["ci"][0]
    if up > hi_null and inc_lo > 0:
        v = "PASS: discriminates beyond the null AND beats the free reversal/momentum rule."
    elif up > hi_null:
        v = "ATTRIBUTED: beats the null but not the free reversal/momentum rule — a one-line rule does as well."
    else:
        v = "FAIL: indistinguishable from the no-skill null."
    if abs(s["bias_bps"]) > 100:
        v += f" WARNING: probabilities miscalibrated ({s['bias_bps']:+.0f} bps bias)."
    lines.append("VERDICT: " + v)
    return "\n".join(lines)


GARCH_FAIL_MAX = 0.10     # above this share of failed fits, "competitive" isn't earned
TIE_MAX_FRACTION = 0.10   # a tie's CI may not reach past this share of the model's own CRPS


def density_verdict(d):
    s, deltas = d["summary"], d["deltas"]
    if not d.get("scored"):
        return (f"NO VERDICT: 0 rows scored ({d.get('skipped', 0)} skipped for insufficient "
                f"pre-origin history). Extend --start so each origin has >=600 prior bars.")
    def fmt(v):
        import math
        return "      —" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:7.3f}"
    lines = [f"scored {d['scored']} rows"
             + (f" (skipped {d['skipped']}: <600 pre-origin bars)" if d.get("skipped") else "")
             + (f" (garch failed on {d['garch_failures']})" if d.get("garch_failures") else ""),
             f"{'model':11s} {'CRPS(bps)':>9s} {'PIT-KS':>7s} {'cov80':>6s}"]
    for m, v in s.items():
        lines.append(f"{m:11s} {v['crps_bps']:9.1f} {fmt(v.get('pit_ks'))} "
                     + ("     —" if v.get("cov80") is None else f"{v['cov80']:6.2f}"))
    for k, (mu, (lo, hi)) in deltas.items():
        lines.append(f"ΔCRPS {k:14s} {mu*1e4:+6.1f} bps [{lo*1e4:+.1f},{hi*1e4:+.1f}]")
    for k, hi in d.get("anch_hi", {}).items():
        lines.append(f"worst hi over L=5,10,20 {k}: {hi*1e4:+.1f} bps")
    import math
    ctr_valid = [k for k, (mu, (lo, hi)) in deltas.items()
                 if k.startswith("ctr") and all(math.isfinite(x) for x in (mu, lo, hi))]
    if "model_ctr" not in s or not ctr_valid:
        lines.append("VERDICT: NO VERDICT: no valid model-vs-baseline comparison "
                     "(missing model scores, or too few dates for a block bootstrap).")
        return "\n".join(lines)
    ctr_lose = [k for k, (mu, (lo, _)) in deltas.items() if k.startswith("ctr") and lo > 0]
    anch_win = [k for k, hi in d.get("anch_hi", {}).items() if hi < 0]   # NaN (too few dates) is no win
    garch_failed = d.get("garch_failures", 0)
    tie_max = TIE_MAX_FRACTION * s["model_ctr"]["crps_bps"] / 1e4
    wide = [k for k in ctr_valid if deltas[k][1][1] > tie_max]
    if ctr_lose:
        v = "FAIL: centered density loses to free baselines (" + ", ".join(ctr_lose) + ")."
        if anch_win:
            v += " Shape-only (variance-anchored) survives (" + ", ".join(anch_win) + ") — marginal at best."
    elif garch_failed > GARCH_FAIL_MAX * d["scored"]:
        lines.append(f"VERDICT: NO VERDICT: GARCH failed on {garch_failed} of {d['scored']} rows, "
                     "so that baseline wasn't really tested.")
        return "\n".join(lines)
    elif wide:
        v = ("INCONCLUSIVE: the model could still be materially worse than free baselines ("
             + ", ".join(wide) + ").")
    else:
        v = "density is competitive with free baselines after centering."
    lines.append("VERDICT: " + v)
    return "\n".join(lines)
