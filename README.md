# model-wringer

Put any "AI price model" through the evaluation battery that killed Kronos
(see the case study below).

## Quick start
```bash
python -m wringer.cli run --adapter noise --out /tmp/noise.csv
python -m wringer.cli score --csv /tmp/noise.csv            # must FAIL
python -m wringer.cli score --csv /tmp/noise.csv --density  # density battery too
```

## Wiring a real model
Subclass `wringer.adapters.ModelAdapter`, implement
`sample_terminal_returns(hist, horizon, n_samples, rng)` returning
`(n_samples,)` terminal simple returns vs the last close in `hist`
(use ONLY `hist` — no lookahead), then:
```bash
python -m wringer.cli run --adapter mypkg.mymod:MyAdapter --out run.csv
python -m wringer.cli score --csv run.csv --density
```

## What the verdict means
- **FAIL** — within-ticker AUC inside the circular-shift no-skill null.
- **ATTRIBUTED** — beats the null but not the free reversal rule (the signal
  is a disguised baseline; a one-line rule gets you the same thing free).
- **PASS** — beats the null AND shows incremental signal beyond reversal
  (block-bootstrap CI above 0.5). Only then is the model worth a deeper look.
- Density: fair-CRPS vs EWMA / GARCH(1,1) / trailing histogram; a model whose
  centered density loses to those is not usable as a scenario generator.

Known-answer validated: certifies an oracle, fails pure noise, and attributes a
disguised reversal clone to its baseline — `tests/test_known_answer.py` (offline,
synthetic data, seconds). The package was also adversarially reviewed against the
original audit scripts (directional estimators reproduce them
digit-for-digit; density fixes from that review are applied).

Scope notes: score-time data is verified per row against the stored origin close
(alignment failures raise, never silently mis-fit baselines); horizon is stamped
into the run CSV and read back at score time; the audited vol-MAE metric is
deliberately omitted (underpowered at 5-bar horizons).

## Case study: Kronos

This battery was extracted from an audit of [Kronos](https://github.com/shiyu-coder/Kronos), a published candlestick foundation model, for US equities and BTC (August to September 2026).

- **Direction**: 3,767 out-of-sample walk-forward forecasts across 8 configurations (small and base models, 90 and 400-bar lookbacks, 1-hour to 10-day horizons). Zero-shot Kronos never beat chance on within-ticker AUC against permutation nulls, and never beat a one-line short-term reversal rule. Probabilities were pulled toward the window mean in every run.
- **Fine-tune**: a predictor-only fine-tune on US large caps following the authors' recipe cut token loss by about 5% but left directional skill unchanged and made calibration worse. The mean reversion comes from the objective, not the data domain.
- **Density**: 415 fifty-path forecast fans scored with fair CRPS, PIT and 80% coverage against EWMA, GARCH(1,1) and a trailing histogram. Raw fans lost to every baseline and covered 47% instead of about 77%. The free trailing histogram was the best-calibrated density.

Verdict: no detectable edge as a direction signal or as a scenario generator. The scorers were adversarially reviewed before the verdict, and three estimator biases were found and fixed first.

## License

MIT. See `LICENSE`.
