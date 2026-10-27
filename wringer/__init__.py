"""model-wringer: put any price model through a rigorous walk-forward evaluation.

Born from the Kronos evaluation (2026-08/09, see the README case study): every metric here
survived an adversarial multi-agent audit there. The battery:
  directional: within-ticker AUC, circular-shift permutation null, momentum/reversal
               baselines, incremental-signal-beyond-reversal, date-block bootstrap
  density:     fair (Ferro) CRPS, PIT, 80% coverage (weibull quantiles), vs
               EWMA / GARCH(1,1) / trailing-histogram baselines
"""
__version__ = "0.1.0"
