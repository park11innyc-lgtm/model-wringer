"""Adapter protocol + reference adapters.

A model adapter turns history into a fan of terminal simple returns.
Contract: use ONLY `hist` (rows strictly before the forecast origin's first
future bar). `future_ret` is passed ONLY to adapters declaring wants_future=True
(test oracles); real model adapters never see it.
"""
import numpy as np


class ModelAdapter:
    name = "base"
    wants_future = False          # True only for known-answer test adapters

    def sample_terminal_returns(self, hist, horizon, n_samples, rng, future_ret=None):
        """Return (n_samples,) terminal simple returns vs last close. Override me."""
        raise NotImplementedError


class NoiseAdapter(ModelAdapter):
    """Pure noise. The wringer must NOT certify this."""
    name = "noise"
    def __init__(self, sigma=0.03): self.sigma = sigma
    def sample_terminal_returns(self, hist, horizon, n_samples, rng, future_ret=None):
        return rng.normal(0.0, self.sigma, n_samples)


class OracleAdapter(ModelAdapter):
    """Cheats: centers the fan on a noisy copy of the realized return.
    edge controls signal-to-noise. The wringer MUST certify this."""
    name = "oracle"
    wants_future = True
    def __init__(self, edge=1.0, sigma=0.03): self.edge, self.sigma = edge, sigma
    def sample_terminal_returns(self, hist, horizon, n_samples, rng, future_ret=None):
        mu = self.edge * future_ret + rng.normal(0.0, self.sigma * 0.5)
        return rng.normal(mu, self.sigma, n_samples)


class ReversalCloneAdapter(ModelAdapter):
    """Disguised free baseline: fan centered on minus the recent return.
    The wringer must attribute it to reversal (zero incremental signal)."""
    name = "reversal-clone"
    def __init__(self, k=0.5, sigma=0.03): self.k, self.sigma = k, sigma
    def sample_terminal_returns(self, hist, horizon, n_samples, rng, future_ret=None):
        c = hist["close"].values
        past = c[-1] / c[-1 - horizon] - 1.0 if len(c) > horizon else 0.0
        return rng.normal(-self.k * past, self.sigma, n_samples)


class KronosAdapter(ModelAdapter):
    """Kronos K-line foundation model (needs the repo + its venv importable).
    Set repo path via KRONOS_REPO env or repo= kwarg. Evaluated verdict for this
    model: no edge (see the README case study) - adapter kept as the reference integration."""
    name = "kronos"
    def __init__(self, repo=None, model="NeoQuasar/Kronos-small",
                 tokenizer="NeoQuasar/Kronos-Tokenizer-base", device="mps",
                 T=1.0, top_p=0.99):
        import os, sys
        repo = repo or os.environ.get("KRONOS_REPO")
        if not repo: raise RuntimeError("KronosAdapter needs repo= or KRONOS_REPO env")
        sys.path.insert(0, repo)
        import model.kronos as K
        K.RETURN_ALL_SAMPLES = True
        from model import Kronos, KronosTokenizer, KronosPredictor
        self._K = K
        self.T, self.top_p = T, top_p
        tok = KronosTokenizer.from_pretrained(tokenizer)
        mdl = Kronos.from_pretrained(model)
        self.p = KronosPredictor(mdl, tok, device=device, max_context=512)

    def sample_terminal_returns(self, hist, horizon, n_samples, rng, future_ret=None):
        import numpy as np, torch, pandas as pd
        K, p = self._K, self.p
        d = hist.copy()
        if "amount" not in d:
            d["amount"] = d["volume"] * d[["open", "high", "low", "close"]].mean(axis=1)
        cols = ["open", "high", "low", "close", "volume", "amount"]
        x = d[cols].values.astype(np.float32)
        mu, sd = x.mean(0), x.std(0)
        xn = np.clip((x - mu) / (sd + 1e-5), -p.clip, p.clip)[None, :]
        xs = K.calc_time_stamps(d["timestamps"]).values.astype(np.float32)[None, :]
        # future stamps: extend business days from last hist timestamp
        fut = pd.bdate_range(d["timestamps"].iloc[-1], periods=horizon + 1)[1:]
        ys = K.calc_time_stamps(pd.Series(fut)).values.astype(np.float32)[None, :]
        t = lambda a: torch.from_numpy(np.array(a, dtype=np.float32)).to(p.device)
        out = K.auto_regressive_inference(p.tokenizer, p.model, t(xn), t(xs), t(ys),
                p.max_context, horizon, p.clip, self.T, 0, self.top_p, n_samples, False)
        out = np.asarray(out)[0][:, -horizon:, :] * (sd + 1e-5) + mu
        last = float(hist["close"].iloc[-1])
        return out[:, -1, 3] / last - 1.0


REGISTRY = {"noise": NoiseAdapter, "oracle": OracleAdapter,
            "reversal-clone": ReversalCloneAdapter, "kronos": KronosAdapter}
