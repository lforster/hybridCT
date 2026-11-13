"""Numpy UQ metrics for MIMO (equal-weight mixture of S Laplace or Gaussian members).

Conventions: member parameters are on the *last* axis, ``loc[..., S]`` and
``scale[..., S]`` (Laplace b or Gaussian sigma), in physical units.

The predictive distribution is the exact mixture p(y) = 1/S sum_s p_s(y).
Its variance is mean_s Var_s + (1/S) sum_s (mu_s - mu)^2; the reported
epistemic term uses the unbiased (S - 1) denominator to match
``compute_uncertainties``. Two simpler predictive approximations are provided
for comparison with the earlier analysis:

* ``moment``    : single Laplace/Gaussian at the mean with the total variance
                  (Laplace b_eff = sqrt(var_total / 2)),
* ``aleatoric`` : single distribution at the mean with aleatoric variance only
                  (what test_cloud_uq.py used, via scale = aleatoric_std / sqrt(2)).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import special, stats

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 / numpy<2

SQRT2 = np.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Single-component helpers
# --------------------------------------------------------------------------- #
def _cdf(y, loc, scale, family):
    z = (y - loc) / scale
    if family == "laplace":
        return np.where(z < 0, 0.5 * np.exp(np.minimum(z, 0)), 1.0 - 0.5 * np.exp(-np.maximum(z, 0)))
    return special.ndtr(z)


def _logpdf(y, loc, scale, family):
    if family == "laplace":
        return -np.log(2 * scale) - np.abs(y - loc) / scale
    return -0.5 * np.log(2 * np.pi) - np.log(scale) - 0.5 * ((y - loc) / scale) ** 2


def _ppf(p, loc, scale, family):
    if family == "laplace":
        return loc - scale * np.sign(p - 0.5) * np.log1p(-2 * np.abs(p - 0.5))
    return loc + scale * special.ndtri(p)


def var_to_scale(var, family):
    return np.sqrt(var / 2.0) if family == "laplace" else np.sqrt(var)


# --------------------------------------------------------------------------- #
# Predictive distributions
# --------------------------------------------------------------------------- #
def member_moments(loc, scale, family="laplace"):
    S = loc.shape[-1]
    mvar = 2.0 * scale ** 2 if family == "laplace" else scale ** 2
    mean = loc.mean(-1)
    ale = mvar.mean(-1)
    epi = loc.var(-1, ddof=1) if S > 1 else np.zeros_like(mean)
    return mean, ale, epi


def predictive_params(loc, scale, family, kind="mixture"):
    """Return (loc[..., K], scale[..., K]) of an equal-weight mixture for a predictive kind."""
    if kind == "mixture":
        return loc, scale
    mean, ale, epi = member_moments(loc, scale, family)
    var = ale + epi if kind == "moment" else ale
    return mean[..., None], var_to_scale(var, family)[..., None]


def pit(y, loc, scale, family="laplace"):
    """Probability integral transform F(y) of the mixture."""
    return _cdf(y[..., None], loc, scale, family).mean(-1)


def nll(y, loc, scale, family="laplace"):
    lp = _logpdf(y[..., None], loc, scale, family)
    return -(special.logsumexp(lp, axis=-1) - np.log(loc.shape[-1]))


def _pair_abs_laplace(d, bi, bj):
    """E|d + Li - Lj| for independent Laplace(0, bi), Laplace(0, bj)."""
    g = lambda b: np.abs(d) + b * np.exp(-np.abs(d) / b)
    close = np.abs(bi - bj) < 1e-4 * (bi + bj)
    bj_ = np.where(close, bj * (1 + 1e-3), bj)  # smooth limit for equal scales
    return (bi ** 2 * g(bi) - bj_ ** 2 * g(bj_)) / (bi ** 2 - bj_ ** 2)


def _abs_gauss(m, s):
    """E|X| for X ~ N(m, s^2)."""
    z = m / s
    return s * np.sqrt(2 / np.pi) * np.exp(-0.5 * z ** 2) + m * (1 - 2 * special.ndtr(-z))


def crps(y, loc, scale, family="laplace"):
    """Closed-form CRPS of the equal-weight mixture: E|X - y| - 0.5 E|X - X'|."""
    S = loc.shape[-1]
    y = y[..., None]
    if family == "laplace":
        d = np.abs(y - loc)
        term1 = (d + scale * np.exp(-d / scale)).mean(-1)
    else:
        term1 = _abs_gauss(y - loc, scale).mean(-1)
    term2 = np.zeros(loc.shape[:-1])
    for i in range(S):
        for j in range(S):
            if family == "laplace":
                if i == j:
                    term2 += 1.5 * scale[..., i]
                else:
                    term2 += _pair_abs_laplace(loc[..., i] - loc[..., j], scale[..., i], scale[..., j])
            else:
                term2 += _abs_gauss(loc[..., i] - loc[..., j], np.sqrt(scale[..., i] ** 2 + scale[..., j] ** 2))
    return term1 - 0.5 * term2 / S ** 2


def quantile(p, loc, scale, family="laplace", n_iter=60):
    """Mixture quantile by vectorised bisection (p scalar)."""
    if loc.shape[-1] == 1:
        return _ppf(p, loc[..., 0], scale[..., 0], family)
    lo = (_ppf(p, loc, scale, family)).min(-1)
    hi = (_ppf(p, loc, scale, family)).max(-1)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        below = _cdf(mid[..., None], loc, scale, family).mean(-1) < p
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def central_coverage(pit_values, level):
    """Hit indicator of the equal-tailed central interval at `level` (from PIT)."""
    return np.abs(pit_values - 0.5) <= level / 2.0


# --------------------------------------------------------------------------- #
# Summary metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y, yhat) -> Dict[str, float]:
    err = yhat - y
    out = {"N": int(y.size), "RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err))),
           "bias": float(np.mean(err))}
    if y.size > 2 and np.std(y) > 0 and np.std(yhat) > 0:
        out["r"] = float(np.corrcoef(y, yhat)[0, 1])
        out["slope"], out["intercept"] = (float(v) for v in np.polyfit(y, yhat, 1))
    else:
        out["r"] = out["slope"] = out["intercept"] = np.nan
    return out


def calibration_curve(pit_values, levels: Sequence[float]):
    """Observed vs nominal coverage of central intervals, and one-sided PIT ECDF."""
    levels = np.asarray(levels)
    central = np.array([central_coverage(pit_values, a).mean() for a in levels])
    one_sided = np.array([(pit_values <= a).mean() for a in levels])
    return levels, central, one_sided


def ece(observed, nominal):
    return float(np.mean(np.abs(np.asarray(observed) - np.asarray(nominal))))


def spread_skill(err, var, n_bins=10):
    """Binned RMSE vs root-mean-variance (RMV) on quantile bins of the predicted std.

    Returns dict with bin centres and ENCE = mean_k |RMV_k - RMSE_k| / RMV_k.
    """
    std = np.sqrt(var)
    edges = np.unique(np.quantile(std, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, std, side="right") - 1, 0, len(edges) - 2)
    rmv, rmse, cnt = [], [], []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() == 0:
            continue
        rmv.append(np.sqrt(np.mean(var[m])))
        rmse.append(np.sqrt(np.mean(err[m] ** 2)))
        cnt.append(int(m.sum()))
    rmv, rmse = np.array(rmv), np.array(rmse)
    ence = float(np.mean(np.abs(rmv - rmse) / np.maximum(rmv, 1e-12)))
    # spread-skill correlation (pixel level, Spearman)
    rho = float(stats.spearmanr(std, np.abs(err))[0]) if std.size > 2 else np.nan
    return {"rmv": rmv, "rmse": rmse, "count": np.array(cnt), "ENCE": ence, "spearman": rho,
            "cv_std": float(np.std(std) / max(np.mean(std), 1e-12))}


def sparsification(err, unc, n_points=50):
    """Error-retention curve (sorted by uncertainty, most uncertain removed first).

    Returns fractions removed, RMSE of the retained set for the uncertainty ordering,
    the oracle (|error|) ordering, and AUSE (area between them, RMSE normalised by
    the full-set RMSE) plus the AURG-style gain over a random ordering.
    """
    n = err.size
    se = err ** 2
    order_u = np.argsort(-unc)
    order_o = np.argsort(-np.abs(err))
    frac = np.linspace(0, 0.99, n_points)
    cs_u = np.cumsum(se[order_u][::-1])[::-1]
    cs_o = np.cumsum(se[order_o][::-1])[::-1]
    cut = (frac * n).astype(int)
    keep = n - cut
    rmse_u = np.sqrt(cs_u[cut] / keep)
    rmse_o = np.sqrt(cs_o[cut] / keep)
    full = np.sqrt(se.mean())
    ause = float(_trapz((rmse_u - rmse_o) / full, frac))
    gain = float(_trapz((full - rmse_u) / full, frac))
    return {"fraction_removed": frac, "rmse_unc": rmse_u, "rmse_oracle": rmse_o,
            "rmse_random": np.full_like(frac, full), "AUSE": ause, "AURG": gain}


def interval_width(loc, scale, family, level, max_n=200_000, rng=None):
    """Mean width of the central `level` interval (subsampled; bisection is costly)."""
    n = loc.shape[0]
    if n > max_n:
        rng = rng or np.random.default_rng(0)
        sel = rng.choice(n, max_n, replace=False)
        loc, scale = loc[sel], scale[sel]
    lo = quantile(0.5 - level / 2, loc, scale, family)
    hi = quantile(0.5 + level / 2, loc, scale, family)
    return float(np.mean(hi - lo))


def distribution_shift(a, b) -> Dict[str, float]:
    """Compare uncertainty distributions (b = shifted / OOD vs a = reference)."""
    a, b = np.asarray(a), np.asarray(b)
    ks = stats.ks_2samp(a, b)
    mw = stats.mannwhitneyu(b, a, alternative="greater")
    return {"median_ref": float(np.median(a)), "median_shift": float(np.median(b)),
            "median_ratio": float(np.median(b) / max(np.median(a), 1e-12)),
            "KS": float(ks.statistic), "KS_p": float(ks.pvalue),
            "MW_p_greater": float(mw.pvalue),
            "AUROC": float(mw.statistic / (a.size * b.size)),
            "W1": float(stats.wasserstein_distance(a, b))}


def full_summary(y, loc, scale, family, levels=(0.5, 0.8, 0.9, 0.95),
                 curve_levels=None, width_level=0.8) -> Dict[str, float]:
    """All scalar metrics for one pixel set (flattened arrays, members last)."""
    mean, ale, epi = member_moments(loc, scale, family)
    tot = ale + epi
    err = mean - y
    out = regression_metrics(y, mean)
    out.update({"std_ale": float(np.sqrt(ale.mean())), "std_epi": float(np.sqrt(epi.mean())),
                "std_tot": float(np.sqrt(tot.mean())), "epi_frac": float(epi.sum() / max(tot.sum(), 1e-12))})
    pv = pit(y, loc, scale, family)
    for a in levels:
        out[f"cov{int(round(a * 100))}"] = float(central_coverage(pv, a).mean())
    curve_levels = curve_levels if curve_levels is not None else np.linspace(0.05, 0.95, 19)
    lv, cen, one = calibration_curve(pv, curve_levels)
    out["ECE"] = ece(cen, lv)
    out["ECE_onesided"] = ece(one, lv)
    for kind in ("moment", "aleatoric"):
        l2, s2 = predictive_params(loc, scale, family, kind)
        _, c2, _ = calibration_curve(pit(y, l2, s2, family), curve_levels)
        out[f"ECE_{kind}"] = ece(c2, lv)
        out[f"cov80_{kind}"] = float(central_coverage(pit(y, l2, s2, family), 0.8).mean())
    out["NLL"] = float(np.mean(nll(y, loc, scale, family)))
    out["CRPS"] = float(np.mean(crps(y, loc, scale, family)))
    ss = spread_skill(err, tot)
    out["ENCE"], out["spearman_std_abs_err"], out["Cv"] = ss["ENCE"], ss["spearman"], ss["cv_std"]
    sp = sparsification(err, tot)
    out["AUSE"], out["AURG"] = sp["AUSE"], sp["AURG"]
    out[f"width{int(width_level * 100)}"] = interval_width(loc, scale, family, width_level)
    return out
