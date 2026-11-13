"""Post-hoc recalibration of deployed MIMO emulators (per model and view angle).

    python recalibrate.py -y config/recalibration_40m.yaml

Transform of the S-member Laplace (Gaussian) mixture, per view angle:

    loc_s'   = m + t_spread * (loc_s - m),    m = mean_s loc_s     (epistemic spread)
    scale_s' = t_scale * scale_s                                   (aleatoric width)

``method: scale_only`` fits t_scale (t_spread = 1); ``two_param`` fits both. The parameters
minimise the exact mixture NLL (a strictly proper score, so the mean is unchanged and no
coverage level is targeted) on the calibration region, optimised in log space with L-BFGS-B
(bounds +/- 3, i.e. factors 0.05 - 20).

Data: ``fit_dataset`` (default 10.5 h scene, y in [0.5, 1]: never used in training or model
selection) and ``check_dataset`` (10.5 h, y in [0, 0.5]: the validation half, an
independent spatial check of the fitted transform). Cloudy pixels (truth or prediction
>= cloud_threshold), randomly subsampled to ``max_pixels``.

Writes per model ``calibration.json`` ({"method", "angles": {"+26.1": {"t_scale", "t_spread"}}})
to ``calibration_file`` (default <model_dir>/calibration.json), read by mimo_inference with
``apply_calibration: true``; plus in ``output_dir``: T_recal_params.{csv,tex},
R1_reliability_before_after.{pdf,png}, R2_temperatures_vs_vza.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "emulator", "mimo", "pyct")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from emulator.mimo.pyct import uq_metrics as U  # noqa: E402
from emulator.mimo.pyct.utils import read_yaml  # noqa: E402
from emulator.mimo.pyct.mimo_inference import (angle_key, calibration_path, load_model,  # noqa: E402
                                               predict_scene_views)
from emulator.mimo.run_uq_analysis import STYLE, WIDTH, save_fig, to_latex  # noqa: E402

LEVELS = np.round(np.linspace(0.05, 0.95, 19), 2)


def transform(loc, scale, ts, te):
    m = loc.mean(-1, keepdims=True)
    return m + te * (loc - m), ts * scale


def pixels(out, c, thr, max_px, rng):
    y = out["tau_true"][..., c].reshape(-1).astype(np.float64)
    loc = out["loc"][..., c, :].reshape(y.size, -1).astype(np.float64)
    scale = np.maximum(out["scale"][..., c, :].reshape(loc.shape).astype(np.float64), 1e-6)
    sel = np.flatnonzero(((y >= thr) | (loc.mean(-1) >= thr)) & np.isfinite(y) & np.isfinite(loc).all(-1))
    if max_px and sel.size > max_px:
        sel = np.sort(rng.choice(sel, max_px, replace=False))
    return y[sel], loc[sel], scale[sel]


def fit(y, loc, scale, family, method, bound=3.0):
    def obj(p):
        ts = np.exp(p[0])
        te = np.exp(p[1]) if method == "two_param" else 1.0
        l2, s2 = transform(loc, scale, ts, te)
        return float(U.nll(y, l2, s2, family).mean())
    n = 2 if method == "two_param" else 1
    best = None
    for x0 in ([0.0] * n, [0.5] * n, [-0.5] * n):
        r = minimize(obj, np.array(x0), method="L-BFGS-B", bounds=[(-bound, bound)] * n)
        if best is None or r.fun < best.fun:
            best = r
    ts = float(np.exp(best.x[0]))
    te = float(np.exp(best.x[1])) if method == "two_param" else 1.0
    at_bound = bool(np.any(np.abs(best.x) > bound - 1e-3))
    return ts, te, float(best.fun), obj(np.zeros(n)), at_bound


def scores(y, loc, scale, family):
    pv = U.pit(y, loc, scale, family)
    _, cen, _ = U.calibration_curve(pv, LEVELS)
    return {"NLL": float(U.nll(y, loc, scale, family).mean()), "ECE": U.ece(cen, LEVELS),
            "cov80": float(U.central_coverage(pv, 0.8).mean()),
            "CRPS": float(U.crps(y, loc, scale, family).mean()), "curve": cen}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--dry-run", action="store_true", help="do not write calibration.json files")
    a = ap.parse_args()
    cfg = read_yaml(a.yaml)
    import torch
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    method = cfg.get("method", "two_param")
    thr = cfg.get("cloud_threshold", 0.05)
    max_px = cfg.get("max_pixels", 500_000)
    rng = np.random.default_rng(cfg.get("seed", 0))
    fd, cd = cfg["fit_dataset"], cfg.get("check_dataset")
    rows, curves = [], {}
    for m in cfg["models"]:
        mcfg = read_yaml(m["config"])
        mcfg.update(m.get("overrides", {}))
        group, tag = m.get("group", "model"), m.get("tag", os.path.basename(m["config"]))
        model = load_model(mcfg, device)
        print(f"[recalibrate] {tag}: predicting fit region {fd.get('region')}")
        o_fit = predict_scene_views(model, mcfg, fd["input_fname"], fd["target_fname"], device=device,
                                    region=fd.get("region"), calibration=None)
        o_chk = (predict_scene_views(model, mcfg, cd["input_fname"], cd["target_fname"], device=device,
                                     region=cd.get("region"), calibration=None) if cd else None)
        fam = o_fit["family"]
        calib = {"method": method, "family": fam, "fitted_on": fd, "cloud_threshold": thr,
                 "created": datetime.now().isoformat(timespec="seconds"), "angles": {}}
        for c, ang in enumerate(o_fit["angles"]):
            y, loc, scale = pixels(o_fit, c, thr, max_px, rng)
            ts, te, nll_after, nll_before, at_b = fit(y, loc, scale, fam, method)
            calib["angles"][angle_key(ang)] = {"t_scale": ts, "t_spread": te}
            row = {"group": group, "network": tag, "vza": float(ang), "t_scale": ts, "t_spread": te,
                   "N_fit": int(y.size), "NLL_fit_before": nll_before, "NLL_fit_after": nll_after,
                   "at_bound": at_b}
            if o_chk is not None:
                yc, lc, sc = pixels(o_chk, c, thr, max_px, rng)
                b = scores(yc, lc, sc, fam)
                l2, s2 = transform(lc, sc, ts, te)
                af = scores(yc, l2, s2, fam)
                curves[(group, float(ang))] = (b.pop("curve"), af.pop("curve"))
                row.update({f"{k}_check_before": v for k, v in b.items()})
                row.update({f"{k}_check_after": v for k, v in af.items()})
            rows.append(row)
            print(f"   vza {ang:+5.1f}: t_scale={ts:.3f} t_spread={te:.3f}  NLL {nll_before:.3f} -> {nll_after:.3f}"
                  + (f"  check cov80 {row['cov80_check_before']:.3f} -> {row['cov80_check_after']:.3f}" if o_chk else ""))
        fcal = mcfg.get("calibration_file") or calibration_path(mcfg)
        if not a.dry_run:
            os.makedirs(os.path.dirname(fcal) or ".", exist_ok=True)
            with open(fcal, "w") as fh:
                json.dump(calib, fh, indent=2)
            print(f"   wrote {fcal}")
        del model

    t = pd.DataFrame(rows).sort_values(["group", "vza"])
    t.to_csv(os.path.join(out_dir, "T_recal_params.csv"), index=False)
    cols = [c for c in ["group", "vza", "t_scale", "t_spread", "NLL_fit_before", "NLL_fit_after",
                        "NLL_check_before", "NLL_check_after", "ECE_check_before", "ECE_check_after",
                        "cov80_check_before", "cov80_check_after"] if c in t.columns]
    to_latex(t[cols], os.path.join(out_dir, "T_recal_params.tex"),
             f"Recalibration parameters ({method}) fitted on {fd.get('name', 'calibration region')} and "
             "checked on the independent half (cloudy pixels).", "tab:recal")
    figures(t, curves, out_dir, cfg.get("formats", ["pdf", "png"]))
    print(f"[recalibrate] -> {out_dir}")


def figures(t, curves, out_dir, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    fc = {"formats": formats, "dpi": 300}
    groups = list(dict.fromkeys(t.group))
    if curves:
        angles = sorted({a for _, a in curves})
        nc = min(len(angles), 5)
        nr = int(np.ceil(len(angles) / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(WIDTH, 1.55 * nr + 0.3), squeeze=False, constrained_layout=True)
        for axi, ang in zip(ax.flat, angles):
            for k, g in enumerate(groups):
                if (g, ang) not in curves:
                    continue
                b, af = curves[(g, ang)]
                axi.plot(LEVELS, b, color=f"C{k}", ls="--", lw=1.0, label=f"{g} raw")
                axi.plot(LEVELS, af, color=f"C{k}", ls="-", lw=1.0, label=f"{g} cal")
            axi.plot([0, 1], [0, 1], color="0.6", lw=0.7, zorder=0)
            axi.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title=f"{ang:+.1f}°" if ang else "0° (nadir)")
            axi.tick_params(labelsize=6)
        for axi in ax.flat[len(angles):]:
            axi.set_visible(False)
        h, lab = [], []
        for axi in ax.flat:
            for hh, ll in zip(*axi.get_legend_handles_labels()):
                if ll not in lab:
                    h.append(hh); lab.append(ll)
        if len(angles) < nr * nc:   # free slot -> legend there, else in the first panel
            ax.flat[-1].set_visible(True); ax.flat[-1].axis("off")
            ax.flat[-1].legend(h, lab, frameon=False, fontsize=6, loc="center")
        else:
            ax[0, 0].legend(h, lab, frameon=False, fontsize=5, loc="upper left")
        fig.supxlabel("Nominal central-interval probability", fontsize=7)
        fig.supylabel("Observed frequency", fontsize=7)
        save_fig(fig, out_dir, "R1_reliability_before_after", fc)

    fig, ax = plt.subplots(1, 2, figsize=(WIDTH * 0.75, 2.2), constrained_layout=True)
    for k, g in enumerate(groups):
        d = t[t.group == g].sort_values("vza")
        ax[0].plot(d.vza, d.t_scale, "o-", ms=3, color=f"C{k}", label=g)
        ax[1].plot(d.vza, d.t_spread, "s-", ms=3, color=f"C{k}", label=g)
    for axi, ttl in zip(ax, (r"$t_\mathrm{scale}$ (aleatoric)", r"$t_\mathrm{spread}$ (epistemic)")):
        axi.axhline(1.0, color="0.6", lw=0.7)
        axi.set(xlabel="VZA [deg]", title=ttl, yscale="log")
    ax[0].legend(frameon=False, fontsize=6)
    save_fig(fig, out_dir, "R2_temperatures_vs_vza", fc)


if __name__ == "__main__":
    main()
