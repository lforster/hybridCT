"""Single-entry UQ analysis for the MIMO emulators (manuscript tables and figures).

Merges ``pyct/test_cloud_uq.py`` (calibration curve, precision-recall /
error-retention curve, noise loop) with the extended test plan:

  T1  overall accuracy + UQ metrics per dataset x angle x subset
      (RMSE, MAE, bias, r, slope, std_ale/epi/tot, epistemic fraction, coverage
      50/80/90/95, ECE (mixture, moment-matched, aleatoric-only), NLL, CRPS,
      ENCE, Spearman(std, |err|), Cv, AUSE, AURG, 80 % interval width)
  T2  OOD vs ID uncertainty shift (KS, Mann-Whitney / AUROC, W1, median ratios)
  T3  stratified performance (true COT, patch COT range, patch cloud fraction,
      patch COT std): slope, intercept, RMSE, bias, r, coverage, ECE, std median/IQR
  T4  input-noise robustness
  T5  focus-angle summary (nadir and +/-26.1 deg by default)
  T6  registration floor of the multi-angle model (true-COT round trip vs model RMSE)
  F1  reliability diagrams          F2  PIT histograms
  F3  spread-skill (RMSE vs RMV)    F4  sparsification / error retention + oracle
  F5  OOD uncertainty distributions F6  stratified metrics
  F7  coverage / ECE vs VZA         F8  spatial maps incl. 80 % hit map and z-score
  F9  noise robustness              F10 |error| vs predicted std density

Stages (``--stage``):
  predict  run each model (single-angle, or one multi-angle model giving all views) on each
           dataset (optional y ``region``) and noise level; one cache .npz per view
  analyze  read caches, write tables (CSV + LaTeX) and figures (PDF + PNG)
  all      both (default)

Run from ``emulator/mimo`` with the repo installed (``pip install -e .``):
    python run_uq_analysis.py -y config/uq_analysis.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import os as _os, sys as _sys
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
for _p in (_REPO, _os.path.join(_REPO, "emulator", "mimo", "pyct")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)  # run from emulator/mimo without PYTHONPATH
from emulator.mimo.pyct import uq_metrics as U
from emulator.mimo.pyct.utils import read_yaml

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
STYLE = {
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 150,
    "lines.linewidth": 1.2, "axes.grid": False, "pdf.fonttype": 42,
}
WIDTH = 7.0  # in, two-column manuscript width


def _angle_colors(angles):
    cmap = plt.get_cmap("coolwarm")
    amax = max(abs(a) for a in angles) or 1.0
    return {a: cmap(0.5 + 0.5 * a / amax) if a != 0 else "k" for a in angles}


def save_fig(fig, out_dir, name, cfg):
    os.makedirs(out_dir, exist_ok=True)
    for ext in cfg.get("formats", ["pdf", "png"]):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=cfg.get("dpi", 300))
    plt.close(fig)


def to_latex(df: pd.DataFrame, path: str, caption: str = "", label: str = "", fmt="{:.3g}"):
    """Minimal booktabs LaTeX writer (no jinja2 dependency)."""
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{table}[t]", r"\centering", r"\small"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines += [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
              " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                cells.append("--" if not np.isfinite(v) else fmt.format(v))
            else:
                cells.append(esc(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_table(df, out_dir, name, caption="", cols=None):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    to_latex(df[cols] if cols else df, os.path.join(out_dir, f"{name}.tex"), caption, f"tab:{name}")


# --------------------------------------------------------------------------- #
# Stage 1: predictions
# --------------------------------------------------------------------------- #
def _cache_name(cache_dir, dataset, angle, noise):
    return os.path.join(cache_dir, f"{dataset}__vza{angle:+05.1f}__noise{noise:.3f}.npz")


def stage_predict(cfg: Dict[str, Any], overwrite=False):
    """Predict every (model, dataset[, noise]) once; one cache file per view angle.

    Single-angle models give one view each; a multi-angle model (``multiview: true``)
    gives all of its views from one forward pass (plus the registration round trip
    ``tau_roundtrip``). ``region`` per dataset restricts the evaluated y band;
    ``overrides`` (per model, or global ``model_overrides``) e.g. ``apply_calibration: true``.
    """
    import torch
    from emulator.mimo.pyct.mimo_inference import load_model, predict_scene_views

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = cfg["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    noise_levels = cfg.get("noise_levels", [0.0])
    noise_ds = cfg.get("noise_dataset", cfg["reference_dataset"])

    for m in cfg["models"]:
        mcfg = read_yaml(m["config"])
        mcfg.update(cfg.get("model_overrides", {}))
        mcfg.update(m.get("overrides", {}))
        angles = [float(a) for a in mcfg["angles"]]
        model = None
        for ds in cfg["datasets"]:
            levels = noise_levels if ds["name"] == noise_ds else [0.0]
            for noise in levels:
                fouts = [_cache_name(cache_dir, ds["name"], a, noise) for a in angles]
                if all(os.path.exists(f) for f in fouts) and not overwrite:
                    continue
                model = model or load_model(mcfg, device)
                print(f"[predict] model vza={angles}  dataset={ds['name']}  noise={noise}  region={ds.get('region')}")
                out = predict_scene_views(model, mcfg, ds["input_fname"], ds["target_fname"],
                                          input_noise=noise, seed=cfg.get("seed", 0), device=device,
                                          region=ds.get("region"))
                for c, (a, fout) in enumerate(zip(out["angles"], fouts)):
                    extra = {"tau_roundtrip": out["tau_roundtrip"][..., c]} if "tau_roundtrip" in out else {}
                    np.savez_compressed(fout, angle=a, dataset=ds["name"], group=ds.get("group", "ID"),
                                        noise=noise, family=out["family"], resolution=mcfg["resolution"],
                                        multiview=bool(mcfg.get("multiview", False)),
                                        calibrated=bool(out["calibrated"]),
                                        tau_true=out["tau_true"][..., c], loc=out["loc"][..., c, :],
                                        scale=out["scale"][..., c, :], **extra)
        if cfg.get("include_test_split", False):
            if mcfg.get("multiview", False) or len(angles) > 1:
                print("[predict] include_test_split skipped for multi-angle models")
            else:
                fout = _cache_name(cache_dir, "test_split", angles[0], 0.0)
                if overwrite or not os.path.exists(fout):
                    model = model or load_model(mcfg, device)
                    _predict_test_split(model, mcfg, fout, angles[0], device)
        del model


def _predict_test_split(model, mcfg, fout, angle, device):
    """Random-tile test split of CloudDataModule (what test_cloud_uq.py evaluated)."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyct"))
    from cloud_datamodule import CloudDataModule
    from emulator.mimo.pyct.mimo_inference import affine_inverse
    from emulator.mimo.pyct.utils import load_scalers

    dm = CloudDataModule(mcfg)
    dm.setup()
    X, y = dm.data_test.X.astype(np.float32), dm.data_test.y.astype(np.float32)
    _, out_scaler = load_scalers(mcfg)
    a, c = (float(v[0]) for v in affine_inverse(out_scaler))
    locs, scales = [], []
    with torch.no_grad():
        for i in range(0, len(X), mcfg.get("batch_size", 32)):
            l, s = model.predict_members(torch.from_numpy(X[i:i + mcfg.get("batch_size", 32)]).to(device))
            locs.append(l[:, :, 0].cpu().numpy())
            scales.append(s[:, :, 0].cpu().numpy())
    loc = (a * np.concatenate(locs) + c).transpose(0, 2, 3, 1)       # [N, H, W, S]
    scale = (abs(a) * np.concatenate(scales)).transpose(0, 2, 3, 1)
    np.savez_compressed(fout, angle=angle, dataset="test_split", group="ID_split", noise=0.0,
                        family=model.loss_fn.name.split("_")[0], resolution=mcfg["resolution"],
                        tau_true=a * y[:, 0] + c, loc=loc.astype(np.float32), scale=scale.astype(np.float32))


# --------------------------------------------------------------------------- #
# Stage 2: analysis
# --------------------------------------------------------------------------- #
SCALAR_KEYS = ("angle", "dataset", "group", "noise", "family", "resolution", "multiview", "calibrated")


def load_records(cache_dir) -> List[Dict[str, Any]]:
    """Scalar metadata + path of every cache file (arrays are loaded on demand)."""
    recs = []
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        with np.load(f, allow_pickle=False) as z:
            r = {k: z[k].item() for k in SCALAR_KEYS if k in z.files}
            r["has_roundtrip"] = "tau_roundtrip" in z.files
        r["path"] = f
        recs.append(r)
    return recs


def load_arrays(rec) -> Dict[str, Any]:
    """Record with its arrays (tau_true, loc, scale[, tau_roundtrip]) loaded."""
    with np.load(rec["path"], allow_pickle=False) as z:
        out = dict(rec)
        for k in z.files:
            if k not in SCALAR_KEYS:
                out[k] = z[k]
    return out


def patch_ids(shape, patch_size):
    """Patch index per pixel for a 2-D map, or tile index for a tile stack [N, H, W]."""
    if len(shape) == 3:
        return np.repeat(np.arange(shape[0]), shape[1] * shape[2]).reshape(shape)
    nx, ny = shape
    npy = int(np.ceil(ny / patch_size))
    ii, jj = np.meshgrid(np.arange(nx) // patch_size, np.arange(ny) // patch_size, indexing="ij")
    return ii * npy + jj


def flatten_record(rec, thr, patch_size):
    """Pixel table (dict of 1-D arrays) with patch metrics attached by patch id.

    Patch metrics are computed on the full maps *before* masking, then mapped
    to pixels by index, so there is no length mismatch after filtering.
    """
    if "tau_true" not in rec:
        rec = load_arrays(rec)
    y = rec["tau_true"]
    pid = patch_ids(y.shape, patch_size).ravel()
    yf = y.ravel().astype(np.float64)
    loc = rec["loc"].reshape(-1, rec["loc"].shape[-1]).astype(np.float64)
    scale = np.maximum(rec["scale"].reshape(loc.shape).astype(np.float64), 1e-6)
    n_p = pid.max() + 1
    cnt = np.bincount(pid, minlength=n_p).astype(float)
    s1 = np.bincount(pid, yf, n_p)
    s2 = np.bincount(pid, yf ** 2, n_p)
    pmax = np.full(n_p, -np.inf); np.maximum.at(pmax, pid, yf)
    pmin = np.full(n_p, np.inf); np.minimum.at(pmin, pid, yf)
    cf = np.bincount(pid, (yf >= thr).astype(float), n_p) / cnt
    pstd = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0))
    mean, ale, epi = U.member_moments(loc, scale, rec["family"])
    # pixels never covered by a trimmed tile are stitched as loc=scale=0; exclude them
    covered = (rec["scale"].reshape(loc.shape) > 0).all(-1)
    valid = covered & np.isfinite(yf) & np.all(np.isfinite(loc), -1) & np.all(np.isfinite(scale), -1)
    t = {"y": yf, "loc": loc, "scale": scale, "mean": mean, "ale": ale, "epi": epi,
         "tot": ale + epi, "patch_id": pid, "cot_range_patch": (pmax - pmin)[pid],
         "cloud_fraction_patch": cf[pid], "cot_std_patch": pstd[pid],
         "cloudy": (yf >= thr) | (mean >= thr)}
    return {k: v[valid] for k, v in t.items()}


def make_tables(rec, thr, patch, subsets, max_pixels, rng):
    """Pre-subsampled float32 pixel tables per subset (keeps memory bounded at 40 m)."""
    t = flatten_record(rec, thr, patch)
    out = {}
    for name in subsets:
        s = _subset(t, name, max_pixels, rng)
        out[name] = {k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in s.items()}
    return out


def subset(t, name, max_pixels=None, rng=None):
    """Return the pre-subsampled table (``t`` from make_tables), sub-sampling further if needed."""
    s = t[name]
    if max_pixels and s["y"].size > max_pixels:
        idx = np.sort(rng.choice(s["y"].size, max_pixels, replace=False))
        s = {k: v[idx] for k, v in s.items()}
    return {k: (v.astype(np.float64) if v.dtype == np.float32 else v) for k, v in s.items()}


def _subset(t, name, max_pixels, rng):
    m = t["cloudy"] if name == "cloudy" else np.ones_like(t["y"], bool)
    idx = np.flatnonzero(m)
    if max_pixels and idx.size > max_pixels:
        idx = np.sort(rng.choice(idx, max_pixels, replace=False))
    return {k: v[idx] for k, v in t.items()}


def analyze(cfg: Dict[str, Any]):
    plt.rcParams.update(STYLE)
    A = cfg.get("analysis", {})
    out_dir = cfg["output_dir"]
    tdir, fdir = os.path.join(out_dir, "tables"), os.path.join(out_dir, "figures")
    thr = A.get("cloud_threshold", 0.05)
    patch = A.get("patch_size", 32)
    max_px = A.get("max_pixels", 2_000_000)
    subsets = A.get("subsets", ["all", "cloudy"])
    curve_lv = np.round(np.linspace(0.05, 0.95, 19), 2)
    focus = [float(a) for a in A.get("focus_angles", [0.0, -26.1, 26.1])]
    ref = cfg["reference_dataset"]
    fig_cfg = cfg.get("figures", {})
    rng = np.random.default_rng(cfg.get("seed", 0))

    recs = load_records(cfg["cache_dir"])
    if not recs:
        raise SystemExit(f"no cached predictions in {cfg['cache_dir']}; run --stage predict")
    base = [r for r in recs if float(r["noise"]) == 0.0]
    angles = sorted({float(r["angle"]) for r in base})
    colors = _angle_colors(angles)
    datasets = [d["name"] for d in cfg.get("datasets", [])] or sorted({r["dataset"] for r in base})
    if any(r["dataset"] == "test_split" for r in base) and "test_split" not in datasets:
        datasets.append("test_split")
    groups = {d["name"]: d.get("group", "ID") for d in cfg.get("datasets", [])}
    groups.setdefault("test_split", "ID_split")
    tables = {}  # (dataset, angle) -> {subset: pre-subsampled pixel table}
    need = sorted(set(subsets) | {"cloudy"})
    for r in base:
        tables[(r["dataset"], float(r["angle"]))] = make_tables(r, thr, patch, need, max_px, rng)
    fam = base[0]["family"]

    # ---- T1 overall ------------------------------------------------------ #
    rows = []
    for (ds, ang), t in sorted(tables.items()):
        for sname in subsets:
            s = subset(t, sname, max_px, rng)
            if s["y"].size < 10:
                continue
            m = U.full_summary(s["y"], s["loc"], s["scale"], fam, curve_levels=curve_lv)
            rows.append({"dataset": ds, "group": groups.get(ds, "ID"), "vza": ang, "subset": sname, **m})
    t1 = pd.DataFrame(rows)
    write_table(t1, tdir, "T1_overall", "Accuracy and uncertainty metrics per dataset, view angle and subset.",
                cols=["dataset", "vza", "subset", "RMSE", "bias", "r", "std_tot", "epi_frac",
                      "cov80", "ECE", "ECE_aleatoric", "CRPS", "NLL", "ENCE", "AUSE"])

    # ---- T5 focus angles -------------------------------------------------- #
    t5 = t1[(t1.vza.isin(focus)) & (t1.subset == "cloudy")][
        ["dataset", "vza", "RMSE", "cov50", "cov80", "cov95", "ECE", "CRPS", "width80"]]
    write_table(t5, tdir, "T5_focus_angles", "Cloudy-pixel calibration at the focus view angles.")

    # ---- T2 OOD shift ----------------------------------------------------- #
    rows = []
    for ang in angles:
        if (ref, ang) not in tables:
            continue
        r0 = subset(tables[(ref, ang)], "cloudy", max_px, rng)
        for ds in datasets:
            if ds == ref or (ds, ang) not in tables:
                continue
            r1 = subset(tables[(ds, ang)], "cloudy", max_px, rng)
            for comp in ("ale", "epi", "tot"):
                sh = U.distribution_shift(np.sqrt(r0[comp]), np.sqrt(r1[comp]))
                rows.append({"dataset": ds, "group": groups.get(ds, "OOD"), "vza": ang, "component": comp, **sh,
                             "RMSE_ratio": float(np.sqrt(np.mean((r1["mean"] - r1["y"]) ** 2)) /
                                                 max(np.sqrt(np.mean((r0["mean"] - r0["y"]) ** 2)), 1e-12)),
                             "epi_frac_ref": float(r0["epi"].sum() / r0["tot"].sum()),
                             "epi_frac_shift": float(r1["epi"].sum() / r1["tot"].sum())})
    t2 = pd.DataFrame(rows)
    if len(t2):
        write_table(t2, tdir, "T2_ood_shift", f"Uncertainty shift relative to '{ref}' (cloudy pixels).",
                    cols=["dataset", "vza", "component", "median_ratio", "KS", "AUROC", "W1", "RMSE_ratio"])

    # ---- T3 stratified ---------------------------------------------------- #
    strat_vars = A.get("stratify", {"y": "true COT", "cot_range_patch": "patch COT range",
                                     "cloud_fraction_patch": "patch cloud fraction",
                                     "cot_std_patch": "patch COT std"})
    n_bins = A.get("n_bins", 5)
    rows = []
    for (ds, ang), t in sorted(tables.items()):
        s = subset(t, "cloudy", max_px, rng)
        for var in strat_vars:
            edges = np.unique(np.quantile(s[var], np.linspace(0, 1, n_bins + 1)))
            if edges.size < 2:
                continue
            b = np.clip(np.searchsorted(edges, s[var], side="right") - 1, 0, len(edges) - 2)
            for k in range(len(edges) - 1):
                m = b == k
                if m.sum() < 20:
                    continue
                reg = U.regression_metrics(s["y"][m], s["mean"][m])
                pv = U.pit(s["y"][m], s["loc"][m], s["scale"][m], fam)
                _, cen, _ = U.calibration_curve(pv, curve_lv)
                sd = np.sqrt(s["tot"][m])
                rows.append({"dataset": ds, "vza": ang, "stratifier": var, "bin": k,
                             "lo": edges[k], "hi": edges[k + 1], "centre": float(np.median(s[var][m])),
                             **reg, "cov80": float(U.central_coverage(pv, 0.8).mean()),
                             "ECE": U.ece(cen, curve_lv), "std_med": float(np.median(sd)),
                             "std_iqr": float(np.subtract(*np.percentile(sd, [75, 25]))),
                             "epi_frac": float(s["epi"][m].sum() / s["tot"][m].sum())})
    t3 = pd.DataFrame(rows)
    write_table(t3, tdir, "T3_stratified", "Stratified performance and calibration (cloudy pixels).",
                cols=["dataset", "vza", "stratifier", "bin", "centre", "RMSE", "bias", "slope", "cov80", "ECE", "std_med"])
    if len(t3):
        for var in strat_vars:
            pv = t3[(t3.dataset == ref) & (t3.stratifier == var)].pivot(index="vza", columns="bin", values="cov80")
            if len(pv):
                pv = pv.reset_index()
                pv.columns = [str(c) if c == "vza" else f"bin{c}" for c in pv.columns]
                to_latex(pv, os.path.join(tdir, f"T3_cov80_{var}.tex"),
                         f"80\\% coverage by {strat_vars[var]} quantile bin ({ref}).", f"tab:cov80_{var}")

    # ---- T4 noise --------------------------------------------------------- #
    noisy = [r for r in recs if r["dataset"] == cfg.get("noise_dataset", ref)]
    rows = []
    for r in noisy:
        s = subset(make_tables(r, thr, patch, ["cloudy"], max_px, rng), "cloudy")
        pv = U.pit(s["y"], s["loc"], s["scale"], fam)
        _, cen, _ = U.calibration_curve(pv, curve_lv)
        rows.append({"vza": float(r["angle"]), "noise": float(r["noise"]),
                     **U.regression_metrics(s["y"], s["mean"]),
                     "std_ale": float(np.sqrt(s["ale"].mean())), "std_epi": float(np.sqrt(s["epi"].mean())),
                     "cov80": float(U.central_coverage(pv, 0.8).mean()), "ECE": U.ece(cen, curve_lv),
                     "NLL": float(U.nll(s["y"], s["loc"], s["scale"], fam).mean())})
    t4 = pd.DataFrame(rows).sort_values(["vza", "noise"]) if rows else pd.DataFrame()
    if len(t4) and t4.noise.nunique() > 1:
        write_table(t4, tdir, "T4_noise", "Robustness to relative radiance noise (cloudy pixels).",
                    cols=["vza", "noise", "RMSE", "std_ale", "std_epi", "cov80", "ECE", "NLL"])

    # ---- T6 registration floor (multi-angle model) -------------------------- #
    rows = []
    for r in base:
        if not r.get("has_roundtrip"):
            continue
        z = load_arrays(r)
        y, rt = z["tau_true"].astype(np.float64), z["tau_roundtrip"].astype(np.float64)
        mean = z["loc"].astype(np.float64).mean(-1)
        cl = y >= thr
        rmse = lambda a, b, m: float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan
        floor, model_rmse = rmse(rt, y, cl), rmse(mean, y, cl)
        rows.append({"dataset": r["dataset"], "vza": float(r["angle"]),
                     "RMSE_registration_floor": floor, "RMSE_model": model_rmse,
                     "floor_fraction": floor / model_rmse if model_rmse else np.nan,
                     "rel_floor": floor / max(float(np.std(y[cl])), 1e-12) if cl.any() else np.nan})
    t6 = pd.DataFrame(rows).sort_values(["dataset", "vza"]) if rows else pd.DataFrame()
    if len(t6):
        write_table(t6, tdir, "T6_registration_floor",
                    "Multi-angle registration round trip of the true COT (cloudy pixels): the RMSE the "
                    "multi-angle design cannot go below, vs. the model RMSE.")

    # ---- Figures ---------------------------------------------------------- #
    ref_tabs = {a: subset(tables[(ref, a)], "cloudy", max_px, rng) for a in angles if (ref, a) in tables}
    fig_reliability(ref_tabs, fam, curve_lv, colors, focus, fdir, fig_cfg)
    fig_pit(ref_tabs, fam, fdir, fig_cfg)
    fig_spread_skill(ref_tabs, colors, fdir, fig_cfg, A.get("n_bins_spread", 10))
    fig_sparsification(ref_tabs, focus, fdir, fig_cfg)
    if len(t2):
        fig_ood(tables, datasets, ref, focus, groups, max_px, rng, fdir, fig_cfg)
    if len(t3):
        fig_stratified(t3[t3.dataset == ref], strat_vars, colors, fdir, fig_cfg)
    fig_vs_vza(t1[t1.subset == "cloudy"], datasets, fdir, fig_cfg)
    map_angle = float(A.get("map_angle", 0.0))
    rec_map = next((r for r in base if r["dataset"] == ref and float(r["angle"]) == map_angle), None)
    rec_map = load_arrays(rec_map) if rec_map is not None else None
    if rec_map is not None and rec_map["tau_true"].ndim == 2:
        fig_maps(rec_map, thr, A.get("map_window"), fdir, fig_cfg)
    if len(t4) and t4.noise.nunique() > 1:
        fig_noise(t4, colors, fdir, fig_cfg)
    fig_err_vs_std(ref_tabs, focus, fdir, fig_cfg)

    with open(os.path.join(out_dir, "analysis_config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"[analyze] tables -> {tdir}\n[analyze] figures -> {fdir}")
    return t1, t2, t3, t4


# --------------------------------------------------------------------------- #
# Figure helpers
# --------------------------------------------------------------------------- #
def _vza_label(a):
    return f"{a:+.1f}°" if a else "0° (nadir)"


def fig_reliability(tabs, fam, lv, colors, focus, fdir, fc):
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.4), constrained_layout=True)
    for a, s in tabs.items():
        _, cen, one = U.calibration_curve(U.pit(s["y"], s["loc"], s["scale"], fam), lv)
        ax[0].plot(lv, cen, color=colors[a], label=_vza_label(a))
        ax[1].plot(lv, one, color=colors[a])
    a0 = next((a for a in focus if a in tabs), next(iter(tabs)))
    s = tabs[a0]
    for kind, ls in (("mixture", "-"), ("moment", "--"), ("aleatoric", ":")):
        l2, s2 = U.predictive_params(s["loc"], s["scale"], fam, kind)
        _, cen, _ = U.calibration_curve(U.pit(s["y"], l2, s2, fam), lv)
        ax[2].plot(lv, cen, "k", ls=ls, label=f"{kind} (ECE {U.ece(cen, lv):.3f})")
    titles = ["Central-interval coverage", "One-sided (PIT ECDF)", f"Predictive form, {_vza_label(a0)}"]
    for i, axi in enumerate(ax):
        axi.plot([0, 1], [0, 1], color="0.6", lw=0.8, zorder=0)
        axi.set(xlim=(0, 1), ylim=(0, 1), xlabel="Nominal probability", title=titles[i], aspect="equal")
    ax[0].set_ylabel("Observed frequency")
    # lower right stays clear for over-dispersed curves; well-calibrated curves sit on the diagonal
    ax[0].legend(ncol=1, frameon=False, fontsize=6, loc="lower right")
    ax[2].legend(frameon=False, fontsize=6, loc="lower right")
    save_fig(fig, fdir, "F1_reliability", fc)


def fig_pit(tabs, fam, fdir, fc):
    n = len(tabs)
    nc = min(n, 3)
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(WIDTH, 1.8 * nr), sharex=True, constrained_layout=True, squeeze=False)
    for axi, (a, s) in zip(axs.flat, sorted(tabs.items())):
        pv = U.pit(s["y"], s["loc"], s["scale"], fam)
        axi.hist(pv, bins=20, range=(0, 1), density=True, color="0.5")
        axi.axhline(1, color="k", lw=0.8)
        axi.set_title(_vza_label(a))
    for axi in axs.flat[n:]:
        axi.set_visible(False)
    for axi in axs[-1]:
        axi.set_xlabel("PIT")
    save_fig(fig, fdir, "F2_pit_histograms", fc)


def fig_spread_skill(tabs, colors, fdir, fc, n_bins):
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.4), constrained_layout=True)
    for i, (comp, title) in enumerate((("tot", "Total"), ("ale", "Aleatoric"), ("epi", "Epistemic"))):
        hi, lo = 0.0, np.inf
        for a, s in tabs.items():
            ss = U.spread_skill(s["mean"] - s["y"], s[comp], n_bins)
            ax[i].plot(ss["rmv"], ss["rmse"], "o-", ms=2.5, color=colors[a], label=_vza_label(a))
            hi = max(hi, ss["rmv"].max(), ss["rmse"].max())
            lo = min(lo, np.nanmin(np.r_[ss["rmv"], ss["rmse"]][np.r_[ss["rmv"], ss["rmse"]] > 0]))
        if hi / max(lo, 1e-12) > 100:  # spread spans decades: log axes keep ticks readable
            ax[i].set(xscale="log", yscale="log")
            ax[i].plot([lo, hi], [lo, hi], color="0.6", lw=0.8, zorder=0)
        else:
            ax[i].plot([0, hi], [0, hi], color="0.6", lw=0.8, zorder=0)
        ax[i].set(xlabel="RMV (predicted std) [COT]", title=f"{title} uncertainty")
    ax[0].set_ylabel("RMSE [COT]")
    ax[0].legend(frameon=False, fontsize=6)
    save_fig(fig, fdir, "F3_spread_skill", fc)


def fig_sparsification(tabs, focus, fdir, fc):
    use = [a for a in focus if a in tabs] or list(tabs)[:3]
    fig, ax = plt.subplots(1, len(use), figsize=(WIDTH, 2.2), constrained_layout=True, squeeze=False)
    for axi, a in zip(ax[0], use):
        s = tabs[a]
        err = s["mean"] - s["y"]
        for comp, ls in (("tot", "-"), ("ale", "--"), ("epi", ":")):
            sp = U.sparsification(err, s[comp])
            axi.plot(sp["fraction_removed"], sp["rmse_unc"] / sp["rmse_random"][0], "k", ls=ls,
                     label=f"{comp} (AUSE {sp['AUSE']:.2f})")
        axi.plot(sp["fraction_removed"], sp["rmse_oracle"] / sp["rmse_random"][0], color="C3", label="oracle")
        axi.axhline(1, color="0.6", lw=0.8)
        axi.set(xlabel="Fraction removed (most uncertain first)", title=_vza_label(a), ylim=(0, None))
        axi.legend(frameon=False, fontsize=6)
    ax[0, 0].set_ylabel("Normalised RMSE of retained pixels")
    save_fig(fig, fdir, "F4_sparsification", fc)


def fig_ood(tables, datasets, ref, focus, groups, max_px, rng, fdir, fc):
    use = [a for a in focus if any((d, a) in tables for d in datasets)]
    fig, ax = plt.subplots(2, len(use), figsize=(WIDTH, 4.0), constrained_layout=True, squeeze=False)
    for j, a in enumerate(use):
        names = [d for d in datasets if (d, a) in tables]
        data_t, med_a, med_e = [], [], []
        for d in names:
            s = subset(tables[(d, a)], "cloudy", min(max_px, 200_000), rng)
            data_t.append(np.log10(np.sqrt(s["tot"]) + 1e-6))
            med_a.append(np.median(np.sqrt(s["ale"])))
            med_e.append(np.median(np.sqrt(s["epi"])))
        parts = ax[0, j].violinplot(data_t, showmedians=True, widths=0.8)
        for pc, d in zip(parts["bodies"], names):
            pc.set_facecolor("C0" if d == ref else ("C1" if groups.get(d) == "OOD" else "0.6"))
            pc.set_alpha(0.6)
        ax[0, j].set_xticks(range(1, len(names) + 1), names, rotation=45, ha="right")
        ax[0, j].set(title=_vza_label(a), ylabel=r"log$_{10}$ total std" if j == 0 else "")
        i0 = names.index(ref) if ref in names else 0
        x = np.arange(len(names))
        ax[1, j].bar(x - 0.2, np.array(med_a) / med_a[i0], 0.4, label="aleatoric", color="C0")
        ax[1, j].bar(x + 0.2, np.array(med_e) / med_e[i0], 0.4, label="epistemic", color="C1")
        ax[1, j].axhline(1, color="k", lw=0.8)
        ax[1, j].set_xticks(x, names, rotation=45, ha="right")
        ax[1, j].set_ylabel(f"median std / {ref}" if j == 0 else "")
    ax[1, 0].legend(frameon=False)
    save_fig(fig, fdir, "F5_ood_uncertainty", fc)


def fig_stratified(t3, strat_vars, colors, fdir, fc):
    mets = [("RMSE", "RMSE [COT]"), ("cov80", "80% coverage"), ("std_med", "median total std"), ("slope", "slope")]
    fig, ax = plt.subplots(len(strat_vars), len(mets), figsize=(WIDTH, 1.6 * len(strat_vars)),
                           constrained_layout=True, squeeze=False)
    for i, (var, lab) in enumerate(strat_vars.items()):
        d = t3[t3.stratifier == var]
        for j, (m, ml) in enumerate(mets):
            for a, g in d.groupby("vza"):
                ax[i, j].plot(g["centre"], g[m], "o-", ms=2, color=colors.get(a, "k"), label=_vza_label(a))
            if m == "cov80":
                ax[i, j].axhline(0.8, color="0.6", lw=0.8)
            if m == "slope":
                ax[i, j].axhline(1.0, color="0.6", lw=0.8)
            ax[i, j].set_xlabel(lab)
            if i == 0:
                ax[i, j].set_title(ml)
    ax[0, 0].legend(frameon=False, fontsize=5)
    fig.get_layout_engine().set(h_pad=0.08, hspace=0.06)
    save_fig(fig, fdir, "F6_stratified", fc)


def fig_vs_vza(t1c, datasets, fdir, fc):
    fig, ax = plt.subplots(1, 4, figsize=(WIDTH, 2.0), constrained_layout=True)
    for k, d in enumerate(datasets):
        g = t1c[t1c.dataset == d].sort_values("vza")
        if not len(g):
            continue
        for axi, m in zip(ax, ("RMSE", "cov80", "ECE", "epi_frac")):
            axi.plot(g.vza, g[m], "o-", ms=2.5, color=f"C{k}", label=d)
    ax[1].axhline(0.8, color="0.6", lw=0.8)
    for axi, t in zip(ax, ("RMSE [COT]", "80% coverage", "ECE", "epistemic fraction")):
        axi.set(xlabel="VZA [deg]", title=t)
    ax[0].legend(frameon=False, fontsize=6)
    save_fig(fig, fdir, "F7_metrics_vs_vza", fc)


def fig_maps(rec, thr, window, fdir, fc):
    y = rec["tau_true"]
    mean, ale, epi = U.member_moments(rec["loc"], rec["scale"], rec["family"])
    pv = U.pit(y, rec["loc"], rec["scale"], rec["family"])
    hit = U.central_coverage(pv, 0.8).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mean - y) / np.sqrt(ale + epi)
    covered = (rec["scale"] > 0).all(-1)  # stitched pixels outside every trimmed tile
    y = np.where(covered, y, np.nan).astype(float)
    mean, ale, epi, hit, z = (np.where(covered, v, np.nan) for v in (mean, ale, epi, hit, z))
    if window:
        sl = (slice(*window[0]), slice(*window[1]))
        y, mean, ale, epi, hit, z = (v[sl] for v in (y, mean, ale, epi, hit, z))
    clear = (y < thr) & (mean < thr)
    hit[clear] = np.nan
    vmax = np.nanpercentile(y, 99.5)
    err = mean - y
    el = np.nanpercentile(np.abs(err), 99)
    panels = [(y, "True COT", "viridis", dict(vmin=0, vmax=vmax)),
              (mean, "Predicted COT", "viridis", dict(vmin=0, vmax=vmax)),
              (err, "Error", "RdBu_r", dict(vmin=-el, vmax=el)),
              (np.sqrt(ale), "Aleatoric std", "magma", dict(norm=LogNorm(1e-2, max(np.nanpercentile(np.sqrt(ale), 99.5), 0.1)))),
              (np.sqrt(epi), "Epistemic std", "magma", dict(norm=LogNorm(1e-3, max(np.nanpercentile(np.sqrt(epi), 99.5), 0.01)))),
              (z, "z = error / total std", "RdBu_r", dict(vmin=-3, vmax=3)),
              (hit, "80% interval hit (cloudy)", "RdYlGn", dict(vmin=0, vmax=1))]
    fig, ax = plt.subplots(2, 4, figsize=(WIDTH, 3.6), constrained_layout=True)
    for axi, (d, t, cm, kw) in zip(ax.flat, panels):
        im = axi.imshow(d.T, origin="lower", cmap=cm, interpolation="nearest", **kw)
        axi.set_title(t)
        axi.set_axis_off()
        fig.colorbar(im, ax=axi, shrink=0.8)
    ax.flat[-1].hist(z[np.isfinite(z) & ~clear].ravel(), bins=60, range=(-5, 5), density=True, color="0.5")
    zz = np.linspace(-5, 5, 200)
    ax.flat[-1].plot(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k", lw=0.8)
    ax.flat[-1].set(title="z (cloudy) vs N(0,1)", xlabel="z", ylabel="density")
    save_fig(fig, fdir, f"F8_maps_vza{float(rec['angle']):+.1f}", fc)


def fig_noise(t4, colors, fdir, fc):
    fig, ax = plt.subplots(1, 4, figsize=(WIDTH, 2.0), constrained_layout=True)
    for a, g in t4.groupby("vza"):
        for axi, m in zip(ax, ("RMSE", "std_ale", "std_epi", "cov80")):
            axi.plot(g.noise, g[m], "o-", ms=2.5, color=colors.get(a, "k"), label=_vza_label(a))
    for axi, t in zip(ax, ("RMSE", "aleatoric std", "epistemic std", "80% coverage")):
        axi.set(xlabel="relative radiance noise", title=t)
    ax[3].axhline(0.8, color="0.6", lw=0.8)
    ax[0].legend(frameon=False, fontsize=5)
    save_fig(fig, fdir, "F9_noise_robustness", fc)


def fig_err_vs_std(tabs, focus, fdir, fc):
    use = [a for a in focus if a in tabs] or list(tabs)[:3]
    fig, ax = plt.subplots(1, len(use), figsize=(WIDTH, 2.3), constrained_layout=True, squeeze=False)
    for axi, a in zip(ax[0], use):
        s = tabs[a]
        sd, ae = np.sqrt(s["tot"]), np.abs(s["mean"] - s["y"])
        lo, hi = np.percentile(sd, 0.5), np.percentile(sd, 99.5)
        bins = np.logspace(np.log10(max(lo, 1e-4)), np.log10(hi), 60)
        ebins = np.logspace(np.log10(max(np.percentile(ae, 1), 1e-5)), np.log10(np.percentile(ae, 99.9) + 1e-6), 60)
        axi.hist2d(sd, np.maximum(ae, ebins[0]), bins=[bins, ebins], cmap="Greys", norm=LogNorm())
        axi.plot(bins, bins, "C3", lw=0.8, label="|err| = std")
        axi.plot(bins, -np.log(0.2) * bins / np.sqrt(2), "C0--", lw=0.8, label="80% Laplace bound")
        axi.set(xscale="log", yscale="log", xlabel="predicted total std", title=_vza_label(a))
    ax[0, 0].set_ylabel("|error|")
    ax[0, 0].legend(frameon=False, fontsize=6)
    save_fig(fig, fdir, "F10_error_vs_std", fc)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-y", "--yaml", required=True)
    p.add_argument("--stage", choices=["predict", "analyze", "all"], default="all")
    p.add_argument("--overwrite", action="store_true", help="recompute cached predictions")
    args = p.parse_args()
    cfg = read_yaml(args.yaml)
    cfg.setdefault("cache_dir", os.path.join(cfg["output_dir"], "cache"))
    if args.stage in ("predict", "all"):
        stage_predict(cfg, args.overwrite)
    if args.stage in ("analyze", "all"):
        analyze(cfg)


if __name__ == "__main__":
    main()
