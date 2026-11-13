"""Single-cloud UQ case study: per-angle emulator UQ and its propagation to 3-D.

Reproduces the reconstruction geometry of ``analyze_mystic_scene.py`` exactly (cloud
selection, COM shift, cube padding, view generation, sinogram padding, per-slice SART with
the truth mask) and adds:

1. Per-angle (2-D) evaluation of the MIMO predictive distribution on the cloud views
   (maps, metrics table, reliability, cross-sections at ``yloc``, sinograms).
2. Monte Carlo propagation of the per-angle predictive distribution through SART
   (member draws + Laplace/Gaussian noise, optional spatially correlated copula noise).
3. 3-D evaluation of the propagated uncertainty (coverage, rank histogram, CRPS,
   spread-skill, vertical profiles, slice panels) against the true extinction and
   against the reconstruction from true COT (isolates the emulator part of the error).
4. Optional per-angle attribution of the 3-D variance.
4b. Optional visual-hull SART masks (``mask.mode``), including per-sample hulls that turn the
   support mask itself into an uncertain quantity (P(voxel in mask), IoU/precision/recall).
5. 3-D isosurfaces of the mean / quantile envelopes / std field, plus a NetCDF of all
   3-D fields for ParaView and an optional interactive plotly HTML.

Run from ``reconstruction/``::

    python cloud_case_study_uq.py -y config/case_study_uq.yaml

Inputs are the merged MIMO file written by ``emulator/mimo/evaluate_and_merge_mimo.py``
(``tau_pre``, ``tau_member_loc``, ``tau_member_scale``; dims x, y, vza[, member]), the
ground-truth COT file and the LES file used in the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy import ndimage, special
from skimage import measure

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_HERE, os.path.join(_REPO, "emulator", "mimo", "pyct")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import projection_utils as proj_utils  # noqa: E402
from custom_sart import custom_radon, iradon_sart_custom  # noqa: E402
from analyze_mystic_scene import filter_labels, generate_cloud_file_dataset, get_beta_ext  # noqa: E402
import uq_metrics as U  # noqa: E402

STYLE = {
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 150,
    "lines.linewidth": 1.1, "pdf.fonttype": 42,
}
WIDTH = 7.0
EXT_UNIT = r"km$^{-1}$"

DEFAULTS = {
    "time": 30.0,
    "data_path": "../data",
    "les_file": "ground_truth/wc_les_RICO_40m_80kmx80km_T_qc_{time:.1f}h.nc",
    "truth_cot_file": "ground_truth/MISR_40m_80x80km_{time:.1f}h_optical_thickness_toa.nc",
    "truth_cot_var": "tau",
    "predicted_file": "predicted_data/MISR_40m_80x80km_{time:.1f}h_optical_thickness_predicted_MIMO_40m_toa.nc",
    "output_dir": "results/wc_les_RICO_40m_80kmx80km_T_qc_{time:.1f}h/uq_case_study",
    "cloud_index": 54,          # index into the filtered regionprops list (paper: props[54:55])
    "yloc": None,               # None -> centre slice, which is what analyze_mystic_scene.py runs
    "lwc_threshold": 0.2,
    "min_diameter_km": 0.5,
    "reff": 10.0, "wavelength": 0.67,
    "com_offset": -120.0,       # cloud_COM offset passed to generate_views (as in the paper code)
    "view_offsets": [-0.05, -0.04, -0.03, -0.01, 0, 0, 0, 0, 0],
    "cot_threshold": 1.0,       # cloudy footprint for per-angle metrics
    "sart": {"niter": 100, "relaxation": 0.15},
    "uq": {
        "n_samples": 64,
        "member_pairing": "independent",  # independent | aligned (members of different angle models are unrelated)
        "noise": "correlated",           # correlated | independent | none
        "correlation_length_px": 2.0,     # Gaussian-filter sigma of the copula field, in 40 m pixels
        "levels": [0.5, 0.8, 0.95],
        "seed": 0,
        "n_jobs": None,                   # None -> CPUs allocated to the job (SLURM_CPUS_PER_TASK)
    },
    "angle_attribution": {"enabled": True, "n_samples": 16},
    # SART support mask. truth: true-extinction support (paper / analyze_mystic_scene.py);
    # hull_mean: visual hull carved from the predicted mean COT views; hull_sample: one hull per
    # Monte Carlo sample (mask uncertainty propagates to 3-D); hull_truecot: hull from the true COT
    # views (error of the carving itself, without emulator error).
    "mask": {"mode": "truth", "cot_threshold": 1.0, "min_views": None, "dilation": 0,
             "geometry_check": True},
    "surface": {"iso_level": None, "iso_quantile": 0.2, "std_iso_quantile": 0.8,
                "step_size": 1, "elev": 22, "azim": -58, "plotly_html": True},
    "figure_formats": ["png", "pdf"],
}


# --------------------------------------------------------------------------- #
# Config / IO
# --------------------------------------------------------------------------- #
def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def load_config(path: Optional[str]) -> dict:
    cfg = DEFAULTS
    if path:
        with open(path) as f:
            cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    for k in ("les_file", "truth_cot_file", "predicted_file", "output_dir"):
        cfg[k] = cfg[k].format(time=cfg["time"])
    return cfg


def save_fig(fig, out_dir, name, formats):
    os.makedirs(out_dir, exist_ok=True)
    for f in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{f}"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Geometry (identical to analyze_mystic_scene.py)
# --------------------------------------------------------------------------- #
def select_cloud(cfg) -> dict:
    les_path = os.path.join(cfg["data_path"], cfg["les_file"])
    with xr.open_dataset(les_path) as les_:
        les = generate_cloud_file_dataset(les_.lwc.transpose("ny", "nx", "nz"), 10, les_.z, les_.dx, pad=0)
        les = les.load()
    lwc = les.lwc
    dx = float(les.delx.data)
    beta_ext = get_beta_ext(reff=cfg["reff"], wvl=cfg["wavelength"])
    binary_image = (lwc.sum(axis=2) > cfg["lwc_threshold"]).data
    props_ = measure.regionprops(measure.label(binary_image))
    _, props = filter_labels(binary_image, props_, cfg["min_diameter_km"] / dx)
    if cfg["cloud_index"] >= len(props):
        raise IndexError(f"cloud_index {cfg['cloud_index']} >= {len(props)} filtered clouds")
    region = props[cfg["cloud_index"]]
    slx, sly = region.slice
    yloc = cfg["yloc"]
    if yloc is None:
        yloc = int(np.ceil((sly.stop - sly.start) / 2.0))
    cloud_lwc = lwc[slx, sly, :]
    cloud_ext = cloud_lwc * beta_ext
    levels = cloud_lwc.z.where(np.sum(cloud_lwc, axis=(0, 1)) > 0).dropna("z").data
    cbh, cth = levels.min(), levels.max()
    cloud_COM = (cbh + cth) / 2

    shift_z = int(len(cloud_ext.z.data) // 2 - np.argmin(abs(cloud_ext.z.data - cloud_COM)))
    tc = cloud_ext.shift(z=shift_z, fill_value=0)
    nx, ny, nz = tc.shape
    M = max(nx, ny, nz)
    ox, oz = int((M - nx) / 2), int((M - nz) / 2)
    pad_x, pad_z = (ox, M - nx - ox), (oz, M - nz - oz)
    tc = tc.pad({"x": pad_x, "z": pad_z}, constant_values=0)
    data_test = np.ascontiguousarray(tc.data.T)  # (z, y, x) with z and x padded to M
    return {"region": region, "label": int(region.label), "slx": slx, "sly": sly, "yloc": yloc,
            "dx": dx, "cloud_COM": float(cloud_COM), "cbh": float(cbh), "cth": float(cth),
            "data_test": data_test, "M": M, "pad_z": pad_z, "beta_ext": float(beta_ext),
            "z_levels": cloud_ext.z.data}


def make_views(field: xr.DataArray, geo: dict, angles, cfg) -> np.ndarray:
    """generate_views() for one (x, y, vza) field -> (x_view, y, angle) numpy, angles ascending."""
    v = proj_utils.generate_views(field, geo["slx"], geo["sly"], geo["dx"],
                                  cloud_COM=geo["cloud_COM"] + cfg["com_offset"], angles=angles,
                                  offset=cfg["view_offsets"])
    return np.asarray(v.data, dtype=np.float64)


def to_sinogram(views: np.ndarray, M: int) -> np.ndarray:
    """views[..., ::-1] padded in x to M rows (the paper code pads x only)."""
    s = views[:, :, ::-1]
    nx = s.shape[0]
    ox = int((M - nx) / 2)
    return np.pad(s, ((ox, M - nx - ox), (0, 0), (0, 0)))


# --------------------------------------------------------------------------- #
# SART (module-level for process pools)
# --------------------------------------------------------------------------- #
_W: dict = {}


def _allocated_cpus() -> int:
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.isdigit():
        return int(v)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _init_worker(mask, angles, dx, niter, relaxation):
    _W.update(mask=mask, angles=angles, dx=dx, niter=niter, relaxation=relaxation)


def sart_volume(sino, mask, angles, dx, niter, relaxation):
    """Per-y-slice SART exactly as in analyze_mystic_scene.py (mask, clip, warm start)."""
    rec = np.zeros(mask.shape, dtype=np.float64)
    for i in range(sino.shape[1]):
        m = mask[:, i]
        prior = np.zeros(m.shape, dtype=np.float64)
        sl = prior
        for _ in range(niter):
            sl = iradon_sart_custom(np.ascontiguousarray(sino[:, i, :]), theta=angles, image=prior,
                                    resolution=dx, relaxation=relaxation)
            sl[m] = 0
            sl[sl < 0] = 0
            prior = sl
        rec[:, i] = sl
    return rec


def _sart_job(job):
    sino, mask = job
    return sart_volume(sino, _W["mask"] if mask is None else mask, _W["angles"], _W["dx"], _W["niter"],
                       _W["relaxation"])


def run_sart_batch(sinos: List[np.ndarray], mask, angles, cfg, label="") -> np.ndarray:
    """mask (True = outside cloud): one array shared by all jobs, or a list with one per sinogram."""
    n_jobs = cfg["uq"]["n_jobs"] or _allocated_cpus()
    n_jobs = max(1, min(n_jobs, len(sinos)))
    per_job = isinstance(mask, (list, tuple))
    if per_job and len(mask) != len(sinos):
        raise ValueError("need one mask per sinogram")
    jobs = [(sn, mask[k] if per_job else None) for k, sn in enumerate(sinos)]
    args = (None if per_job else mask, np.asarray(angles, float), cfg["_dx"], cfg["sart"]["niter"],
            cfg["sart"]["relaxation"])
    t0 = _time.time()
    if n_jobs == 1:
        _init_worker(*args)
        out = [_sart_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=args) as ex:
            out = list(ex.map(_sart_job, jobs, chunksize=max(1, len(jobs) // (4 * n_jobs))))
    print(f"[sart] {label}: {len(sinos)} reconstructions in {_time.time() - t0:.1f}s ({n_jobs} workers)")
    return np.stack(out)


# --------------------------------------------------------------------------- #
# Visual-hull (space-carving) SART masks
# --------------------------------------------------------------------------- #
def detector_coords(M: int, angles, sign: float = 1.0) -> np.ndarray:
    """Detector row t[a, z, x] of voxel (z, x) of an M x M slice (rows z, cols x) per angle.

    t = C + cos(theta) (X - C) - sign * sin(theta) (Z - C), C = M // 2 (custom_radon centre).
    """
    C = M // 2
    Z, X = np.meshgrid(np.arange(M, dtype=float), np.arange(M, dtype=float), indexing="ij")
    th = np.deg2rad(np.asarray(angles, float))[:, None, None]
    return C + np.cos(th) * (X - C) - sign * np.sin(th) * (Z - C)


def geometry_sign(M: int, angles, dx: float) -> float:
    """Choose the sign of the z term so detector_coords matches custom_radon (point-voxel test)."""
    C = M // 2
    img = np.zeros((M, M))
    z0, x0 = C + max(2, M // 6), C - max(1, M // 8)
    img[z0, x0] = 1.0
    sino = np.asarray(custom_radon(img, np.asarray(angles, float), dx))
    if sino.shape[0] != M:
        sino = sino.T
    w = np.clip(sino, 0, None)
    peak = (w * np.arange(M)[:, None]).sum(0) / np.maximum(w.sum(0), 1e-12)
    err = {sg: float(np.abs(detector_coords(M, angles, sg)[:, z0, x0] - peak).max()) for sg in (1.0, -1.0)}
    best = min(err, key=err.get)
    print(f"[mask] geometry check vs custom_radon: max |dt| = {err[best]:.2f} px with sign {best:+.0f} "
          f"(other sign {err[-best]:.2f} px)")
    if err[best] > 1.5:
        print("[warn] visual-hull geometry does not match custom_radon within 1.5 px; inspect C16")
    return best


def visual_hull(sino_cot: np.ndarray, tcoords: np.ndarray, thr: float, min_views=None, dilation: int = 0):
    """Space carving from COT sinograms.

    sino_cot (M, ny, A): COT per detector row / y / angle (to_sinogram layout); tcoords (A, M, M).
    A voxel (z, y, x) is kept if its projection is cloudy (COT >= thr) in >= min_views views
    (default: all views). Returns bool (z, y, x), True inside the hull.
    """
    M, ny, A = sino_cot.shape
    need = A if min_views is None else int(min_views)
    cloudy = sino_cot >= thr
    ti = np.rint(tcoords).astype(int)
    inside = (ti >= 0) & (ti < M)
    tic = np.clip(ti, 0, M - 1)
    count = np.zeros((M, M, ny), np.int16)          # (z, x, y)
    for a in range(A):
        count += (cloudy[tic[a], :, a] & inside[a][..., None])
    hull = np.transpose(count >= need, (0, 2, 1))   # (z, y, x)
    if dilation:
        hull = ndimage.binary_dilation(hull, iterations=int(dilation))
    return hull


def mask_scores(hull: np.ndarray, support: np.ndarray) -> Dict[str, float]:
    tp = float((hull & support).sum()); fp = float((hull & ~support).sum()); fn = float((~hull & support).sum())
    return {"IoU": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "N_hull": int(hull.sum()), "N_truth": int(support.sum())}


def plot_masks(support, hull, p_in, geo, out_fig, fmts, iou_samples=None):
    """C16: x-z slice at yloc: true support, hull overlap, P(voxel in mask) [+ per-sample IoU]."""
    yl = geo["yloc"]
    sl = bbox(support | hull | (p_in > 0), pad=3)
    cut = lambda v: v[sl[0], yl, sl[2]]
    n = 3 if iou_samples is None else 4
    fig, ax = plt.subplots(1, n, figsize=(WIDTH, 2.1), constrained_layout=True)
    t, h = cut(support), cut(hull)
    rgb = np.ones(t.shape + (3,))
    rgb[t & h] = (0.35, 0.35, 0.35); rgb[t & ~h] = (0.85, 0.2, 0.2); rgb[~t & h] = (0.25, 0.5, 0.9)
    ax[0].imshow(t, origin="lower", cmap="Greys", interpolation="nearest"); ax[0].set_title("True support")
    ax[1].imshow(rgb, origin="lower", interpolation="nearest")
    ax[1].set_title("Hull: grey hit, red missed,\nblue extra", fontsize=7)
    im = ax[2].imshow(cut(p_in), origin="lower", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    if t.any():
        ax[2].contour(t.astype(float), levels=[0.5], colors="w", linewidths=0.6)
    ax[2].set_title("P(voxel in mask)")
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    for axi in ax[:3]:
        axi.set(xlabel="x [px]", ylabel="z [px]")
    if iou_samples is not None:
        ax[3].hist(iou_samples, bins=12, color="0.5")
        ax[3].set(xlabel="IoU per sample", ylabel="count", title="Sample-hull IoU")
    save_fig(fig, out_fig, "C16_mask_slices", fmts)


def plot_mask_profiles(support, hull, p_in, z_km, out_fig, fmts):
    """C17: mask size and quality by height."""
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH * 0.75, 2.3), constrained_layout=True)
    ax[0].plot(support.sum((1, 2)), z_km, "k", label="true support")
    ax[0].plot(hull.sum((1, 2)), z_km, "C0", label="hull")
    ax[0].plot(p_in.sum((1, 2)), z_km, "C1--", label="E[hull]")
    ax[0].set(xlabel="voxels per level", ylabel="z [km]", title="Mask size")
    ax[0].legend(frameon=False, fontsize=6)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = (p_in * support).sum((1, 2)) / support.sum((1, 2))
        ext = (p_in * ~support).sum((1, 2)) / p_in.sum((1, 2))
    ax[1].plot(rec, z_km, "C2", label="recall")
    ax[1].plot(ext, z_km, "C3", label="false-volume share")
    ax[1].set(xlabel="fraction", xlim=(0, 1), title="Mask quality")
    ax[1].legend(frameon=False, fontsize=6)
    save_fig(fig, out_fig, "C17_mask_profiles", fmts)


# --------------------------------------------------------------------------- #
# Sampling in view space
# --------------------------------------------------------------------------- #
def _std_noise(shape, family, corr_len, rng, kind):
    if kind == "none":
        return np.zeros(shape)
    g = rng.standard_normal(shape)
    if kind == "correlated" and corr_len > 0:
        g = ndimage.gaussian_filter(g, sigma=(corr_len, corr_len), mode="wrap")
        g = (g - g.mean()) / max(g.std(), 1e-12)
    if family == "gaussian":
        return g
    u = special.ndtr(g) - 0.5  # Gaussian copula -> standard Laplace(0, 1)
    return -np.sign(u) * np.log1p(-2.0 * np.abs(u) + 1e-15)


def member_draws(K, A, S, pairing, rng):
    """[K, A] member indices, stratified so each member is used ~K/S times per angle."""
    if S == 1:
        return np.zeros((K, A), int)
    base = np.arange(K) % S
    if pairing == "aligned":
        return np.repeat(rng.permutation(base)[:, None], A, 1)
    return np.stack([rng.permutation(base) for _ in range(A)], 1)


def sample_views(loc_v, scale_v, draws, family, cfg, rng, angle_mask=None, mean_v=None):
    """Return (with_noise, loc_only) view samples [K, x, y, A].

    angle_mask (bool[A]): perturb only these angles; others are fixed at mean_v.
    """
    K, A = draws.shape
    noisy, loc_only = [], []
    q = cfg["uq"]
    for k in range(K):
        vl = np.empty(loc_v.shape[:3])
        vn = np.empty(loc_v.shape[:3])
        for a in range(A):
            if angle_mask is not None and not angle_mask[a]:
                vl[..., a] = vn[..., a] = mean_v[..., a]
                continue
            m = draws[k, a]
            mu, b = loc_v[..., a, m], scale_v[..., a, m]
            eps = _std_noise(mu.shape, family, q["correlation_length_px"], rng, q["noise"])
            vl[..., a] = mu
            vn[..., a] = mu + b * eps
        noisy.append(np.clip(vn, 0, None))
        loc_only.append(np.clip(vl, 0, None))
    return np.stack(noisy), np.stack(loc_only)


# --------------------------------------------------------------------------- #
# Ensemble statistics
# --------------------------------------------------------------------------- #
def ens_crps(samples, y):
    """Fair ensemble CRPS per element; samples [K, ...]."""
    K = samples.shape[0]
    t1 = np.abs(samples - y[None]).mean(0)
    xs = np.sort(samples, 0)
    w = (2 * np.arange(K) - K + 1).reshape((K,) + (1,) * y.ndim)
    t2 = 2.0 * (w * xs).sum(0) / (K * (K - 1))
    return t1 - 0.5 * t2


def rank_of(samples, y, rng):
    below = (samples < y[None]).sum(0)
    ties = (samples == y[None]).sum(0)
    return below + np.floor(rng.random(y.shape) * (ties + 1)).astype(int)


def ens_coverage(samples, y, levels):
    out = {}
    for a in levels:
        lo, hi = np.quantile(samples, [(1 - a) / 2, (1 + a) / 2], axis=0)
        out[a] = (y >= lo) & (y <= hi)
    return out


# --------------------------------------------------------------------------- #
# Per-angle (2-D) analysis
# --------------------------------------------------------------------------- #
def per_angle_analysis(V, angles, cfg, out_fig, fmts, geo):
    fam = V["family"]
    thr = cfg["cot_threshold"]
    rows, curves = [], {}
    lv_curve = np.linspace(0.05, 0.95, 19)
    for a, ang in enumerate(angles):
        y = V["truth"][..., a]
        loc, sc = V["loc"][..., a, :], np.maximum(V["scale"][..., a, :], 1e-6)
        mean = loc.mean(-1)
        fp = np.isfinite(y) & ((y >= thr) | (mean >= thr))
        if fp.sum() < 10:
            continue
        m = U.full_summary(y[fp], loc[fp], sc[fp], fam, curve_levels=lv_curve)
        m.update({"vza": float(ang), "N": int(fp.sum())})
        rows.append(m)
        pv = U.pit(y[fp], loc[fp], sc[fp], fam)
        curves[float(ang)] = (U.calibration_curve(pv, lv_curve), pv)
    tab = pd.DataFrame(rows)
    keep = ["vza", "N", "RMSE", "MAE", "bias", "r", "slope", "std_ale", "std_epi", "std_tot", "epi_frac",
            "cov50", "cov80", "cov95", "ECE", "CRPS", "NLL", "ENCE", "spearman_std_abs_err", "width80"]
    tab = tab[[c for c in keep if c in tab]]

    # P1: per-angle maps
    mean_all = V["loc"].mean(-1)
    _, ale, epi = U.member_moments(V["loc"], np.maximum(V["scale"], 1e-6), fam)
    tot = ale + epi
    err = mean_all - V["truth"]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = err / np.sqrt(tot)
    pv_all = U.pit(V["truth"], V["loc"], np.maximum(V["scale"], 1e-6), fam)
    hit = U.central_coverage(pv_all, 0.8).astype(float)
    hit[(V["truth"] < thr) & (mean_all < thr)] = np.nan
    vmax = np.nanpercentile(V["truth"], 99.5)
    el = np.nanpercentile(np.abs(err), 99)
    smax = np.nanpercentile(np.sqrt(tot), 99.5)
    spec = [("True COT", V["truth"], "viridis", Normalize(0, vmax)),
            ("Mean COT", mean_all, "viridis", Normalize(0, vmax)),
            ("Error", err, "RdBu_r", Normalize(-el, el)),
            (r"$\sigma_\mathrm{ale}$", np.sqrt(ale), "magma", Normalize(0, smax)),
            (r"$\sigma_\mathrm{epi}$", np.sqrt(epi), "magma", Normalize(0, smax)),
            (r"$\sigma_\mathrm{tot}$", np.sqrt(tot), "magma", Normalize(0, smax)),
            ("z", z, "RdBu_r", Normalize(-3, 3)),
            ("80% hit", hit, "RdYlGn", Normalize(0, 1))]
    A = len(angles)
    fig, ax = plt.subplots(len(spec), A, figsize=(WIDTH, 0.95 * len(spec) * V["truth"].shape[1] /
                                                   V["truth"].shape[0] * 1.15 + 0.6),
                           squeeze=False, constrained_layout=True)
    for r, (name, d, cm, norm) in enumerate(spec):
        for a in range(A):
            im = ax[r, a].imshow(d[..., a].T, origin="lower", cmap=cm, norm=norm, interpolation="nearest")
            ax[r, a].set_axis_off()
            if r == 0:
                ax[r, a].set_title(f"{angles[a]:+.1f}°")
        cb = fig.colorbar(im, ax=ax[r, :].tolist(), shrink=0.9, pad=0.01, aspect=8)
        cb.set_label(name)
    save_fig(fig, out_fig, "C1_per_angle_maps", fmts)

    # P2: metrics vs VZA
    fig, ax = plt.subplots(1, 4, figsize=(WIDTH, 1.9), constrained_layout=True)
    ax[0].plot(tab.vza, tab.RMSE, "o-k", ms=3, label="RMSE")
    ax[0].plot(tab.vza, tab.std_tot, "s--", color="C3", ms=3, label=r"$\sigma_\mathrm{tot}$")
    ax[0].plot(tab.vza, tab.std_epi, "^:", color="C0", ms=3, label=r"$\sigma_\mathrm{epi}$")
    ax[0].set(ylabel="COT", title="Error vs spread"); ax[0].legend(frameon=False, fontsize=6)
    for c, l, mk in (("cov50", 0.5, "o"), ("cov80", 0.8, "s"), ("cov95", 0.95, "^")):
        if c in tab:
            ax[1].plot(tab.vza, tab[c], mk + "-", ms=3, label=f"{int(l * 100)}%")
            ax[1].axhline(l, color="0.7", lw=0.7)
    ax[1].set(title="Coverage", ylim=(0, 1.02)); ax[1].legend(frameon=False, fontsize=6)
    ax[2].plot(tab.vza, tab.ECE, "o-k", ms=3); ax[2].set(title="ECE")
    ax[3].plot(tab.vza, tab.CRPS, "o-k", ms=3); ax[3].set(title="CRPS [COT]")
    for axi in ax:
        axi.set_xlabel("VZA [deg]")
    save_fig(fig, out_fig, "C2_per_angle_metrics", fmts)

    # P3: reliability per angle + pooled PIT histogram
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH * 0.66, 2.3), constrained_layout=True)
    allp = []
    for i, (ang, ((lv, cen, _), pv)) in enumerate(sorted(curves.items())):
        ax[0].plot(lv, cen, color=cmap(i / max(len(curves) - 1, 1)), label=f"{ang:+.0f}°")
        allp.append(pv)
    ax[0].plot([0, 1], [0, 1], color="0.6", lw=0.8)
    ax[0].set(xlabel="Nominal", ylabel="Observed", title="Central-interval coverage", xlim=(0, 1), ylim=(0, 1))
    ax[0].legend(frameon=False, fontsize=5, ncol=2, loc="lower right")
    if allp:
        ax[1].hist(np.concatenate(allp), bins=20, range=(0, 1), density=True, color="0.5")
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set(xlabel="PIT", title="PIT (all angles, footprint)")
    save_fig(fig, out_fig, "C3_per_angle_reliability", fmts)

    # P4: cross-sections at yloc with predictive bands
    yl = geo["yloc"]
    levels = sorted(cfg["uq"]["levels"])[::-1]
    nr = int(np.ceil(A / 3))
    fig, ax = plt.subplots(nr, 3, figsize=(WIDTH, 1.6 * nr), sharex=True, constrained_layout=True, squeeze=False)
    xk = np.arange(V["truth"].shape[0]) * geo["dx"]
    for a in range(A):
        axi = ax.flat[a]
        loc, sc = V["loc"][:, yl, a, :], np.maximum(V["scale"][:, yl, a, :], 1e-6)
        for j, lev in enumerate(levels):
            lo = U.quantile((1 - lev) / 2, loc, sc, fam)
            hi = U.quantile((1 + lev) / 2, loc, sc, fam)
            axi.fill_between(xk, lo, hi, color="C0", alpha=0.18 + 0.17 * j, lw=0, label=f"{int(lev * 100)}%")
        axi.plot(xk, loc.mean(-1), color="C0", lw=1.0, label="mean")
        axi.plot(xk, V["truth"][:, yl, a], "k", lw=0.9, label="truth")
        axi.set_title(f"{angles[a]:+.1f}°")
    for axi in ax.flat[A:]:
        axi.set_visible(False)
    for axi in ax[-1]:
        axi.set_xlabel("x [km]")
    for axi in ax[:, 0]:
        axi.set_ylabel("COT")
    ax.flat[0].legend(frameon=False, fontsize=5, loc="upper left")
    save_fig(fig, out_fig, "C4_cross_sections_yloc", fmts)

    # P5: sinogram (x vs angle) at yloc: the exact SART input of this slice
    sl = lambda d: d[:, yl, :]
    sp = [("True COT", sl(V["truth"]), "viridis", Normalize(0, vmax)),
          ("Mean COT", sl(mean_all), "viridis", Normalize(0, vmax)),
          ("Error", sl(err), "RdBu_r", Normalize(-el, el)),
          (r"$\sigma_\mathrm{tot}$", sl(np.sqrt(tot)), "magma", Normalize(0, smax)),
          ("z", sl(z), "RdBu_r", Normalize(-3, 3))]
    fig, ax = plt.subplots(1, len(sp), figsize=(WIDTH, 2.0), sharey=True, constrained_layout=True)
    ext = [-0.5, A - 0.5, 0, xk[-1]]  # one column per camera (MISR angles are not equally spaced)
    for axi, (t, d, cm, norm) in zip(ax, sp):
        im = axi.imshow(d, origin="lower", aspect="auto", cmap=cm, norm=norm, extent=ext, interpolation="nearest")
        axi.set(title=t, xlabel="VZA [deg]")
        axi.set_xticks(np.arange(A)[::2], [f"{v:.0f}" for v in angles[::2]])
        fig.colorbar(im, ax=axi, shrink=0.85, pad=0.02)
    ax[0].set_ylabel("x [km]")
    save_fig(fig, out_fig, "C5_sinogram_yloc", fmts)
    return tab


# --------------------------------------------------------------------------- #
# 3-D analysis
# --------------------------------------------------------------------------- #
def bbox(mask, pad=3):
    idx = [np.where(mask.any(axis=tuple(j for j in range(mask.ndim) if j != i)))[0] for i in range(mask.ndim)]
    return tuple(slice(max(ix[0] - pad, 0), min(ix[-1] + pad + 1, n)) for ix, n in zip(idx, mask.shape))


def volume_metrics(truth, samples, mask, levels, rng, name):
    y = truth[mask]
    X = samples[:, mask]
    mean, std = X.mean(0), X.std(0, ddof=1)
    err = mean - y
    cov = ens_coverage(X, y, levels)
    lv = np.linspace(0.05, 0.95, 19)
    cc = np.array([ens_coverage(X, y, [a])[a].mean() for a in lv])
    ss = U.spread_skill(err, std ** 2, n_bins=10)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    row = {"reference": name, "N_voxels": int(mask.sum()), "RMSE": rmse, "bias": float(err.mean()),
           "r": float(np.corrcoef(y, mean)[0, 1]) if y.std() > 0 else np.nan,
           "mean_std": float(np.sqrt(np.mean(std ** 2))), "spread_skill_ratio": float(np.sqrt(np.mean(std ** 2)) / max(rmse, 1e-12)),
           "CRPS": float(ens_crps(X, y).mean()), "ECE": U.ece(cc, lv), "ENCE": ss["ENCE"],
           "spearman_std_abs_err": ss["spearman"]}
    for a in levels:
        row[f"cov{int(round(a * 100))}"] = float(cov[a].mean())
    ranks = rank_of(X, y, rng)
    return row, {"curve": (lv, cc), "ranks": ranks, "K": X.shape[0], "ss": ss}


def plot_3d_calibration(diag, out_fig, fmts):
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.2), constrained_layout=True)
    for i, (name, d) in enumerate(diag.items()):
        lv, cc = d["curve"]
        ax[0].plot(lv, cc, "o-", ms=2.5, color=f"C{i}", label=name)
        ax[1].hist(d["ranks"], bins=np.arange(d["K"] + 2) - 0.5, density=True, histtype="step",
                   color=f"C{i}", label=name)
        ss = d["ss"]
        ax[2].plot(ss["rmv"], ss["rmse"], "o-", ms=2.5, color=f"C{i}", label=name)
    ax[0].plot([0, 1], [0, 1], color="0.6", lw=0.8)
    ax[0].set(xlabel="Nominal", ylabel="Observed", title="Voxel coverage", xlim=(0, 1), ylim=(0, 1))
    ax[0].legend(frameon=False, fontsize=6, loc="upper left")
    K = next(iter(diag.values()))["K"]
    ax[1].axhline(1 / (K + 1), color="k", lw=0.8)
    ax[1].set(xlabel="rank of reference in ensemble", title="Rank histogram")
    hi = max(max(d["ss"]["rmv"].max(), d["ss"]["rmse"].max()) for d in diag.values())
    ax[2].plot([0, hi], [0, hi], color="0.6", lw=0.8)
    ax[2].set(xlabel=f"RMV ensemble std [{EXT_UNIT}]", ylabel=f"RMSE [{EXT_UNIT}]", title="Spread-skill")
    save_fig(fig, out_fig, "C8_voxel_calibration", fmts)


def plot_profiles(F, mask, z_km, levels, out_fig, fmts, yloc):
    """Mean extinction over cloudy voxels per level (full cloud and slice yloc)."""
    def prof(vol, m):
        num = (vol * m).sum(axis=tuple(range(1, vol.ndim)))
        den = m.sum(axis=tuple(range(1, m.ndim)))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH * 0.66, 2.6), sharey=True, constrained_layout=True)
    for axi, (title, sel) in zip(ax, (("Full cloud", np.s_[:, :, :]), (f"Slice y={yloc}", np.s_[:, yloc:yloc + 1, :]))):
        m = mask[sel]
        sp = np.stack([prof(s[sel], m) for s in F["samples"]])
        keep = np.isfinite(sp).all(0)  # levels with cloudy voxels
        zk, sp = z_km[keep], sp[:, keep]
        for j, lev in enumerate(sorted(levels)[::-1]):
            lo, hi = np.quantile(sp, [(1 - lev) / 2, (1 + lev) / 2], axis=0)
            axi.fill_betweenx(zk, lo, hi, color="C0", alpha=0.18 + 0.17 * j, lw=0, label=f"{int(lev * 100)}%")
        axi.plot(sp.mean(0), zk, "C0", label="ensemble mean")
        axi.plot(prof(F["rec_det"][sel], m)[keep], zk, "C1--", label="SART(mean COT)")
        axi.plot(prof(F["rec_truecot"][sel], m)[keep], zk, "C2:", label="SART(true COT)")
        axi.plot(prof(F["truth"][sel], m)[keep], zk, "k", label="truth")
        axi.set(title=title, xlabel=f"Extinction [{EXT_UNIT}]")
    ax[0].set_ylabel("z rel. COM [km]")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, loc="outside lower center", ncol=4)
    save_fig(fig, out_fig, "C7_vertical_profiles", fmts)


def plot_slices(F, mask, geo, out_fig, fmts):
    """x-z slice panels at yloc (axis0 = z, axis1 = x)."""
    yl, dx = geo["yloc"], geo["dx"]
    m2 = mask[:, yl, :]
    if not m2.any():
        return
    bz, bx = bbox(m2)
    cut = lambda v: v[:, yl, :][bz, bx]
    mk = lambda v: np.ma.masked_array(cut(v), ~cut(mask))
    vmax = np.nanmax(cut(F["truth"]))
    err = F["ens_mean"] - F["truth"]
    el = np.nanmax(np.abs(cut(err) * cut(mask))) or 1
    smax = np.nanmax(cut(F["ens_std"]) * cut(mask)) or 1
    with np.errstate(divide="ignore", invalid="ignore"):
        zsc = err / F["ens_std"]
    panels = [("Truth", F["truth"], "viridis", Normalize(0, vmax)),
              ("SART(mean COT)", F["rec_det"], "viridis", Normalize(0, vmax)),
              ("Ensemble mean", F["ens_mean"], "viridis", Normalize(0, vmax)),
              ("Error (ens. mean)", err, "RdBu_r", Normalize(-el, el)),
              (r"$\sigma_\mathrm{tot}$", F["ens_std"], "magma", Normalize(0, smax)),
              (r"$\sigma_\mathrm{epi}$ (member)", F["std_epi"], "magma", Normalize(0, smax)),
              (r"$\sigma_\mathrm{ale}$ (residual)", F["std_ale"], "magma", Normalize(0, smax)),
              ("z = error / σ", zsc, "RdBu_r", Normalize(-3, 3)),
              ("80% hit (truth)", F["hit80"].astype(float), "RdYlGn", Normalize(0, 1))]
    nzc, nxc = cut(F["truth"]).shape
    ext = [0, nxc * dx, -nzc * dx / 2, nzc * dx / 2]
    fig, ax = plt.subplots(3, 3, figsize=(WIDTH, WIDTH * 0.9 * nzc / max(nxc, 1) + 0.8), sharex=True, sharey=True,
                           constrained_layout=True)
    for axi, (t, v, cm, norm) in zip(ax.flat, panels):
        im = axi.imshow(mk(v), origin="lower", cmap=cm, norm=norm, extent=ext, interpolation="nearest")
        axi.contour(np.linspace(ext[0], ext[1], nxc), np.linspace(ext[2], ext[3], nzc), cut(mask).astype(float),
                    levels=[0.5], colors="k", linewidths=0.5)
        axi.set_title(t)
        axi.set_aspect("equal")
        fig.colorbar(im, ax=axi, shrink=0.8, pad=0.02)
    for axi in ax[-1]:
        axi.set_xlabel("x [km]")
    for axi in ax[:, 0]:
        axi.set_ylabel("z [km]")
    save_fig(fig, out_fig, "C6_xz_slice_yloc", fmts)


def plot_attribution(contrib, angles, F, mask, geo, out_fig, fmts):
    tot = np.array([c[mask].mean() for c in contrib])
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH * 0.66, 2.1), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.2, 1]})
    ax[0].bar(np.arange(len(angles)), tot / max(F["ens_std"][mask].__pow__(2).mean(), 1e-12), color="0.4")
    ax[0].set_xticks(np.arange(len(angles)), [f"{a:+.0f}" for a in angles], rotation=45)
    ax[0].set(xlabel="perturbed VZA [deg]", ylabel="share of voxel variance", title="Per-angle attribution")
    ax[0].text(0.03, 0.95, f"sum = {tot.sum() / max(F['ens_std'][mask].__pow__(2).mean(), 1e-12):.2f}",
               transform=ax[0].transAxes, ha="left", va="top", fontsize=6)
    dom = np.argmax(np.stack(contrib), 0).astype(float)
    dom[~mask] = np.nan
    yl = geo["yloc"]
    m2 = mask[:, yl, :]
    if m2.any():
        bz, bx = bbox(m2)
        im = ax[1].imshow(dom[:, yl, :][bz, bx], origin="lower", cmap="coolwarm",
                          norm=Normalize(-0.5, len(angles) - 0.5), interpolation="nearest")
        cb = fig.colorbar(im, ax=ax[1], ticks=np.arange(len(angles)), shrink=0.85)
        cb.ax.set_yticklabels([f"{a:+.0f}" for a in angles], fontsize=5)
        ax[1].set_axis_off()
        ax[1].set_title("Dominant angle (x-z, yloc)")
    save_fig(fig, out_fig, "C10_angle_attribution", fmts)


# --------------------------------------------------------------------------- #
# 3-D surfaces
# --------------------------------------------------------------------------- #
def _iso(vol, level, dx, step):
    if not (np.nanmin(vol) < level < np.nanmax(vol)):
        return None
    v, f, _, _ = measure.marching_cubes(np.nan_to_num(vol), level=level, spacing=(dx, dx, dx), step_size=step)
    return v, f


def _sample_at(vol, verts, dx):
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return ndimage.map_coordinates(vol, (verts / dx).T, order=1, mode="nearest")


def _add_mesh(ax, verts, faces, color=None, values=None, cmap="magma", norm=None, alpha=1.0, edge=False):
    tri = verts[faces][:, :, [2, 1, 0]]  # (z, y, x) -> (x, y, z)
    pc = Poly3DCollection(tri, alpha=alpha, linewidths=0.05 if edge else 0)
    if values is not None:
        fv = values[faces].mean(1)
        pc.set_facecolor(plt.get_cmap(cmap)(norm(fv)))
    else:
        pc.set_facecolor(color)
    if edge:
        pc.set_edgecolor((0, 0, 0, 0.15))
    ax.add_collection3d(pc)
    return pc


def _setup3d(ax, shape, dx, title, elev, azim, zlabel=True):
    nz, ny, nx = shape
    ax.set_xlim(0, nx * dx); ax.set_ylim(0, ny * dx); ax.set_zlim(0, nz * dx)
    ax.set_box_aspect((nx, ny, nz))
    ax.set_xlabel("x [km]", labelpad=-8); ax.set_ylabel("y [km]", labelpad=-8); ax.set_zlabel("z [km]" if zlabel else "", labelpad=-8)
    ax.tick_params(labelsize=5, pad=-3)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, pad=0)


def plot_surfaces(F, mask, geo, cfg, out_fig, fmts):
    sc = cfg["surface"]
    dx, step = geo["dx"], sc["step_size"]
    sl = bbox(mask, pad=2)
    C = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim == 3 else v) for k, v in F.items()}
    msk = mask[sl]
    level = sc["iso_level"] or float(np.quantile(F["truth"][mask], sc["iso_quantile"]))
    shape = msk.shape
    el, az = sc["elev"], sc["azim"]
    q_lo, q_hi = C["q_lo80"], C["q_hi80"]
    smax = float(np.quantile(F["ens_std"][mask], 0.99)) or 1.0
    norm_s = Normalize(0, smax)

    # S1: ensemble-mean isosurface coloured by total std, and by epistemic fraction
    fig = plt.figure(figsize=(WIDTH, 3.0), constrained_layout=True)
    mesh = _iso(C["ens_mean"], level, dx, step)
    for i, (key, title, cmap, norm) in enumerate(
            (("ens_std", r"mean isosurface, colour $\sigma_\mathrm{tot}$", "magma", norm_s),
             ("epi_frac", "mean isosurface, colour epistemic fraction", "viridis", Normalize(0, 1)))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        if mesh is not None:
            v, f = mesh
            _add_mesh(ax, v, f, values=_sample_at(C[key], v, dx), cmap=cmap, norm=norm)
        _setup3d(ax, shape, dx, title, el, az)
        m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.12)
        cb.set_label(f"[{EXT_UNIT}]" if key == "ens_std" else "")
    fig.suptitle(f"Isosurface at {level:.1f} {EXT_UNIT}", fontsize=8)
    save_fig(fig, out_fig, "C11_surface_mean_std", fmts)

    # S2: 80% envelope (q10 core, q90 hull) vs truth
    fig = plt.figure(figsize=(WIDTH, 2.6), constrained_layout=True)
    for i, (key, title, col) in enumerate((("q_lo80", "q10 (confident core)", "C0"),
                                           ("truth", "truth", "0.5"),
                                           ("q_hi80", "q90 (possible hull)", "C3"))):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        mesh_i = _iso(C[key], level, dx, step)
        if mesh_i is not None:
            _add_mesh(ax, *mesh_i, color=col, alpha=0.9, edge=True)
        _setup3d(ax, shape, dx, title, el, az, zlabel=(i == 2))
    save_fig(fig, out_fig, "C12_surface_envelope", fmts)

    # S3: truth surface with envelope overlay, coloured by z-score at truth surface
    fig = plt.figure(figsize=(WIDTH, 3.0), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for key, col, al in (("q_hi80", (0.85, 0.2, 0.2, 1), 0.15), ("truth", (0.3, 0.3, 0.3, 1), 0.6),
                         ("q_lo80", (0.1, 0.3, 0.8, 1), 0.9)):
        mesh_i = _iso(C[key], level, dx, step)
        if mesh_i is not None:
            _add_mesh(ax, *mesh_i, color=col, alpha=al)
    _setup3d(ax, shape, dx, "q10 (blue) / truth (grey) / q90 (red)", el, az)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    mesh_t = _iso(C["truth"], level, dx, step)
    if mesh_t is not None:
        v, f = mesh_t
        _add_mesh(ax, v, f, values=_sample_at(np.nan_to_num(C["zscore"]), v, dx), cmap="RdBu_r",
                  norm=Normalize(-3, 3))
    _setup3d(ax, shape, dx, "truth isosurface, colour z", el, az)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-3, 3), cmap="RdBu_r"), ax=ax, shrink=0.6, pad=0.12)
    save_fig(fig, out_fig, "C13_surface_truth_vs_envelope", fmts)

    # S4: std isosurface inside translucent mean surface
    std_level = float(np.quantile(F["ens_std"][mask], sc["std_iso_quantile"]))
    fig = plt.figure(figsize=(WIDTH * 0.5, 3.0), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    if mesh is not None:
        _add_mesh(ax, *mesh, color=(0.6, 0.6, 0.6, 1), alpha=0.15)
    ms = _iso(np.where(msk, C["ens_std"], 0), std_level, dx, step)
    if ms is not None:
        _add_mesh(ax, *ms, color=(0.9, 0.4, 0.0, 1), alpha=0.9)
    _setup3d(ax, shape, dx, rf"$\sigma_\mathrm{{tot}}$ = {std_level:.1f} {EXT_UNIT} (orange)", el, az)
    save_fig(fig, out_fig, "C14_surface_std", fmts)

    if sc.get("plotly_html"):
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("[surface] plotly not installed; skipping interactive HTML")
            return level
        data = []
        for key, name, op in (("ens_mean", "ensemble mean", 1.0), ("q_hi80", "q90 hull", 0.2),
                              ("q_lo80", "q10 core", 0.5), ("truth", "truth", 0.4)):
            mesh_i = _iso(C[key], level, dx, step)
            if mesh_i is None:
                continue
            v, f = mesh_i
            kw = dict(intensity=_sample_at(C["ens_std"], v, dx), colorscale="Magma", cmin=0, cmax=smax,
                      colorbar=dict(title="σ_tot")) if key == "ens_mean" else dict(color={"q_hi80": "red", "q_lo80": "blue", "truth": "gray"}[key])
            data.append(go.Mesh3d(x=v[:, 2], y=v[:, 1], z=v[:, 0], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                  name=name, opacity=op, showlegend=True, visible=True if key == "ens_mean" else "legendonly", **kw))
        go.Figure(data=data, layout=dict(scene=dict(aspectmode="data", xaxis_title="x [km]", yaxis_title="y [km]",
                                                    zaxis_title="z [km]"),
                                         title=f"Cloud UQ isosurfaces ({level:.1f} km^-1)")).write_html(
            os.path.join(out_fig, "C15_surfaces_interactive.html"), include_plotlyjs="cdn")
    return level


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_fields(cfg):
    p_pred = os.path.join(cfg["data_path"], cfg["predicted_file"])
    p_true = os.path.join(cfg["data_path"], cfg["truth_cot_file"])
    pred = xr.open_dataset(p_pred).load()
    truth = xr.open_dataset(p_true)[cfg["truth_cot_var"]].load()
    if "tau_member_loc" in pred:
        loc, scale = pred.tau_member_loc, pred.tau_member_scale
        family = pred.attrs.get("distribution", "laplace")
    else:  # old merged file: single moment-matched member (epistemic not separable in sampling)
        print("[warn] predicted file has no member fields; using one moment-matched member")
        family = "laplace"
        loc = pred.tau_pre.expand_dims(member=[0], axis=-1)
        var = pred.aleatoric_uncertainty + pred.epistemic_uncertainty
        scale = np.sqrt(var / 2.0).expand_dims(member=[0], axis=-1)
    return pred, truth, loc, scale, family


def main(cfg):
    plt.rcParams.update(STYLE)
    out = cfg["output_dir"]
    out_fig, out_tab = os.path.join(out, "figures"), os.path.join(out, "tables")
    os.makedirs(out_fig, exist_ok=True); os.makedirs(out_tab, exist_ok=True)
    fmts = cfg["figure_formats"]
    rng = np.random.default_rng(cfg["uq"]["seed"])

    geo = select_cloud(cfg)
    cfg["_dx"] = geo["dx"]
    print(f"[cloud] label={geo['label']} slx={geo['slx']} sly={geo['sly']} yloc={geo['yloc']} M={geo['M']}")
    pred, truth, loc_da, scale_da, family = load_fields(cfg)
    angles = np.sort(pred.vza.data)
    S = loc_da.sizes["member"]

    # --- views (2-D, per angle) ---
    t0 = _time.time()
    V = {"family": family, "truth": make_views(truth, geo, angles, cfg),
         "mean": make_views(pred.tau_pre, geo, angles, cfg)}
    V["loc"] = np.stack([make_views(loc_da.isel(member=s), geo, angles, cfg) for s in range(S)], -1)
    V["scale"] = np.stack([np.maximum(make_views(scale_da.isel(member=s), geo, angles, cfg), 0) for s in range(S)], -1)
    print(f"[views] {V['truth'].shape} x {S} members in {_time.time() - t0:.1f}s")
    tab_angle = per_angle_analysis(V, angles, cfg, out_fig, fmts, geo)
    tab_angle.to_csv(os.path.join(out_tab, "T_case_per_angle.csv"), index=False)

    # --- SART masks (True = outside cloud, zeroed after every SART iteration) ---
    M, data_test = geo["M"], geo["data_test"]
    cloud = data_test > 0                                 # true support = metrics domain
    K = cfg["uq"]["n_samples"]
    draws = member_draws(K, len(angles), S, cfg["uq"]["member_pairing"], rng)
    vn, vl = sample_views(V["loc"], V["scale"], draws, family, cfg, rng)
    sino_mean, sino_true = to_sinogram(V["mean"], M), to_sinogram(V["truth"], M)
    sino_n = [to_sinogram(v, M) for v in vn]
    sino_l = [to_sinogram(v, M) for v in vl]
    mk = cfg["mask"]
    mode = mk["mode"]
    mask_rows, hull_fn, h_show = [], None, cloud
    if mode == "truth":
        m_mean = m_true = m_samples = ~cloud
        p_in = cloud.astype(float)
    elif mode in ("hull_mean", "hull_sample", "hull_truecot"):
        sign = geometry_sign(M, angles, geo["dx"]) if mk.get("geometry_check", True) else 1.0
        tco = detector_coords(M, angles, sign)
        hull_fn = lambda sn: visual_hull(sn, tco, mk["cot_threshold"], mk.get("min_views"), mk.get("dilation", 0))
        h_mean, h_true = hull_fn(sino_mean), hull_fn(sino_true)
        if not h_mean.any() or not h_true.any():
            raise ValueError("visual hull is empty: lower mask.cot_threshold, set mask.min_views below the "
                             "number of views, or add mask.dilation (the views may not be ray-consistent)")
        mask_rows += [{"mask": "hull(mean COT)", **mask_scores(h_mean, cloud)},
                      {"mask": "hull(true COT)", **mask_scores(h_true, cloud)}]
        if mode == "hull_mean":
            m_mean, m_true, m_samples, h_show = ~h_mean, ~h_true, ~h_mean, h_mean
            p_in = h_mean.astype(float)
        elif mode == "hull_truecot":
            m_mean = m_true = m_samples = ~h_true
            p_in, h_show = h_true.astype(float), h_true
        else:
            m_mean, m_true, h_show = ~h_mean, ~h_true, h_mean
            h_n = [hull_fn(sn) for sn in sino_n]
            m_samples = [~h for h in h_n] + [~hull_fn(sn) for sn in sino_l]
            p_in = np.mean(h_n, 0)
            sc = pd.DataFrame([mask_scores(h, cloud) for h in h_n])
            mask_rows.append({"mask": "hull(samples), mean", **sc.mean().to_dict(), "IoU_sd": float(sc["IoU"].std())})
            mask_rows.append({"mask": "P(in mask) >= 0.5", **mask_scores(p_in >= 0.5, cloud)})
        print("[mask] " + mode + ": " + "; ".join(f"{r['mask']} IoU={r['IoU']:.3f}" for r in mask_rows))
    else:
        raise ValueError(f"unknown mask.mode {mode!r}")
    mask_sart = m_mean                                    # shared mask for non-sampled runs

    # --- deterministic references ---
    det = np.concatenate([run_sart_batch([sino_mean], m_mean, angles, cfg, "deterministic (mean COT)"),
                          run_sart_batch([sino_true], m_true, angles, cfg, "deterministic (true COT)")])

    # --- Monte Carlo propagation ---
    recs = run_sart_batch(sino_n + sino_l, m_samples, angles, cfg, "Monte Carlo")
    rec_n, rec_l = recs[:K], recs[K:]

    levels = sorted(cfg["uq"]["levels"])
    F = {"truth": data_test, "rec_det": det[0], "rec_truecot": det[1], "samples": rec_n,
         "ens_mean": rec_n.mean(0), "ens_std": rec_n.std(0, ddof=1)}
    var_tot = F["ens_std"] ** 2
    var_epi = rec_l.var(0, ddof=1)  # spread from member choice alone (paired draws)
    F["std_epi"] = np.sqrt(var_epi)
    F["std_ale"] = np.sqrt(np.clip(var_tot - var_epi, 0, None))  # residual (law of total variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        F["epi_frac"] = np.where(var_tot > 0, np.clip(var_epi / var_tot, 0, 1), 0)
        F["zscore"] = np.where(cloud & (F["ens_std"] > 1e-9), (F["ens_mean"] - data_test) / F["ens_std"], np.nan)
    for a in levels:
        lo, hi = np.quantile(rec_n, [(1 - a) / 2, (1 + a) / 2], axis=0)
        F[f"q_lo{int(a * 100)}"], F[f"q_hi{int(a * 100)}"] = lo, hi
    if "q_lo80" not in F:
        F["q_lo80"], F["q_hi80"] = np.quantile(rec_n, [0.1, 0.9], axis=0)
    F["hit80"] = np.where(cloud, (data_test >= F["q_lo80"]) & (data_test <= F["q_hi80"]), False)

    # --- 3-D metrics ---
    rows, diag = [], {}
    for name, ref in (("truth extinction", data_test), ("SART(true COT)", det[1])):
        row, d = volume_metrics(ref, rec_n, cloud, levels, rng, name)
        rows.append(row); diag[name] = d
    det_rows = []
    for name, rec in (("SART(mean COT)", det[0]), ("SART(true COT)", det[1]), ("ensemble mean", F["ens_mean"])):
        e = rec[cloud] - data_test[cloud]
        det_rows.append({"reconstruction": name, "RMSE": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()),
                         "r": float(np.corrcoef(rec[cloud], data_test[cloud])[0, 1])})
    # union domain (true support or possibly-in-mask): penalises false cloud volume
    union = cloud | (p_in > 0)
    for name, rec in (("SART(mean COT)", det[0]), ("SART(true COT)", det[1]), ("ensemble mean", F["ens_mean"])):
        e = rec[union] - data_test[union]
        det_rows.append({"reconstruction": f"{name} [union domain]", "RMSE": float(np.sqrt(np.mean(e ** 2))),
                         "bias": float(e.mean()), "r": float(np.corrcoef(rec[union], data_test[union])[0, 1])})
    F["p_in_mask"] = p_in
    if mode != "truth":
        mask_rows.append({"mask": "summary", "mode": mode, "cot_threshold": mk["cot_threshold"],
                          "min_views": mk.get("min_views") or len(angles), "dilation": mk.get("dilation", 0),
                          "RMSE_ens_truth_support": det_rows[2]["RMSE"], "RMSE_ens_union": det_rows[5]["RMSE"],
                          "N_union": int(union.sum())})
        pd.DataFrame(mask_rows).to_csv(os.path.join(out_tab, "T_case_mask.csv"), index=False)
        ious = [mask_scores(~m, cloud)["IoU"] for m in m_samples[:K]] if mode == "hull_sample" else None
        plot_masks(cloud, h_show, p_in, geo, out_fig, fmts, ious)
        plot_mask_profiles(cloud, h_show, p_in, (np.arange(M) - M // 2) * geo["dx"], out_fig, fmts)
    t3 = pd.DataFrame(rows)
    t3["mask_mode"] = mode
    t3["epi_frac_volume"] = float(var_epi[cloud].sum() / max(var_tot[cloud].sum(), 1e-12))
    t3.to_csv(os.path.join(out_tab, "T_case_3d_uq.csv"), index=False)
    pd.DataFrame(det_rows).to_csv(os.path.join(out_tab, "T_case_3d_deterministic.csv"), index=False)

    z_km = (np.arange(M) - M // 2) * geo["dx"]
    plot_slices(F, cloud, geo, out_fig, fmts)
    plot_profiles(F, cloud, z_km, levels, out_fig, fmts, geo["yloc"])
    plot_3d_calibration(diag, out_fig, fmts)

    # --- per-angle attribution of 3-D variance ---
    contrib = None
    if cfg["angle_attribution"]["enabled"]:
        Ka = cfg["angle_attribution"]["n_samples"]
        contrib = []
        mean_v = V["loc"].mean(-1)
        for a in range(len(angles)):
            am = np.zeros(len(angles), bool); am[a] = True
            dr = member_draws(Ka, len(angles), S, cfg["uq"]["member_pairing"], rng)
            vn_a, _ = sample_views(V["loc"], V["scale"], dr, family, cfg, rng, angle_mask=am, mean_v=mean_v)
            sinos_a = [to_sinogram(v, M) for v in vn_a]
            m_a = [~hull_fn(sn) for sn in sinos_a] if mode == "hull_sample" else mask_sart
            r_a = run_sart_batch(sinos_a, m_a, angles, cfg, f"attribution {angles[a]:+.1f}")
            contrib.append(r_a.var(0, ddof=1))
        plot_attribution(contrib, angles, F, cloud, geo, out_fig, fmts)
        share = [float(c[cloud].mean() / max(var_tot[cloud].mean(), 1e-12)) for c in contrib]
        pd.DataFrame({"vza": angles, "variance_share": share}).to_csv(os.path.join(out_tab, "T_case_angle_attribution.csv"), index=False)

    level = plot_surfaces(F, cloud, geo, cfg, out_fig, fmts)

    # --- NetCDF for ParaView / re-plotting ---
    nz, ny, nx = data_test.shape
    coords = {"z": z_km, "y": np.arange(ny) * geo["dx"], "x": np.arange(nx) * geo["dx"]}
    dv = {k: (("z", "y", "x"), np.asarray(v, np.float32)) for k, v in F.items()
          if isinstance(v, np.ndarray) and v.shape == data_test.shape}
    dv["samples"] = (("sample", "z", "y", "x"), rec_n.astype(np.float32))
    if contrib is not None:
        dv["angle_variance"] = (("vza", "z", "y", "x"), np.stack(contrib).astype(np.float32))
    for k in ("truth", "mean"):
        dv[f"view_{k}"] = (("xv", "y", "vza"), V[k].astype(np.float32))
    dv["view_member_loc"] = (("xv", "y", "vza", "member"), V["loc"].astype(np.float32))
    dv["view_member_scale"] = (("xv", "y", "vza", "member"), V["scale"].astype(np.float32))
    coords.update(vza=angles, sample=np.arange(K), member=np.arange(S), xv=np.arange(V["truth"].shape[0]) * geo["dx"])
    attrs = {"cloud_label": geo["label"], "cloud_index": cfg["cloud_index"], "yloc": geo["yloc"],
             "family": family, "iso_level": level, "units": "extinction km-1 (3-D), COT (views)",
             "mask_mode": mode,
             "note": ("SART uses the true-extinction support as mask (as in analyze_mystic_scene.py)"
                      if mode == "truth" else
                      f"SART mask: visual hull ({mode}); p_in_mask = fraction of samples whose hull contains the voxel")}
    xr.Dataset(dv, coords=coords, attrs=attrs).to_netcdf(os.path.join(out, f"cloud_id{geo['label']:03d}_uq_fields.nc"))
    with open(os.path.join(out, "case_study_config_used.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, indent=2, default=str)
    print(tab_angle.round(3).to_string(index=False))
    print(t3.round(3).to_string(index=False))
    print(f"[done] outputs in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-y", "--yaml", default=None, help="case-study yaml (defaults reproduce the paper case)")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args()
    c = load_config(a.yaml)
    if a.n_samples:
        c["uq"]["n_samples"] = a.n_samples
    if a.n_jobs:
        c["uq"]["n_jobs"] = a.n_jobs
    main(c)
