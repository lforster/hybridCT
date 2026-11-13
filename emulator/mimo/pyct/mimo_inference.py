"""Scene-level MIMO inference in physical units (shared by evaluation, recalibration,
UQ analysis and the reconstruction case study).

* Affine unscaling: for MinMax / Standard scalers ``tau = a * tau_scaled + c``, so a
  Laplace / Gaussian member ``(mu, b)`` maps to ``(a mu + c, |a| b)``.
* Periodic tiling: LES scenes are doubly periodic, so tiles wrap around the domain and
  every pixel is predicted (trimmed tile centres, 25 % trim on each side). The previous
  ``tiles2map`` stitching left the outer ``trim`` pixels uncovered.
* Single-angle models (``multiview: false``): one radiance view in, one COT view out.
* Multi-angle model (``multiview: true``): the 9 views are registered exactly as in
  training (``project2com`` to ``cloud_height_avg`` and the per-tile cosine projection
  of ``generate_training_dataset.map2tiles``), predicted jointly, and then mapped back
  to each view's native grid (inverse cosine projection + inverse ``project2com``), so
  the output can be used by ``evaluate_and_merge_mimo.py`` / SART like the single-angle
  outputs. The forward/inverse round trip of the true COT (``tau_roundtrip``) is returned
  as the registration floor of the multi-angle design.
* Optional post-hoc recalibration (``apply_calibration: true``; file written by
  ``recalibrate.py``): ``loc_s <- m + t_spread (loc_s - m)``, ``b_s <- t_scale b_s``.
* Optional ``region = (y0, y1)`` (fractions of the y extent) restricts prediction and
  output to a y band (used for the validation / calibration halves of a scene).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import xarray as xr

try:
    from emulator.mimo.pyct.generate_training_dataset import (coarsen_resolution, find_cosine_window,
                                                              get_cos_projection, get_projection_sortIndex,
                                                              project2com)
    from emulator.mimo.pyct.utils import load_scalers, log_scale_data, scaler_transform
    from emulator.mimo.pyct.mimo_unet import EnsembleModule, load_mimo_checkpoint
except ImportError:  # flat imports from emulator/mimo/pyct
    from generate_training_dataset import (coarsen_resolution, find_cosine_window, get_cos_projection,
                                           get_projection_sortIndex, project2com)
    from utils import load_scalers, log_scale_data, scaler_transform
    from mimo_unet import EnsembleModule, load_mimo_checkpoint


def angle_key(a: float) -> str:
    return f"{float(a):+.1f}"


# --------------------------------------------------------------------------- #
# Model / scalers / calibration
# --------------------------------------------------------------------------- #
def affine_inverse(scaler, n_features: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Return (a, c) with x_phys = a * x_scaled + c for sklearn MinMax/Standard scalers."""
    if scaler is None:
        return np.ones(n_features), np.zeros(n_features)
    name = type(scaler).__name__
    if name == "MinMaxScaler":
        a = 1.0 / scaler.scale_
        return a, -scaler.min_ * a
    if name == "StandardScaler":
        scale = scaler.scale_ if scaler.scale_ is not None else np.ones(n_features)
        mean = scaler.mean_ if scaler.mean_ is not None else np.zeros(n_features)
        return np.asarray(scale), np.asarray(mean)
    raise NotImplementedError(f"Uncertainty unscaling not implemented for {name}.")


def load_model(config: Dict[str, Any], device: str = "cpu"):
    """Single MIMO checkpoint (model_dir/state_dict) or a deep ensemble of them."""
    paths = config.get("model_checkpoint_paths")
    if paths and len(paths) > 1:
        model = EnsembleModule(paths, map_location=device)
    else:
        path = paths[0] if paths else os.path.join(config["model_dir"], config["state_dict"])
        model = load_mimo_checkpoint(path, map_location=device)
    return model.to(device).eval()


def calibration_path(config: Dict[str, Any]) -> str:
    return config.get("calibration_file") or os.path.join(config["model_dir"], "calibration.json")


def load_calibration(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not config.get("apply_calibration", False):
        return None
    p = calibration_path(config)
    if not os.path.exists(p):
        raise FileNotFoundError(f"apply_calibration=true but {p} does not exist; run recalibrate.py first.")
    with open(p) as f:
        return json.load(f)


def apply_calibration(loc: np.ndarray, scale: np.ndarray, angles: Sequence[float],
                      calib: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """loc, scale [..., C, S] with C = len(angles). Returns calibrated copies."""
    if calib is None:
        return loc, scale
    loc, scale = loc.copy(), scale.copy()
    for c, a in enumerate(angles):
        p = calib["angles"].get(angle_key(a))
        if p is None:
            raise KeyError(f"no calibration parameters for vza {angle_key(a)} in calibration file")
        m = loc[..., c, :].mean(-1, keepdims=True)
        loc[..., c, :] = m + float(p.get("t_spread", 1.0)) * (loc[..., c, :] - m)
        scale[..., c, :] = float(p.get("t_scale", 1.0)) * scale[..., c, :]
    return loc, scale


# --------------------------------------------------------------------------- #
# Scene loading (same projection / coarsening order as preprocess_single_file)
# --------------------------------------------------------------------------- #
def _open(fname: str, field: str) -> xr.DataArray:
    with xr.open_dataset(fname) as ds:
        da = ds[field].load()
    return da.transpose("x", "y", "vza") if set(("x", "y", "vza")) <= set(da.dims) else da


def _select_angles(da: xr.DataArray, angles: Sequence[float]) -> np.ndarray:
    return np.stack([da.sel(vza=a, method="nearest").data for a in angles], -1).astype(np.float32)


def load_scene(cfg: Dict[str, Any], input_fname: str, target_fname: str) -> Dict[str, Any]:
    """Native (unregistered) and, for multiview, project2com-registered fields [x, y, C]."""
    rad = _open(input_fname, cfg["input_field"])
    tau = _open(target_fname, cfg["target_field"])
    angles = [float(a) for a in cfg["angles"]]
    multiview = bool(cfg.get("multiview", False))
    res = cfg["dx"]
    h = cfg.get("cloud_height_avg")
    if multiview and h is None:
        raise ValueError("multiview=true requires cloud_height_avg (as in training)")
    rad_p = project2com(rad, h, cfg["dx"]) if h is not None else rad
    tau_p = project2com(tau, h, cfg["dx"]) if h is not None else tau
    if cfg.get("resolution", res) > cfg["dx"]:
        res = cfg["resolution"]
        rad, tau = coarsen_resolution(rad, cfg["dx"], res), coarsen_resolution(tau, cfg["dx"], res)
        rad_p, tau_p = coarsen_resolution(rad_p, cfg["dx"], res), coarsen_resolution(tau_p, cfg["dx"], res)
    x = rad_p.x.data
    perms = None
    if h is not None:  # permutation of project2com on the (possibly coarsened) grid
        perms = [get_projection_sortIndex(a, h, x.copy(), res) for a in angles]
    return {"rad": _select_angles(rad, angles), "tau": _select_angles(tau, angles),
            "rad_p": _select_angles(rad_p, angles), "tau_p": _select_angles(tau_p, angles),
            "angles": angles, "res": float(res), "x": x, "perms": perms, "multiview": multiview}


# --------------------------------------------------------------------------- #
# Registration geometry (exactly the convention of map2tiles(multiview=True))
# --------------------------------------------------------------------------- #
def _view_positions(xs: int, T: int, nx: int, res: float, vza: float, multiview: bool):
    """Fractional native x index (projected frame) and weight of each of the T tile rows.

    Identity for nadir / single-view models. For off-nadir multiview views, the training
    code (find_cosine_window + get_cos_projection on a nadir window) is applied to an
    index field and a ones field, so pos/weight reproduce its interpolation exactly.
    """
    j = np.arange(T, dtype=np.float64)
    if not multiview or vza == 0:
        return xs + j, np.ones(T)
    f = np.cos(np.deg2rad(vza))
    P = int(np.ceil(T / f)) + 4
    ie = np.arange(xs - P, xs + T + P, dtype=np.float64)  # extended (unwrapped) native indices
    xe = ie * res
    nadir_x = xr.DataArray(xe[P:P + T], dims=["x"], coords={"x": xe[P:P + T]})
    win = find_cosine_window(abs(vza), nadir_x.data, res)
    da_i = xr.DataArray(ie[:, None], dims=["x", "y"], coords={"x": xe, "y": [0.0]}).sel(x=win)
    da_1 = xr.DataArray(np.ones((len(ie), 1)), dims=["x", "y"], coords={"x": xe, "y": [0.0]}).sel(x=win)
    pi = get_cos_projection(vza, res, nadir_x, da_i).data[:, 0]
    w = get_cos_projection(vza, res, nadir_x, da_1).data[:, 0]
    pos = np.where(w > 1e-9, pi / np.maximum(w, 1e-12), np.nan)
    return pos, w


def _gather(field: np.ndarray, pos: np.ndarray, w: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """field [nx, ny]; pos, w [T]; iy [nt, T] -> tiles [nt, T(x), T(y)] = w * lerp_x(field)."""
    nx = field.shape[0]
    p = np.nan_to_num(pos)
    i0 = np.floor(p).astype(int)
    t = (p - i0)[:, None]
    rows = (1 - t) * field[i0 % nx] + t * field[(i0 + 1) % nx]   # [T, ny]
    rows = rows * w[:, None]
    return np.transpose(rows[:, iy], (1, 0, 2))                # [nt, T, T]


def _scatter_plan(pos: np.ndarray, trim: int, T: int):
    """Native indices n covered by the trimmed tile centre and their fractional tile row."""
    jj = np.arange(trim, T - trim, dtype=np.float64)
    pc = pos[trim:T - trim]
    ok = np.isfinite(pc)
    jj, pc = jj[ok], pc[ok]
    n = np.arange(int(np.ceil(pc[0] - 1e-9)), int(np.floor(pc[-1] + 1e-9)) + 1)
    jf = np.interp(n, pc, jj)
    j0 = np.clip(np.floor(jf).astype(int), 0, T - 2)
    return n, j0, (jf - j0)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def _scale_inputs(x_phys: np.ndarray, cfg, in_scaler) -> np.ndarray:
    """[N, T, T, C] physical -> [N, C, T, T] scaled float32."""
    x = x_phys
    if cfg.get("log_scale_input", False):
        x = log_scale_data(x, True, cfg["fill_value"])
    if in_scaler is not None:
        x = scaler_transform(in_scaler, x, method="transform")
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32)


def _predict_batch(model, x: np.ndarray, batch_size: int, device: str):
    locs, scales = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            loc, scale = model.predict_members(torch.from_numpy(x[i:i + batch_size]).to(device))
            locs.append(loc.float().cpu().numpy())
            scales.append(scale.float().cpu().numpy())
    return np.concatenate(locs), np.concatenate(scales)   # [N, S, C, T, T]


def predict_scene_views(model, config: Dict[str, Any], input_fname: str, target_fname: str,
                        input_noise: float = 0.0, seed: int = 0, device: str = "cpu",
                        batch_size: Optional[int] = None, region: Optional[Sequence[float]] = None,
                        calibration: Optional[Dict[str, Any]] = "auto") -> Dict[str, Any]:
    """Tile (periodic), predict and stitch one scene for a single- or multi-angle model.

    Returns (physical units, native view grids, y restricted to ``region``):
        angles [C]; rad, tau_true [nx, ny_r, C]; loc, scale [nx, ny_r, C, S];
        family; tau_roundtrip [nx, ny_r, C] (multiview only); y_slice (rows of the scene).
    """
    if config.get("log_scale_target", False):
        raise NotImplementedError("log_scale_target=True is not supported for UQ (members would not be "
                                  "Laplace/Gaussian in COT space). Train with log_scale_target=False.")
    cfg = dict(config)
    T = int(cfg["tile_size"])
    trim = T // 4
    stride = T - 2 * trim
    batch_size = batch_size or cfg.get("batch_size", 32)
    sc = load_scene(cfg, input_fname, target_fname)
    angles, multiview, res = sc["angles"], sc["multiview"], sc["res"]
    C = len(angles)
    src_rad = sc["rad_p"] if sc["perms"] is not None else sc["rad"]
    src_tau = sc["tau_p"] if sc["perms"] is not None else sc["tau"]
    nx, ny = src_rad.shape[:2]
    if input_noise > 0:
        rng = np.random.default_rng(seed)
        src_rad = src_rad * (1.0 + input_noise * rng.standard_normal(src_rad.shape).astype(np.float32))

    r0, r1 = (0, ny) if region is None else (int(round(region[0] * ny)), int(round(region[1] * ny)))
    ny_r = r1 - r0
    ys = np.arange(r0 - trim, r1 - trim, stride)
    iy = (ys[:, None] + np.arange(T)[None]) % ny                 # [nt, T]
    xs_all = np.arange(-trim, nx - trim, stride)

    in_scaler, out_scaler = load_scalers(cfg)
    a_aff, c_aff = affine_inverse(out_scaler, C)
    a_aff, c_aff = np.asarray(a_aff, np.float64).reshape(-1), np.asarray(c_aff, np.float64).reshape(-1)
    if a_aff.size == 1 and C > 1:
        a_aff, c_aff = np.repeat(a_aff, C), np.repeat(c_aff, C)

    S = None
    loc_sum = scale_sum = None
    rt_sum = np.zeros((nx, ny_r, C), np.float32) if multiview else None
    cnt = np.zeros((C, nx), np.float32)
    for xs in xs_all:
        geo = [_view_positions(int(xs), T, nx, res, a, multiview) for a in angles]
        x_tiles = np.stack([_gather(src_rad[..., c], *geo[c], iy) for c in range(C)], -1)  # [nt,T,T,C]
        loc_t, scale_t = _predict_batch(model, _scale_inputs(x_tiles, cfg, in_scaler), batch_size, device)
        if S is None:
            S = loc_t.shape[1]
            loc_sum = np.zeros((nx, ny_r, C, S), np.float32)
            scale_sum = np.zeros((nx, ny_r, C, S), np.float32)
        if multiview:
            t_tiles = np.stack([_gather(src_tau[..., c], *geo[c], iy) for c in range(C)], -1)
        # rows of the output covered by the trimmed tile centres in y
        yc = (ys[:, None] + np.arange(trim, T - trim)[None])             # [nt, stride] scene rows
        keep = (yc >= r0) & (yc < r1)
        for c in range(C):
            n, j0, t = _scatter_plan(geo[c][0], trim, T)
            ni = n % nx
            cnt[c, ni] += 1
            lt = a_aff[c] * loc_t[:, :, c] + c_aff[c]                    # [nt, S, T, T]
            st = abs(a_aff[c]) * scale_t[:, :, c]
            for arr, out in ((lt, loc_sum), (st, scale_sum)):
                v = (1 - t)[None, None, :, None] * arr[:, :, j0] + t[None, None, :, None] * arr[:, :, j0 + 1]
                v = v[:, :, :, trim:T - trim]                            # [nt, S, len(n), stride]
                for k in range(len(ys)):
                    rows = yc[k][keep[k]] - r0
                    out[np.ix_(ni, rows, [c], np.arange(S))] += \
                        np.transpose(v[k][:, :, keep[k]], (1, 2, 0))[:, :, None, :]
            if multiview:
                tt = t_tiles[..., c]                                     # [nt, T, T]
                v = (1 - t)[None, :, None] * tt[:, j0] + t[None, :, None] * tt[:, j0 + 1]
                v = v[:, :, trim:T - trim]
                for k in range(len(ys)):
                    rows = yc[k][keep[k]] - r0
                    rt_sum[np.ix_(ni, rows, [c])] += v[k][:, keep[k]][:, :, None]
    if (cnt == 0).any():
        raise RuntimeError("stitching left uncovered pixels; check tile_size / geometry")
    loc = loc_sum / cnt.T[:, None, :, None]
    scale = scale_sum / cnt.T[:, None, :, None]
    rt = rt_sum / cnt.T[:, None, :] if multiview else None

    if sc["perms"] is not None:  # undo project2com: native[idx] = projected
        def unproj(arr):
            out = np.empty_like(arr)
            for c, idx in enumerate(sc["perms"]):
                out[idx, :, c] = arr[:, :, c]
            return out
        loc, scale = unproj(loc), unproj(scale)
        rt = unproj(rt) if rt is not None else None

    calib = load_calibration(cfg) if calibration == "auto" else calibration
    loc, scale = apply_calibration(loc, scale, angles, calib)
    family = "laplace" if model.loss_fn.name.startswith("laplace") else "gaussian"
    ysl = slice(r0, r1)
    rad_out = sc["rad"][:, ysl]
    if input_noise > 0 and sc["perms"] is None:
        rad_out = src_rad[:, ysl]
    out = {"angles": angles, "rad": rad_out, "tau_true": sc["tau"][:, ysl],
           "loc": loc.astype(np.float32), "scale": np.maximum(scale, 0).astype(np.float32),
           "family": family, "y_slice": (r0, r1), "calibrated": calib is not None,
           "unscale_a": a_aff, "unscale_c": c_aff}
    if rt is not None:
        out["tau_roundtrip"] = rt.astype(np.float32)
    return out


def predict_scene(model, config, input_fname, target_fname, **kw) -> Dict[str, Any]:
    """Backward-compatible single-angle wrapper: loc/scale [nx, ny, S], tau_true/rad [nx, ny]."""
    out = predict_scene_views(model, config, input_fname, target_fname, **kw)
    if len(out["angles"]) != 1:
        raise ValueError("predict_scene is for single-angle models; use predict_scene_views")
    o = dict(out)
    for k in ("rad", "tau_true"):
        o[k] = out[k][:, :, 0]
    for k in ("loc", "scale"):
        o[k] = out[k][:, :, 0, :]
    o["unscale_a"], o["unscale_c"] = float(out["unscale_a"][0]), float(out["unscale_c"][0])
    return o


def moments(loc: np.ndarray, scale: np.ndarray, family: str = "laplace"):
    """Mean, aleatoric and epistemic variance from members on the last axis.

    aleatoric = mean_s Var_s (2 b^2 for Laplace), epistemic = unbiased variance of member locations.
    """
    S = loc.shape[-1]
    member_var = 2.0 * scale ** 2 if family == "laplace" else scale ** 2
    mean = loc.mean(-1)
    ale = member_var.mean(-1)
    epi = loc.var(-1, ddof=1) if S > 1 else np.zeros_like(mean)
    return mean, ale, epi
