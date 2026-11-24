"""Export MIMO U-Net representations as Rep_Taxonomy paired-view bundles.

    python export_representations.py -y config/reptax_40m.yaml [--only n705 multiangle] [--overwrite]

Every deployed network (9 single-angle + 1 multi-angle, optional random-weight controls) is
evaluated on the SAME sample of pixels, so Rep_Taxonomy (``reptaxonomy.intercompare.paired_views``)
can align rows by key across models, view angles and datasets.

Common frame
------------
Rows live in the registered frame of the multi-angle design: the nadir tile frame after
``project2com`` to ``registration.cloud_height_avg`` (1.55 km), with the tile-local cosine
projection of ``generate_training_dataset.map2tiles(multiview=True)`` for the off-nadir views.

* multi-angle model: tiles are built exactly as in training / ``predict_scene_views``; a row is the
  tile pixel (j, k).
* single-angle model at vza a: the model sees its own (unregistered, uncompressed) view on a wider
  tile (``single_margin`` px of context on each side); its feature maps are sampled at the
  fractional view-a positions pos_a(j) that the multi-angle registration assigns to nadir row j,
  so the row refers to the same cloud-height location. project2com is a cyclic shift in x, so
  running the CNN on the projected field equals shifting its native features.
* coarse layers are sampled with bilinear interpolation (``grid_sample``, pixel-centre
  convention, align_corners=False) at the same full-resolution positions.

Rows are a ``rows_per_tile`` = g x g grid inside the trimmed tile centre (25 % trim). Keys are
``<cloud field>|t<tile>|<j>|<k>``: datasets that share a cloud field and region (the 10.5 h
baseline and its SZA / AOD / wind variants) share keys, so a model's baseline and OOD
representations of the same cloud pixels can be compared directly.

Outputs (``output_dir``)
------------------------
bundles/<tag>/<dataset>.npz     paired-views bundle (keys, groups, domain, view__*, labels__*, target__*)
geo_id/embeddings/<dataset>/<tag>__<layer>/test/embd_<stem>.npy   (D, g, g) per tile (geo_id layout)
geo_id/work/manifests/<dataset>.json                              tile pseudo-coordinates (see below)
meta/views.csv                  one row per exported view (tag, kind, vza, layer, member, dataset, dim, ref)
meta/models.csv, meta/resources.csv, meta/samples_<dataset>.csv, manifest.json

Views per bundle: view__<layer> (members concatenated for per-member layers) and, for
``member_layers``, view__<layer>__m<s>. Layers: enc1, enc2 (per member, levels 1-2), enc3, enc4,
bott (shared core), d4, d3, d2 (shared decoder), d1 (per member), out (member locations, scaled).
Targets: tau_nadir, log_tau_nadir, cot_hetero (5x5 std of nadir COT), cloud_frac_tile,
tau_v<vza> / rad_v<vza> for every registered view, and the model's own prediction mean / aleatoric /
epistemic std / absolute error per predicted view (physical units). Labels: cloudy, cot_bin.

Pseudo-coordinates for geo_id: the doubly periodic LES domain has no geolocation; tile centres
are placed on an equatorial equirectangular grid (lat = y_km / 111.195, lon = x_km / 111.195), so
great-circle distances equal in-domain distances to < 0.01 %. Use km variogram bins (config).

Missing models / datasets / scalers are skipped with a ``[WARN] skip`` line; finished bundles are
reused unless the export settings changed (fingerprint) or ``--overwrite`` is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "emulator", "mimo", "pyct")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from emulator.mimo.pyct.utils import read_yaml, load_scalers  # noqa: E402
from emulator.mimo.pyct.mimo_inference import (  # noqa: E402
    _gather, _scale_inputs, _view_positions, affine_inverse, angle_key, apply_calibration,
    load_calibration, load_model, load_scene, missing_files, warn_skip)
from emulator.mimo.pyct.mimo_unet import EnsembleModule, MimoUnetModel  # noqa: E402

LAYERS = ("enc1", "enc2", "enc3", "enc4", "bott", "d4", "d3", "d2", "d1", "out")
PER_MEMBER = {"enc1", "enc2", "d1", "out"}
COT_BINS = (0.05, 1.0, 3.0, 10.0, 30.0)   # cot_bin: 0 clear, 1..5
KM_PER_DEG = 111.195
EXPORT_VERSION = 1


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def unets_of(model) -> List[Any]:
    """(MimoUNet, lightning module) for a single checkpoint or every member of a deep ensemble."""
    if isinstance(model, EnsembleModule):
        return [(m.model, m) for m in model.models]
    return [(model.model, model)]


def forward_features(net, x: torch.Tensor, layers: Sequence[str]) -> Dict[str, Any]:
    """MimoUNet.forward with intermediate maps. x [B, S, C, H, W].

    Returns layer -> tensor [B, D, h, w] (shared) or list over members of [B, d, h, w].
    ``out`` is the list of per-member [loc, log_param] maps (scaled units).
    """
    S = net.num_subnetworks
    enc1 = [net.encoder1[i](x[:, i]) for i in range(S)]
    enc2 = [net.encoder2[i](net.pool(enc1[i])) for i in range(S)]
    x2 = torch.cat(enc2, dim=1)
    enc3 = net.encoder3(net.pool(x2))
    enc4 = net.encoder4(net.pool(enc3))
    bott = net.center_dropout(net.bottleneck(net.pool(enc4)))
    d4 = net.decoder4(torch.cat((net.upconv4(bott), enc4), dim=1))
    d3 = net.decoder3(torch.cat((net.upconv3(d4), enc3), dim=1))
    d2 = net.decoder2(torch.cat((net.upconv2(d3), x2), dim=1))
    d1, out = [], []
    for i in range(S):
        h = net.decoder1[i](torch.cat((net.upconv1[i](d2), enc1[i]), dim=1))
        d1.append(h)
        out.append(net.conv[i](net.final_dropout[i](h)))
    allf = {"enc1": enc1, "enc2": enc2, "enc3": enc3, "enc4": enc4, "bott": bott, "d4": d4, "d3": d3,
            "d2": d2, "d1": d1, "out": out}
    return {k: allf[k] for k in layers}


def _grid(ux: np.ndarray, uy: np.ndarray, H: int, W: int, device) -> torch.Tensor:
    """Full-resolution pixel coordinates (ux along H = x, uy along W = y), [B, P] each ->
    grid_sample grid [B, 1, P, 2] (pixel-centre convention, align_corners=False)."""
    gx = 2.0 * (uy + 0.5) / W - 1.0     # width axis = y
    gy = 2.0 * (ux + 0.5) / H - 1.0     # height axis = x
    g = np.stack([gx, gy], -1)[:, None].astype(np.float32)
    return torch.from_numpy(g).to(device)


def _sample(t: torch.Tensor, grid: torch.Tensor) -> np.ndarray:
    """[B, D, h, w] sampled at grid [B, 1, P, 2] -> [B, P, D] float32."""
    s = F.grid_sample(t.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return s[:, :, 0].permute(0, 2, 1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def sample_features(model, x: np.ndarray, ux: np.ndarray, uy: np.ndarray, layers: Sequence[str],
                    batch_size: int, device: str) -> Tuple[Dict[str, np.ndarray], Dict[str, List[np.ndarray]],
                                                           np.ndarray, np.ndarray]:
    """x [N, C, H, W] scaled inputs; ux, uy [N, P] full-res positions.

    Returns (shared {layer: [N, P, D]}, members {layer: [S_total x [N, P, d]]},
    loc [N, P, C_out, S_total], scale [N, P, C_out, S_total]) - loc/scale in scaled units."""
    shared: Dict[str, List[np.ndarray]] = {}
    members: Dict[str, List[List[np.ndarray]]] = {}
    locs, scales = [], []
    H, W = x.shape[-2:]
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        grid = _grid(ux[i:i + batch_size], uy[i:i + batch_size], H, W, device)
        sh_b: Dict[str, List[np.ndarray]] = {}
        mem_b: Dict[str, List[np.ndarray]] = {}
        lb, sb = [], []
        for net, lm in unets_of(model):          # deep ensemble: features concatenated over checkpoints
            S = net.num_subnetworks
            xin = xb[:, None].repeat(1, S, 1, 1, 1)
            feats = forward_features(net, xin, tuple(set(layers) | {"out"}))
            half = feats["out"][0].shape[1] // 2
            for k in layers:
                if k in PER_MEMBER:
                    mem_b.setdefault(k, []).extend(_sample(f[:, :half] if k == "out" else f, grid) for f in feats[k])
                else:
                    sh_b.setdefault(k, []).append(_sample(feats[k], grid))
            for f in feats["out"]:
                lb.append(_sample(f[:, :half], grid))                                        # [B, P, C]
                sb.append(_sample(lm.loss_fn.scale(f[:, half:]), grid))
        for k, v in sh_b.items():
            shared.setdefault(k, []).append(np.concatenate(v, -1))
        for k, v in mem_b.items():
            members.setdefault(k, []).append(v)
        locs.append(np.stack(lb, -1))
        scales.append(np.stack(sb, -1))
    sh = {k: np.concatenate(v) for k, v in shared.items()}
    mem = {k: [np.concatenate([b[s] for b in v]) for s in range(len(v[0]))] for k, v in members.items()}
    return sh, mem, np.concatenate(locs), np.concatenate(scales)


# --------------------------------------------------------------------------- #
# Sampling plan (shared by all models)
# --------------------------------------------------------------------------- #
def field_id(ds: Dict[str, Any]) -> str:
    stem = os.path.splitext(os.path.basename(ds["target_fname"]))[0]
    reg = ds.get("region")
    return stem if reg is None else f"{stem}@y{float(reg[0]):.2f}-{float(reg[1]):.2f}"


def _seed_of(s: str, seed: int) -> int:
    return int(hashlib.sha1(f"{s}|{seed}".encode()).hexdigest()[:8], 16)


def sample_plan(tau_nadir: np.ndarray, ds: Dict[str, Any], T: int, S: Dict[str, Any], seed: int) -> pd.DataFrame:
    """Tile origins (xs, ys) in the nadir frame and the g x g row grid of each tile.

    Seeded by the cloud field (not the dataset name), so variants of one field share rows."""
    nx, ny = tau_nadir.shape
    reg = ds.get("region")
    r0, r1 = (0, ny) if reg is None else (int(round(reg[0] * ny)), int(round(reg[1] * ny)))
    trim = T // 4
    g = int(round(math.sqrt(int(S.get("rows_per_tile", 36)))))
    off = trim + ((np.arange(g) + 0.5) * (T - 2 * trim) / g).astype(int)
    rng = np.random.default_rng(_seed_of(field_id(ds), seed))
    n_tiles = int(S.get("tiles_per_dataset", 96))
    thr = float(S.get("cloud_threshold", 0.05))
    min_cf = float(S.get("min_cloud_fraction", 0.05))
    keep_clear = float(S.get("clear_tile_keep_prob", 0.25))
    if r1 - r0 < T:
        raise ValueError(f"region {reg} is narrower than one tile ({T} px)")
    tiles, tries = [], 0
    while len(tiles) < n_tiles and tries < 200 * n_tiles:
        tries += 1
        xs, ys = int(rng.integers(0, nx)), int(rng.integers(r0, r1 - T + 1))
        sub = tau_nadir[np.ix_((xs + np.arange(T)) % nx, ys + np.arange(T))]
        cf = float(np.mean(sub >= thr))
        if cf < min_cf and rng.random() > keep_clear:
            continue
        tiles.append((xs, ys, cf))
    rows = []
    bpx = int(S.get("block_px", 256))
    for t, (xs, ys, cf) in enumerate(tiles):
        for j in off:
            for k in off:
                rows.append({"tile": t, "xs": xs, "ys": ys, "j": int(j), "k": int(k), "tile_cloud_frac": cf,
                             "block": f"b{((xs + T // 2) % nx) // bpx}_{(ys + T // 2) // bpx}"})
    df = pd.DataFrame(rows)
    fid = field_id(ds)
    df["key"] = [f"{fid}|t{t:04d}|{j:02d}|{k:02d}" for t, j, k in zip(df.tile, df.j, df.k)]
    df["group"] = fid + "|" + df.block
    return df


# --------------------------------------------------------------------------- #
# Shared scene quantities
# --------------------------------------------------------------------------- #
def registration_cfg(cfg: Dict[str, Any], model_cfgs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    R = dict(cfg.get("registration") or {})
    multi = next((c for c in model_cfgs.values() if c.get("multiview")), None)
    base = dict(multi or next(iter(model_cfgs.values())))
    for k in ("input_field", "target_field", "dx", "resolution", "tile_size", "cloud_height_avg", "angles"):
        if k not in R and k in base:
            R[k] = base[k]
    R.setdefault("angles", [-70.5, -60.0, -45.6, -26.1, 0.0, 26.1, 45.6, 60.0, 70.5])
    if R.get("cloud_height_avg") is None:
        raise ValueError("registration.cloud_height_avg is required (no multi-angle config to take it from)")
    R["multiview"] = True
    return R


def scene_targets(sc: Dict[str, Any], plan: pd.DataFrame, T: int, thr: float) -> Dict[str, np.ndarray]:
    """Truth / radiance at every row in the registered frame (identical for all models)."""
    from scipy.ndimage import uniform_filter
    angles, res = sc["angles"], sc["res"]
    nx, ny = sc["tau_p"].shape[:2]
    c0 = int(np.argmin(np.abs(np.asarray(angles))))
    tau_n = sc["tau_p"][..., c0].astype(np.float64)
    m1 = uniform_filter(tau_n, 5, mode="wrap")
    m2 = uniform_filter(tau_n ** 2, 5, mode="wrap")
    hetero = np.sqrt(np.maximum(m2 - m1 ** 2, 0))
    xn = (plan["xs"].to_numpy() + plan.j.to_numpy()) % nx
    yn = (plan.ys.to_numpy() + plan.k.to_numpy()) % ny
    out = {"tau_nadir": tau_n[xn, yn], "cot_hetero": hetero[xn, yn], "cloud_frac_tile": plan.tile_cloud_frac.to_numpy()}
    out["log_tau_nadir"] = np.log1p(np.maximum(out["tau_nadir"], 0))
    for c, a in enumerate(angles):
        tv, rv = np.full(len(plan), np.nan), np.full(len(plan), np.nan)
        for (xs, ys), ix in plan.groupby(["xs", "ys"]).indices.items():
            pos, w = _view_positions(int(xs), T, nx, res, a, True)
            iy = ((ys + np.arange(T)) % ny)[None]
            tt = _gather(sc["tau_p"][..., c], pos, w, iy)[0]
            rr = _gather(sc["rad_p"][..., c], pos, w, iy)[0]
            jj, kk = plan.j.to_numpy()[ix], plan.k.to_numpy()[ix]
            ok = w[jj] > 1e-6
            tv[ix] = np.where(ok, tt[jj, kk] / np.where(ok, w[jj], 1), np.nan)
            rv[ix] = np.where(ok, rr[jj, kk] / np.where(ok, w[jj], 1), np.nan)
        out[f"tau_v{angle_key(a)}"] = tv
        out[f"rad_v{angle_key(a)}"] = rv
    labels = {"cloudy": (out["tau_nadir"] >= thr).astype(np.int64),
              "cot_bin": np.digitize(out["tau_nadir"], COT_BINS).astype(np.int64)}
    return {"targets": out, "labels": labels}


def model_inputs(mcfg: Dict[str, Any], sc: Dict[str, Any], plan: pd.DataFrame, T: int, margin: int,
                 in_scaler) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Scaled input tiles and full-res row positions for one model.

    Returns x [n_tiles, C, H, W], ux / uy [n_tiles, P], tile list, and a validity mask [n_tiles, P]."""
    angles_reg = [float(a) for a in sc["angles"]]
    m_angles = [float(a) for a in mcfg["angles"]]
    c_idx = [int(np.argmin(np.abs(np.asarray(angles_reg) - a))) for a in m_angles]
    nx, ny = sc["rad_p"].shape[:2]
    res = sc["res"]
    tiles = list(plan.groupby(["xs", "ys"], sort=False).indices.items())
    P = max(len(ix) for _, ix in tiles)
    multiview = bool(mcfg.get("multiview", False))
    xs_l, ux_l, uy_l, ok_l = [], [], [], []
    if not multiview and len(m_angles) != 1:
        raise ValueError("single-view models must have exactly one angle")
    if not multiview:
        a = m_angles[0]
        spans = []
        for (xs, ys), ix in tiles:
            pos, w = _view_positions(int(xs), T, nx, res, a, True)
            spans.append(np.nanmax(pos) - np.nanmin(pos))
        Wd = int(16 * math.ceil((max(spans) + 2 + 2 * margin) / 16))
    for (xs, ys), ix in tiles:
        iy = ((ys + np.arange(T)) % ny)[None]
        jj, kk = plan.j.to_numpy()[ix], plan.k.to_numpy()[ix]
        if multiview:
            geo = [_view_positions(int(xs), T, nx, res, angles_reg[c], True) for c in c_idx]
            xt = np.stack([_gather(sc["rad_p"][..., c], *geo[i], iy) for i, c in enumerate(c_idx)], -1)
            ux, ok = jj.astype(np.float64), np.ones(len(ix), bool)
        else:
            pos, w = _view_positions(int(xs), T, nx, res, m_angles[0], True)
            lo = int(math.floor(np.nanmin(pos))) - (Wd - int(math.ceil(np.nanmax(pos) - np.nanmin(pos)))) // 2
            rows = np.arange(lo, lo + Wd)
            field = sc["rad_p"][..., c_idx[0]]
            xt = field[np.ix_(rows % nx, iy[0])][None, ..., None]          # [1, Wd, T, 1]
            ux = pos[jj] - lo
            ok = np.isfinite(ux) & (w[jj] > 1e-6)
            ux = np.where(ok, ux, 0.0)
        xs_l.append(_scale_inputs(xt.astype(np.float32), mcfg, in_scaler)[0])
        pad = P - len(ix)
        ux_l.append(np.pad(ux, (0, pad)))
        uy_l.append(np.pad(kk.astype(np.float64), (0, pad)))
        ok_l.append(np.pad(ok, (0, pad)))
    return np.stack(xs_l), np.stack(ux_l), np.stack(uy_l), tiles, np.stack(ok_l)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def random_init_like(model, seed: int):
    """Same architecture and hyper-parameters, freshly initialised weights (representation floor)."""
    torch.manual_seed(int(seed))
    if isinstance(model, EnsembleModule):
        raise ValueError("random_init control is not implemented for deep ensembles")
    hp = {k: v for k, v in dict(model.hparams).items() if k not in ("architecture", "trainable_params")}
    m = MimoUnetModel(**hp)
    return m.to(next(model.parameters()).device).eval()


def model_info(tag: str, model, mcfg: Dict[str, Any], kind: str) -> Dict[str, Any]:
    nets = unets_of(model)
    net = nets[0][0]
    conv = next((m for m in net.encoder1[0].modules() if isinstance(m, torch.nn.Conv2d)), None)
    f = int(conv.out_channels) if conv is not None else None
    return {"tag": tag, "kind": kind, "multiview": bool(mcfg.get("multiview", False)),
            "vza": float(mcfg["angles"][0]) if not mcfg.get("multiview") else np.nan,
            "n_angles": len(mcfg["angles"]), "num_members": int(sum(n.num_subnetworks for n, _ in nets)),
            "n_checkpoints": len(nets), "filter_base": f,
            "params": int(sum(p.numel() for p in model.parameters())),
            "deployed_from": json.dumps((mcfg.get("deployed_from") or {}).get("runs", []))}


def _fp(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _atomic_savez(path: str, **arrs) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def export(cfg: Dict[str, Any], only: Optional[Sequence[str]] = None, overwrite: bool = False) -> None:
    out = cfg["output_dir"]
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    S = dict(cfg.get("sampling") or {})
    layers = [l for l in cfg.get("layers", ["enc2", "enc3", "enc4", "bott", "d2", "d1", "out"]) if l in LAYERS]
    member_layers = [l for l in cfg.get("member_layers", ["enc2", "d1"]) if l in PER_MEMBER and l in layers]
    geo_layers = [l for l in (cfg.get("geo_id") or {}).get("layers", []) if l in layers]
    margin = int(cfg.get("single_margin", 16))
    seed = int(cfg.get("seed", 0))
    thr = float(S.get("cloud_threshold", 0.05))
    bs = int(cfg.get("batch_size", 16))

    # ---- model configs --------------------------------------------------- #
    mcfgs: Dict[str, Dict[str, Any]] = {}
    for m in cfg["models"]:
        tag = str(m.get("tag") or "") or os.path.basename(m["config"]).replace("config_", "").replace("_mimo_evaluate.yaml", "")
        if only and tag not in only and not any(o == f"{tag}_rand" for o in only):
            continue
        if missing_files(m["config"]):
            warn_skip(f"model {tag}", [m["config"]])
            continue
        mc = read_yaml(m["config"])
        mc.update(cfg.get("model_overrides", {}))
        mc.update(m.get("overrides", {}))
        mcfgs[tag] = mc
    if not mcfgs:
        print("[WARN] skip export: no model configs found")
        return
    R = registration_cfg(cfg, mcfgs)
    T = int(R["tile_size"])
    for tag, mc in mcfgs.items():
        if int(mc["tile_size"]) != T or float(mc.get("resolution", mc["dx"])) != float(R.get("resolution", R["dx"])):
            print(f"[WARN] {tag}: tile_size/resolution differ from the registration frame; rows still align, "
                  f"but receptive fields differ")
    rand_tags = set(cfg.get("controls", {}).get("random_init", []) or [])

    # ---- datasets: scenes, plans, shared targets ------------------------- #
    scenes: Dict[str, Any] = {}
    for ds in cfg["datasets"]:
        miss = missing_files(ds["input_fname"], ds["target_fname"])
        if miss:
            warn_skip(f"dataset {ds['name']}", miss)
            continue
        sc = load_scene(R, ds["input_fname"], ds["target_fname"])
        c0 = int(np.argmin(np.abs(np.asarray(sc["angles"]))))
        plan = sample_plan(sc["tau_p"][..., c0], ds, T, S, seed)
        tg = scene_targets(sc, plan, T, thr)
        plan.to_csv(os.path.join(out, "meta", f"samples_{ds['name']}.csv"), index=False)
        scenes[ds["name"]] = {"ds": ds, "sc": sc, "plan": plan, **tg}
        print(f"[plan] {ds['name']}: {plan.tile.nunique()} tiles x {len(plan) // max(plan.tile.nunique(), 1)} rows "
              f"(field {field_id(ds)}, cloudy rows {tg['labels']['cloudy'].mean():.2f})", flush=True)
    if not scenes:
        print("[WARN] skip export: no dataset inputs found")
        return

    settings = {"v": EXPORT_VERSION, "layers": layers, "member_layers": member_layers, "sampling": S, "seed": seed,
                "margin": margin, "registration": {k: R[k] for k in ("cloud_height_avg", "dx", "tile_size")},
                "geo_layers": geo_layers}
    view_rows, model_rows, res_rows = [], [], []
    jobs = [(t, "trained") for t in mcfgs] + [(t, "random") for t in mcfgs if t in rand_tags]
    for tag, kind in jobs:
        mc = mcfgs[tag]
        otag = tag if kind == "trained" else f"{tag}_rand"
        if only and otag not in only and tag not in only:
            continue
        model = None
        try:
            in_scaler, out_scaler = load_scalers(mc)
            if in_scaler is None or out_scaler is None:
                warn_skip(f"model {otag}", "input/output scaler not found in model_dir")
                continue
            ckpt_missing = missing_files(mc.get("model_checkpoint_paths") or
                                         os.path.join(mc["model_dir"], mc["state_dict"]))
            if ckpt_missing:
                warn_skip(f"model {otag}", ckpt_missing)
                continue
            model = load_model(mc, device)
            if kind == "random":
                model = random_init_like(model, seed + 1000)
        except Exception as e:  # noqa: BLE001
            warn_skip(f"model {otag}", repr(e))
            continue
        info = model_info(otag, model, mc, kind)
        model_rows.append(info)
        m_angles = [float(a) for a in mc["angles"]]
        a_aff, c_aff = affine_inverse(out_scaler, len(m_angles))
        a_aff = np.broadcast_to(np.asarray(a_aff, float).reshape(-1), (len(m_angles),))
        c_aff = np.broadcast_to(np.asarray(c_aff, float).reshape(-1), (len(m_angles),))
        calib = load_calibration(mc) if kind == "trained" else None
        family = "laplace" if model.loss_fn.name.startswith("laplace") else "gaussian"
        for dname, D in scenes.items():
            path = os.path.join(out, "bundles", otag, f"{dname}.npz")
            fp = _fp({**settings, "model": otag, "cfg": mc.get("deployed_from"), "ds": D["ds"],
                      "calibrated": calib is not None})
            side = path + ".json"
            done = os.path.exists(path) and os.path.exists(side) and json.load(open(side)).get("fingerprint") == fp
            if done and not overwrite:
                meta = json.load(open(side))
                view_rows += meta["views"]
                print(f"[export] {otag} / {dname}: done (reuse)", flush=True)
                continue
            t0 = time.time()
            plan = D["plan"]
            x, ux, uy, tiles, okm = model_inputs(mc, D["sc"], plan, T, margin, in_scaler)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.time()
            sh, mem, loc, scl = sample_features(model, x, ux, uy, layers, bs, device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t_fwd = time.time() - t1
            # rows back to plan order
            order = np.concatenate([ix for _, ix in tiles])
            flat_ok = np.concatenate([okm[i, :len(ix)] for i, (_, ix) in enumerate(tiles)])

            def _rows(a):  # [n_tiles, P, ...] -> [n_rows, ...] in plan order
                parts = [a[i, :len(ix)] for i, (_, ix) in enumerate(tiles)]
                v = np.concatenate(parts)
                o = np.empty_like(v)
                o[order] = v
                return o

            ok = np.empty_like(flat_ok)
            ok[order] = flat_ok
            views: Dict[str, np.ndarray] = {}
            for k, v in sh.items():
                views[f"view__{k}"] = _rows(v)
            for k, lst in mem.items():
                views[f"view__{k}"] = np.concatenate([_rows(v) for v in lst], -1)
                if k in member_layers:
                    for s, v in enumerate(lst):
                        views[f"view__{k}__m{s}"] = _rows(v)
            for k in views:
                views[k][~ok] = np.nan
            L = _rows(loc)                                                   # [n, C, S] scaled
            Sc = _rows(scl)
            Lp = a_aff[None, :, None] * L + c_aff[None, :, None]
            Sp = np.abs(a_aff)[None, :, None] * Sc
            if calib is not None:
                Lp, Sp = apply_calibration(Lp, Sp, m_angles, calib)
            mvar = 2.0 * Sp ** 2 if family == "laplace" else Sp ** 2
            targets = {f"target__{k}": v for k, v in D["targets"].items()}
            for c, a in enumerate(m_angles):
                ak = angle_key(a)
                mean = Lp[:, c].mean(-1)
                epi = Lp[:, c].var(-1, ddof=1) if Lp.shape[-1] > 1 else np.zeros_like(mean)
                truth = D["targets"][f"tau_v{ak}"]
                for nm, val in (("pred_mean", mean), ("ale_std", np.sqrt(mvar[:, c].mean(-1))),
                                ("epi_std", np.sqrt(epi)), ("abs_err", np.abs(mean - truth))):
                    val = val.astype(np.float64)
                    val[~ok] = np.nan
                    targets[f"target__{nm}_v{ak}"] = val
            labels = {f"labels__{k}": v for k, v in D["labels"].items()}
            arrs = {"keys": plan.key.to_numpy().astype(str), "groups": plan.group.to_numpy().astype(str),
                    "domain": np.full(len(plan), dname).astype(str), **views, **labels, **targets}
            _atomic_savez(path, **arrs)
            vr = []
            for k, v in views.items():
                parts = k[len("view__"):].split("__")
                vr.append({"ref": f"{otag}@{dname}:{k}", "input": f"{otag}@{dname}", "view": k, "tag": otag,
                           "base_tag": tag, "kind": kind, "multiview": info["multiview"], "vza": info["vza"],
                           "layer": parts[0], "member": int(parts[1][1:]) if len(parts) > 1 else -1,
                           "dataset": dname, "group": D["ds"].get("group", ""), "field": field_id(D["ds"]),
                           "dim": int(v.shape[1]), "path": os.path.abspath(path)})
            view_rows += vr
            with open(side, "w") as fh:
                json.dump({"fingerprint": fp, "views": vr, "n_rows": int(len(plan)),
                           "valid_frac": float(ok.mean())}, fh, indent=1)
            res_rows.append({"tag": otag, "dataset": dname, "n_tiles": int(len(tiles)), "tile_hw": list(x.shape[-2:]),
                             "forward_s": t_fwd, "forward_ms_per_tile": 1e3 * t_fwd / max(len(tiles), 1),
                             "prep_s": t1 - t0, "device": device, "params": info["params"],
                             "num_members": info["num_members"]})
            # geo_id layout (tile-level grids of sampled rows)
            for layer in geo_layers:
                key = f"view__{layer}"
                if key not in views:
                    continue
                g = int(round(math.sqrt(len(plan) / plan.tile.nunique())))
                edir = os.path.join(out, "geo_id", "embeddings", dname, f"{otag}__{layer}", "test")
                os.makedirs(edir, exist_ok=True)
                V = views[key]
                for t, ix in plan.groupby("tile").indices.items():
                    a = V[ix]
                    if len(a) != g * g:
                        continue
                    np.save(os.path.join(edir, f"embd_t{t:04d}.npy"), a.T.reshape(a.shape[1], g, g).astype(np.float32))
            print(f"[export] {otag} / {dname}: {len(plan)} rows, {len(views)} views, valid {ok.mean():.3f}, "
                  f"{1e3 * t_fwd / max(len(tiles), 1):.1f} ms/tile -> {path}", flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- geo_id manifests -------------------------------------------------- #
    if geo_layers:
        mdir = os.path.join(out, "geo_id", "work", "manifests")
        os.makedirs(mdir, exist_ok=True)
        for dname, D in scenes.items():
            p = D["plan"].drop_duplicates("tile").sort_values("tile")
            res = D["sc"]["res"]
            nx = D["sc"]["tau_p"].shape[0]
            xkm = (((p["xs"] + T // 2) % nx) * res).to_numpy()
            ykm = ((p.ys + T // 2) * res).to_numpy()
            man = {"stems": [f"t{t:04d}" for t in p.tile], "lats": (ykm / KM_PER_DEG).tolist(),
                   "lons": (xkm / KM_PER_DEG).tolist(), "times": [None] * len(p),
                   "tags": {"dataset": dname, "split": D["ds"].get("group", "")},
                   "point_tags": {"tile_cloud_frac": p.tile_cloud_frac.round(4).tolist(),
                                  "region": [D["ds"].get("group", "")] * len(p)},
                   "coordinates": "pseudo (equatorial equirectangular of in-domain km; LES has no geolocation)"}
            with open(os.path.join(mdir, f"{dname}.json"), "w") as fh:
                json.dump(man, fh)

    # ---- meta ------------------------------------------------------------ #
    vdf = pd.DataFrame(view_rows).drop_duplicates("ref")
    vdf.to_csv(os.path.join(out, "meta", "views.csv"), index=False)
    if model_rows:
        mp = os.path.join(out, "meta", "models.csv")
        md = pd.DataFrame(model_rows)
        if os.path.exists(mp) and only:
            old = pd.read_csv(mp)
            md = pd.concat([old[~old.tag.isin(md.tag)], md], ignore_index=True)
        md.to_csv(mp, index=False)
    if res_rows:
        rp = os.path.join(out, "meta", "resources.csv")
        rd = pd.DataFrame(res_rows)
        if os.path.exists(rp):
            old = pd.read_csv(rp)
            rd = pd.concat([old[~(old.tag + old.dataset).isin(rd.tag + rd.dataset)], rd], ignore_index=True)
        rd.to_csv(rp, index=False)
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump({"settings": settings, "models": sorted(set(vdf.tag)) if len(vdf) else [],
                   "datasets": sorted(scenes), "registration": {k: v for k, v in R.items() if k != "angles"},
                   "angles": R["angles"], "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=1, default=str)
    print(f"[export] views: {len(vdf)} -> {os.path.join(out, 'meta', 'views.csv')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--only", nargs="*", help="model tags to export (e.g. nadir multiangle nadir_rand)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    export(read_yaml(a.yaml), [str(o) for o in a.only] if a.only else None, a.overwrite)


if __name__ == "__main__":
    main()
