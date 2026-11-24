"""Run the nine single-angle MIMO U-Nets on a scene and merge the predictions
into one netCDF file for the tomography step.

Output variables (dims x, y, vza [, member]; physical COT units):
    tau_pre                 member-mean COT
    aleatoric_uncertainty   aleatoric variance  (mean_s 2 b_s^2 for Laplace)  [COT^2]
    epistemic_uncertainty   epistemic variance  (member spread, ddof=1)       [COT^2]
    tau_member_loc          per-member location   (with --keep-members, default on)
    tau_member_scale        per-member Laplace b / Gaussian sigma

Changes vs. the previous version: no external ``mimo`` package; uncertainties
are converted to physical units (previously left in scaled units); per-member
parameters are stored so that downstream code can use the exact mixture
predictive distribution; time / resolution / config directory are CLI options.

A multi-angle model (``multiview: true`` config) provides all of its views from one
config; mix freely with single-angle configs (duplicate angles are an error).
Views are written in ascending VZA order.

Example (from emulator/mimo):
    python evaluate_and_merge_mimo.py --time 10.5 --resolution 280
    python evaluate_and_merge_mimo.py --time 30.0 --resolution 40 \
        --config-dir config/30h --suffix _40m
    # deployed sweep winners, calibrated, on the held-out 30 h scene
    python evaluate_and_merge_mimo.py --time 30.0 --resolution 40 \
        --configs config/deployed_40m/config_{n705,n60,n456,n261,nadir,261,456,60,705}_mimo_evaluate.yaml \
        --apply-calibration --tag _MIMO_single_40m_toa_cal
    python evaluate_and_merge_mimo.py --time 30.0 --resolution 40 \
        --configs config/deployed_40m/config_multiangle_mimo_evaluate.yaml \
        --apply-calibration --tag _MIMO_multi_40m_toa_cal
"""
import argparse
import os

import numpy as np
import torch
import xarray as xr

import os as _os, sys as _sys
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
for _p in (_REPO, _os.path.join(_REPO, "emulator", "mimo", "pyct")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)  # run from emulator/mimo without PYTHONPATH
from emulator.mimo.pyct.utils import read_yaml
from emulator.mimo.pyct.mimo_inference import load_model, missing_files, moments, predict_scene_views, warn_skip

ANGLE_TAGS = ["nadir", "n261", "261", "n456", "456", "n60", "60", "n705", "705"]


def _merge_complete(fout, configs, args) -> bool:
    """fout exists, skipped no config, used exactly these configs, and is newer than their inputs."""
    if not os.path.exists(fout):
        return False
    with xr.open_dataset(fout) as d:
        skipped, used = d.attrs.get("skipped_configs", ""), d.attrs.get("configs", "")
    if skipped:
        print(f"[evaluate_and_merge_mimo] {fout} skipped {skipped}; redoing")
        return False
    if set(filter(None, used.split(";"))) != set(configs):
        print(f"[evaluate_and_merge_mimo] {fout} was made from other configs; redoing")
        return False
    t = os.path.getmtime(fout)
    for fname in configs:
        cfg = read_yaml(fname)
        deps = [fname, args.input_fname or cfg["input_fname"][0], args.target_fname or cfg["target_fname"][0]]
        deps += list(cfg.get("model_checkpoint_paths") or [os.path.join(cfg["model_dir"], cfg["state_dict"])])
        if args.apply_calibration or cfg.get("apply_calibration"):
            deps.append(cfg.get("calibration_file") or os.path.join(cfg["model_dir"], "calibration.json"))
        newer = [p for p in deps if os.path.exists(p) and os.path.getmtime(p) > t]
        if newer:
            print(f"[evaluate_and_merge_mimo] {fout} is older than {newer[0]}; redoing")
            return False
    return True


def run_config(config, device, input_fname=None, target_fname=None):
    model = load_model(config, device)
    fi = input_fname or config["input_fname"][0]
    ft = target_fname or config["target_fname"][0]
    out = predict_scene_views(model, config, fi, ft, device=device)
    return out, config["resolution"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", type=float, default=10.5)
    p.add_argument("--resolution", type=int, default=280, choices=[40, 280])
    p.add_argument("--config-dir", default=None, help="default: config/<int(time)>h")
    p.add_argument("--suffix", default=None, help="config suffix, default '' (280 m) or '_40m'")
    p.add_argument("--configs", nargs="*", default=None, help="explicit list of config files")
    p.add_argument("--save-dir", default="../../data/predicted_data")
    p.add_argument("--no-members", action="store_true", help="do not store per-member parameters")
    p.add_argument("--input-fname", default=None, help="scene radiances (default: each config's input_fname[0])")
    p.add_argument("--target-fname", default=None, help="scene truth COT (default: config target_fname[0])")
    p.add_argument("--tag", default=None, help="output tag, default _MIMO_40m_toa / _MIMO_toa")
    p.add_argument("--skip-complete", action="store_true",
                   help="do nothing if the output exists, has no skipped configs, was made from the same "
                        "configs and is newer than their checkpoints / calibration.json")
    p.add_argument("--apply-calibration", action="store_true",
                   help="apply <model_dir>/calibration.json (from recalibrate.py) to every model")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dir = args.config_dir or f"config/{args.time:.0f}h"
    suffix = args.suffix if args.suffix is not None else ("_40m" if args.resolution == 40 else "")
    configs = args.configs or [os.path.join(cfg_dir, f"config_{t}_mimo_single_angle_evaluate{suffix}.yaml")
                               for t in ANGLE_TAGS]

    tag = args.tag or ("_MIMO_40m_toa" if args.resolution == 40 else "_MIMO_toa")
    fout = os.path.join(args.save_dir, f"MISR_{args.resolution}m_80x80km_{args.time:.1f}h_optical_thickness_predicted{tag}.nc")
    if args.skip_complete and _merge_complete(fout, configs, args):
        print(f"[evaluate_and_merge_mimo] {fout}: complete and up to date; skipped (drop --skip-complete to redo)")
        return
    loc, scale, vza = {}, {}, []
    family, dx, calibrated, used, skipped = None, None, [], [], []
    for fname in configs:
        print(f"[evaluate_and_merge_mimo] {fname}")
        if missing_files(fname):
            warn_skip(fname, [fname]); skipped.append(fname)
            continue
        cfg = read_yaml(fname)
        if args.apply_calibration:
            cfg["apply_calibration"] = True
        fi = args.input_fname or cfg["input_fname"][0]
        ft = args.target_fname or cfg["target_fname"][0]
        miss = missing_files(fi, ft)
        if miss:
            warn_skip(fname, miss); skipped.append(fname)
            continue
        try:
            out, dx = run_config(cfg, device, fi, ft)
        except (FileNotFoundError, OSError) as e:   # checkpoint, scaler or calibration.json missing
            warn_skip(fname, e); skipped.append(fname)
            continue
        used.append(fname)
        for c, a in enumerate(out["angles"]):
            if a in loc:
                raise ValueError(f"vza {a} provided by more than one config")
            loc[a], scale[a] = out["loc"][..., c, :], out["scale"][..., c, :]
        calibrated.append(bool(out["calibrated"]))
        family = out["family"]
    if skipped:
        print(f"[evaluate_and_merge_mimo] WARNING: {len(skipped)} config(s) skipped: {', '.join(skipped)}")
    if not loc:
        print("[WARN] skip merge: no config produced predictions; no file written")
        return
    vza = sorted(loc)
    # Single-angle networks from the sweep can have different member counts (mimo_num_subnetworks).
    # Moments are computed per view from its own members; the member axis is then padded with NaN
    # to the largest count and num_members[vza] records how many entries of each view are real.
    n_mem = np.array([loc[a].shape[-1] for a in vza], dtype=np.int32)
    S = int(n_mem.max())
    if len(set(n_mem.tolist())) > 1:
        print(f"[evaluate_and_merge_mimo] member counts differ across views "
              f"{dict(zip(map(float, vza), n_mem.tolist()))}: padding to {S} with NaN (see num_members)")
    mom = [moments(loc[a], scale[a], family) for a in vza]
    mean, ale, epi = (np.stack([m[i] for m in mom], axis=2) for i in range(3))
    pad = lambda v: np.concatenate([v, np.full(v.shape[:-1] + (S - v.shape[-1],), np.nan, v.dtype)], -1)
    loc = np.stack([pad(loc[a]) for a in vza], axis=2)      # [x, y, vza, S]
    scale = np.stack([pad(scale[a]) for a in vza], axis=2)
    nx, ny = mean.shape[:2]
    coords = {"x": np.arange(nx) * dx, "y": np.arange(ny) * dx, "vza": vza}
    var_attrs = {"units": "COT^2", "description": "variance in physical COT units"}
    data_vars = {
        "tau_pre": (("x", "y", "vza"), mean.astype(np.float32), {"units": "1"}),
        "aleatoric_uncertainty": (("x", "y", "vza"), ale.astype(np.float32), var_attrs),
        "epistemic_uncertainty": (("x", "y", "vza"), epi.astype(np.float32), var_attrs),
        "num_members": (("vza",), n_mem, {"description": "real members per view; member entries >= "
                                                         "num_members are NaN padding"}),
    }
    if not args.no_members:
        data_vars["tau_member_loc"] = (("x", "y", "vza", "member"), loc)
        data_vars["tau_member_scale"] = (("x", "y", "vza", "member"), scale,
                                         {"description": "Laplace b" if family == "laplace" else "Gaussian sigma"})
    ds = xr.Dataset(data_vars, coords=coords, attrs={"distribution": family,
                                                      "num_members": S,
                                                      "calibrated": int(all(calibrated)),
                                                      "configs": ";".join(used),
                                                      "skipped_configs": ";".join(skipped)})

    os.makedirs(args.save_dir, exist_ok=True)
    fout = os.path.join(args.save_dir, f"MISR_{dx * 1e3:.0f}m_80x80km_{args.time:.1f}h_optical_thickness_predicted{tag}.nc")
    ds.to_netcdf(fout)
    print(f"wrote {fout}")


if __name__ == "__main__":
    main()
