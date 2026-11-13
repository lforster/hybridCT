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
from emulator.mimo.pyct.mimo_inference import load_model, moments, predict_scene_views

ANGLE_TAGS = ["nadir", "n261", "261", "n456", "456", "n60", "60", "n705", "705"]


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
    p.add_argument("--apply-calibration", action="store_true",
                   help="apply <model_dir>/calibration.json (from recalibrate.py) to every model")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dir = args.config_dir or f"config/{args.time:.0f}h"
    suffix = args.suffix if args.suffix is not None else ("_40m" if args.resolution == 40 else "")
    configs = args.configs or [os.path.join(cfg_dir, f"config_{t}_mimo_single_angle_evaluate{suffix}.yaml")
                               for t in ANGLE_TAGS]

    loc, scale, vza = {}, {}, []
    family, dx, calibrated = None, None, []
    for fname in configs:
        print(f"[evaluate_and_merge_mimo] {fname}")
        cfg = read_yaml(fname)
        if args.apply_calibration:
            cfg["apply_calibration"] = True
        out, dx = run_config(cfg, device, args.input_fname, args.target_fname)
        for c, a in enumerate(out["angles"]):
            if a in loc:
                raise ValueError(f"vza {a} provided by more than one config")
            loc[a], scale[a] = out["loc"][..., c, :], out["scale"][..., c, :]
        calibrated.append(bool(out["calibrated"]))
        family = out["family"]
    vza = sorted(loc)
    if len({loc[a].shape[-1] for a in vza}) > 1:
        raise ValueError("all views need the same number of members")
    loc = np.stack([loc[a] for a in vza], axis=2)      # [x, y, vza, S]
    scale = np.stack([scale[a] for a in vza], axis=2)
    mean, ale, epi = moments(loc, scale, family)
    nx, ny = mean.shape[:2]
    coords = {"x": np.arange(nx) * dx, "y": np.arange(ny) * dx, "vza": vza}
    var_attrs = {"units": "COT^2", "description": "variance in physical COT units"}
    data_vars = {
        "tau_pre": (("x", "y", "vza"), mean.astype(np.float32), {"units": "1"}),
        "aleatoric_uncertainty": (("x", "y", "vza"), ale.astype(np.float32), var_attrs),
        "epistemic_uncertainty": (("x", "y", "vza"), epi.astype(np.float32), var_attrs),
    }
    if not args.no_members:
        data_vars["tau_member_loc"] = (("x", "y", "vza", "member"), loc)
        data_vars["tau_member_scale"] = (("x", "y", "vza", "member"), scale,
                                         {"description": "Laplace b" if family == "laplace" else "Gaussian sigma"})
    ds = xr.Dataset(data_vars, coords=coords, attrs={"distribution": family,
                                                      "num_members": int(loc.shape[-1]),
                                                      "calibrated": int(all(calibrated)),
                                                      "configs": ";".join(configs)})

    os.makedirs(args.save_dir, exist_ok=True)
    tag = args.tag or ("_MIMO_40m_toa" if args.resolution == 40 else "_MIMO_toa")
    fout = os.path.join(args.save_dir, f"MISR_{dx * 1e3:.0f}m_80x80km_{args.time:.1f}h_optical_thickness_predicted{tag}.nc")
    ds.to_netcdf(fout)
    print(f"wrote {fout}")


if __name__ == "__main__":
    main()
