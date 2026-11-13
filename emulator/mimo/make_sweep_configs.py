"""Generate the 40 m MIMO hyper-parameter sweep configs (9 single-angle + 1 multi-angle).

    python make_sweep_configs.py                     # -> config/sweep_40m/
    python make_sweep_configs.py --out config/sweep_40m --run-cap 40 --epochs 60

Design (all paths relative to emulator/mimo):
* train  : 12.0, 15.0, 18.0, 34.5 h scenes (full domain).
* val    : 10.5 h scene, y in [0, 0.5]   -> early stopping and model selection.
* cal    : 10.5 h scene, y in [0.5, 1.0] -> recalibration (recalibrate.py); never seen in training.
* test   : 30.0 h (toa) scene, held out entirely (it is the case-study scene).
  The primary 40 m study also trained on 30.0 h; it is excluded here so that the
  case study is out-of-sample.
* 40 m, tile 64, stride 32, MinMax scalers, no log scaling, Laplace NLL.
* No flips: solar azimuth 320 deg breaks mirror symmetry, and MultiAngleRandomHorizontalFlip
  also reverses the channel (view) order.
* Selection metric: exact mixture NLL on cloudy validation pixels (``val_nll_mixture_cloudy``).
* Multi-angle network: all 9 views as channels, registered to cloud_height_avg = 1.55 km.
"""
from __future__ import annotations

import argparse
import copy
import os

import yaml

TAGS = ["n705", "n60", "n456", "n261", "nadir", "261", "456", "60", "705"]
ANGLES = [-70.5, -60.0, -45.6, -26.1, 0.0, 26.1, 45.6, 60.0, 70.5]
DATA = "/scratch/3D-RT_LES-clouds/training_data/"
TRAIN_T = ["12.0", "15.0", "18.0", "34.5"]
OUT = "/scratch/3D-RT_LES-clouds/nlahaye/111226/"

def rad(t, suffix=""):
    return f"{DATA}/MISR_40m_radiances_phi0_SZA40_RICO_{t}h_AOD0.1_WIND10.0{suffix}.nc"


def cot(t, suffix=""):
    return f"{DATA}/MISR_40m_80x80km_{t}h_optical_thickness{suffix}.nc"


def sweep_block(run_cap: int, single: bool):
    return {
        "method": "bayes",
        "seed": 0,                      # local sampler only (removed before wandb.sweep)
        "run_cap": run_cap,
        "metric": {"name": "val_nll_mixture_cloudy", "goal": "minimize"},
        "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1.0e-4, "max": 3.0e-3},
            "batch_size": {"values": [16, 32]},
            "weight_decay": {"values": [0.0, 1.0e-5, 1.0e-4]},
            "mimo_num_subnetworks": {"values": [3, 4, 5, 7]},
            "mimo_filter_base_count": {"values": [16, 32, 48, 64] if single else [32, 48, 64]},
            "mimo_core_width_multiplier": {"values": [1, 2]},
            "mimo_input_repetition_probability": {"values": [0.0, 0.2, 0.4, 0.6]},
            "mimo_batch_repetitions": {"values": [1, 2, 4]},
            "mimo_loss_buffer_temperature": {"values": [0.0, 0.3, 1.0]},
            "mimo_core_dropout_rate": {"values": [0.0, 0.1]},
        },
    }


def base_config(tag: str, angles, multiview: bool, epochs: int, run_cap: int):
    out = f"{OUT}/{tag}/"
    c = {
        # ---- data ------------------------------------------------------------
        "split_mode": "scene",
        "input_fname": [rad(t) for t in TRAIN_T],
        "target_fname": [cot(t) for t in TRAIN_T],
        "train_region": None,
        "val_input_fname": [rad("10.5")],
        "val_target_fname": [cot("10.5")],
        "val_region": [0.0, 0.5],
        "calib_input_fname": [rad("10.5")],
        "calib_target_fname": [cot("10.5")],
        "calib_region": [0.5, 1.0],
        "test_input_fname": rad("30.0", "_toa"),
        "test_target_fname": cot("30.0", "_toa"),
        "input_field": "rad", "target_field": "tau", "grid_field": "y", "vza_field": "vza",
        "cloud_mask_field": "cloud_mask",
        "angles": list(angles), "multiview": multiview,
        "dx": 0.04, "resolution": 0.04, "tile_size": 64, "stride": 32, "offset_x": 0, "offset_y": 0,
        "input_scaler": "MinMax", "target_scaler": "MinMax",
        "log_scale_input": False, "log_scale_target": False, "fill_value": -9.999e-9,
        "stratify": False, "stratify_check_center": False, "stratify_center_window_size": 10,
        "test_ratio": 0.2, "val_ratio": 0.1, "cloudy_tile_percent_threshold": 0.8,
        "tau_min_threshold": 0.05, "cloud_threshold": 0.05,
        "transform_fliph_prob": 0.0, "transform_flipv_prob": 0.0,
        "process_image": True, "load_and_scale": False,
        # ---- model (defaults; overridden per trial by the sweep) ---------------
        "use_mimo": True, "mimo_loss": "laplace_nll",
        "mimo_num_subnetworks": 5, "mimo_filter_base_count": 32, "mimo_core_width_multiplier": 1,
        "mimo_encoder_dropout_rate": 0.0, "mimo_core_dropout_rate": 0.0, "mimo_decoder_dropout_rate": 0.0,
        "mimo_center_dropout_rate": 0.0, "mimo_final_dropout_rate": 0.0,
        "mimo_loss_buffer_size": 10, "mimo_loss_buffer_temperature": 0.3,
        "mimo_input_repetition_probability": 0.2, "mimo_batch_repetitions": 2,
        # ---- optimisation ----------------------------------------------------
        "batch_size": 32, "epochs": epochs, "learning_rate": 1.0e-3, "weight_decay": 0.0,
        "scheduler_step_size": 20, "scheduler_gamma": 0.5, "rng_seed": 1,
        "monitor": "val_nll_mixture_cloudy", "monitor_mode": "min", "early_stopping_patience": 15,
        "gradient_clip_val": 1.0,
        # ---- hardware / logging ---------------------------------------------
        "accelerator": "auto", "devices": 1, "num_workers": 8, "progress_bar": False,
        "use_wandb": False, "wandb_project": "HybridCT_MIMO_sweep_40m", "wandb_entity": None,
        # ---- paths -------------------------------------------------------------
        "checkpoint_path": out,
        "out_dir": out,                         # data cache -> <out_dir>/data_<hash>/
        "model_dir": f"{out}/artifacts/mimo_unet/sweep_40m/{tag}/",
        "state_dict": "cloud_unet.ckpt",
        "model_checkpoint_paths": None,
        "device": "auto",
        "sweep": sweep_block(run_cap, single=not multiview),
    }
    if multiview:
        c["cloud_height_avg"] = 1.55
        c["batch_size"] = 16
        c["sweep"]["parameters"]["batch_size"] = {"values": [8, 16]}
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="config/sweep_40m")
    ap.add_argument("--run-cap", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--epochs-multi", type=int, default=80)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = []
    for tag, ang in zip(TAGS, ANGLES):
        c = base_config(tag, [ang], False, a.epochs, a.run_cap)
        f = os.path.join(a.out, f"config_{tag}_mimo_sweep.yaml")
        with open(f, "w") as fh:
            fh.write(f"# 40 m single-angle MIMO sweep, vza {ang:+.1f} (generated by make_sweep_configs.py)\n")
            yaml.safe_dump(c, fh, sort_keys=False, default_flow_style=None, width=120)
        files.append(f)
    c = base_config("multiangle", ANGLES, True, a.epochs_multi, a.run_cap)
    f = os.path.join(a.out, "config_multiangle_mimo_sweep.yaml")
    with open(f, "w") as fh:
        fh.write("# 40 m multi-angle MIMO sweep, all 9 views registered at 1.55 km (generated by make_sweep_configs.py)\n")
        yaml.safe_dump(c, fh, sort_keys=False, default_flow_style=None, width=120)
    files.append(f)
    print("\n".join(files))


if __name__ == "__main__":
    main()
