"""Lightning data module for HybridCT emulator training.

split_mode
----------
``random`` (legacy, default): all tiles of ``input_fname``/``target_fname`` are pooled and
    split randomly (or stratified) into train/val/test. Overlapping tiles (stride < tile)
    leak between splits, so validation scores are optimistic.

``scene`` (recommended for sweeps / UQ): spatially disjoint splits.
    * train: tiles of ``input_fname``/``target_fname`` (optional ``train_region``);
    * val:   tiles of ``val_input_fname``/``val_target_fname`` restricted to ``val_region``
             (default [0, 0.5] of the y extent) -> model selection / early stopping;
    * cal:   tiles of ``calib_input_fname``/``calib_target_fname`` (default: the val files)
             restricted to ``calib_region`` (default [0.5, 1.0]) -> held out for recalibration
             checks (``data_test``).
    Scalers are fitted on the training tiles only and applied to val / cal tiles.

Cached arrays (x/y_{train,val,cal|test}.npy, scalers) are written to ``out_dir``; with
``load_and_scale: true`` they are re-used.
"""
import os
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
from torchvision import transforms

import utils
from generate_training_dataset import preprocess_global, preprocess_single_file
from DataLoader import Cloud_Dataset, MultiAngleRandomHorizontalFlip


def load_and_preprocess_train(config):
    if config["load_and_scale"]:
        train_test_val_dataset = utils.load_train_test_val_dataset(config['out_dir'])
    else:
        os.makedirs(config["out_dir"], exist_ok=True)
        input_tiled, target_tiled, nx, ny = preprocess_global(config)
        utils.save2dataset(input_tiled, target_tiled, nx, ny, config)
        print(f"rad tiled {input_tiled.shape}")
        input_tiles_scaled, target_tiles_scaled = utils.scale_training_data(input_tiled, target_tiled, config)
        train_test_val_dataset = utils.preprocess_training_data(
            input_tiles_scaled.transpose(0, 3, 1, 2), target_tiles_scaled.transpose(0, 3, 1, 2), config)
    print(f"train shape {train_test_val_dataset[0].shape}")
    return train_test_val_dataset


def _tiles(config, inputs, targets, region):
    xs, ys = [], []
    for fi, ft in zip(inputs, targets):
        x, y, *_ = preprocess_single_file(fi, ft, config, region=region)
        xs.append(x)
        ys.append(y)
    return np.vstack(xs).astype(np.float32), np.vstack(ys).astype(np.float32)


def _transform(x, y, config, in_scaler, out_scaler):
    if config.get("log_scale_input", False):
        x = utils.log_scale_data(x, True, config["fill_value"])
    if config.get("log_scale_target", False):
        y = utils.log_scale_data(y, True, config["fill_value"])
    if in_scaler is not None:
        x = utils.scaler_transform(in_scaler, x, method="transform")
    if out_scaler is not None:
        y = utils.scaler_transform(out_scaler, y, method="transform")
    return x.astype(np.float32), y.astype(np.float32)


def _as_list(v):
    return [v] if isinstance(v, str) else list(v)


def load_and_preprocess_scene_split(config):
    """Scene/region-disjoint train / val / cal split (see module docstring)."""
    out_dir = config["out_dir"]
    names = ["x_train", "x_val", "x_cal", "y_train", "y_val", "y_cal"]
    if config["load_and_scale"] and all(os.path.exists(os.path.join(out_dir, n + ".npy")) for n in names):
        arr = {n: np.load(os.path.join(out_dir, n + ".npy")).transpose(0, 3, 1, 2) for n in names}
        return arr["x_train"], arr["x_cal"], arr["x_val"], arr["y_train"], arr["y_cal"], arr["y_val"]
    os.makedirs(out_dir, exist_ok=True)
    val_in = _as_list(config["val_input_fname"])
    val_tg = _as_list(config["val_target_fname"])
    cal_in = _as_list(config.get("calib_input_fname") or val_in)
    cal_tg = _as_list(config.get("calib_target_fname") or val_tg)
    x_tr, y_tr = _tiles(config, config["input_fname"], config["target_fname"], config.get("train_region"))
    x_va, y_va = _tiles(config, val_in, val_tg, config.get("val_region", [0.0, 0.5]))
    x_ca, y_ca = _tiles(config, cal_in, cal_tg, config.get("calib_region", [0.5, 1.0]))
    print(f"scene split tiles: train {len(x_tr)}, val {len(x_va)}, cal {len(x_ca)}")
    # fit scalers on train only (also log-scales + saves input/output_scaler.pkl in out_dir)
    x_tr_s, y_tr_s = utils.scale_training_data(x_tr, y_tr, config)
    in_scaler, out_scaler = utils.load_scalers({**config, "model_dir": out_dir})
    x_va_s, y_va_s = _transform(x_va, y_va, config, in_scaler, out_scaler)
    x_ca_s, y_ca_s = _transform(x_ca, y_ca, config, in_scaler, out_scaler)
    arrs = {"x_train": x_tr_s, "y_train": y_tr_s, "x_val": x_va_s, "y_val": y_va_s,
            "x_cal": x_ca_s, "y_cal": y_ca_s}
    for n, a in arrs.items():  # same NHWC layout as preprocess_training_data
        np.save(os.path.join(out_dir, n), a.astype(np.float32))
    t = lambda a: np.ascontiguousarray(a.transpose(0, 3, 1, 2), dtype=np.float32)
    return t(x_tr_s), t(x_ca_s), t(x_va_s), t(y_tr_s), t(y_ca_s), t(y_va_s)


def cloud_threshold_scaled(config):
    """COT cloud threshold (``cloud_threshold`` or ``tau_min_threshold``) in scaled target units,
    one value per output channel (used for the cloudy-pixel validation metrics)."""
    thr = float(config.get("cloud_threshold", config.get("tau_min_threshold", 0.0)))
    _, out_scaler = utils.load_scalers({**config, "model_dir": config["out_dir"]})
    v = np.array([thr], dtype=np.float64)
    if config.get("log_scale_target", False):
        v = np.log(np.maximum(v, 1e-12))
    if out_scaler is None:
        return [float(v[0])] * len(config["angles"])
    n = out_scaler.n_features_in_
    return [float(s) for s in out_scaler.transform(np.repeat(v[None], n, axis=1))[0]]


class CloudDataModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.batch_size = self.config["batch_size"]
        self.cloud_threshold_scaled = None

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_is_setup", False):
            return
        if self.config.get("split_mode", "random") == "scene":
            x_train, x_test, x_val, y_train, y_test, y_val = load_and_preprocess_scene_split(self.config)
        else:
            x_train, x_test, x_val, y_train, y_test, y_val = load_and_preprocess_train(self.config)
        print(f"shape before training: X {x_train.shape}, y {y_train.shape}, val {x_val.shape}, test/cal {x_test.shape}")
        print("DATA STATS", x_train.min(), x_train.mean(), x_train.std(), x_train.max(),
              y_train.min(), y_train.mean(), y_train.std(), y_train.max())
        self.cloud_threshold_scaled = cloud_threshold_scaled(self.config)

        transformations = [transforms.ToTensor()]
        im_transform = transforms.Compose(list(transformations))
        if self.config.get("transform_fliph_prob", 0):
            transformations.append(MultiAngleRandomHorizontalFlip(p=self.config["transform_fliph_prob"]))
        if self.config.get("transform_flipv_prob", 0):
            transformations.append(transforms.RandomVerticalFlip(p=self.config["transform_flipv_prob"]))
        train_transform = transforms.Compose(transformations)

        self.data_train = Cloud_Dataset(x_train, y_train, transform_image=train_transform)
        self.data_valid = Cloud_Dataset(x_val, y_val, transform_image=im_transform)
        self.data_test = Cloud_Dataset(x_test, y_test, transform_image=im_transform)
        self._is_setup = True

    def _loader(self, ds, shuffle, drop_last=False):
        nw = int(self.config.get("num_workers", 8))
        avail = os.environ.get("SLURM_CPUS_PER_TASK")
        avail = int(avail) if avail and avail.isdigit() else len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        nw = max(0, min(nw, avail - 1))   # leave one core for the main process
        return torch.utils.data.DataLoader(ds, batch_size=self.config["batch_size"], num_workers=nw,
                                           drop_last=drop_last, shuffle=shuffle,
                                           persistent_workers=nw > 0)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self._loader(self.data_train, True, drop_last=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._loader(self.data_valid, False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._loader(self.data_test, False)
