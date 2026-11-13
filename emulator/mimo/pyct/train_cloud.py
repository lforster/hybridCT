"""Train HybridCT emulators (MIMO U-Net or deterministic U-Net), single runs or sweeps.

Run from ``emulator/mimo``:

    # single run (copies best checkpoint + scalers to model_dir/state_dict)
    python pyct/train_cloud.py -y CONFIG

    # build the tiled / scaled data cache once (avoids races between array jobs)
    python pyct/train_cloud.py -y CONFIG --prepare-data

    # local sweep from the config's ``sweep:`` block (wandb sweep format)
    python pyct/train_cloud.py -y CONFIG --sweep local --count 40        # sequential
    python pyct/train_cloud.py -y CONFIG --sweep local --trial-index $SLURM_ARRAY_TASK_ID

    # Weights & Biases sweep (creates the sweep unless --sweep-id ENTITY/PROJECT/ID is given)
    python pyct/train_cloud.py -y CONFIG --sweep wandb --count 10

Sweep parameters are config keys (e.g. ``learning_rate``, ``mimo_num_subnetworks``) and
override the base config per trial. Supported distributions in local mode: ``values``,
``value``, ``uniform``, ``log_uniform_values``, ``int_uniform`` (``q_*`` treated as their
continuous counterparts). ``method: grid`` enumerates the product of ``values``; ``random``
and ``bayes`` draw seeded random samples (trial i always gets the same parameters; bayes is
only adaptive in wandb mode). ``early_terminate`` is honoured by wandb only; locally,
``early_stopping_patience`` applies to every trial.

Every run writes ``<checkpoint_path>/runs/<run_name>/``:
    best.ckpt, last.ckpt, metrics.csv (CSV logger), run_summary.json
``run_summary.json`` (status, params, best_score, best_epoch, best_metrics, checkpoint paths,
data_dir, config, trainable_params) is what ``select_and_deploy.py`` reads.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import itertools
import json
import logging
import math
import os
import shutil
import sys
import traceback
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, "..", "..", ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from mimo_unet import MimoUnetModel
from cloud_datamodule import CloudDataModule
import utils

logger = logging.getLogger(__name__)

DATA_KEYS = ["input_fname", "target_fname", "val_input_fname", "val_target_fname",
             "calib_input_fname", "calib_target_fname", "train_region", "val_region", "calib_region",
             "split_mode", "input_field", "target_field", "angles", "multiview", "cloud_height_avg",
             "tile_size", "stride", "offset_x", "offset_y", "dx", "resolution", "input_scaler",
             "target_scaler", "log_scale_input", "log_scale_target", "fill_value", "stratify",
             "test_ratio", "val_ratio", "rng_seed"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.item() if o.numel() == 1 else o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(obj), f, indent=2)
    os.replace(tmp, path)


def data_dir_for(config: Dict[str, Any]) -> str:
    """Tile/scaler cache shared by all trials with the same data settings."""
    key = json.dumps({k: config.get(k) for k in DATA_KEYS}, sort_keys=True, default=str)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    return os.path.join(config["out_dir"], f"data_{h}")


def _cache_complete(d: str, config) -> bool:
    if config.get("split_mode", "random") == "scene":
        names = ["x_train", "x_val", "x_cal", "y_train", "y_val", "y_cal"]
    else:
        names = ["x_train", "x_val", "x_test", "y_train", "y_val", "y_test"]
    return all(os.path.exists(os.path.join(d, n + ".npy")) for n in names)


class _CacheLock:
    """Exclusive lock on the shared data cache.

    Uses flock; on file systems without flock support (e.g. Lustre mounted without -o flock)
    falls back to an atomic mkdir lock. A mkdir lock older than ``stale_s`` is treated as left
    over from a killed job and removed.
    """

    def __init__(self, d: str, stale_s: float = 6 * 3600):
        self.path, self.stale_s, self.fh, self.dir_lock = os.path.join(d, ".lock"), stale_s, None, None

    def __enter__(self):
        import time
        try:
            self.fh = open(self.path, "w")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
            return self
        except OSError:
            if self.fh:
                self.fh.close()
                self.fh = None
        self.dir_lock = self.path + ".d"
        while True:
            try:
                os.mkdir(self.dir_lock)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.dir_lock) > self.stale_s:
                        os.rmdir(self.dir_lock)
                        continue
                except OSError:
                    pass
                time.sleep(10)

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
        if self.dir_lock:
            try:
                os.rmdir(self.dir_lock)
            except OSError:
                pass


def prepare_data(config: Dict[str, Any]) -> CloudDataModule:
    """Set up the data module with a locked, hashed cache directory."""
    cfg = config
    d = data_dir_for(cfg)
    os.makedirs(d, exist_ok=True)
    cfg["out_dir"] = d
    with _CacheLock(d):
        cfg["load_and_scale"] = _cache_complete(d, cfg)
        if not cfg["load_and_scale"]:
            with open(os.path.join(d, "data_config.json"), "w") as f:
                json.dump(_jsonable({k: cfg.get(k) for k in DATA_KEYS}), f, indent=2)
        dm = CloudDataModule(cfg)
        dm.setup()
    return dm


def _resume_checkpoint(run_dir: str) -> Optional[str]:
    """Newest of last.ckpt and Lightning's SLURM-requeue hpc_ckpt_*.ckpt in an unfinished run."""
    import glob
    cands = [c for c in [os.path.join(run_dir, "last.ckpt")] + glob.glob(os.path.join(run_dir, "hpc_ckpt_*.ckpt"))
             if os.path.exists(c)]
    return max(cands, key=os.path.getmtime) if cands else None


def allocated_cpus() -> int:
    """CPUs granted to this job (SLURM_CPUS_PER_TASK / affinity), not the node total."""
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.isdigit():
        return int(v)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class BestMetricsTracker(Callback):
    """Keeps all logged metrics of the best epoch and refreshes run_summary.json."""

    def __init__(self, monitor: str, mode: str, summary: Dict[str, Any], summary_path: str):
        self.monitor, self.mode = monitor, mode
        self.summary, self.summary_path = summary, summary_path
        self.best = math.inf if mode == "min" else -math.inf

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = {k: float(v) for k, v in trainer.callback_metrics.items() if torch.as_tensor(v).numel() == 1}
        v = m.get(self.monitor)
        self.summary["last_epoch"] = int(trainer.current_epoch)
        self.summary["last_metrics"] = m
        if v is not None and math.isfinite(v) and (v < self.best if self.mode == "min" else v > self.best):
            self.best = v
            self.summary.update(best_score=v, best_epoch=int(trainer.current_epoch), best_metrics=m)
        write_json(self.summary_path, self.summary)


# --------------------------------------------------------------------------- #
# One training run
# --------------------------------------------------------------------------- #
def run_training(config: Dict[str, Any], params: Optional[Dict[str, Any]] = None,
                 run_name: Optional[str] = None, wandb_run=None) -> Dict[str, Any]:
    params = params or {}
    cfg = copy.deepcopy(config)
    cfg.pop("sweep", None)
    cfg.update(params)
    run_name = run_name or cfg.get("run_name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["checkpoint_path"], "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    monitor = cfg.get("monitor", "val_nll_mixture_cloudy" if cfg.get("use_mimo", True) else "val_loss")
    mode = cfg.get("monitor_mode", "min")
    summary_path = os.path.join(run_dir, "run_summary.json")
    resume_from = _resume_checkpoint(run_dir) if cfg.get("resume", True) else None
    previous = {}
    if resume_from and os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                previous = json.load(f)
        except (OSError, ValueError):
            previous = {}
    summary: Dict[str, Any] = {"run_name": run_name, "status": "running", "params": params,
                               "monitor": monitor, "mode": mode, "best_score": None, "best_epoch": None,
                               "best_metrics": {}, "started_at": datetime.now().isoformat(timespec="seconds"),
                               "run_dir": os.path.abspath(run_dir)}
    if resume_from:
        for k in ("best_score", "best_epoch", "best_metrics", "started_at"):
            if previous.get(k) is not None:
                summary[k] = previous[k]
        summary["resumed_from"] = os.path.abspath(resume_from)
        print(f"[{run_name}] resuming from {resume_from}")
    if wandb_run is not None:
        summary["wandb"] = {"id": wandb_run.id, "path": "/".join(wandb_run.path) if isinstance(wandb_run.path, (list, tuple)) else str(wandb_run.path)}
    write_json(summary_path, summary)

    try:
        pl.seed_everything(cfg.get("rng_seed", 1), workers=True)
        dm = prepare_data(cfg)
        summary["data_dir"] = os.path.abspath(cfg["out_dir"])
        cfg["_cloud_threshold_scaled"] = dm.cloud_threshold_scaled
        in_ch, out_ch = dm.data_train.X.shape[1], dm.data_train.y.shape[1]
        if cfg.get("use_mimo", True):
            model = MimoUnetModel.from_config(cfg, in_channels=in_ch, out_channels=out_ch)
            summary["trainable_params"] = int(model.hparams.trainable_params)
        else:
            from cloud_unet_pl import CloudUNetPL
            model = CloudUNetPL(cfg, in_channels=in_ch, out_channels=out_ch)
            summary["trainable_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print(f"[{run_name}] {summary['trainable_params']/1e6:.2f} M trainable parameters; params={params}")
        summary["config"] = cfg
        write_json(summary_path, summary)

        ckpt = ModelCheckpoint(dirpath=run_dir, filename="best", monitor=monitor, mode=mode,
                               save_top_k=1, save_last=True, auto_insert_metric_name=False)
        callbacks: List[Callback] = [ckpt, BestMetricsTracker(monitor, mode, summary, summary_path)]
        if cfg.get("early_stopping_patience"):
            callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=int(cfg["early_stopping_patience"]),
                                           check_finite=True))
        if wandb_run is not None:
            from lightning.pytorch.loggers.wandb import WandbLogger
            run_logger = WandbLogger(experiment=wandb_run, log_model=cfg.get("wandb_log_model", False))
            try:
                from callbacks import WandbMetricsDefiner
                callbacks.append(WandbMetricsDefiner())
            except Exception:
                pass
        else:
            run_logger = CSVLogger(save_dir=run_dir, name="", version="")

        devices = cfg.get("devices", 1)
        trainer = pl.Trainer(
            callbacks=callbacks, logger=run_logger,
            accelerator=cfg.get("accelerator", "auto"), devices=devices,
            strategy=cfg.get("strategy", "ddp" if (isinstance(devices, int) and devices > 1) else "auto"),
            precision=cfg.get("precision", "16-mixed" if torch.cuda.is_available() else "32-true"),
            deterministic=cfg.get("deterministic", False), max_epochs=cfg["epochs"],
            default_root_dir=run_dir, log_every_n_steps=cfg.get("log_every_n_steps", 10),
            gradient_clip_val=cfg.get("gradient_clip_val"),
            enable_progress_bar=cfg.get("progress_bar", True),
            limit_train_batches=cfg.get("limit_train_batches", 1.0),
            limit_val_batches=cfg.get("limit_val_batches", 1.0))
        torch.set_float32_matmul_precision("high")
        trainer.fit(model, dm, ckpt_path=resume_from)

        best = ckpt.best_model_path or ckpt.last_model_path
        summary.update(status="finished", best_checkpoint=os.path.abspath(best) if best else None,
                       last_checkpoint=os.path.abspath(ckpt.last_model_path) if ckpt.last_model_path else None,
                       epochs_trained=int(trainer.current_epoch),
                       stopped_early=bool(trainer.current_epoch < cfg["epochs"]),
                       finished_at=datetime.now().isoformat(timespec="seconds"))
        if ckpt.best_model_score is not None:
            summary["best_score"] = float(ckpt.best_model_score)
    except Exception as e:  # keep a record for the deploy step
        summary.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        write_json(summary_path, summary)
        raise
    write_json(summary_path, summary)
    return summary


def export_single_run(config: Dict[str, Any], summary: Dict[str, Any]) -> None:
    """Single-run convenience: copy best checkpoint + scalers into model_dir."""
    md = config.get("model_dir")
    if not md or not summary.get("best_checkpoint"):
        return
    os.makedirs(md, exist_ok=True)
    shutil.copy2(summary["best_checkpoint"], os.path.join(md, config.get("state_dict", "cloud_unet.ckpt")))
    for s in ("input_scaler.pkl", "output_scaler.pkl"):
        src = os.path.join(summary["data_dir"], s)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(md, s))
    print(f"exported best checkpoint (epoch {summary.get('best_epoch')}) and scalers to {md}")


# --------------------------------------------------------------------------- #
# Local sweep (wandb-format spec, no wandb dependency)
# --------------------------------------------------------------------------- #
def _sample_param(spec: Dict[str, Any], rng: np.random.Generator):
    if "value" in spec:
        return spec["value"]
    if "values" in spec:
        vals = spec["values"]
        return vals[int(rng.integers(len(vals)))]
    dist = spec.get("distribution", "uniform").replace("q_", "")
    lo, hi = spec.get("min"), spec.get("max")
    if dist == "uniform":
        return float(rng.uniform(lo, hi))
    if dist in ("log_uniform_values", "log_uniform"):
        if dist == "log_uniform":  # wandb: min/max are natural-log bounds
            return float(math.exp(rng.uniform(lo, hi)))
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    if dist == "int_uniform":
        return int(rng.integers(int(lo), int(hi) + 1))
    if dist == "normal":
        return float(rng.normal(spec.get("mu", 0.0), spec.get("sigma", 1.0)))
    raise ValueError(f"unsupported distribution {dist}")


def sweep_trial_params(sweep: Dict[str, Any], index: int) -> Dict[str, Any]:
    params = sweep["parameters"]
    if sweep.get("method", "random") == "grid":
        keys = list(params)
        grids = [params[k]["values"] if "values" in params[k] else [params[k]["value"]] for k in keys]
        combos = list(itertools.product(*grids))
        if index >= len(combos):
            raise IndexError(f"grid has {len(combos)} points; trial {index} out of range")
        return dict(zip(keys, combos[index]))
    rng = np.random.default_rng([int(sweep.get("seed", 0)), int(index)])
    return {k: _sample_param(v, rng) for k, v in sorted(params.items())}


def _sweep_size(sweep):
    if sweep.get("method") == "grid":
        n = 1
        for v in sweep["parameters"].values():
            n *= len(v.get("values", [0]))
        return n
    return int(sweep.get("run_cap", 10 ** 9))


def _trial_status(runs_dir: str, i: int) -> Optional[str]:
    f = os.path.join(runs_dir, f"trial_{i:03d}", "run_summary.json")
    if not os.path.exists(f):
        return None
    try:
        with open(f) as fh:
            return json.load(fh).get("status")
    except (OSError, ValueError):
        return None


def local_sweep(config: Dict[str, Any], count: Optional[int], trial_index: Optional[int]) -> None:
    """Finished trials are skipped (re-submitting an array is safe); failed/interrupted ones re-run."""
    sweep = config["sweep"]
    runs_dir = os.path.join(config["checkpoint_path"], "runs")
    os.makedirs(runs_dir, exist_ok=True)
    if trial_index is not None:
        idx = [int(trial_index)]
    else:
        todo = [i for i in range(_sweep_size(sweep)) if _trial_status(runs_dir, i) != "finished"]
        idx = todo[:count] if count else todo
    for i in idx:
        if i >= _sweep_size(sweep):
            print(f"trial {i} beyond run_cap / grid size; skipping")
            continue
        if _trial_status(runs_dir, i) == "finished":
            print(f"trial {i} already finished; skipping")
            continue
        params = sweep_trial_params(sweep, i)
        try:
            run_training(config, params, run_name=f"trial_{i:03d}")
        except Exception as e:
            print(f"trial {i} failed: {e}")
            if trial_index is not None:
                raise


def wandb_sweep(config: Dict[str, Any], count: Optional[int], sweep_id: Optional[str]) -> None:
    import wandb
    project = config.get("wandb_project", "Cloud")
    entity = config.get("wandb_entity")
    if sweep_id is None:
        spec = copy.deepcopy(config["sweep"])
        spec.pop("seed", None)
        sweep_id = wandb.sweep(sweep=spec, project=project, entity=entity)
        print(f"created sweep {sweep_id}")

    def _fn():
        run = wandb.init()
        params = dict(run.config)
        try:
            run_training(config, params, run_name=f"wandb_{run.id}", wandb_run=run)
        finally:
            run.finish()

    wandb.agent(sweep_id, function=_fn, count=count, project=project, entity=entity)


# --------------------------------------------------------------------------- #
def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    ap = ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-y", "--yaml", required=True, help="training / sweep config")
    ap.add_argument("--prepare-data", action="store_true", help="only build the data cache")
    ap.add_argument("--sweep", choices=["local", "wandb"], default=None)
    ap.add_argument("--count", type=int, default=None, help="number of trials for this process")
    ap.add_argument("--trial-index", type=int, default=None, help="local sweep: run exactly this trial")
    ap.add_argument("--sweep-id", default=None, help="wandb sweep id (ENTITY/PROJECT/ID) to join")
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args(argv)
    config = utils.read_yaml(a.yaml)
    config.setdefault("out_dir", os.path.join(config["checkpoint_path"], "data"))
    if a.prepare_data:
        dm = prepare_data(copy.deepcopy(config))
        print(f"data cache ready: {dm.config['out_dir']} (train {len(dm.data_train)}, val {len(dm.data_valid)}, "
              f"cal/test {len(dm.data_test)} tiles; cloud threshold scaled {dm.cloud_threshold_scaled})")
        return
    if a.sweep is None and config.get("run_tune"):
        a.sweep = "wandb"  # legacy flag
    if a.sweep == "local":
        local_sweep(config, a.count, a.trial_index)
    elif a.sweep == "wandb":
        wandb_sweep(config, a.count, a.sweep_id)
    else:
        summary = run_training(config, run_name=a.run_name)
        export_single_run(config, summary)


if __name__ == "__main__":
    main()
