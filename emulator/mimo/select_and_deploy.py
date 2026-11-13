"""Select the best sweep trial(s) per network and deploy them for evaluation.

    python select_and_deploy.py -y config/deploy_40m.yaml [--dry-run]

For every sweep config listed in the deploy config:
  1. read ``<checkpoint_path>/runs/*/run_summary.json`` (written by pyct/train_cloud.py;
     optionally merged with the W&B API for ``source: wandb``);
  2. filter: status finished, finite score, >= ``min_epochs`` epochs, optional
     ``max_rmse_ratio`` guard (cloudy validation RMSE <= ratio x best RMSE of the sweep, so a
     well-calibrated but inaccurate model is never chosen);
  3. rank by the monitored score (default ``val_nll_mixture_cloudy``, lower is better);
  4. copy the top-k checkpoint(s) and the scalers to ``<deploy_root>/<tag>/`` (+ deploy_info.json);
     top_k > 1 deploys a deep ensemble of MIMO models via ``model_checkpoint_paths``;
  5. write ``<eval_config_dir>/config_<tag>_mimo_evaluate.yaml`` (input defaults to the held-out
     test scene ``test_input_fname``; calibration off until recalibrate.py has run).

Outputs in ``output_dir``: leaderboard_<tag>.csv, leaderboard_all.csv, T_sweep_selected.{csv,tex},
F_sweep_summary.{pdf,png}.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "emulator", "mimo", "pyct")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from emulator.mimo.pyct.utils import read_yaml, load_scalers  # noqa: E402
from emulator.mimo.pyct.mimo_inference import affine_inverse  # noqa: E402
from emulator.mimo.run_uq_analysis import to_latex  # noqa: E402


def tag_of(path: str) -> str:
    b = os.path.basename(path)
    for pre, suf in (("config_", "_mimo_sweep.yaml"), ("config_", ".yaml")):
        if b.startswith(pre) and b.endswith(suf):
            return b[len(pre):-len(suf)]
    return os.path.splitext(b)[0]


def load_summaries(sweep_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = []
    for f in sorted(glob.glob(os.path.join(sweep_cfg["checkpoint_path"], "runs", "*", "run_summary.json"))):
        with open(f) as fh:
            s = json.load(fh)
        s["_summary_path"] = f
        runs.append(s)
    return runs


def merge_wandb(runs, sweep_path: str, metric: str):
    """Fill in scores from the W&B API (summary ``<metric>`` min over history)."""
    import wandb
    api = wandb.Api()
    by_id = {r.get("wandb", {}).get("id"): r for r in runs if r.get("wandb")}
    for wr in api.sweep(sweep_path).runs:
        r = by_id.get(wr.id)
        if r is None:
            print(f"  [wandb] run {wr.id} has no local run directory; skipped (no checkpoint)")
            continue
        h = wr.history(keys=[metric], pandas=True)
        if len(h):
            r["best_score"] = float(np.nanmin(h[metric].values))
        r["status"] = {"finished": "finished"}.get(wr.state, r.get("status"))
    return runs


def leaderboard(runs, metric: str, out_a: float) -> pd.DataFrame:
    rows = []
    for r in runs:
        bm = r.get("best_metrics") or {}
        score = r.get("best_score") if metric == r.get("monitor") else bm.get(metric)
        mse = bm.get("metric_val/mse_cloudy")
        rows.append({"run": r["run_name"], "status": r.get("status"), "score": score,
                     "best_epoch": r.get("best_epoch"), "epochs_trained": r.get("epochs_trained"),
                     "val_nll_mixture": bm.get("val_nll_mixture"),
                     "val_rmse_cloudy_cot": math.sqrt(mse) * abs(out_a) if mse is not None and mse >= 0 else np.nan,
                     "trainable_params": r.get("trainable_params"),
                     "checkpoint": r.get("best_checkpoint"), "data_dir": r.get("data_dir"),
                     **{f"p_{k}": v for k, v in (r.get("params") or {}).items()}})
    df = pd.DataFrame(rows)
    if len(df):
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def select(df: pd.DataFrame, sel: Dict[str, Any]) -> pd.DataFrame:
    ok = df[(df.status == "finished") & np.isfinite(df.score)]
    ok = ok[ok.checkpoint.notna() & ok.checkpoint.map(lambda p: isinstance(p, str) and os.path.exists(p))]
    ok = ok[ok.best_epoch.fillna(-1) >= sel.get("min_epochs", 0)]
    if sel.get("max_rmse_ratio") and ok.val_rmse_cloudy_cot.notna().any():
        best_rmse = ok.val_rmse_cloudy_cot.min()
        ok = ok[ok.val_rmse_cloudy_cot <= sel["max_rmse_ratio"] * best_rmse]
    asc = sel.get("mode", "min") == "min"
    return ok.sort_values("score", ascending=asc).head(int(sel.get("top_k", 1)))


def deploy_one(tag: str, sweep_cfg_path: str, chosen: pd.DataFrame, runs, dcfg, dry: bool):
    sweep_cfg = read_yaml(sweep_cfg_path)
    by_name = {r["run_name"]: r for r in runs}
    best = by_name[chosen.iloc[0].run]
    data_dirs = {by_name[n]["data_dir"] for n in chosen.run}
    if len(data_dirs) > 1:
        raise RuntimeError(f"{tag}: top-k runs use different data caches/scalers {data_dirs}; use top_k: 1 "
                           "or do not sweep data keys")
    dep = os.path.join(dcfg["deploy_root"], tag)
    ckpts = []
    if not dry:
        os.makedirs(dep, exist_ok=True)
        for k, n in enumerate(chosen.run):
            dst = os.path.join(dep, "cloud_unet.ckpt" if k == 0 else f"cloud_unet_member{k}.ckpt")
            shutil.copy2(by_name[n]["best_checkpoint"], dst)
            ckpts.append(dst)
        for s in ("input_scaler.pkl", "output_scaler.pkl"):
            src = os.path.join(best["data_dir"], s)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dep, s))
    else:
        ckpts = [os.path.join(dep, "cloud_unet.ckpt")] + [os.path.join(dep, f"cloud_unet_member{k}.ckpt")
                                                         for k in range(1, len(chosen))]
    ev = dict(best["config"])
    for k in ("sweep", "_cloud_threshold_scaled"):
        ev.pop(k, None)
    test_in = dcfg.get("test_input_fname") or sweep_cfg.get("test_input_fname")
    test_tg = dcfg.get("test_target_fname") or sweep_cfg.get("test_target_fname")
    ev.update({"input_fname": [test_in], "target_fname": [test_tg], "model_dir": dep.rstrip("/") + "/",
               "state_dict": "cloud_unet.ckpt", "model_checkpoint_paths": ckpts if len(ckpts) > 1 else None,
               "out_dir": dep.rstrip("/") + "/", "load_and_scale": True, "apply_calibration": False,
               "calibration_file": os.path.join(dep, "calibration.json"),
               "deployed_from": {"sweep_config": sweep_cfg_path, "runs": list(chosen.run)}})
    os.makedirs(dcfg["eval_config_dir"], exist_ok=True)
    fcfg = os.path.join(dcfg["eval_config_dir"], f"config_{tag}_mimo_evaluate.yaml")
    with open(fcfg, "w") as fh:
        fh.write(f"# deployed by select_and_deploy.py from {sweep_cfg_path} (runs {', '.join(chosen.run)})\n")
        yaml.safe_dump(ev, fh, sort_keys=False, default_flow_style=None, width=120)
    info = {"tag": tag, "runs": list(chosen.run), "scores": [float(v) for v in chosen.score],
            "metric": dcfg.get("selection", {}).get("metric", best.get("monitor")),
            "best_epoch": best.get("best_epoch"), "params": best.get("params"),
            "trainable_params": best.get("trainable_params"), "checkpoints": ckpts,
            "source_checkpoints": [by_name[n]["best_checkpoint"] for n in chosen.run],
            "data_dir": best["data_dir"], "eval_config": fcfg}
    if not dry:
        with open(os.path.join(dep, "deploy_info.json"), "w") as fh:
            json.dump(info, fh, indent=2, default=str)
    return info


def summary_figure(lbs: Dict[str, pd.DataFrame], chosen: Dict[str, pd.DataFrame], out_dir, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = list(lbs)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    for i, t in enumerate(tags):
        d = lbs[t][(lbs[t].status == "finished") & np.isfinite(lbs[t].score)]
        jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(d))
        for axi, col in zip(ax, ("score", "val_rmse_cloudy_cot")):
            axi.plot(i + jit, d[col], "o", ms=2.5, color="0.6")
            if t in chosen and len(chosen[t]):
                axi.plot([i] * len(chosen[t]), chosen[t][col], "D", ms=4, color="C3")
    for axi, lab in zip(ax, ("val. NLL, cloudy (scaled units)", "val. RMSE, cloudy [COT]")):
        axi.set_xticks(range(len(tags)), tags, rotation=45, ha="right")
        axi.set_ylabel(lab)
        axi.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Sweep trials (grey) and deployed model (red)", fontsize=8)
    for ext in formats:
        fig.savefig(os.path.join(out_dir, f"F_sweep_summary.{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--dry-run", action="store_true", help="rank and write tables only; no copying")
    a = ap.parse_args()
    dcfg = read_yaml(a.yaml)
    sel = dcfg.get("selection", {})
    out = dcfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    lbs, chosen, infos, rows = {}, {}, [], []
    for spath in dcfg["sweep_configs"]:
        tag = tag_of(spath)
        scfg = read_yaml(spath)
        runs = load_summaries(scfg)
        metric = sel.get("metric") or scfg.get("monitor", "val_nll_mixture_cloudy")
        if dcfg.get("source", "local") == "wandb":
            sp = (dcfg.get("wandb_sweeps") or {}).get(tag)
            if sp:
                runs = merge_wandb(runs, sp, metric)
        if not runs:
            print(f"[{tag}] no runs found under {scfg['checkpoint_path']}runs/ ; skipped")
            continue
        out_a = 1.0
        rd = next((r.get("data_dir") for r in runs if r.get("data_dir")), None)
        if rd:
            _, osc = load_scalers({"out_dir": rd, "model_dir": rd})
            out_a = float(np.asarray(affine_inverse(osc)[0]).ravel()[0]) if osc is not None else 1.0
        lb = leaderboard(runs, metric, out_a)
        lb.sort_values("score", ascending=sel.get("mode", "min") == "min").to_csv(
            os.path.join(out, f"leaderboard_{tag}.csv"), index=False)
        lbs[tag] = lb
        c = select(lb, sel)
        if not len(c):
            print(f"[{tag}] no run passed the filters ({len(lb)} runs)")
            continue
        chosen[tag] = c
        info = deploy_one(tag, spath, c, runs, dcfg, a.dry_run)
        infos.append(info)
        b = c.iloc[0]
        n_fin = int(((lb.status == "finished") & np.isfinite(lb.score)).sum())
        angles = scfg["angles"]
        rows.append({"network": tag, "vza": "all" if len(angles) > 1 else f"{angles[0]:+.1f}",
                     "trials": len(lb), "finished": n_fin, "selected": ",".join(c.run),
                     "score": float(b.score), "val_RMSE_cloudy": float(b.val_rmse_cloudy_cot),
                     "best_epoch": int(b.best_epoch) if pd.notna(b.best_epoch) else -1,
                     "params_M": (b.trainable_params or 0) / 1e6,
                     **{k[2:]: b[k] for k in lb.columns if k.startswith("p_")}})
        print(f"[{tag}] deployed {list(c.run)} score={b.score:.4f} -> {info['eval_config']}")
    if lbs:
        pd.concat([d.assign(network=t) for t, d in lbs.items()]).to_csv(os.path.join(out, "leaderboard_all.csv"), index=False)
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(out, "T_sweep_selected.csv"), index=False)
        cols = [c for c in ["network", "vza", "trials", "score", "val_RMSE_cloudy", "best_epoch", "params_M",
                            "learning_rate", "mimo_num_subnetworks", "mimo_filter_base_count",
                            "mimo_input_repetition_probability", "mimo_batch_repetitions"] if c in t.columns]
        to_latex(t[cols], os.path.join(out, "T_sweep_selected.tex"),
                 "Selected sweep configuration per network (validation: 10.5 h, y < 0.5).", "tab:sweep_selected")
        with open(os.path.join(out, "deployed.json"), "w") as fh:
            json.dump(infos, fh, indent=2, default=str)
        summary_figure(lbs, chosen, out, dcfg.get("formats", ["pdf", "png"]))
    print(f"tables/figure -> {out}")


if __name__ == "__main__":
    main()
