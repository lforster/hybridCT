"""Compare several run_uq_analysis.py runs (e.g. single/multi x raw/calibrated) side by side.

Reads ``<run>/tables/T1_overall.csv`` of each labelled run, keeps one pixel subset
(default ``cloudy``) and writes, to ``output_dir``:

* ``C1_overall_long.csv``         all metrics, one row per variant x dataset x vza
* ``C2_summary.csv/.tex``         mean over view angles per variant x dataset
* ``C3_<metric>_wide.csv``        dataset x vza rows, one column per variant
* ``C4_deterministic.csv``        optional: cloudy RMSE per vza of deterministic prediction files
                                  (e.g. the primary-study U-Net) on the same scene
* ``C5_case_study.csv/.tex``      optional: merged case-study tables (3-D UQ, deterministic, mask)
* ``F_compare_vs_vza.{png,pdf}``  metrics vs VZA per variant for the selected datasets

Run from ``emulator/mimo``::

    python compare_uq_runs.py -y config/compare_40m.yaml
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import yaml  # noqa: E402

DEFAULTS = {
    "runs": {},                     # label -> run_uq_analysis output_dir
    "subset": "cloudy",
    "metrics": ["RMSE", "NLL", "CRPS", "ECE", "cov80", "width80", "epi_frac"],
    "plot_datasets": None,          # default: all datasets in the first run
    "deterministic": [],            # [{label, predicted_file, target_file, pred_var, truth_var, region}]
    "cot_threshold": 1.0,
    "case_studies": {},             # label -> cloud_case_study_uq.py output_dir
    "output_dir": "output/compare_40m",
    "formats": ["png", "pdf"],
}
TARGET = {"cov50": 0.5, "cov80": 0.8, "cov90": 0.9, "cov95": 0.95, "ECE": 0.0}


def load_config(path):
    cfg = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in DEFAULTS.items()}
    if path:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


def read_runs(cfg):
    frames = []
    for label, d in cfg["runs"].items():
        f = os.path.join(d, "tables", "T1_overall.csv")
        if not os.path.exists(f):
            print(f"[skip] {label}: {f} not found")
            continue
        t = pd.read_csv(f)
        t = t[t["subset"] == cfg["subset"]].copy()
        t.insert(0, "variant", label)
        frames.append(t)
    if not frames:
        raise SystemExit("no run tables found")
    return pd.concat(frames, ignore_index=True)


def crop_y(da, region):
    if region is None:
        return da
    ny = da.sizes["y"]
    return da.isel(y=slice(int(round(region[0] * ny)), int(round(region[1] * ny))))


def deterministic_rmse(entries, thr):
    rows = []
    for e in entries:
        p = xr.open_dataset(e["predicted_file"])[e.get("pred_var", "tau_pre")]
        t = xr.open_dataset(e["target_file"])[e.get("truth_var", "tau")]
        p, t = crop_y(p, e.get("region")), crop_y(t, e.get("region"))
        for v in p.vza.values:
            vt = t.sel(vza=v, method="nearest") if "vza" in t.dims else t
            a, b = np.asarray(p.sel(vza=v), float), np.asarray(vt, float)
            if a.shape != b.shape:
                a = a.T if a.T.shape == b.shape else a
            m = b >= thr
            rows.append({"variant": e["label"], "dataset": e.get("dataset", "test_30h"), "vza": float(v),
                         "subset": "cloudy", "N": int(m.sum()),
                         "RMSE": float(np.sqrt(np.mean((a[m] - b[m]) ** 2))), "bias": float(np.mean(a[m] - b[m]))})
    return pd.DataFrame(rows)


def to_tex(df, path, float_fmt="%.3g"):
    with open(path, "w") as f:
        f.write(df.to_latex(index=False, float_format=float_fmt, na_rep="--"))


def plot_vs_vza(long, metrics, datasets, out, fmts):
    variants = list(dict.fromkeys(long["variant"]))
    nr, nc = len(metrics), len(datasets)
    fig, ax = plt.subplots(nr, nc, figsize=(2.3 * nc + 0.6, 1.55 * nr + 0.6), squeeze=False, sharex=True)
    for j, ds in enumerate(datasets):
        sub = long[long["dataset"] == ds]
        for i, m in enumerate(metrics):
            a = ax[i, j]
            for k, v in enumerate(variants):
                s = sub[sub["variant"] == v].sort_values("vza")
                if m in s and s[m].notna().any():
                    a.plot(s["vza"], s[m], marker="o", ms=2.5, lw=1, color=f"C{k}", label=v)
            if m in TARGET:
                a.axhline(TARGET[m], color="0.6", lw=0.6, ls="--")
            if i == 0:
                a.set_title(ds, fontsize=8)
            if j == 0:
                a.set_ylabel(m, fontsize=8)
            if i == nr - 1:
                a.set_xlabel("VZA [deg]", fontsize=8)
            a.tick_params(labelsize=6.5)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=min(len(lab), 6), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (1.55 * nr + 0.6)))
    for f in fmts:
        fig.savefig(os.path.join(out, f"F_compare_vs_vza.{f}"), dpi=200)
    plt.close(fig)


def main(cfg):
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    long = read_runs(cfg)
    if cfg["deterministic"]:
        det = deterministic_rmse(cfg["deterministic"], cfg["cot_threshold"])
        det.to_csv(os.path.join(out, "C4_deterministic.csv"), index=False)
        long = pd.concat([long, det], ignore_index=True)
    long.to_csv(os.path.join(out, "C1_overall_long.csv"), index=False)

    metrics = [m for m in cfg["metrics"] if m in long]
    summ = long.groupby(["dataset", "variant"], sort=False)[metrics].mean().reset_index()
    summ.to_csv(os.path.join(out, "C2_summary.csv"), index=False)
    to_tex(summ, os.path.join(out, "C2_summary.tex"))
    for m in metrics:
        long.pivot_table(index=["dataset", "vza"], columns="variant", values=m, sort=False).to_csv(
            os.path.join(out, f"C3_{m}_wide.csv"))

    if cfg["case_studies"]:
        parts = []
        for label, d in cfg["case_studies"].items():
            for name in ("T_case_3d_uq", "T_case_3d_deterministic", "T_case_mask"):
                f = os.path.join(d, "tables", f"{name}.csv")
                if os.path.exists(f):
                    t = pd.read_csv(f)
                    t.insert(0, "table", name.replace("T_case_", ""))
                    t.insert(0, "variant", label)
                    parts.append(t)
        if parts:
            cs = pd.concat(parts, ignore_index=True)
            cs.to_csv(os.path.join(out, "C5_case_study.csv"), index=False)
            keep = [c for c in ("variant", "table", "reference", "reconstruction", "mask", "RMSE", "CRPS",
                                "cov80", "spread_skill_ratio", "IoU", "precision", "recall") if c in cs]
            to_tex(cs[keep], os.path.join(out, "C5_case_study.tex"))

    datasets = cfg["plot_datasets"] or list(dict.fromkeys(long["dataset"]))
    plot_vs_vza(long, metrics, datasets, out, cfg["formats"])
    print(summ.round(3).to_string(index=False))
    print(f"[done] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-y", "--yaml", required=True)
    main(load_config(ap.parse_args().yaml))
