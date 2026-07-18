import numpy as np
import pandas as pd
from scipy import stats

import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp, mannwhitneyu
from sklearn.metrics import mean_squared_error
from evaluation import rmse
from evaluation_utils import linear_fit_stats, qcut_or_cut 

from utils import read_yaml

def laplace_interval_half_width_from_scale(scale_b, nominal=0.80):
    scale_b = np.maximum(scale_b, 1e-12)
    return -scale_b * np.log(1.0 - nominal)


def laplace_scale_from_variance(variance):
    variance = np.maximum(variance, 1e-12)
    return np.sqrt(variance / 2.0)


def ensure_df(
    y_true, y_pred, var_ale, var_epi,
    angle=None, scene=None, ablation_mask=None,
    cot_range_patch=None, cloud_fraction_patch=None, cot_var_patch=None,
    extra=None,
    drop_invalid=True,
):
    df = pd.DataFrame({
        "y_true": np.asarray(y_true).reshape(-1),
        "y_pred": np.asarray(y_pred).reshape(-1),
        "var_ale": np.asarray(var_ale).reshape(-1),
        "var_epi": np.asarray(var_epi).reshape(-1),
    })
    df["var_total"] = df["var_ale"] + df["var_epi"]
    df["err"] = df["y_pred"] - df["y_true"]
    df["abs_err"] = np.abs(df["err"])
    df["sq_err"] = df["err"] ** 2

    if angle is not None:
        df["angle"] = np.asarray(angle).reshape(-1)
    if scene is not None:
        df["scene"] = np.asarray(scene).reshape(-1)
    if cot_range_patch is not None:
        df["cot_range_patch"] = np.asarray(cot_range_patch).reshape(-1)
    if cloud_fraction_patch is not None:
        df["cloud_fraction_patch"] = np.asarray(cloud_fraction_patch).reshape(-1)
    if cot_var_patch is not None:
        df["cot_var_patch"] = np.asarray(cot_var_patch).reshape(-1)
    if ablation_mask is not None:
        df["ablation_mask"] = np.asarray(ablation_mask).reshape(-1)

    if extra:
        for k, v in extra.items():
            df[k] = np.asarray(v).reshape(-1)

    df = df.replace([np.inf, -np.inf], np.nan)
    if drop_invalid:
        df = df.dropna()

    return df


def add_patch_metrics_to_df(df, cot_patches, cloud_threshold=0.0, ddof=0):
    patches = np.asarray(cot_patches)

    if patches.ndim == 4 and patches.shape[-1] == 1:
        patches = patches[..., 0]

    if patches.ndim != 3:
        raise ValueError(
            f"Expected cot_patches with shape (N,H,W) or (N,H,W,1), got {patches.shape}"
        )

    n_patches, h, w = patches.shape
    pixels_per_patch = h * w
    expected_rows = n_patches * pixels_per_patch

    if len(df) != expected_rows:
        raise ValueError(
            f"df has {len(df)} rows, but cot_patches implies {expected_rows} per-pixel rows "
            f"({n_patches} patches x {h} x {w}). "
            "Call this before dropping invalid rows, or attach a patch_idx and merge."
        )

    flat = patches.reshape(n_patches, pixels_per_patch)

    cot_range_patch = np.nanmax(flat, axis=1) - np.nanmin(flat, axis=1)
    cloud_fraction_patch = np.mean(flat > cloud_threshold, axis=1)
    cot_var_patch = np.nanvar(flat, axis=1, ddof=ddof)

    df_out = df.copy()
    df_out["cot_range_patch"] = np.repeat(cot_range_patch, pixels_per_patch)
    df_out["cloud_fraction_patch"] = np.repeat(cloud_fraction_patch, pixels_per_patch)
    df_out["cot_var_patch"] = np.repeat(cot_var_patch, pixels_per_patch)

    if "patch_idx" not in df_out.columns:
        df_out["patch_idx"] = np.repeat(np.arange(n_patches), pixels_per_patch)

    return df_out


def coverage_from_laplace_scale(y_true, y_pred, scale_b, nominal=0.80):
    half_width = laplace_interval_half_width_from_scale(scale_b, nominal=nominal)
    lo = y_pred - half_width
    hi = y_pred + half_width
    return ((y_true >= lo) & (y_true <= hi)).mean()

def interval_membership_from_laplace_scale(y_true, y_pred, scale_b, nominal=0.80):
    half_width = laplace_interval_half_width_from_scale(scale_b, nominal=nominal)
    lo = y_pred - half_width
    hi = y_pred + half_width
    return (y_true >= lo) & (y_true <= hi)

def reliability_by_variance(
    df,
    variance_col="var_total",
    group_cols=("angle", "scene"),
    n_bins=10,
    nominal=0.80
):
    z = stats.norm.ppf((1 + nominal) / 2)
    out = []

    for keys, g in df.groupby(list(group_cols)):
        g = g.copy()
        if len(g) < n_bins:
            continue

        g["var_bin"] = qcut_or_cut(g[variance_col].values, n_bins=n_bins)

        for b, gb in g.groupby("var_bin", observed=False):
            if len(gb) == 0:
                continue
            row = {}
            if isinstance(keys, tuple):
                for c, k in zip(group_cols, keys):
                    row[c] = k
            else:
                row[group_cols[0]] = keys

            scale_b = laplace_scale_from_variance(gb["var_ale"] + gb["var_epi"])

            row.update({
                "variance_type": variance_col,
                "bin": str(b),
                "n": len(gb),
                "var_mean": gb[variance_col].mean(),
                "var_median": gb[variance_col].median(),
                "rmse": rmse(gb["y_pred"], gb["y_true"]),
                "bias": (gb["y_pred"] - gb["y_true"]).mean(),
                "coverage": coverage_from_laplace_scale(
                    gb["y_true"].values,
                    gb["y_pred"].values,
                    scale_b,
                    nominal=nominal,
                ),
            })
            out.append(row)


    return pd.DataFrame(out)


def plot_reliability_curves(rel_df, angle, scene=None):
    sub = rel_df[rel_df["angle"] == angle].copy()
    if scene is not None and "scene" in sub.columns:
        sub = sub[sub["scene"] == scene]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(
        data=sub, x="var_mean", y="rmse", hue="variance_type",
        marker="o", ax=axes[0]
    )
    axes[0].set_xlabel("Mean predicted variance in bin")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title(f"RMSE vs predicted variance: {angle}")

    sns.lineplot(
        data=sub, x="var_mean", y="coverage", hue="variance_type",
        marker="o", ax=axes[1]
    )
    axes[1].axhline(0.80, ls="--", c="k", alpha=0.7)
    axes[1].set_xlabel("Mean predicted variance in bin")
    axes[1].set_ylabel("Empirical 80% coverage")
    axes[1].set_title(f"Coverage vs predicted variance: {angle}")

    plt.tight_layout()
    return fig


def plot_reliability(results_dir, fig_dir):
    df = _read_if_exists(results_dir / "reliability_all.csv")
    if df is None or df.empty:
        return []

    outputs = []
    group_cols = [c for c in ["angle", "scene"] if c in df.columns]
    if not group_cols:
        group_iter = [("all", df)]
    elif group_cols == ["angle", "scene"]:
        group_iter = df.groupby(group_cols)
    else:
        group_iter = df.groupby(group_cols[0])

    for keys, sub in group_iter:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_label = "_".join(str(k) for k in keys)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.lineplot(data=sub, x="var_mean", y="rmse", hue="variance_type", marker="o", ax=axes[0])
        axes[0].set_xlabel("Mean predicted variance in bin")
        axes[0].set_ylabel("RMSE")
        axes[0].set_title(f"Reliability RMSE: {key_label}")

        sns.lineplot(data=sub, x="var_mean", y="coverage", hue="variance_type", marker="o", ax=axes[1])
        axes[1].axhline(0.80, ls="--", c="k", alpha=0.8)
        axes[1].set_xlabel("Mean predicted variance in bin")
        axes[1].set_ylabel("Observed coverage")
        axes[1].set_title(f"Reliability coverage: {key_label}")

        out_path = fig_dir / f"reliability_{key_label}.png"
        #save(fig, out_path)
        plt.savefig(out_path, dpi=400)
        plt.clf()
        plt.close()
        outputs.append(out_path)
    return outputs





def sample_indices_by_variance_bin(df, variance_col="var_epi", n_bins=5, samples_per_bin=8):
    g = df.copy()
    g["var_bin"] = qcut_or_cut(g[variance_col].values, n_bins=n_bins)
    sampled = (
        g.groupby("var_bin", observed=False)
         .apply(lambda x: x.sample(min(len(x), samples_per_bin), random_state=0))
         .reset_index(drop=True)
    )
    return sampled



def compare_variance_distributions(
    df,
    baseline_mask,
    comparison_col="scene",
    variance_cols=("var_ale", "var_epi", "var_total")
):
    base = df[baseline_mask].copy()
    results = []

    for cond, g in df.groupby(comparison_col):
        if g.index.equals(base.index):
            continue

        row = {"condition": cond, "n": len(g)}
        for vc in variance_cols:
            x = base[vc].values
            y = g[vc].values

            ks = ks_2samp(x, y)
            mw = mannwhitneyu(x, y, alternative="two-sided")
            row[f"{vc}_baseline_median"] = np.median(x)
            row[f"{vc}_cond_median"] = np.median(y)
            row[f"{vc}_median_ratio"] = np.median(y) / max(np.median(x), 1e-12)
            row[f"{vc}_mean_ratio"] = np.mean(y) / max(np.mean(x), 1e-12)
            row[f"{vc}_wasserstein"] = wasserstein_distance(x, y)
            row[f"{vc}_ks_stat"] = ks.statistic
            row[f"{vc}_ks_p"] = ks.pvalue
            row[f"{vc}_mw_p"] = mw.pvalue
        results.append(row)

    return pd.DataFrame(results)



def plot_ood_id_variances(df, out_dir, comparison_col="scene"):
    long_df = df.melt(
        id_vars=[comparison_col],
        value_vars=["var_ale", "var_epi", "var_total"],
        var_name="variance_type",
        value_name="variance"
    )

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=long_df, x=comparison_col, y="variance", hue="variance_type", showfliers=False)
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    #plt.show()
    plt_dir = os.path.join(out_dir, "OOD_ID_variances.png")
    plt.savefig(plt_dir, dpi=400)
    plt.clf()
    plt.close()

def add_variance_ratios(df):
    df = df.copy()
    df["epi_to_ale_ratio"] = df["var_epi"] / np.maximum(df["var_ale"], 1e-12)
    df["epi_fraction"] = df["var_epi"] / np.maximum(df["var_total"], 1e-12)
    return df


def stratified_summary(
    df,
    strat_col,
    angle_col="angle",
    n_bins=8,
    log_bins=False
):
    out = []

    for angle, ga in df.groupby(angle_col):
        g = ga.copy()
        g["strat_bin"] = qcut_or_cut(g[strat_col].values, n_bins=n_bins, logspace=log_bins)

        for b, gb in g.groupby("strat_bin", observed=False):
            if len(gb) < 5:
                continue

            fit = linear_fit_stats(gb["y_true"].values, gb["y_pred"].values)
            out.append({
                "angle": angle,
                "strat_col": strat_col,
                "bin": str(b),
                "n": len(gb),
                "bin_center": gb[strat_col].median(),
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "rmse": fit["rmse"],
                "bias": fit["bias"],
                "r": fit["r"],
                "var_ale_median": gb["var_ale"].median(),
                "var_ale_iqr": gb["var_ale"].quantile(0.75) - gb["var_ale"].quantile(0.25),
                "var_epi_median": gb["var_epi"].median(),
                "var_epi_iqr": gb["var_epi"].quantile(0.75) - gb["var_epi"].quantile(0.25),
                "var_total_median": gb["var_total"].median(),
                "var_total_iqr": gb["var_total"].quantile(0.75) - gb["var_total"].quantile(0.25),
            })

    return pd.DataFrame(out)


def plot_stratified_metrics(df, results_dir, fig_dir):
    outputs = [] 
    for strat_col in ["cot_range_patch", "cloud_fraction_patch", "cot_var_patch"]:
        angle_groups = df.groupby("angle") if "angle" in df.columns else [("all", df)]
        for angle, sub in angle_groups:
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            metric_map = [
                ("slope", "Slope"),
                ("intercept", "Intercept"),
                ("rmse", "RMSE"),
                ("bias", "Bias"),
                ("r", "Correlation"),
                ("var_total_median", "Median total variance"),
            ]
            for ax, (metric, title) in zip(axes.ravel(), metric_map):
                sns.lineplot(data=sub, x="bin_center", y=metric, marker="o", ax=ax)
                ax.set_title(title)
                ax.set_xlabel(strat_col)
            fig.suptitle(f"Stratified metrics: {strat_col} | {angle}", y=1.02)
            out_path = fig_dir + f"stratified_{strat_col}_{angle}.png"
            plt.savefig(out_path, dpi=400)
            plt.clf()
            plt.close()
            outputs.append(out_path)
    return outputs


def plot_ece(df, results_dir, fig_dir):
    outputs = []
    for strat_col in ["cot_range_patch", "cloud_fraction_patch", "cot_var_patch"]:
        df = _read_if_exists(results_dir / f"ece_{strat_col}.csv")
        if df is None or df.empty:
            continue
        angle_groups = df.groupby("angle") if "angle" in df.columns else [("all", df)]
        for angle, sub in angle_groups:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.lineplot(data=sub, x="bin_center", y="ece", marker="o", ax=axes[0])
            axes[0].set_xlabel(strat_col)
            axes[0].set_ylabel("ECE")
            axes[0].set_title(f"ECE vs {strat_col}: {angle}")

            sns.lineplot(data=sub, x="bin_center", y="coverage_80", marker="o", ax=axes[1])
            axes[1].axhline(0.80, ls="--", c="k", alpha=0.8)
            axes[1].set_xlabel(strat_col)
            axes[1].set_ylabel("Observed 80% coverage")
            axes[1].set_title(f"80% coverage vs {strat_col}: {angle}")

            out_path = fig_dir / f"ece_{strat_col}_{angle}.png"
            #_save(fig, out_path)
            plt.savefig(out_path, dpi=400)
            outputs.append(out_path)
    return outputs


def patch_metrics_from_truth(cot_patch, cloud_threshold=0.0):
    cot_patch = np.asarray(cot_patch)
    return {
        "cot_range_patch": float(np.nanmax(cot_patch) - np.nanmin(cot_patch)),
        "cloud_fraction_patch": float(np.mean(cot_patch > cloud_threshold)),
        "cot_var_patch": float(np.nanvar(cot_patch)),
    }


def regression_ece(y_true, y_pred, variance, nominal_levels=None):
    if nominal_levels is None:
        nominal_levels = np.linspace(0.1, 0.9, 9)

    abs_gaps = []
    rows = []

    for p in nominal_levels:
        half_width = laplace_interval_half_width_from_scale(variance, nominal=p)
        lo = y_pred - half_width
        hi = y_pred + half_width
        obs = ((y_true >= lo) & (y_true <= hi)).mean()
        gap = abs(obs - p)
        abs_gaps.append(gap)
        rows.append({"nominal": p, "observed": obs, "abs_gap": gap})


    return np.mean(abs_gaps), pd.DataFrame(rows)


def stratified_ece(df, strat_col, angle_col="angle", n_bins=8, log_bins=False):
    out = []
    detail = []

    for angle, ga in df.groupby(angle_col):
        g = ga.copy()
        g["strat_bin"] = qcut_or_cut(g[strat_col].values, n_bins=n_bins, logspace=log_bins)

        for b, gb in g.groupby("strat_bin", observed=False):
            if len(gb) < 10:
                continue

            scale_b = laplace_scale_from_variance(gb["var_total"].values)


            ece, curves = regression_ece(
                gb["y_true"].values,
                gb["y_pred"].values,
                gb["var_total"].values
            )

            out.append({
                "angle": angle,
                "strat_col": strat_col,
                "bin": str(b),
                "bin_center": gb[strat_col].median(),
                "n": len(gb),
                "ece": ece,
                "coverage_80": coverage_from_laplace_scale(
                    gb["y_true"].values, gb["y_pred"].values, scale_b, nominal=0.80)
            })

            curves = curves.copy()
            curves["angle"] = angle
            curves["strat_col"] = strat_col
            curves["bin"] = str(b)
            curves["bin_center"] = gb[strat_col].median()
            detail.append(curves)

    return pd.DataFrame(out), pd.concat(detail, ignore_index=True)






def plot_ece_vs_strat(summary_df, angle, strat_col):
    sub = summary_df[(summary_df["angle"] == angle) & (summary_df["strat_col"] == strat_col)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(data=sub, x="bin_center", y="ece", marker="o", ax=axes[0])
    axes[0].set_title(f"ECE vs {strat_col} ({angle})")
    axes[0].set_xlabel(strat_col)
    axes[0].set_ylabel("ECE")

    sns.lineplot(data=sub, x="bin_center", y="coverage_80", marker="o", ax=axes[1])
    axes[1].axhline(0.80, ls="--", c="k")
    axes[1].set_title(f"80% interval coverage vs {strat_col} ({angle})")
    axes[1].set_xlabel(strat_col)
    axes[1].set_ylabel("Observed coverage")

    plt.tight_layout()
    return fig



def coverage80_by_group(df, group_cols, interval_convention="aleatoric_b2_plus_epistemic_variance"):
    rows = []
    for keys, g in df.groupby(group_cols):
        scale_b = laplace_scale_from_variance(g["var_total"].values)

        row = {}
        if isinstance(keys, tuple):
            for c, k in zip(group_cols, keys):
                row[c] = k
        else:
            row[group_cols[0]] = keys

        cov80 = coverage_from_laplace_scale(
            g["y_true"].values,
            g["y_pred"].values,
            scale_b,
            nominal=0.80,
        )
        row["n"] = len(g)
        row["coverage_80"] = cov80
        row["exceeds_target"] = cov80 > 0.80
        rows.append(row)
    return pd.DataFrame(rows)


def pixelwise_interval_hit_map_laplace(y_true_img, y_pred_img, scale_b_img, nominal=0.80):
    half_width = laplace_interval_half_width_from_scale(scale_b_img, nominal=nominal)
    lo = y_pred_img - half_width
    hi = y_pred_img + half_width
    return ((y_true_img >= lo) & (y_true_img <= hi)).astype(np.uint8)

def coverage80_stratified(df, strat_col, angle_col="angle", n_bins=8, log_bins=False):
    g = df.copy()
    g["strat_bin"] = None

    for angle, idx in g.groupby(angle_col).groups.items():
        bins = qcut_or_cut(g.loc[idx, strat_col].values, n_bins=n_bins, logspace=log_bins)
        g.loc[idx, "strat_bin"] = bins.astype(str)

    out = []
    for (angle, b), gb in g.groupby([angle_col, "strat_bin"]):

        scale_b = laplace_scale_from_variance(gb["var_ale"] + gb["var_epi"])
 
        out.append({
            "angle": angle,
            "strat_col": strat_col,
            "bin": b,
            "bin_center": gb[strat_col].median(),
            "n": len(gb),


            "coverage_80": coverage_from_laplace_scale(
                gb["y_true"].values,
                gb["y_pred"].values,
                scale_b,
                nominal=0.80,
            )

        })
    return pd.DataFrame(out)



def plot_coverage80_focus(summary_df, focus_angles=("An", "Aa", "Af"), strat_col="cot_range_patch"):
    sub = summary_df[
        (summary_df["angle"].isin(focus_angles)) &
        (summary_df["strat_col"] == strat_col)
    ].copy()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=sub, x="bin_center", y="coverage_80", hue="angle", marker="o")
    plt.axhline(0.80, ls="--", c="k")
    plt.xlabel(strat_col)
    plt.ylabel("Observed 80% coverage")
    plt.title("80% coverage in nadir / one-off-nadir angles")
    plt.tight_layout()
    #plt.show()
    plt_dir = os.path.join(out_dir, "Coverage_80_Focus.png")
    plt.savefig(plt_dir, dpi=400)
    plt.clf()
    plt.close()


def pixelwise_interval_hit_map(y_true_img, y_pred_img, var_total_img, nominal=0.80):
    z = stats.norm.ppf((1 + nominal) / 2)
    sigma = np.sqrt(np.maximum(var_total_img, 1e-12))
    lo = y_pred_img - z * sigma
    hi = y_pred_img + z * sigma
    return ((y_true_img >= lo) & (y_true_img <= hi)).astype(np.uint8)


def mimo_uncertainty_from_heads(mu_heads, b_heads, axis=0):
    mu_heads = np.asarray(mu_heads)
    b_heads = np.asarray(b_heads)

    mu_bar = np.mean(mu_heads, axis=axis)
    var_ale = np.mean(b_heads ** 2, axis=axis)
    var_epi = np.mean((mu_heads - np.expand_dims(mu_bar, axis=axis)) ** 2, axis=axis)
    var_total = var_ale + var_epi

    return {
        "y_pred": mu_bar,
        "var_ale": var_ale,
        "var_epi": var_epi,
        "var_total": var_total,
    }


"""
def run_all_uq_diagnostics(df):
    outputs = {}

    # Reliability curves
    rel_frames = []
    for vc in ["var_total", "var_ale", "var_epi"]:
        rel_frames.append(reliability_by_variance(df, variance_col=vc, group_cols=("angle", "scene")))
    outputs["reliability"] = pd.concat(rel_frames, ignore_index=True)

    # Stratified summaries
    outputs["strat_cot_range"] = stratified_summary(df, "cot_range_patch", n_bins=8, log_bins=True)
    outputs["strat_cloud_fraction"] = stratified_summary(df, "cloud_fraction_patch", n_bins=8, log_bins=False)
    outputs["strat_cot_var"] = stratified_summary(df, "cot_var_patch", n_bins=8, log_bins=True)

    # Stratified ECE
    outputs["ece_cot_range"], outputs["ece_cot_range_detail"] = stratified_ece(df, "cot_range_patch", n_bins=8, log_bins=True)
    outputs["ece_cloud_fraction"], outputs["ece_cloud_fraction_detail"] = stratified_ece(df, "cloud_fraction_patch", n_bins=8)
    outputs["ece_cot_var"], outputs["ece_cot_var_detail"] = stratified_ece(df, "cot_var_patch", n_bins=8, log_bins=True)

    # 80% coverage summaries
    outputs["coverage80_angle_scene"] = coverage80_by_group(df, ["angle", "scene"])
    outputs["coverage80_cot_range"] = coverage80_stratified(df, "cot_range_patch", n_bins=8, log_bins=True)
    outputs["coverage80_cloud_fraction"] = coverage80_stratified(df, "cloud_fraction_patch", n_bins=8)
    outputs["coverage80_cot_var"] = coverage80_stratified(df, "cot_var_patch", n_bins=8, log_bins=True)

    return outputs
"""

def run_all_uq_diagnostics(df):
    """
    Convenience function to reproduce the full set of experiment plan diagnostics:

    - Reliability curves (total, aleatoric, epistemic variance)
    - Stratified performance summaries vs COT range, cloud fraction, COT variability
    - ECE and 80% coverage vs those stratifications
    - 80% coverage by angle / scene

    Returns a dictionary of dataframes keyed by analysis name. [file:1]
    """
    outputs = {}

    # Reliability curves
    rel_frames = []
    for vc in ["var_total", "var_ale", "var_epi"]:
        rel_frames.append(
            reliability_by_variance(df, variance_col=vc, group_cols=("angle", "scene"))
        )
    outputs["reliability"] = pd.concat(rel_frames, ignore_index=True)

    # Stratified performance summaries
    if "cot_range_patch" in df.columns:
        outputs["strat_cot_range"] = stratified_summary(
            df,
            "cot_range_patch",
            n_bins=8,
            log_bins=True,
        )
    if "cloud_fraction_patch" in df.columns:
        outputs["strat_cloud_fraction"] = stratified_summary(
            df,
            "cloud_fraction_patch",
            n_bins=8,
            log_bins=False,
        )
    if "cot_var_patch" in df.columns:
        outputs["strat_cot_var"] = stratified_summary(
            df,
            "cot_var_patch",
            n_bins=8,
            log_bins=True,
        )

    # Stratified ECE
    if "cot_range_patch" in df.columns:
        outputs["ece_cot_range"], outputs["ece_cot_range_detail"] = stratified_ece(
            df,
            "cot_range_patch",
            n_bins=8,
            log_bins=True,
        )
    if "cloud_fraction_patch" in df.columns:
        outputs["ece_cloud_fraction"], outputs["ece_cloud_fraction_detail"] = stratified_ece(
            df,
            "cloud_fraction_patch",
            n_bins=8,
            log_bins=False,
        )
    if "cot_var_patch" in df.columns:
        outputs["ece_cot_var"], outputs["ece_cot_var_detail"] = stratified_ece(
            df,
            "cot_var_patch",
            n_bins=8,
            log_bins=True,
        )

    # 80% coverage summaries
    outputs["coverage80_angle_scene"] = coverage80_by_group(df, ["angle", "scene"])

    if "cot_range_patch" in df.columns:
        outputs["coverage80_cot_range"] = coverage80_stratified(
            df,
            "cot_range_patch",
            n_bins=8,
            log_bins=True,
        )
    if "cloud_fraction_patch" in df.columns:
        outputs["coverage80_cloud_fraction"] = coverage80_stratified(
            df,
            "cloud_fraction_patch",
            n_bins=8,
            log_bins=False,
        )
    if "cot_var_patch" in df.columns:
        outputs["coverage80_cot_var"] = coverage80_stratified(
            df,
            "cot_var_patch",
            n_bins=8,
            log_bins=True,
        )

    return outputs



def main(config):

    scenes = config["scenes"]

    #TODO - loop through scenes keys, build angles and scnes arrs (angles from keys, scenes from uid), read data, stack

    angles = config["angles"]
    ablation_mask = config["ablation_mask"]

    
    rad_full = None
    output_full = None
    target_full = None
    ep_u_full = None
    al_u_full = None
    angles_full = None
    scenes_full = None
    ablation_mask_full = None

    rad_tiled_full = None
    output_tiled_full = None
    target_tiled_full = None
    ep_u_tiled_full = None
    al_u_tiled_full = None
    angles_tiled_full = None
    scenes_tiled_full = None
    ablation_mask_tiled_full = None

    for a in range(len(angles)):
        angle = angles[a]
        out_dir = config["dir_base"] + angle + config["dir_end"]
        for s in range(len(scenes)):
            abl_mask = ablation_mask[s]
            scene = scenes[s]
            uid = scene
            out_subdir = os.path.join(out_dir, uid)
 

            rad = np.load(os.path.join(out_subdir, f"{uid}_unscaled_inputs.npy"))
            output = np.load(os.path.join(out_subdir, f"{uid}_unscaled_y_preds.npy"))
            target = np.load(os.path.join(out_subdir, f"{uid}_unscaled_y_trues.npy"))
            ep_u = np.load(os.path.join(out_subdir, f"{uid}_unscaled_epistemic_vars.npy"))
            al_u = np.load(os.path.join(out_subdir, f"{uid}_unscaled_aleatoric_vars.npy"))

            if rad_full is not None:
                rad_full = np.concatenate((rad_full, rad.flatten()))
                output_full = np.concatenate((output_full, output.flatten()))
                target_full = np.concatenate((target_full, target.flatten()))
                ep_u_full = np.concatenate((ep_u_full, ep_u.flatten()))
                al_u_full = np.concatenate((al_u_full, al_u.flatten()))
               
                print(angles_full.shape, scenes_full.shape, np.full(al_u.flatten().shape, angle).shape, np.full(al_u.flatten().shape, scene).shape)
                angles_full = np.concatenate((angles_full, np.full(al_u.flatten().shape, angle)))
                scenes_full = np.concatenate((scenes_full, np.full(al_u.flatten().shape, scene)))
                ablation_mask_full = np.concatenate((ablation_mask_full, np.full(al_u.flatten().shape, abl_mask)))
            else:
                rad_full = rad.flatten()
                output_full = output.flatten()
                target_full = target.flatten()
                ep_u_full = ep_u.flatten()
                al_u_full = al_u.flatten()
                angles_full = np.full(al_u.flatten().shape, angle)
                scenes_full = np.full(al_u.flatten().shape, scene)
                ablation_mask_full = np.full(al_u.flatten().shape, abl_mask)
                          
 
 

            new_rad_tiled = np.load(os.path.join(out_subdir, f"{uid}_unscaled_inputs_tiled.npy"))
            new_output_tiled = np.load(os.path.join(out_subdir, f"{uid}_unscaled_y_preds_tiled.npy"))
            new_target_tiled = np.load(os.path.join(out_subdir, f"{uid}_unscaled_y_trues_tiled.npy"))
            new_ep_u_tiled = np.load(os.path.join(out_subdir, f"{uid}_unscaled_epistemic_vars_tiled.npy"))
            new_al_u_tiled = np.load(os.path.join(out_subdir, f"{uid}_unscaled_aleatoric_vars_tiled.npy"))

            if rad_tiled_full is not None:
                rad_tiled_full = np.concatenate((rad_tiled_full, np.squeeze(new_rad_tiled)), axis=0)
                output_tiled_full = np.concatenate((output_tiled_full, np.squeeze(new_output_tiled)), axis=0)
                target_tiled_full = np.concatenate((target_tiled_full, np.squeeze(new_target_tiled)), axis=0)
                ep_u_tiled_full = np.concatenate((ep_u_tiled_full, np.squeeze(new_ep_u_tiled)), axis=0)
                al_u_tiled_full = np.concatenate((al_u_tiled_full, np.squeeze(new_al_u_tiled)), axis=0)
                angles_tiled_full = np.concatenate((angles_tiled_full, np.full(np.squeeze(new_ep_u_tiled).shape, angle)), axis=0)
                scenes_tiled_full = np.concatenate((scenes_tiled_full, np.full(np.squeeze(new_ep_u_tiled).shape, scene)), axis=0)
                ablation_mask_tiled_full = np.concatenate((ablation_mask_tiled_full, np.full(np.squeeze(new_ep_u_tiled).shape, abl_mask)), axis=0)
            else:
                rad_tiled_full = np.squeeze(new_rad_tiled)
                output_tiled_full = np.squeeze(new_output_tiled)
                target_tiled_full = np.squeeze(new_target_tiled)
                ep_u_tiled_full = np.squeeze(new_ep_u_tiled)
                al_u_tiled_full = np.squeeze(new_al_u_tiled)
                angles_tiled_full = np.full(np.squeeze(new_ep_u_tiled).shape, angle)
                scenes_tiled_full = np.full(np.squeeze(new_ep_u_tiled).shape, scene)
                ablation_mask_tiled_full = np.full(np.squeeze(new_ep_u_tiled).shape, abl_mask)
 
    df = ensure_df(target_full, output_full, al_u_full, ep_u_full, angle=angles_full, scene=scenes_full, ablation_mask=ablation_mask_full)
    run_all_uq_diagnostics(df) #TODO

    rel_total = reliability_by_variance(df, variance_col="var_total")
    rel_ale   = reliability_by_variance(df, variance_col="var_ale")
    rel_epi   = reliability_by_variance(df, variance_col="var_epi")

    rel_all = pd.concat([rel_total, rel_ale, rel_epi], ignore_index=True)

    outputs = run_all_uq_diagnostics(df)
    # Inspect, export, or plot as needed
    rel_df = outputs["reliability"]
    print(rel_df.head())

    df = add_variance_ratios(df)

    df["ablation_name"] = df["scene"]
    for angl in angles:
        out_dir = config["dir_base"] + angl + config["dir_end"]

        #TODO - all angles and fix plotting (add savefig and clf())
        fig = plot_reliability_curves(rel_all, angle=angl)
        #plt.show()
        plt_dir = os.path.join(out_dir, "reliability_curves_init_" + angl + ".png")
        plt.savefig(plt_dir, dpi=400)
        plt.clf()
        plt.close()

        # Reliability curves for specific angle/scene
        fig = plot_reliability_curves(rel_df, angle=angl)
        #plt.show()
        plt_dir = os.path.join(out_dir, "reliability_curves_output_" + angl + ".png")
        plt.savefig(plt_dir, dpi=400)
        plt.clf()
        plt.close()


        # Stratified performance vs COT range
    if "strat_cot_range" in outputs:
            strat_cot = outputs["strat_cot_range"]
            plot_stratified_metrics(strat_cot, out_dir, out_dir)
 
    # ECE vs cloud fraction 
    if "ece_cloud_fraction" in outputs:
            ece_cf = outputs["ece_cloud_fraction"]
            plot_ece_vs_strat(ece_cf, out_dir, out_dir)

    # 80% coverage focus on nadir and one-off-nadir
    if "coverage80_cot_range" in outputs:
        cov_cot = outputs["coverage80_cot_range"]
        # annotate with strat_col label
        cov_cot["strat_col"] = "cot_range_patch"
        plot_coverage80_focus(
            cov_cot,
            focus_angles=angles,
            strat_col="cot_range_patch",
        )


    patch_idx_tiled_full = np.broadcast_to(
        np.arange(target_tiled_full.shape[0])[:, None, None],
        target_tiled_full.shape
    )   

    print(target_tiled_full.shape, output_tiled_full.shape, al_u_tiled_full.shape, "HERE SHAPE ISSUE?")

    df= ensure_df(
        target_tiled_full,
        output_tiled_full,
        al_u_tiled_full,
        ep_u_tiled_full,
        angle=angles_tiled_full,
        scene=scenes_tiled_full,
        ablation_mask=ablation_mask_tiled_full,
        extra={"patch_idx": patch_idx_tiled_full},
        drop_invalid=False,
    )   

    print(len(df), "DF SHAPE 4?")

    df = add_patch_metrics_to_df(df, cot_patches=target_tiled_full, cloud_threshold=0.0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

 
    ece_cot, ece_cot_detail = stratified_ece(df, "cot_range_patch", n_bins=8, log_bins=True)
    ece_cf, ece_cf_detail = stratified_ece(df, "cloud_fraction_patch", n_bins=8)
    ece_var, ece_var_detail = stratified_ece(df, "cot_var_patch", n_bins=8, log_bins=True)

    baseline_mask = ~df["ablation_mask"].astype(bool)
    print(df.head())
    ood_summary = compare_variance_distributions(df, baseline_mask=baseline_mask)
    plot_ood_id_variances(df, out_dir)
    print(ood_summary)

    # df must already contain patch-level covariates
    cot_range_summary = stratified_summary(df, "cot_range_patch", n_bins=8, log_bins=True)
    cloud_frac_summary = stratified_summary(df, "cloud_fraction_patch", n_bins=8)
    cot_var_summary = stratified_summary(df, "cot_var_patch", n_bins=8, log_bins=True)

    plot_stratified_metrics(cot_range_summary, out_dir, out_dir)
 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config.")
    args = parser.parse_args()
    config = read_yaml(args.yaml)
    main(config)


