import numpy as np
import pandas as pd
from scipy import stats

from sklearn.calibration import calibration_curve

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from evaluation_utils import coverage_from_variance, interval_membership, linear_fit_stats, qcut_or_cut 
 

def linear_fit_stats(y_true, y_pred):
     if len(y_true) < 2:
         return {
             "slope": np.nan,
             "intercept": np.nan,
             "r": np.nan,
             "rmse": np.nan,
             "bias": np.nan,
         }   
     slope, intercept, r, _, _ = stats.linregress(y_true, y_pred)
     return {
         "slope": slope,
         "intercept": intercept,
         "r": r,
         "rmse": rmse(y_true, y_pred),
         "bias": np.mean(y_pred - y_true),
     }   


def qcut_or_cut(values, n_bins=10, logspace=False, eps=1e-8):
    values = np.asarray(values)
    if logspace:
        v = np.maximum(values, eps)
        edges = np.geomspace(v.min(), v.max(), n_bins + 1)
        return pd.cut(v, bins=edges, include_lowest=True, duplicates="drop")
    return pd.qcut(values, q=n_bins, duplicates="drop")

def ensure_df(
    y_true, y_pred, var_ale, var_epi,
    angle=None, scene=None,
    cot_range_patch=None, cloud_fraction_patch=None, cot_var_patch=None,
    extra=None
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

    if extra:
        for k, v in extra.items():
            df[k] = np.asarray(v).reshape(-1)

    return df.replace([np.inf, -np.inf], np.nan).dropna()


import matplotlib.pyplot as plt
import seaborn as sns

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

            row.update({
                "variance_type": variance_col,
                "bin": str(b),
                "n": len(gb),
                "var_mean": gb[variance_col].mean(),
                "var_median": gb[variance_col].median(),
                "rmse": rmse(gb["y_true"], gb["y_pred"]),
                "bias": (gb["y_pred"] - gb["y_true"]).mean(),
                "coverage": coverage_from_variance(
                    gb["y_true"].values, gb["y_pred"].values, gb["var_total"].values, z=z
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


df = ensure_df(y_true, y_pred, var_ale, var_epi, angle=angle, scene=scene)

rel_total = reliability_by_variance(df, variance_col="var_total")
rel_ale   = reliability_by_variance(df, variance_col="var_ale")
rel_epi   = reliability_by_variance(df, variance_col="var_epi")

rel_all = pd.concat([rel_total, rel_ale, rel_epi], ignore_index=True)

fig = plot_reliability_curves(rel_all, angle="An")
plt.show()


def sample_indices_by_variance_bin(df, variance_col="var_epi", n_bins=5, samples_per_bin=8):
    g = df.copy()
    g["var_bin"] = qcut_or_cut(g[variance_col].values, n_bins=n_bins)
    sampled = (
        g.groupby("var_bin", observed=False)
         .apply(lambda x: x.sample(min(len(x), samples_per_bin), random_state=0))
         .reset_index(drop=True)
    )
    return sampled


from scipy.stats import wasserstein_distance, ks_2samp, mannwhitneyu

def compare_variance_distributions(
    df,
    baseline_mask,
    comparison_col="ablation_name",
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

def plot_ood_id_variances(df, comparison_col="ablation_name"):
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
    plt.show()



# Example metadata column:
# df["ablation_name"] in {"baseline", "SZA20", "SZA30", "SZA50", "wind5", "wind20", "AOD0"}
baseline_mask = df["ablation_name"] == "baseline"
ood_summary = compare_variance_distributions(df, baseline_mask=baseline_mask)
plot_ood_id_variances(df)


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

def plot_stratified_metrics(summary_df, angle, strat_col):
    sub = summary_df[(summary_df["angle"] == angle) & (summary_df["strat_col"] == strat_col)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    metrics = ["slope", "intercept", "rmse", "bias", "r", "var_total_median"]
    titles = ["Slope", "Intercept", "RMSE", "Bias", "Correlation", "Median total variance"]

    for ax, m, t in zip(axes.ravel(), metrics, titles):
        sns.lineplot(data=sub, x="bin_center", y=m, marker="o", ax=ax)
        ax.set_title(t)
        ax.set_xlabel(strat_col)

    plt.tight_layout()
    return fig



# df must already contain patch-level covariates
cot_range_summary = stratified_summary(df, "cot_range_patch", n_bins=8, log_bins=True)
cloud_frac_summary = stratified_summary(df, "cloud_fraction_patch", n_bins=8)
cot_var_summary = stratified_summary(df, "cot_var_patch", n_bins=8, log_bins=True)

fig = plot_stratified_metrics(cot_range_summary, angle="An", strat_col="cot_range_patch")
plt.show()




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

    sigma = np.sqrt(np.maximum(variance, 1e-12))
    abs_gaps = []
    rows = []

    for p in nominal_levels:
        z = stats.norm.ppf((1 + p) / 2)
        lo = y_pred - z * sigma
        hi = y_pred + z * sigma
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
                "coverage_80": coverage_from_variance(
                    gb["y_true"].values, gb["y_pred"].values, gb["var_total"].values, z=Z80_TWO_SIDED
                )
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


ece_cot, ece_cot_detail = stratified_ece(df, "cot_range_patch", n_bins=8, log_bins=True)
ece_cf, ece_cf_detail = stratified_ece(df, "cloud_fraction_patch", n_bins=8)
ece_var, ece_var_detail = stratified_ece(df, "cot_var_patch", n_bins=8, log_bins=True)


def coverage80_by_group(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        row = {}
        if isinstance(keys, tuple):
            for c, k in zip(group_cols, keys):
                row[c] = k
        else:
            row[group_cols[0]] = keys

        cov80 = coverage_from_variance(
            g["y_true"].values, g["y_pred"].values, g["var_total"].values, z=Z80_TWO_SIDED
        )
        row["n"] = len(g)
        row["coverage_80"] = cov80
        row["exceeds_target"] = cov80 > 0.80
        rows.append(row)
    return pd.DataFrame(rows)

def coverage80_stratified(df, strat_col, angle_col="angle", n_bins=8, log_bins=False):
    g = df.copy()
    g["strat_bin"] = None

    for angle, idx in g.groupby(angle_col).groups.items():
        bins = qcut_or_cut(g.loc[idx, strat_col].values, n_bins=n_bins, logspace=log_bins)
        g.loc[idx, "strat_bin"] = bins.astype(str)

    out = []
    for (angle, b), gb in g.groupby([angle_col, "strat_bin"]):
        out.append({
            "angle": angle,
            "strat_col": strat_col,
            "bin": b,
            "bin_center": gb[strat_col].median(),
            "n": len(gb),
            "coverage_80": coverage_from_variance(
                gb["y_true"].values, gb["y_pred"].values, gb["var_total"].values, z=Z80_TWO_SIDED
            ),
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
    plt.show()


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




