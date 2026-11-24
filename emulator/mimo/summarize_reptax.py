"""Figures and tables from the Rep_Taxonomy analyses of the HybridCT networks.

    python summarize_reptax.py -y config/reptax_40m.yaml

Reads ``<reptax.out_dir>/{cross_angle,single_vs_multi,probes_*,probes_uq_*,ood_shift,members,controls,
geo_id,spectra}`` (written by the commands of make_reptax_configs.py), the exporter's meta/views.csv and
the UQ tables (``reptax.uq_tables``/{single,multi}_{cal,raw}/tables/T1_overall.csv, T2_ood_shift.csv).
Every panel whose inputs are missing is skipped with a [WARN] line.

Outputs in ``<reptax.out_dir>/summary``:
  F_rep_cross_angle_cka.{pdf,png}     CKA heatmaps (networks ordered by view zenith) per layer
  F_rep_cka_vs_dvza.{pdf,png}         CKA vs |delta vza| per layer with Mantel r / p
  F_rep_single_vs_multi.{pdf,png}     CKA(multi, single@vza) per layer + ridge asymmetry
  F_rep_probes.{pdf,png}              probe R^2 of log COT / heterogeneity by layer and network
  F_rep_ood.{pdf,png}                 representation shift (1 - CKA) vs RMSE ratio per OOD dataset
  F_rep_members.{pdf,png}             member CKA (diversity) vs epistemic fraction
  F_rep_id.{pdf,png}                  intrinsic dimension by layer
  F_rep_spectra.{pdf,png}             median layer alpha vs sweep validation score
  T_rep_*.csv / .tex                  the tables behind each figure
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LAYER_ORDER = ["enc1", "enc2", "enc3", "enc4", "bott", "d4", "d3", "d2", "d1", "out"]


def warn(msg: str) -> None:
    print(f"[WARN] skip {msg}")


def read(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        warn(f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}: not found")
        return None
    df = pd.read_csv(path)
    if df.empty:
        warn(f"{path}: empty")
        return None
    return df


def save(fig, out: str, name: str, fmts=("pdf", "png"), dpi=300) -> None:
    for f in fmts:
        fig.savefig(os.path.join(out, f"{name}.{f}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[summary] {name}")


def table(df: pd.DataFrame, out: str, name: str) -> None:
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    try:
        df.to_latex(os.path.join(out, f"{name}.tex"), index=False, float_format="%.3f")
    except Exception:  # noqa: BLE001 - jinja2 missing etc.
        pass


def attach(df: pd.DataFrame, V: pd.DataFrame) -> pd.DataFrame:
    """Add tag / vza / layer / dataset / multiview for view_a and view_b."""
    m = V.set_index("ref")[["tag", "base_tag", "kind", "vza", "layer", "member", "dataset", "multiview"]]
    a = m.add_suffix("_a").reindex(df["view_a"]).reset_index(drop=True)
    b = m.add_suffix("_b").reindex(df["view_b"]).reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), a, b], axis=1)


def tag_label(tag: str, vza: float, multiview: bool) -> str:
    return "multi" if multiview else f"{vza:+.1f}"


def _layers(df, col="layer_a"):
    return [l for l in LAYER_ORDER if l in set(df[col])]


# --------------------------------------------------------------------------- panels
def cross_angle(A: str, V: pd.DataFrame, out: str, ds: str) -> None:
    pw = read(os.path.join(A, "cross_angle", "pairwise.csv"))
    if pw is None:
        return
    pw = attach(pw, V)
    pw = pw[pw.dataset_a == ds]
    if pw.empty:
        return warn(f"cross-angle heatmaps: no rows for {ds}")
    tags = (V[(V.kind == "trained") & (V.member == -1)].drop_duplicates("tag")
            .assign(o=lambda d: np.where(d.multiview.astype(bool), 999, d.vza)).sort_values("o"))
    order = list(tags.tag)
    lab = [tag_label(t, v, m) for t, v, m in zip(tags.tag, tags.vza, tags.multiview.astype(bool))]
    Ls = _layers(pw)
    nc = min(5, len(Ls))
    nr = int(np.ceil(len(Ls) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.0 * nr), squeeze=False)
    rows = []
    for ax, L in zip(axs.flat, Ls):
        M = pd.DataFrame(np.eye(len(order)), index=order, columns=order)
        for r in pw[pw.layer_a == L].itertuples():
            if r.tag_a in order and r.tag_b in order:
                M.loc[r.tag_a, r.tag_b] = M.loc[r.tag_b, r.tag_a] = r.linear_cka
        im = ax.imshow(M.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
        ax.set_title(L)
        ax.set_xticks(range(len(order)), lab, rotation=90, fontsize=6)
        ax.set_yticks(range(len(order)), lab, fontsize=6)
        rows.append({"layer": L, "mean_cka_single_pairs": float(pw[(pw.layer_a == L) & ~pw.multiview_a.astype(bool)
                                                                  & ~pw.multiview_b.astype(bool)].linear_cka.mean())})
    for ax in list(axs.flat)[len(Ls):]:
        ax.axis("off")
    fig.colorbar(im, ax=axs, shrink=0.6, label="linear CKA")
    save(fig, out, "F_rep_cross_angle_cka")

    cov = read(os.path.join(A, "cross_angle", "covariates.csv"))
    sg = pw[~pw.multiview_a.astype(bool) & ~pw.multiview_b.astype(bool)].copy()
    sg["dvza"] = (sg.vza_a - sg.vza_b).abs()
    fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.6 * nr), squeeze=False, sharey=True)
    for ax, L in zip(axs.flat, Ls):
        d = sg[sg.layer_a == L]
        same = np.sign(d.vza_a) == np.sign(d.vza_b)
        ax.scatter(d.dvza[same], d.linear_cka[same], s=12, label="same side / nadir")
        ax.scatter(d.dvza[~same], d.linear_cka[~same], s=12, marker="^", label="fore vs aft")
        if "linear_cka_lo95" in d:
            ax.vlines(d.dvza, d.linear_cka_lo95, d.linear_cka_hi95, color="0.6", lw=0.6)
        t = ""
        if cov is not None:
            c = cov[(cov.stratum == f"{L}|{ds}") & (cov.term == "abs_delta_vza")]
            if len(c):
                t = f"\nMantel r={c.mantel_r.iloc[0]:.2f}, p={c.p_perm.iloc[0]:.3f}"
                rows_l = [r for r in rows if r["layer"] == L]
                if rows_l:
                    rows_l[0].update(mantel_r=float(c.mantel_r.iloc[0]), mantel_p=float(c.p_perm.iloc[0]))
        ax.set_title(L + t, fontsize=8)
        ax.set_xlabel("|delta vza| (deg)")
    axs[0, 0].set_ylabel("linear CKA")
    axs[0, 0].legend(fontsize=6)
    for ax in list(axs.flat)[len(Ls):]:
        ax.axis("off")
    save(fig, out, "F_rep_cka_vs_dvza")
    con = read(os.path.join(A, "cross_angle", "contrasts.csv"))
    if con is not None:
        table(con, out, "T_rep_contrasts")
    table(pd.DataFrame(rows), out, "T_rep_cross_angle")
    rel = read(os.path.join(A, "cross_angle", "relative.csv"))
    if rel is not None and "mean_cosine_between_representations" in rel:
        rel = attach(rel, V)
        cols = ["layer_a", "tag_a", "tag_b", "vza_b", "representation", "mean_cosine_between_representations",
                "mean_cosine_lo95", "mean_cosine_hi95", "pairwise_distance_spearman", "top1", "mrr", "rel_cka"]
        table(rel[[c for c in cols if c in rel.columns]], out, "T_rep_relative")


def single_vs_multi(A: str, V: pd.DataFrame, out: str, ds: str) -> None:
    pw = read(os.path.join(A, "single_vs_multi", "pairwise.csv"))
    if pw is None:
        return
    pw = attach(pw, V)
    pw = pw[pw.dataset_a == ds].sort_values("vza_b")
    Ls = _layers(pw)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    cm = plt.get_cmap("viridis")
    for i, L in enumerate(Ls):
        d = pw[pw.layer_a == L]
        axs[0].errorbar(d.vza_b, d.linear_cka, yerr=[(d.linear_cka - d.linear_cka_lo95).clip(lower=0), (d.linear_cka_hi95 - d.linear_cka).clip(lower=0)]
                        if "linear_cka_lo95" in d else None, marker="o", ms=3, capsize=2,
                        color=cm(i / max(len(Ls) - 1, 1)), label=L)
        if "ridge_r2_ab" in d:
            axs[1].plot(d.vza_b, d.ridge_r2_ba - d.ridge_r2_ab, marker="o", ms=3, color=cm(i / max(len(Ls) - 1, 1)))
    axs[0].set(xlabel="single-angle network vza (deg)", ylabel="CKA(multi, single)")
    axs[0].legend(fontsize=6, ncol=2)
    axs[1].axhline(0, color="k", lw=0.5)
    axs[1].set(xlabel="single-angle network vza (deg)",
               ylabel="R2(single->multi) - R2(multi->single)")
    axs[1].set_title("> 0: multi-angle features are predictable from the single view\n"
                     "< 0: multi-angle features contain information the view lacks", fontsize=7)
    save(fig, out, "F_rep_single_vs_multi")
    cols = [c for c in ["layer_a", "tag_b", "vza_b", "linear_cka", "linear_cka_lo95", "linear_cka_hi95", "procrustes",
                        "knn_jaccard", "ridge_r2_ab", "ridge_r2_ba"] if c in pw]
    table(pw[cols], out, "T_rep_single_vs_multi")


def probes(A: str, V: pd.DataFrame, out: str, ds: str) -> None:
    pr = read(os.path.join(A, f"probes_{ds}", "probes.csv"))
    if pr is None or "r2" not in pr:
        return
    m = V.set_index("ref")
    pr = pr.join(m[["tag", "vza", "layer", "multiview"]], on="view")
    pr["target_name"] = pr.target.str.split("target__").str[-1]
    tnames = [t for t in ["log_tau_nadir", "cot_hetero", "cloud_frac_tile"] if t in set(pr.target_name)]
    if not tnames:
        return warn("probe figure: no cloud-property targets")
    fig, axs = plt.subplots(1, len(tnames), figsize=(4 * len(tnames), 3.4), squeeze=False, sharey=True)
    tags = pr.drop_duplicates("tag").assign(o=lambda d: np.where(d.multiview.astype(bool), 999, d.vza)).sort_values("o")
    cm = plt.get_cmap("coolwarm")
    for ax, tn in zip(axs[0], tnames):
        for t, v, mv in zip(tags.tag, tags.vza, tags.multiview.astype(bool)):
            d = pr[(pr.tag == t) & (pr.target_name == tn)]
            d = d.assign(li=d.layer.map({l: i for i, l in enumerate(LAYER_ORDER)})).sort_values("li")
            ax.plot(d.layer, d.r2, marker="o", ms=3, lw=2.2 if mv else 1,
                    color="k" if mv else cm((v + 75) / 150), label=tag_label(t, v, mv))
        ax.set_title(tn)
        ax.set_xlabel("layer")
    axs[0, 0].set_ylabel("grouped-CV ridge R2")
    axs[0, -1].legend(fontsize=6, ncol=2)
    save(fig, out, "F_rep_probes")
    table(pr[["tag", "vza", "layer", "target_name", "r2", "spearman"]], out, "T_rep_probes")
    uq = [pd.read_csv(f).assign(src=f) for f in glob.glob(os.path.join(A, f"probes_uq_{ds}", "*", "probes.csv"))]
    uq = [u for u in uq if len(u) and "r2" in u]
    if uq:
        u = pd.concat(uq).join(m[["tag", "vza", "layer"]], on="view")
        u["target_name"] = u.target.str.split("target__").str[-1].str.replace(r"_v[+-]\d+\.\d$", "", regex=True)
        table(u.pivot_table(index=["tag", "vza", "layer"], columns="target_name", values="r2").reset_index(),
              out, "T_rep_probes_uq")


def _uq_tables(U: str, kind: str, net: str) -> Optional[pd.DataFrame]:
    for v in (f"{net}_cal", f"{net}_raw"):
        p = os.path.join(U, v, "tables", f"{kind}.csv")
        if os.path.exists(p):
            return pd.read_csv(p).assign(uq_run=v)
    warn(f"UQ table {kind} for {net}: not found under {U}")
    return None


def ood(A: str, V: pd.DataFrame, out: str, U: str) -> None:
    pw = read(os.path.join(A, "ood_shift", "pairwise.csv"))
    if pw is None:
        return
    pw = attach(pw, V)
    pw["shift"] = 1 - pw.linear_cka
    T2 = [t for t in (_uq_tables(U, "T2_ood_shift", "single"), _uq_tables(U, "T2_ood_shift", "multi")) if t is not None]
    rows = pw[["tag_a", "vza_a", "multiview_a", "layer_a", "dataset_b", "shift", "procrustes", "knn_jaccard"]].copy()
    tr = read(os.path.join(A, "ood_shift", "transfer.csv"))
    if tr is not None and "r2_transfer" in tr:
        tr = tr.assign(view_a=tr["fit"], view_b=tr["apply"])
        tr = attach(tr, V)[["tag_a", "layer_a", "dataset_b", "r2_within", "r2_transfer"]]
        rows = rows.merge(tr, on=["tag_a", "layer_a", "dataset_b"], how="left")
    if T2:
        t2 = pd.concat(T2)
        net = np.where(t2.uq_run.str.startswith("multi"), "multi", "single")
        t2 = t2.assign(net=net).groupby(["net", "dataset", "vza"], as_index=False)[["RMSE_ratio", "median_ratio"]].mean()
        rows["net"] = np.where(rows.multiview_a.astype(bool), "multi", "single")
        rows["vza_m"] = np.where(rows.multiview_a.astype(bool), 0.0, rows.vza_a)
        rows = rows.merge(t2.rename(columns={"dataset": "dataset_b", "vza": "vza_m"}), on=["net", "dataset_b", "vza_m"],
                          how="left")
    table(rows, out, "T_rep_ood")
    Ls = [l for l in ("enc2", "bott", "d1") if l in set(rows.layer_a)]
    if not Ls:
        return
    fig, axs = plt.subplots(1, len(Ls), figsize=(4 * len(Ls), 3.4), squeeze=False, sharey=True)
    for ax, L in zip(axs[0], Ls):
        d = rows[rows.layer_a == L]
        y = d.RMSE_ratio if "RMSE_ratio" in d and d.RMSE_ratio.notna().any() else None
        for ds_, g in d.groupby("dataset_b"):
            if y is not None:
                ax.scatter(g["shift"], g.RMSE_ratio, s=14, label=ds_)
            else:
                ax.scatter(g.vza_a, g["shift"], s=14, label=ds_)
        ax.set_title(L)
        ax.set_xlabel("1 - CKA(baseline, OOD)" if y is not None else "vza")
    axs[0, 0].set_ylabel("RMSE ratio (OOD / baseline)" if "RMSE_ratio" in rows else "1 - CKA")
    axs[0, -1].legend(fontsize=6)
    save(fig, out, "F_rep_ood")


def members(A: str, V: pd.DataFrame, out: str, U: str, ds: str) -> None:
    pw = read(os.path.join(A, "members", "pairwise.csv"))
    if pw is None:
        return
    pw = attach(pw, V)
    agg = pw[pw.dataset_a == ds].groupby(["tag_a", "vza_a", "multiview_a", "layer_a"], as_index=False).linear_cka.mean()
    agg = agg.rename(columns={"linear_cka": "member_cka"})
    T1 = [t for t in (_uq_tables(U, "T1_overall", "single"), _uq_tables(U, "T1_overall", "multi")) if t is not None]
    if T1:
        t1 = pd.concat(T1)
        t1 = t1[(t1.dataset == ds) & (t1.subset.astype(str) == "cloudy")] if "subset" in t1 else t1[t1.dataset == ds]
        t1 = t1.assign(net=np.where(t1.uq_run.str.startswith("multi"), "multi", "single"))
        agg["net"] = np.where(agg.multiview_a.astype(bool), "multi", "single")
        agg["vza_m"] = np.where(agg.multiview_a.astype(bool), 0.0, agg.vza_a)
        cols = [c for c in ("epi_frac", "ECE", "NLL", "RMSE") if c in t1]
        agg = agg.merge(t1[["net", "vza"] + cols].rename(columns={"vza": "vza_m"}), on=["net", "vza_m"], how="left")
    table(agg, out, "T_rep_members")
    Ls = list(dict.fromkeys(agg.layer_a))
    fig, axs = plt.subplots(1, len(Ls), figsize=(4 * len(Ls), 3.2), squeeze=False)
    for ax, L in zip(axs[0], Ls):
        d = agg[agg.layer_a == L]
        x = d.epi_frac if "epi_frac" in d else d.vza_a
        ax.scatter(x, d.member_cka, s=16)
        for xi, yi, t in zip(x, d.member_cka, d.tag_a):
            ax.annotate(str(t), (xi, yi), fontsize=6)
        ax.set(title=f"{L} members", xlabel="epistemic fraction (T1)" if "epi_frac" in d else "vza",
               ylabel="mean CKA between members")
    save(fig, out, "F_rep_members")


def intrinsic_dimension(A: str, out: str, est: str = "MLE", k: int = 20) -> None:
    g = read(os.path.join(A, "geo_id", "hybridct", "global_id.csv"))
    if g is None:
        return
    g = g[(g.method == est) & (g.k == k)] if (g.method == est).any() else g[g.method == g.method.iloc[0]]
    g = g.assign(tag=g.model.str.rsplit("__", n=1).str[0], layer=g.model.str.rsplit("__", n=1).str[1])
    table(g, out, "T_rep_id")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for (sc, t), d in g.groupby(["scope", "tag"]):
        d = d.assign(li=d.layer.map({l: i for i, l in enumerate(LAYER_ORDER)})).sort_values("li")
        ax.plot(d.layer, d.global_id_mean, marker="o", ms=3, lw=1, label=f"{t} ({sc})")
    ax.set(xlabel="layer", ylabel=f"global ID ({g.method.iloc[0]})")
    ax.legend(fontsize=5, ncol=2)
    save(fig, out, "F_rep_id")
    s = read(os.path.join(A, "geo_id", "hybridct", "spatial_stats.csv"))
    if s is not None:
        table(s, out, "T_rep_id_spatial")


def spectra(A: str, out: str) -> None:
    m = read(os.path.join(A, "spectra", "models.csv"))
    if m is None or "alpha_median" not in m:
        return
    c = read(os.path.join(A, "spectra", "correlations.csv"))
    if c is not None:
        table(c, out, "T_rep_spectra_corr")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for g, d in m.groupby("group"):
        axs[0].scatter(d.alpha_median, d.score, s=10, label=g)
        if "val_rmse_cloudy_cot" in d:
            axs[1].scatter(d.alpha_median, d.val_rmse_cloudy_cot, s=10, label=g)
    axs[0].set(xlabel="median layer alpha", ylabel="sweep validation score")
    axs[1].set(xlabel="median layer alpha", ylabel="val RMSE, cloudy COT")
    axs[0].legend(fontsize=6, ncol=2)
    save(fig, out, "F_rep_spectra")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.yaml))
    rc = cfg.get("reptax") or {}
    A = os.path.abspath(rc.get("out_dir", os.path.join(cfg["output_dir"], "analysis")))
    U = os.path.abspath(rc.get("uq_tables", "output/uq_eval_40m"))
    out = os.path.join(A, "summary")
    os.makedirs(out, exist_ok=True)
    vp = os.path.join(os.path.abspath(cfg["output_dir"]), "meta", "views.csv")
    if not os.path.exists(vp):
        return warn(f"summary: {vp} not found")
    V = pd.read_csv(vp, dtype={"tag": str, "base_tag": str})
    ids = [d for d in rc.get("id_datasets", ["test_30h"]) if d in set(V.dataset)]
    if not ids:
        return warn("summary: no ID dataset exported")
    ds = ids[0]
    for fn, args in ((cross_angle, (A, V, out, ds)), (single_vs_multi, (A, V, out, ds)), (probes, (A, V, out, ds)),
                     (ood, (A, V, out, U)), (members, (A, V, out, U, ds)), (intrinsic_dimension, (A, out)),
                     (spectra, (A, out))):
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001 - one panel must not stop the others
            warn(f"{fn.__name__}: {e!r}")


if __name__ == "__main__":
    main()
