"""Write Rep_Taxonomy configs (and a command script) for the exported HybridCT representations.

    python make_reptax_configs.py -y config/reptax_40m.yaml [--rt-dir ~/Rep_Taxonomy]

Reads ``<output_dir>/meta/views.csv`` written by export_representations.py and the ``reptax``
block of the same YAML, and writes into ``<reptax.out_dir>/configs``:

  pv_cross_angle.yaml      every pair of single-angle networks + multi-angle, per layer and ID dataset
                           (CKA, debiased CKA, Procrustes, kNN Jaccard; block-bootstrap CIs),
                           fore/aft symmetry and nadir-anchoring contrasts, relative / relative-geodesic
                           representations, Mantel tests of similarity vs |delta vza|
  pv_single_vs_multi.yaml  multi-angle vs each single-angle network per layer (CKA, Procrustes,
                           kNN, ridge cross-predictability in both directions) and linear probes of
                           cloud properties, per-view COT and the models' own uncertainty / error
  pv_ood.yaml              each network's representation of the same cloud pixels at the 10.5 h
                           baseline vs each SZA / AOD / wind variant, plus probe transfer (fit at the
                           baseline, applied to the shifted representation)
  pv_members.yaml          MIMO member diversity: pairs of members' enc2 / d1 features per network
  pv_controls.yaml         trained vs random-weight networks and the random-weight similarity floor
  geo_id.yaml              intrinsic dimension (global / local) and its spatial structure per layer
  spectra.yaml             weight-spectrum (heavy-tail alpha, stable rank) of every sweep checkpoint vs
                           its validation score
  run_reptax.sh            the Rep_Taxonomy commands (PYTHONPATH=$RT_DIR, HybridCT environment)

Views or datasets missing from views.csv are skipped with a [WARN] line.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys
from typing import Any, Dict, List

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(p)


def _dump(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=None, width=140)
    print(f"[configs] {path}")


def vkey(v: float) -> str:
    return f"v{float(v):+.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--rt-dir", default=os.environ.get("RT_DIR", "~/Rep_Taxonomy"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.yaml))
    out = _abs(cfg["output_dir"])
    rc = cfg.get("reptax") or {}
    ana = _abs(rc.get("out_dir", os.path.join(out, "analysis")))
    cdir = os.path.join(ana, "configs")
    os.makedirs(cdir, exist_ok=True)
    meta_csv = os.path.join(out, "meta", "views.csv")
    if not os.path.exists(meta_csv):
        print(f"[WARN] skip: {meta_csv} not found (run export_representations.py first)")
        return
    V = pd.read_csv(meta_csv, dtype={"tag": str, "base_tag": str})
    V["member"] = V["member"].astype(int)
    have = set(V["ref"])
    datasets = list(dict.fromkeys(V["dataset"]))
    id_ds = [d for d in rc.get("id_datasets", ["test_30h", "baseline_10.5h"]) if d in datasets]
    for d in rc.get("id_datasets", []):
        if d not in datasets:
            print(f"[WARN] skip ID dataset {d}: not exported")
    ref_ds = rc.get("reference_dataset", "baseline_10.5h")
    ood_ds = [d for d in datasets if d not in (ref_ds, *id_ds)] if ref_ds in datasets else []
    if ref_ds not in datasets:
        print(f"[WARN] skip OOD shift analysis: reference dataset {ref_ds} not exported")
    layers = [l for l in rc.get("compare_layers", sorted(set(V.layer))) if l in set(V.layer)]
    trained = V[(V.kind == "trained") & (V.member == -1)]
    singles = [t for t in dict.fromkeys(trained[~trained.multiview.astype(bool)].tag)]
    multi = next(iter(trained[trained.multiview.astype(bool)].tag.unique()), None)
    vza = dict(zip(trained.tag, trained.vza))
    if multi is None:
        print("[WARN] no multi-angle network exported: single-vs-multi analyses are skipped")

    def ref(tag, ds, layer, member=-1):
        r = f"{tag}@{ds}:view__{layer}" + (f"__m{member}" if member >= 0 else "")
        return r if r in have else None

    def tref(tag, ds, name):
        return f"{tag}@{ds}:target__{name}"

    common = {"meta_csv": meta_csv, "max_rows": int(rc.get("max_rows", 4000)),
              "bootstrap": rc.get("bootstrap", {"n": 200, "seed": 0}), "knn": rc.get("knn", {"k": 10}),
              "ridge": rc.get("ridge", {"alphas": [1, 10, 100, 1000, 10000], "n_folds": 5}),
              "debiased_max_rows": 3000}
    rel = rc.get("relative") or {}
    cmds: List[str] = []
    PV = "$RTPY -m reptaxonomy.intercompare.paired_views -y"

    # ---------------------------------------------------------------- 1 cross-angle
    contrasts, rel_pairs = [], []
    by_abs: Dict[float, Dict[int, str]] = {}
    for t in singles:
        v = float(vza[t])
        if v != 0:
            by_abs.setdefault(round(abs(v), 1), {})[int(v > 0)] = t
    nadir = next((t for t in singles if float(vza[t]) == 0), None)
    for ds in id_ds:
        for L in layers:
            for av, side in sorted(by_abs.items()):
                if nadir and 0 in side and 1 in side:      # fore/aft symmetry about nadir
                    p1, p2 = [ref(side[0], ds, L), ref(nadir, ds, L)], [ref(side[1], ds, L), ref(nadir, ds, L)]
                    if None not in p1 + p2:
                        contrasts.append({"name": f"foreaft_{av}|{L}|{ds}", "metric": "linear_cka", "pair_1": p1, "pair_2": p2})
            if multi and nadir:                            # is the fused representation anchored at nadir?
                far = [t for t in singles if abs(float(vza[t])) == max(abs(float(vza[s])) for s in singles)]
                for f in far:
                    p1, p2 = [ref(multi, ds, L), ref(nadir, ds, L)], [ref(multi, ds, L), ref(f, ds, L)]
                    if None not in p1 + p2:
                        contrasts.append({"name": f"multi_nadir_vs_{f}|{L}|{ds}", "metric": "linear_cka",
                                          "pair_1": p1, "pair_2": p2})
        for L in rel.get("layers", ["bott", "d1"]):
            anchor = [t for t in (multi, nadir) if t]
            for x in anchor:
                for t in singles + ([multi] if multi and x != multi else []):
                    if t != x and ref(x, ds, L) and ref(t, ds, L):
                        rel_pairs.append([ref(x, ds, L), ref(t, ds, L)])
    cross = {"paired_views": {
        **common, "out_dir": os.path.join(ana, "cross_angle"),
        "meta_select": {"kind": "trained", "member": -1, "dataset": id_ds, "layer": layers},
        "metrics": ["linear_cka", "linear_cka_debiased", "knn_jaccard", "procrustes"],
        "pairs": [],
        "pair_sets": [{"name": "cross_angle", "same": ["layer", "dataset"], "differ": ["tag"]}],
        "contrasts": contrasts,
        "relative": {"pairs": rel_pairs, "kinds": rel.get("kinds", ["cos", "euc", "geo_euc"]),
                     "n_anchors": rel.get("n_anchors", 256), "anchor_scheme": rel.get("anchor_scheme", "uniform"),
                     "max_rows": rel.get("max_rows", 3000), "steps": rel.get("steps", 10),
                     "decoder": rel.get("decoder", {"labels": "labels__cot_bin"})},
        "covariates": {"metric": "linear_cka", "strata": ["layer", "dataset"],
                       "n_perm": int((rc.get("covariates") or {}).get("n_perm", 999)),
                       "terms": [{"name": "abs_delta_vza", "column": "vza", "op": "absdiff"},
                                 {"name": "same_side", "column": "vza", "op": "same_sign"}]}}}
    if not id_ds or len(singles) + bool(multi) < 2:
        print("[WARN] skip cross-angle config: fewer than two networks or no ID dataset")
    else:
        _dump(os.path.join(cdir, "pv_cross_angle.yaml"), cross)
        cmds.append(f"{PV} {cdir}/pv_cross_angle.yaml")

    # ---------------------------------------------------------------- 2 single vs multi + probes
    if multi and id_ds:
        pairs = []
        for ds in id_ds:
            for L in layers:
                for t in singles:
                    if ref(multi, ds, L) and ref(t, ds, L):
                        pairs.append([ref(multi, ds, L), ref(t, ds, L)])
        # probes: the same targets from every network, one config per dataset row space
        probe_layers = [l for l in rc.get("probe_layers", ["enc2", "enc4", "bott", "d2", "d1"]) if l in layers]
        probes_by_ds = {}
        for ds in id_ds:
            views = [ref(t, ds, L) for t in singles + [multi] for L in probe_layers if ref(t, ds, L)]
            tg = [tref(multi, ds, n) for n in rc.get("probe_targets", ["log_tau_nadir", "cot_hetero", "cloud_frac_tile"])]
            tg += [tref(multi, ds, f"tau_{vkey(vza[t])}") for t in singles]          # per-view COT
            probes_by_ds[ds] = (views, tg)
        svm = {"paired_views": {**common, "out_dir": os.path.join(ana, "single_vs_multi"),
                                "metrics": ["linear_cka", "procrustes", "knn_jaccard", "ridge_r2"], "pairs": pairs}}
        _dump(os.path.join(cdir, "pv_single_vs_multi.yaml"), svm)
        cmds.append(f"{PV} {cdir}/pv_single_vs_multi.yaml")
        for ds, (views, tg) in probes_by_ds.items():
            # cloud-property probes: every network's layers, truth targets (identical across inputs)
            _dump(os.path.join(cdir, f"pv_probes_{ds}.yaml"),
                  {"paired_views": {**common, "out_dir": os.path.join(ana, f"probes_{ds}"), "pairs": [],
                                    "metrics": ["linear_cka"], "probes": {"views": views, "targets": tg}}})
            cmds.append(f"{PV} {cdir}/pv_probes_{ds}.yaml")
            # uncertainty / error probes: each network's own prediction quantities (own angle; nadir for multi)
            for t in singles + [multi]:
                k = vkey(0.0) if t == multi else vkey(vza[t])
                vs = [ref(t, ds, L) for L in probe_layers if ref(t, ds, L)]
                if not vs:
                    continue
                _dump(os.path.join(cdir, f"pv_probes_uq_{ds}_{t}.yaml"),
                      {"paired_views": {**common, "out_dir": os.path.join(ana, f"probes_uq_{ds}", t), "pairs": [],
                                        "metrics": ["linear_cka"],
                                        "probes": {"views": vs, "targets": [tref(t, ds, f"{q}_{k}") for q in
                                                                            ("ale_std", "epi_std", "abs_err", "pred_mean")]}}})
                cmds.append(f"{PV} {cdir}/pv_probes_uq_{ds}_{t}.yaml")

    # ---------------------------------------------------------------- 3 OOD shift
    if ood_ds:
        pairs, cv = [], []
        for t in singles + ([multi] if multi else []):
            for L in layers:
                b = ref(t, ref_ds, L)
                for d in ood_ds:
                    o = ref(t, d, L)
                    if b and o:
                        pairs.append([b, o])
                        if L in ("bott", "d1"):
                            cv.append({"fit": b, "apply": o, "targets": [tref(t, ref_ds, "log_tau_nadir")]})
        ood = {"paired_views": {**common, "out_dir": os.path.join(ana, "ood_shift"),
                                "metrics": ["linear_cka", "procrustes", "knn_jaccard"], "pairs": pairs,
                                "transfer": {"cross_view": cv}}}
        _dump(os.path.join(cdir, "pv_ood.yaml"), ood)
        cmds.append(f"{PV} {cdir}/pv_ood.yaml")

    # ---------------------------------------------------------------- 4 members
    mem = V[V.member >= 0]
    pairs = []
    for (inp, L), g in mem[mem.dataset.isin(id_ds)].groupby(["input", "layer"], sort=False):
        pairs += [list(p) for p in itertools.combinations(sorted(g.ref), 2)]
    if pairs:
        _dump(os.path.join(cdir, "pv_members.yaml"),
              {"paired_views": {**common, "out_dir": os.path.join(ana, "members"),
                                "metrics": ["linear_cka", "procrustes", "knn_jaccard"], "pairs": pairs}})
        cmds.append(f"{PV} {cdir}/pv_members.yaml")
    else:
        print("[WARN] skip member config: no per-member views exported")

    # ---------------------------------------------------------------- 5 random-weight controls
    rnd = V[(V.kind == "random") & (V.member == -1)]
    pairs, con = [], []
    for ds in id_ds:
        for L in layers:
            for rt in rnd.base_tag.unique():
                a_, b_ = ref(rt, ds, L), ref(f"{rt}_rand", ds, L)
                if a_ and b_:
                    pairs.append([a_, b_])
            rts = list(rnd.base_tag.unique())
            if multi in rts and nadir in rts:
                p1 = [ref(multi, ds, L), ref(nadir, ds, L)]
                p2 = [ref(f"{multi}_rand", ds, L), ref(f"{nadir}_rand", ds, L)]
                if None not in p1 + p2:
                    pairs.append(p2)
                    con.append({"name": f"trained_vs_random_similarity|{L}|{ds}", "metric": "linear_cka",
                                "pair_1": p1, "pair_2": p2})
    if pairs:
        probes = {"views": [ref(f"{t}_rand", ds, L) for t in rnd.base_tag.unique() for ds in id_ds[:1]
                            for L in rc.get("probe_layers", ["bott", "d1"]) if ref(f"{t}_rand", ds, L)],
                  "targets": [tref(f"{t}_rand", id_ds[0], n) for t in rnd.base_tag.unique()[:1]
                              for n in rc.get("probe_targets", ["log_tau_nadir"])]}
        _dump(os.path.join(cdir, "pv_controls.yaml"),
              {"paired_views": {**common, "out_dir": os.path.join(ana, "controls"),
                                "metrics": ["linear_cka", "procrustes", "knn_jaccard"], "pairs": pairs,
                                "contrasts": con, "probes": probes if probes["views"] else None}})
        cmds.append(f"{PV} {cdir}/pv_controls.yaml")
    else:
        print("[WARN] skip control config: no random-weight views exported")

    # ---------------------------------------------------------------- 6 geo_id
    emb = os.path.join(out, "geo_id", "embeddings")
    models = sorted({os.path.basename(p) for p in glob.glob(os.path.join(emb, "*", "*"))})
    conds = sorted({os.path.basename(p) for p in glob.glob(os.path.join(emb, "*"))})
    if models and conds:
        gp = {"unit": "token", "max_tokens_per_chip": 64, "pool": False, "per_condition": True,
              "min_points": 40, "global_estimators": ["MLE", "TwoNN", "MOM", "TLE", "FisherS", "CorrInt"],
              "ks_global": [10, 20, 50], "ks_local": [20, 50], "n_runs": 3, "max_points": 5000,
              "group_keys": ["region", "split", "scope"], "make_maps": True, "seed": 0,
              **((rc.get("geo_id") or {}).get("params") or {})}
        _dump(os.path.join(cdir, "geo_id.yaml"), {"geo_id": gp})
        cmds.append(f"$RTPY -m reptaxonomy.robustness.geo_intrinsic_dimension --embed-dir {emb} "
                    f"--work-dir {os.path.join(out, 'geo_id', 'work')} --out-dir {ana} --dataset hybridct "
                    f"--conditions {' '.join(conds)} --models {' '.join(models)} --params {cdir}/geo_id.yaml")
    else:
        print("[WARN] skip geo_id: no embeddings exported")

    # ---------------------------------------------------------------- 7 spectra
    sp = rc.get("spectra") or {}
    lbs = _abs(sp.get("leaderboards", "output/deploy_40m/leaderboard_*.csv"))
    if glob.glob(lbs):
        lb_all = [f for f in glob.glob(lbs) if not f.endswith("leaderboard_all.csv")]
        _dump(os.path.join(cdir, "spectra.yaml"), {"checkpoint_spectra": {
            "out_dir": os.path.join(ana, "spectra"), "sys_path": [REPO],
            "loader": "emulator.mimo.pyct.mimo_unet:load_mimo_checkpoint",
            "models_csv": sorted(lb_all), "csv_columns": {"name": "run", "checkpoint": "checkpoint"},
            "metrics": sp.get("metrics", ["score", "val_rmse_cloudy_cot", "val_nll_mixture"]),
            "layer_groups": {"member_encoder": r"encoder[12]\.", "core": r"(encoder[34]|bottleneck|decoder[234]|upconv[234])",
                             "member_decoder": r"(decoder1|upconv1|conv)\."},
            "min_eigs": 10, "max_models": sp.get("max_models"),
            "backend": sp.get("backend", "auto"), "ww_min_evals": sp.get("ww_min_evals", 10)}})
        cmds.append(f"$RTPY -m reptaxonomy.robustness.checkpoint_spectra -y {cdir}/spectra.yaml")
    else:
        print(f"[WARN] skip spectra: no leaderboards at {lbs}")

    sh = os.path.join(cdir, "run_reptax.sh")
    with open(sh, "w") as f:
        f.write("#!/usr/bin/env bash\n# Generated by make_reptax_configs.py. Run in the HybridCT environment.\n"
                "# Each command resumes from run_state markers; a failed one does not stop the others.\n"
                f"export RT_DIR=${{RT_DIR:-{a.rt_dir}}}\nexport PYTHONPATH=\"$RT_DIR:{REPO}:${{PYTHONPATH:-}}\"\n"
                "RTPY=${RT_PYTHON:-python}\nrc=0\n")
        for c in cmds:
            f.write(f"{c} || {{ echo \"[WARN] failed: {c.split()[-1]}\"; rc=1; }}\n")
        f.write("exit $rc\n")
    os.chmod(sh, 0o755)
    print(f"[configs] {sh} ({len(cmds)} commands)")


if __name__ == "__main__":
    sys.exit(main())
