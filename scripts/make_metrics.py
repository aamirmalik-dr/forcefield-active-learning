"""Aggregate all benchmark JSONs into results/metrics.json.

Every number in the README, RESULTS.md, and model card traces back to this
file, which itself only restates values measured by the benchmark runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from falearn.calibration import binned_reliability, label_reduction_factor

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def _load(name: str) -> dict | None:
    path = RESULTS / name
    if path.exists():
        return json.loads(path.read_text())
    return None


def curves_by_strategy(main: dict, metric: str, eval_name: str = "test") -> dict:
    """Budget curve mean and spread across seeds for each strategy."""
    out: dict = {}
    for strategy in ("committee", "committee_capped", "random"):
        runs = [r for r in main["runs"] if r["strategy"] == strategy]
        if not runs:
            continue
        budgets = [rec["budget"] for rec in runs[0]["rounds"]]
        values = np.array(
            [[rec["metrics"][eval_name][metric] for rec in run["rounds"]] for run in runs]
        )
        out[strategy] = {
            "budgets": budgets,
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "per_seed": values.tolist(),
            "seeds": [run["seed"] for run in runs],
        }
    return out


def main() -> int:
    metrics: dict = {}
    main_res = _load("main.json")
    if main_res:
        metrics["dataset"] = {
            "n_pool": main_res["n_pool"],
            "n_test": main_res["n_test"],
            "n_transfer": main_res["n_transfer"],
            "pool_kind_fractions": main_res.get("pool_kind_fractions", {}),
            "pool_composition_fractions": main_res.get("pool_composition_fractions", {}),
        }
        curves_f = curves_by_strategy(main_res, "f_mae_mev_per_a")
        curves_e = curves_by_strategy(main_res, "e_mae_mev_per_atom")
        metrics["curves"] = {"f_mae_mev_per_a": curves_f, "e_mae_mev_per_atom": curves_e}
        metrics["curves_transfer"] = {
            "f_mae_mev_per_a": curves_by_strategy(main_res, "f_mae_mev_per_a", "transfer"),
            "e_mae_mev_per_atom": curves_by_strategy(main_res, "e_mae_mev_per_atom", "transfer"),
        }
        if "random" in curves_f:
            metrics["label_reduction"] = {}
            for variant in ("committee", "committee_capped"):
                if variant not in curves_f:
                    continue
                metrics["label_reduction"][variant] = {
                    "force": label_reduction_factor(
                        np.array(curves_f[variant]["budgets"]),
                        np.array(curves_f[variant]["mean"]),
                        np.array(curves_f["random"]["mean"]),
                    ),
                    "energy": label_reduction_factor(
                        np.array(curves_e[variant]["budgets"]),
                        np.array(curves_e[variant]["mean"]),
                        np.array(curves_e["random"]["mean"]),
                    ),
                }
        # calibration per committee run
        cal_rows = []
        for run in main_res["runs"]:
            if run["strategy"].startswith("committee") and "calibration" in run:
                cal = run["calibration"]
                rel = binned_reliability(
                    np.array(cal["disagreement_mean_mev_per_a"]),
                    np.array(cal["f_rmse_mev_per_a"]),
                    n_bins=8,
                )
                cal_rows.append(
                    {
                        "strategy": run["strategy"],
                        "seed": run["seed"],
                        "spearman_rho": rel["spearman_rho"],
                        "pearson_r": rel["pearson_r"],
                        "n_frames": rel["n_frames"],
                    }
                )
        metrics["calibration"] = cal_rows
        # split-protocol exhibit: honest held-out-group test vs sibling frames
        split_rows = []
        for run in main_res["runs"]:
            if "sibling_test" not in run:
                continue
            final = run["rounds"][-1]["metrics"]["test"]
            split_rows.append(
                {
                    "strategy": run["strategy"],
                    "seed": run["seed"],
                    "honest_f_mae": final["f_mae_mev_per_a"],
                    "sibling_f_mae": run["sibling_test"]["f_mae_mev_per_a"],
                    "honest_e_mae": final["e_mae_mev_per_atom"],
                    "sibling_e_mae": run["sibling_test"]["e_mae_mev_per_atom"],
                }
            )
        metrics["split_protocol"] = split_rows
        # per-batch-kind breakdown of the final test error: the calibration
        # arrays are per-frame in split.test order, so kinds can be recovered
        # by re-deriving the (deterministic) split from the dataset
        ds_path = REPO / main_res["config"]["dataset"]
        if ds_path.exists():
            from falearn.data import group_kind, load_frames, pool_test_split

            frames = load_frames(ds_path)
            sc = main_res["config"].get("split", {})
            split = pool_test_split(
                frames,
                holdout_compositions=tuple(sc.get("holdout_compositions", ())),
                test_fraction=float(sc.get("test_fraction", 0.2)),
                seed=int(sc.get("seed", 0)),
            )
            test_kinds = [group_kind(frames[k].group, frames[k].composition) for k in split.test]
            by_kind: dict[str, dict[str, list[float]]] = {}
            for run in main_res["runs"]:
                if "calibration" not in run:
                    continue
                errs = run["calibration"]["f_rmse_mev_per_a"]
                if len(errs) != len(test_kinds):
                    continue
                acc = by_kind.setdefault(run["strategy"], {})
                per: dict[str, list[float]] = {}
                for kd, e in zip(test_kinds, errs):
                    per.setdefault(kd, []).append(e)
                for kd, vals in per.items():
                    acc.setdefault(kd, []).append(float(np.mean(vals)))
            metrics["test_error_by_kind"] = {
                strategy: {
                    kd: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                    for kd, v in kinds.items()
                }
                for strategy, kinds in by_kind.items()
            }
        metrics["timing"] = {
            "wall_seconds_total": main_res.get("wall_seconds"),
            "per_run": [
                {
                    "strategy": r["strategy"],
                    "seed": r["seed"],
                    "wall_seconds": r.get("wall_seconds"),
                }
                for r in main_res["runs"]
            ],
        }
    scarce = _load("scarce.json")
    if scarce:
        sc_f = curves_by_strategy(scarce, "f_mae_mev_per_a")
        sc_e = curves_by_strategy(scarce, "e_mae_mev_per_atom")
        metrics["scarce"] = {
            "n_pool": scarce["n_pool"],
            "pool_kind_fractions": scarce.get("pool_kind_fractions", {}),
            "curves": {"f_mae_mev_per_a": sc_f, "e_mae_mev_per_atom": sc_e},
        }
        if "committee" in sc_f and "random" in sc_f:
            metrics["scarce"]["label_reduction"] = {
                "force": label_reduction_factor(
                    np.array(sc_f["committee"]["budgets"]),
                    np.array(sc_f["committee"]["mean"]),
                    np.array(sc_f["random"]["mean"]),
                ),
                "energy": label_reduction_factor(
                    np.array(sc_e["committee"]["budgets"]),
                    np.array(sc_e["committee"]["mean"]),
                    np.array(sc_e["random"]["mean"]),
                ),
            }
    ceiling = _load("ceiling.json")
    if ceiling and ceiling.get("runs"):
        rec = ceiling["runs"][0]["rounds"][-1]["metrics"]
        metrics["ceiling"] = {
            "n_labels": ceiling["runs"][0]["rounds"][-1]["budget"],
            "f_mae_mev_per_a": rec["test"]["f_mae_mev_per_a"],
            "e_mae_mev_per_atom": rec["test"]["e_mae_mev_per_atom"],
            "transfer_f_mae_mev_per_a": rec.get("transfer", {}).get("f_mae_mev_per_a"),
        }
    explore = _load("md_explore.json")
    if explore:
        run = explore["runs"][0]
        metrics["md_explore"] = {
            "seed": run["seed"],
            "budgets": [rec["budget"] for rec in run["rounds"]],
            "f_mae_mev_per_a": [rec["metrics"]["test"]["f_mae_mev_per_a"] for rec in run["rounds"]],
            "e_mae_mev_per_atom": [
                rec["metrics"]["test"]["e_mae_mev_per_atom"] for rec in run["rounds"]
            ],
            "oracle_calls": run.get(
                "oracle_calls", explore.get("runs", [{}])[0].get("oracle_calls")
            ),
            "n_explore_selected": sum(
                1
                for rec in run["rounds"]
                if "selected" in rec
                for s in rec["selected"]
                if "ensmd" in s["kind"]
            ),
        }
    eos = _load("eos.json")
    if eos:
        table = {}
        for comp, entry in eos["compositions"].items():
            row = {}
            for name in ("active", "random", "committee"):
                if name in entry:
                    row[f"{name}_a0"] = entry[name]["a0_bcc_a"]
                    row[f"{name}_b0"] = entry[name]["b0_gpa"]
            if "teacher" in entry:
                row["teacher_a0"] = entry["teacher"]["a0_bcc_a"]
                row["teacher_b0"] = entry["teacher"]["b0_gpa"]
            if "anchor" in entry:
                row["anchor_a0"] = entry["anchor"].get("a0_bcc_a")
                row["anchor_b0"] = entry["anchor"].get("b0_gpa")
            table[comp] = row
        metrics["eos"] = {"table": table, "anchors_source": eos.get("anchors_source")}
    nve = _load("nve.json")
    if nve:
        metrics["nve"] = [
            {
                "temperature_init_k": r["temperature_init_k"],
                "drift_mev_per_atom_per_ps": r["drift_mev_per_atom_per_ps"],
                "n_steps": r["n_steps"],
                "dt_fs": r["dt_fs"],
                "n_atoms": r["n_atoms"],
            }
            for r in nve["runs"]
        ]
    for extra in ("provenance.json", "teacher_choice.json"):
        path = REPO / "data" / extra
        if path.exists():
            metrics[extra.replace(".json", "")] = json.loads(path.read_text())
    out = RESULTS / "metrics.json"
    out.write_text(json.dumps(metrics, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
