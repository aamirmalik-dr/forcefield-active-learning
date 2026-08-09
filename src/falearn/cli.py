"""The falearn command line.

Subcommands:
    demo      Offline micro active-learning loop on the committed sample.
    run       Run a YAML-configured benchmark (al | eos | nve).
    eval      Evaluate a saved committee on a labeled extxyz file.
    generate  Generate and teacher-label the full dataset (needs chgnet).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _cmd_demo(args: argparse.Namespace) -> int:
    from falearn.bench import load_config, run_al_benchmark
    from falearn.calibration import binned_reliability

    cfg = load_config(args.config)
    results = run_al_benchmark(cfg)
    print(
        f"pool {results['n_pool']} frames, test {results['n_test']}, "
        f"transfer {results['n_transfer']}"
    )
    print(f"{'strategy':<10} {'seed':>4} {'budget':>7} {'E MAE':>10} {'F MAE':>10}")
    for run in results["runs"]:
        last = run["rounds"][-1]
        m = last["metrics"]["test"]
        print(
            f"{run['strategy']:<10} {run['seed']:>4} {last['budget']:>7} "
            f"{m['e_mae_mev_per_atom']:>7.2f} meV/atom {m['f_mae_mev_per_a']:>7.1f} meV/A"
        )
    for run in results["runs"]:
        if run["strategy"] == "committee" and "calibration" in run:
            cal = run["calibration"]
            rel = binned_reliability(
                np.array(cal["disagreement_mean_mev_per_a"]),
                np.array(cal["f_rmse_mev_per_a"]),
                n_bins=min(6, max(2, len(cal["f_rmse_mev_per_a"]) // 8)),
            )
            print(
                f"calibration (committee, seed {run['seed']}): "
                f"Spearman rho {rel['spearman_rho']:.3f} over {rel['n_frames']} test frames"
            )
            break
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from falearn.bench import (
        load_config,
        run_al_benchmark,
        run_eos_benchmark,
        run_nve_benchmark,
    )

    cfg = load_config(args.config)
    kind = cfg.get("kind", "al")
    if kind == "al":
        oracle = None
        if cfg.get("explore"):
            from falearn.teacher import get_teacher, make_oracle

            calc, info = get_teacher(cfg.get("teacher", "chgnet"))
            print(f"teacher oracle: {info['package']} {info['version']}")
            oracle = make_oracle(calc)
        results = run_al_benchmark(cfg, oracle=oracle)
    elif kind == "eos":
        teacher_calc = None
        if cfg.get("teacher_scan", True):
            from falearn.teacher import get_teacher

            teacher_calc, _ = get_teacher(cfg.get("teacher", "chgnet"))
        results = run_eos_benchmark(cfg, teacher_calc=teacher_calc)
    elif kind == "nve":
        results = run_nve_benchmark(cfg)
    else:
        print(f"unknown benchmark kind {kind}", file=sys.stderr)
        return 2
    out = cfg.get("out")
    print(f"done; results written to {out}" if out else json.dumps(results, indent=1))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from falearn.data import load_frames
    from falearn.descriptors import AcsfParams, frame_graph
    from falearn.ensemble import evaluate_mean
    from falearn.model import load_members

    members, extra = load_members(args.committee)
    frames = [fr for fr in load_frames(args.frames) if fr.energy is not None]
    if not frames:
        print("no labeled frames found", file=sys.stderr)
        return 2
    acsf = AcsfParams()
    graphs = [frame_graph(fr, acsf) for fr in frames]
    m = evaluate_mean(
        members, graphs, np.array([fr.energy for fr in frames]), [fr.forces for fr in frames]
    )
    m.pop("per_frame")
    print(
        f"committee of {len(members)} members "
        f"(trained at budget {extra.get('budget', '?')}) on {len(frames)} frames:"
    )
    for key in ("e_mae_mev_per_atom", "e_rmse_mev_per_atom", "f_mae_mev_per_a", "f_rmse_mev_per_a"):
        print(f"  {key}: {m[key]:.3f}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from falearn.data import save_frames
    from falearn.structures import (
        POOL_COMPOSITIONS,
        TRANSFER_COMPOSITIONS,
        build_dataset,
    )
    from falearn.teacher import get_teacher, label_frames

    calc, info = get_teacher(args.teacher)
    print(f"teacher: {info['package']} {info['version']} ({info['checkpoint']})")
    frames = build_dataset(
        POOL_COMPOSITIONS,
        rattle_amplitudes=(0.04, 0.08, 0.14, 0.20, 0.28),
        batches_per_kind=2,
        frames_per_batch=16,
        seed=args.seed,
    )
    frames += build_dataset(
        TRANSFER_COMPOSITIONS,
        rattle_amplitudes=(0.04, 0.08, 0.14, 0.20, 0.28),
        batches_per_kind=1,
        frames_per_batch=12,
        seed=args.seed + 1,
    )
    print(f"generated {len(frames)} frames; labeling with the teacher...")
    sec = label_frames(frames, calc)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_frames(frames, args.out)
    groups = {fr.group for fr in frames}
    comps = sorted({fr.composition for fr in frames})
    provenance = {
        "teacher": info,
        "label_kind": "surrogate model labels from a pretrained universal potential, not DFT",
        "n_frames": len(frames),
        "n_groups": len(groups),
        "compositions": comps,
        "atoms_min": int(min(fr.n_atoms for fr in frames)),
        "atoms_max": int(max(fr.n_atoms for fr in frames)),
        "seconds_per_frame": round(sec, 4),
        "seed": args.seed,
    }
    prov_path = Path(args.out).parent.parent / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"labeled {len(frames)} frames at {sec:.3f} s/frame -> {args.out}")
    print(f"provenance -> {prov_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="falearn",
        description="Query-by-committee active learning for TiZrNb interatomic potentials",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="offline micro loop on the committed sample")
    p.add_argument("--config", default="configs/fast.yaml")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("run", help="run a YAML-configured benchmark")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("eval", help="evaluate a saved committee on labeled frames")
    p.add_argument("--committee", default="models/committee_active.pt")
    p.add_argument("--frames", default="data/sample_frames.extxyz")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("generate", help="generate and teacher-label the full dataset")
    p.add_argument("--out", default="data/full/dataset.extxyz")
    p.add_argument("--teacher", default="chgnet")
    p.add_argument("--seed", type=int, default=20260809)
    p.set_defaults(func=_cmd_generate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
