"""Render all committed figures from the benchmark JSONs.

Fixed entity colors across every figure: committee acquisition is blue,
random selection is orange, MD exploration is aqua, the teacher is dark
gray, and real-DFT anchors are black. Rattle amplitudes use one sequential
blue ramp because amplitude is ordered, not categorical.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from falearn.calibration import binned_reliability

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGS = RESULTS / "figures"

C_ACTIVE = "#2a78d6"
C_RANDOM = "#eb6834"
C_CAPPED = "#1baf7a"
C_EXPLORE = "#4a3aa7"
C_TEACHER = "#52514e"
C_ANCHOR = "#0b0b0b"
C_MUTED = "#898781"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e2",
        "grid.linewidth": 0.6,
        "axes.edgecolor": C_MUTED,
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
    }
)


def _load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def _curve(ax, curves: dict, ylabel: str, title: str) -> None:
    for strategy, color, label in (
        ("committee", C_ACTIVE, "committee, pure top-k"),
        ("committee_capped", C_CAPPED, "committee, group cap 4"),
        ("random", C_RANDOM, "random selection"),
    ):
        if strategy not in curves:
            continue
        c = curves[strategy]
        budgets = np.array(c["budgets"])
        mean = np.array(c["mean"])
        std = np.array(c["std"])
        ax.plot(budgets, mean, color=color, lw=2, marker="o", ms=4, label=label)
        ax.fill_between(budgets, mean - std, mean + std, color=color, alpha=0.18, lw=0)
    ax.set_xlabel("teacher labels purchased")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)


def _selection_kind_fractions(main: dict) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    """Per-round batch-kind fractions of committee purchases, plus pool shares."""
    runs = [r for r in main["runs"] if r["strategy"] == "committee"]
    kinds = ["rattle04", "rattle08", "rattle14", "rattle20", "rattle28"]
    kinds += ["strain_vol", "strain_shear"]
    n_rounds = len(runs[0]["rounds"]) - 1
    frac = {k: np.zeros(n_rounds + 1) for k in kinds}
    # column 0 = the shared initial random draw; columns 1..R = acquisition rounds
    for col in range(n_rounds + 1):
        rows = []
        for run in runs:
            if col == 0:
                rows.extend(run["initial_purchase"])
            elif "selected" in run["rounds"][col - 1]:
                rows.extend(run["rounds"][col - 1]["selected"])
        total = max(1, len(rows))
        for k in kinds:
            frac[k][col] = sum(1 for s in rows if s["kind"] == k) / total
    # pool composition reference from the config counts is not stored; use the
    # union of everything the runs could see: initial + all selections is biased,
    # so the pool bar is written by make_metrics from the dataset instead.
    pool = np.zeros(len(kinds))
    meta = _load("metrics.json")
    if meta and "pool_kind_fractions" in meta.get("dataset", {}):
        pool = np.array([meta["dataset"]["pool_kind_fractions"].get(k, 0.0) for k in kinds])
    return kinds, frac, pool


def _acquisition_axis(ax, main: dict) -> None:
    kinds, frac, pool = _selection_kind_fractions(main)
    n_cols = len(next(iter(frac.values())))
    x = np.arange(n_cols + 1)
    colors = {
        "rattle04": BLUE_RAMP[0],
        "rattle08": BLUE_RAMP[1],
        "rattle14": BLUE_RAMP[2],
        "rattle20": BLUE_RAMP[4],
        "rattle28": BLUE_RAMP[6],
        "strain_vol": "#1baf7a",
        "strain_shear": "#eda100",
    }
    bottom = np.zeros(n_cols + 1)
    for k in kinds:
        vals = np.concatenate([[pool[kinds.index(k)]], frac[k]])
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=colors[k],
            width=0.82,
            label=k.replace("rattle", "rattle "),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(["pool", "init"] + [f"r{k}" for k in range(1, n_cols)], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of purchases")
    ax.set_title("what the committee bought, by batch kind", fontsize=9.5)
    ax.grid(False)
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        handlelength=1.2,
        columnspacing=1.0,
    )


def fig_hero(metrics: dict, main: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    _curve(
        axes[0],
        metrics["curves"]["f_mae_mev_per_a"],
        "test force MAE (meV/A)",
        "forces versus label budget (3 seeds)",
    )
    _curve(
        axes[1],
        metrics["curves"]["e_mae_mev_per_atom"],
        "test energy MAE (meV/atom)",
        "energies versus label budget (3 seeds)",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    ceiling = metrics.get("ceiling")
    if ceiling:
        for ax, key, xpos, ha in (
            (axes[0], "f_mae_mev_per_a", 0.97, "right"),
            (axes[1], "e_mae_mev_per_atom", 0.03, "left"),
        ):
            ax.axhline(ceiling[key], color=C_ANCHOR, lw=1.2, ls=":")
            ax.annotate(
                f"same committee, entire pool ({ceiling['n_labels']} labels)",
                xy=(xpos, ceiling[key]),
                xycoords=("axes fraction", "data"),
                va="bottom",
                ha=ha,
                fontsize=7,
                color="#52514e",
            )
    _acquisition_axis(axes[2], main)
    fig.tight_layout()
    fig.savefig(FIGS / "hero.png", bbox_inches="tight")
    plt.close(fig)


def fig_calibration(main: dict) -> None:
    committees = [r for r in main["runs"] if r["strategy"] == "committee" and "calibration" in r]
    if not committees:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    cal = committees[0]["calibration"]
    d = np.array(cal["disagreement_mean_mev_per_a"])
    e = np.array(cal["f_rmse_mev_per_a"])
    axes[0].scatter(d, e, s=12, color=C_ACTIVE, alpha=0.45, edgecolors="none")
    rel = binned_reliability(d, e, n_bins=8)
    axes[0].plot(
        rel["bin_median_disagreement"],
        rel["bin_mean_error"],
        color=C_ANCHOR,
        lw=2,
        marker="s",
        ms=5,
        label="bin mean",
    )
    lim = [min(d.min(), e.min()) * 0.8, max(d.max(), e.max()) * 1.2]
    axes[0].plot(lim, lim, color=C_MUTED, lw=1, ls="--", label="y = x")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("committee force disagreement (meV/A)")
    axes[0].set_ylabel("actual force RMSE of mean (meV/A)")
    axes[0].set_title(
        f"held-out test frames, seed {committees[0]['seed']} "
        f"(Spearman rho {rel['spearman_rho']:.2f})",
        fontsize=9.5,
    )
    axes[0].legend(frameon=False, fontsize=8)
    rhos = []
    for run in committees:
        c = run["calibration"]
        r = binned_reliability(
            np.array(c["disagreement_mean_mev_per_a"]),
            np.array(c["f_rmse_mev_per_a"]),
            n_bins=8,
        )
        rhos.append((run["seed"], r["spearman_rho"], r["pearson_r"]))
    seeds = [f"seed {s}" for s, _, _ in rhos]
    axes[1].bar(
        np.arange(len(rhos)) - 0.18,
        [r for _, r, _ in rhos],
        width=0.36,
        color=C_ACTIVE,
        label="Spearman rho",
    )
    axes[1].bar(
        np.arange(len(rhos)) + 0.18,
        [p for _, _, p in rhos],
        width=0.36,
        color=BLUE_RAMP[1],
        label="Pearson r",
    )
    axes[1].set_xticks(np.arange(len(rhos)))
    axes[1].set_xticklabels(seeds)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("correlation with actual error")
    axes[1].set_title("disagreement-error correlation per seed", fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="x")
    fig.tight_layout()
    fig.savefig(FIGS / "calibration.png", bbox_inches="tight")
    plt.close(fig)


def fig_acquisition_detail(main: dict) -> None:
    runs = [r for r in main["runs"] if r["strategy"] == "committee"]
    comps = ["Ti", "Zr", "Nb", "TiZr", "TiNb", "ZrNb", "TiZrNb"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    # composition of purchases vs pool
    meta = _load("metrics.json") or {}
    pool_comp = meta.get("dataset", {}).get("pool_composition_fractions", {})
    bought = {c: 0 for c in comps}
    for run in runs:
        for rec in run["rounds"]:
            for s in rec.get("selected", []):
                if s["composition"] in bought:
                    bought[s["composition"]] += 1
    total = max(1, sum(bought.values()))
    x = np.arange(len(comps))
    axes[0].bar(
        x - 0.2,
        [bought[c] / total for c in comps],
        width=0.4,
        color=C_ACTIVE,
        label="committee purchases",
    )
    axes[0].bar(
        x + 0.2,
        [pool_comp.get(c, 0.0) for c in comps],
        width=0.4,
        color=C_MUTED,
        label="pool share",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comps, fontsize=8)
    axes[0].set_ylabel("fraction")
    axes[0].set_title("purchases by composition (all seeds, rounds 1+)", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="x")
    # mean disagreement score of purchases per round
    n_rounds = len(runs[0]["rounds"]) - 1
    for run in runs:
        scores = [
            np.mean([s["score"] for s in rec["selected"]]) * 1000
            for rec in run["rounds"]
            if "selected" in rec
        ]
        axes[1].plot(
            np.arange(1, n_rounds + 1),
            scores,
            color=C_ACTIVE,
            alpha=0.55,
            lw=1.5,
            marker="o",
            ms=3,
        )
    axes[1].set_xlabel("acquisition round")
    axes[1].set_ylabel("mean disagreement of purchases (meV/A)")
    axes[1].set_title("uncertainty of what was bought, per seed", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGS / "acquisition_map.png", bbox_inches="tight")
    plt.close(fig)


def fig_scarce(metrics: dict) -> None:
    sc = metrics.get("scarce")
    if not sc:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    n_seeds = len(sc["curves"]["f_mae_mev_per_a"].get("committee", {}).get("seeds", []))
    _curve(
        axes[0],
        sc["curves"]["f_mae_mev_per_a"],
        "test force MAE (meV/A)",
        f"scarce pool ({sc['n_pool']} frames), forces ({n_seeds} seeds)",
    )
    _curve(
        axes[1],
        sc["curves"]["e_mae_mev_per_atom"],
        "test energy MAE (meV/atom)",
        f"scarce pool, energies ({n_seeds} seeds)",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "scarce.png", bbox_inches="tight")
    plt.close(fig)


def fig_error_by_kind(metrics: dict) -> None:
    data = metrics.get("test_error_by_kind")
    if not data:
        return
    kinds = ["rattle04", "rattle08", "rattle14", "rattle20", "rattle28"]
    kinds += ["strain_vol", "strain_shear"]
    kinds = [k for k in kinds if any(k in d for d in data.values())]
    entries = [
        ("committee", C_ACTIVE, "committee, pure top-k"),
        ("committee_capped", C_CAPPED, "committee, group cap 4"),
        ("random", C_RANDOM, "random selection"),
    ]
    entries = [e for e in entries if e[0] in data]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    x = np.arange(len(kinds))
    width = 0.8 / max(1, len(entries))
    for i, (strategy, color, label) in enumerate(entries):
        vals = [data[strategy].get(k, {}).get("mean", np.nan) for k in kinds]
        errs = [data[strategy].get(k, {}).get("std", 0.0) for k in kinds]
        ax.bar(
            x + (i - (len(entries) - 1) / 2) * width,
            vals,
            width=width * 0.92,
            yerr=errs,
            color=color,
            label=label,
            error_kw={"elinewidth": 0.8, "ecolor": "#52514e"},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, fontsize=8)
    ax.set_ylabel("test force RMSE (meV/A)")
    ax.set_title("final-budget error by test batch kind (mean over 3 seeds)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(FIGS / "error_by_kind.png", bbox_inches="tight")
    plt.close(fig)


def fig_split_protocol(metrics: dict) -> None:
    rows = metrics.get("split_protocol", [])
    if not rows:
        return
    strategies = [
        s
        for s in ("committee", "committee_capped", "random")
        if any(r["strategy"] == s for r in rows)
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    x = np.arange(len(strategies))
    honest = [np.mean([r["honest_f_mae"] for r in rows if r["strategy"] == s]) for s in strategies]
    sibling = [
        np.mean([r["sibling_f_mae"] for r in rows if r["strategy"] == s]) for s in strategies
    ]
    ax.bar(x - 0.2, honest, width=0.4, color=C_ACTIVE, label="held-out groups (honest)")
    ax.bar(x + 0.2, sibling, width=0.4, color=BLUE_RAMP[1], label="sibling frames (leaky)")
    for xi, (h, s) in enumerate(zip(honest, sibling)):
        ax.text(xi - 0.2, h + 0.6, f"{h:.0f}", ha="center", fontsize=8, color="#0b0b0b")
        ax.text(xi + 0.2, s + 0.6, f"{s:.0f}", ha="center", fontsize=8, color="#0b0b0b")
    ax.set_xticks(x)
    labels = {"committee": "committee", "committee_capped": "committee + cap", "random": "random"}
    ax.set_xticklabels([labels[s] for s in strategies])
    ax.set_ylabel("final test force MAE (meV/A)")
    ax.set_title("what a random-frame protocol would have claimed (mean of 3 seeds)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(FIGS / "split_protocol.png", bbox_inches="tight")
    plt.close(fig)


def fig_md_explore(metrics: dict) -> None:
    ex = metrics.get("md_explore")
    curves = metrics.get("curves", {}).get("f_mae_mev_per_a", {})
    if not ex or not curves:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    seed = ex["seed"]
    for strategy, color, label in (
        ("committee", C_ACTIVE, "committee, fixed pool"),
        ("committee_capped", C_CAPPED, "committee + cap, fixed pool"),
        ("random", C_RANDOM, "random, fixed pool"),
    ):
        c = curves.get(strategy)
        if not c or seed not in c["seeds"]:
            continue
        k = c["seeds"].index(seed)
        ax.plot(
            c["budgets"],
            c["per_seed"][k],
            color=color,
            lw=2,
            marker="o",
            ms=4,
            label=f"{label} (seed {seed})",
        )
    ax.plot(
        ex["budgets"],
        ex["f_mae_mev_per_a"],
        color=C_EXPLORE,
        lw=2,
        marker="s",
        ms=4,
        label=f"committee + MD exploration (seed {seed})",
    )
    ax.annotate(
        f"{ex['n_explore_selected']} of {ex['budgets'][-1]} purchases were\n"
        f"ensemble-MD snapshots ({ex['oracle_calls']} live teacher calls)",
        xy=(0.97, 0.96),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.5,
        color="#52514e",
    )
    ax.set_xlabel("teacher labels purchased")
    ax.set_ylabel("test force MAE (meV/A)")
    ax.set_title("does on-policy MD exploration help?", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "md_explore.png", bbox_inches="tight")
    plt.close(fig)


def fig_eos(eos: dict) -> None:
    comps = list(eos["compositions"])
    fig, axes = plt.subplots(1, len(comps), figsize=(3.3 * len(comps), 3.2))
    for ax, comp in zip(np.atleast_1d(axes), comps):
        entry = eos["compositions"][comp]
        text_lines = []
        for name, color, label in (
            ("active", C_ACTIVE, "active committee"),
            ("random", C_RANDOM, "random committee"),
            ("committee", C_ACTIVE, "committee mean"),
        ):
            if name not in entry:
                continue
            com = entry[name]
            v = np.array(com["volumes_a3_per_atom"])
            e = np.array(com["energies_ev_per_atom"])
            ax.plot(v, e - e.min(), color=color, lw=2, marker="o", ms=3.5, label=label)
            text_lines.append(f"{label} a0 {com['a0_bcc_a']:.3f} A, B0 {com['b0_gpa']:.0f} GPa")
        if "teacher" in entry:
            te = entry["teacher"]
            tv = np.array(te["volumes_a3_per_atom"])
            tee = np.array(te["energies_ev_per_atom"])
            ax.plot(
                tv, tee - tee.min(), color=C_TEACHER, lw=1.8, ls="--", marker="", label="teacher"
            )
            text_lines.append(f"teacher a0 {te['a0_bcc_a']:.3f}, B0 {te['b0_gpa']:.0f}")
        text = "\n".join(text_lines)
        if "anchor" in entry:
            anchor = entry["anchor"]
            v_anchor = anchor["a0_bcc_a"] ** 3 / 2.0
            ax.axvline(v_anchor, color=C_ANCHOR, lw=1.2, ls=":")
            text += f"\nDFT anchor a0 {anchor['a0_bcc_a']:.3f}"
            if anchor.get("b0_gpa"):
                text += f", B0 {anchor['b0_gpa']:.0f}"
        ax.set_title(comp, fontsize=10)
        ax.set_xlabel("volume (A$^3$/atom)")
        ax.set_ylabel("E - E$_{min}$ (eV/atom)")
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", fontsize=7, color="#52514e")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "eos.png", bbox_inches="tight")
    plt.close(fig)


def fig_nve(nve: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    colors = {300: BLUE_RAMP[2], 900: BLUE_RAMP[5]}
    for run in nve["runs"]:
        t = np.array(run["time_ps"])
        e = np.array(run["e_total_ev"])
        temp = int(run["temperature_init_k"])
        ax.plot(
            t,
            (e - e[0]) / run["n_atoms"] * 1000,
            lw=1.5,
            color=colors.get(temp, C_ACTIVE),
            label=f"{temp} K init, drift {run['drift_mev_per_atom_per_ps']:+.4f} meV/atom/ps",
        )
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("total-energy change (meV/atom)")
    ax.set_title(
        f"NVE with committee-mean forces ({nve['runs'][0]['n_atoms']} atoms, "
        f"{nve['runs'][0]['dt_fs']:.0f} fs steps)",
        fontsize=9.5,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "nve.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    metrics = _load("metrics.json")
    main_res = _load("main.json")
    if metrics and main_res:
        fig_hero(metrics, main_res)
        fig_calibration(main_res)
        fig_acquisition_detail(main_res)
        fig_scarce(metrics)
        fig_error_by_kind(metrics)
        fig_split_protocol(metrics)
        fig_md_explore(metrics)
    eos = _load("eos.json")
    if eos:
        fig_eos(eos)
    nve = _load("nve.json")
    if nve:
        fig_nve(nve)
    print(f"figures written to {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
