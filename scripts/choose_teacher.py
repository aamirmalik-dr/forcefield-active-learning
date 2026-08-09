"""Choose the teacher by measured CPU speed and record provenance.

Times one energy-plus-force call on a 54-atom TiZrNb cell for CHGNet and
MACE-MP-0 small, picks the faster, and writes data/teacher_choice.json with
package, version, checkpoint, license, and the measured timings.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from falearn.structures import bcc_supercell
from falearn.teacher import frame_to_atoms, get_teacher

REPO = Path(__file__).resolve().parents[1]


def time_teacher(name: str, n_calls: int = 3) -> tuple[float, dict]:
    calc, info = get_teacher(name)
    fr = bcc_supercell("TiZrNb", 3, seed=0)
    rng = np.random.default_rng(0)
    fr.positions = fr.positions + rng.normal(0, 0.05, fr.positions.shape)
    atoms = frame_to_atoms(fr)
    atoms.calc = calc
    atoms.get_potential_energy()  # warm-up (model load, graph build)
    start = time.perf_counter()
    for k in range(n_calls):
        shaken = atoms.copy()
        shaken.positions = shaken.positions + 1e-4 * (k + 1)
        shaken.calc = calc
        shaken.get_potential_energy()
        shaken.get_forces()
    return (time.perf_counter() - start) / n_calls, info


def main() -> int:
    results = {}
    for name in ("chgnet", "mace-mp0-small"):
        try:
            seconds, info = time_teacher(name)
            results[name] = {"seconds_per_54atom_call": round(seconds, 4), **info}
            print(f"{name}: {seconds:.3f} s per 54-atom energy+force call")
        except Exception as exc:  # a missing optional package should not abort
            results[name] = {"error": str(exc)}
            print(f"{name}: failed ({exc})")
    timed = {k: v for k, v in results.items() if "seconds_per_54atom_call" in v}
    if not timed:
        print("no teacher available; install chgnet or mace-torch")
        return 1
    chosen = min(timed, key=lambda k: timed[k]["seconds_per_54atom_call"])
    payload = {
        "chosen": chosen,
        "criterion": "faster measured CPU wall time per 54-atom energy+force call",
        "candidates": results,
        "note": "labels produced by the chosen teacher are surrogate model labels, not DFT",
    }
    out = REPO / "data" / "teacher_choice.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"chosen teacher: {chosen} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
