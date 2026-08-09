"""Reproduce every committed number from scratch.

Order: teacher choice, dataset generation and labeling, DFT anchor fetch
(optional, needs MP_API_KEY), the main active-versus-random benchmark, the
MD-exploration variant, the EOS and NVE physics checks, the committed sample
carve, metrics aggregation, figures, and the executed tutorial notebook.

Roughly 3 hours on a desktop CPU, dominated by committee retraining.

Usage:
    python scripts/run_all.py [--skip-generate]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def run(cmd: list[str]) -> None:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="reuse an existing data/full/dataset.extxyz",
    )
    args = parser.parse_args()
    py = sys.executable
    run([py, str(SCRIPTS / "choose_teacher.py")])
    if not args.skip_generate or not (REPO / "data" / "full" / "dataset.extxyz").exists():
        run([py, "-m", "falearn.cli", "generate", "--out", "data/full/dataset.extxyz"])
    run([py, str(SCRIPTS / "fetch_mp_anchors.py")])
    for config in ("main", "ceiling", "scarce", "md_explore"):
        run([py, str(SCRIPTS / "run_benchmarks.py"), "--config", f"configs/{config}.yaml"])
    run([py, str(SCRIPTS / "train_random_reference.py")])
    for config in ("eos", "nve"):
        run([py, str(SCRIPTS / "run_benchmarks.py"), "--config", f"configs/{config}.yaml"])
    run([py, str(SCRIPTS / "make_sample_data.py")])
    run([py, str(SCRIPTS / "make_metrics.py")])
    run([py, str(SCRIPTS / "make_figures.py")])
    run([py, str(SCRIPTS / "build_notebook.py")])
    print("\nrun_all complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
