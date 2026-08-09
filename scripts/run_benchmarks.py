"""Run one YAML-configured benchmark (same dispatch as `falearn run`).

Usage:
    python scripts/run_benchmarks.py --config configs/main.yaml
"""

from __future__ import annotations

import argparse

from falearn.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    return cli_main(["run", "--config", args.config])


if __name__ == "__main__":
    raise SystemExit(main())
