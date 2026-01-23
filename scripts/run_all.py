"""
run_all.py

One-click pipeline for the project:
Execution under Adverse Selection

This script:
1. Runs regime grid simulations
2. Runs misspecification grid simulations
3. Generates all reports and figures

Usage:
    python scripts/run_all.py
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80 + "\n")

    res = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    print("\n Starting full pipeline: Execution under Adverse Selection\n")

    steps = [
        # --- Data generation
        ["python", "scripts/run_regime_grid.py"],
        ["python", "scripts/run_misspec_grid.py"],

        # --- Reports
        ["python", "scripts/make_regime_report.py"],
        ["python", "scripts/make_misspec_report.py"],
        ["python", "scripts/make_dominance_report.py"],
    ]

    for cmd in steps:
        run(cmd)

    print("\n Pipeline completed successfully.")
    print("Reports generated in: reports/")
    print("Figures generated in: reports/figures/\n")


if __name__ == "__main__":
    main()
