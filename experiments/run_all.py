"""Run the full LCM-Laplace evaluation campaign and print the aggregated report.

    python experiments/run_all.py

Individual experiments can also be run standalone (exp1_fidelity.py, exp2_uci.py,
exp3_image.py, exp4_scaling.py). Results are written to experiments/results/.
"""
import runpy
import sys

EXPERIMENTS = [
    "experiments/exp1_fidelity.py",
    "experiments/exp2_uci.py",
    "experiments/exp3_image.py",
    "experiments/exp4_scaling.py",
]

if __name__ == "__main__":
    sys.path.insert(0, "experiments")
    for exp in EXPERIMENTS:
        print(f"\n{'#' * 70}\n# {exp}\n{'#' * 70}")
        runpy.run_path(exp, run_name="__main__")
    print(f"\n{'#' * 70}\n# aggregated report\n{'#' * 70}")
    runpy.run_path("experiments/make_report.py", run_name="__main__")
