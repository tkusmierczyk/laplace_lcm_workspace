"""Run the full LCM-Laplace evaluation campaign and print the aggregated report.

    python experiments/run_all.py

Individual experiments can also be run standalone (exp1_fidelity.py, exp2_uci.py,
exp3_image.py, exp4_scaling.py). Results are written to experiments/results/.
"""
import runpy
import sys

EXPERIMENTS = [
    # --- initial evaluation vs MAP / Diag / KFAC / Full ---
    "experiments/exp1_fidelity.py",        # posterior fidelity + when-structure-helps
    "experiments/exp2_uci.py",             # UCI regression UQ
    "experiments/exp3_image.py",           # MNIST last-layer calibration + OOD
    "experiments/exp3b_cifar.py",          # CIFAR-10 last-layer calibration + OOD
    "experiments/exp4_scaling.py",         # cost/scaling + ablations
    "experiments/exp4b_ablation_controlled.py",  # controlled chain-recovery ablation
    # --- search for a compelling use ---
    "experiments/exp5_epistemic.py",       # epistemic (MI/BALD) OOD scores
    "experiments/exp6_perlayer.py",        # per-layer (block-diagonal) LCM
    "experiments/exp7_correlated_glm.py",  # 1-D-correlated GLM (LCM's home regime)
    "experiments/exp8b_joint_robust.py",   # joint vs marginal predictive UQ
    "experiments/exp9_active_learning.py", # batch active learning (negative result)
    # --- publication hardening ---
    "experiments/exp8c_joint_ci.py",       # headline joint-UQ result, 10 seeds + CIs
    "experiments/exp10_ordering.py",       # learned vs fixed vs seriation ordering
    "experiments/exp10b_fragility.py",     # ordering fragility + recovery failure
    "experiments/exp11_realdata.py",       # real ordered-feature data (NIR, CO2)
    "experiments/exp12_cost.py",           # cost/complexity vs KFAC; applicability
    "experiments/exp13_sameclass.py",      # vs banded-Cholesky / low-rank peers
]

# Diagnostics (print-only, no CSV): exp8_joint.py, ctrl_ordering.py,
# diag_matfree_init.py -- run individually if needed.

if __name__ == "__main__":
    sys.path.insert(0, "experiments")
    for exp in EXPERIMENTS:
        print(f"\n{'#' * 70}\n# {exp}\n{'#' * 70}")
        runpy.run_path(exp, run_name="__main__")
    print(f"\n{'#' * 70}\n# aggregated report\n{'#' * 70}")
    runpy.run_path("experiments/make_report.py", run_name="__main__")
