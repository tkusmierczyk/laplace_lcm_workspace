"""Aggregate all experiment result CSVs into markdown tables for EXPERIMENTS.md."""
from __future__ import annotations

import os

import pandas as pd

R = "experiments/results"
ORDER = ["MAP", "Diag", "KFAC", "Full", "LCM"]


def _has(f):
    return os.path.exists(os.path.join(R, f))


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def report():
    print("# Aggregated results\n")

    if _has("exp1_fidelity.csv"):
        df = pd.read_csv(f"{R}/exp1_fidelity.csv")
        print("## Exp1 fidelity: KL(approx || full)\n")
        print(md_table(df[["setting", "P", "KL_Diag", "KL_KFAC", "KL_LCM"]], "{:.1f}"))
        print("\n## Exp1 fidelity: rel. Frobenius(cov)\n")
        print(md_table(df[["setting", "Frob_Diag", "Frob_KFAC", "Frob_LCM"]], "{:.3f}"))
        print()

    if _has("exp2_uci_summary.csv"):
        s = pd.read_csv(f"{R}/exp2_uci_summary.csv")
        piv = s.pivot(index="dataset", columns="method", values="ll_mean")[ORDER]
        print("## Exp2 UCI: test log-likelihood (higher better)\n")
        print(md_table(piv.reset_index(), "{:+.3f}"))
        if _has("exp2_uci_avgrank.csv"):
            ar = pd.read_csv(f"{R}/exp2_uci_avgrank.csv")
            print("\n**Average test-LL rank (1=best):** " +
                  ", ".join(f"{r['method']} {r['rank']:.2f}" for _, r in ar.iterrows()))
        print()

    for fn, title in [("exp3_image_summary.csv", "Exp3 MNIST last-layer"),
                      ("exp3b_cifar_summary.csv", "Exp3b CIFAR-10 last-layer")]:
        if _has(fn):
            s = pd.read_csv(f"{R}/{fn}")
            print(f"## {title} (NLL/ECE lower, AUROC higher)\n")
            keep = [c for c in ["method", "acc", "nll_mean", "ece_mean", "auroc_mean"]
                    if c in s.columns]
            print(md_table(s[keep], "{:.4f}"))
            print()

    if _has("exp4_scaling.csv"):
        df = pd.read_csv(f"{R}/exp4_scaling.csv")
        print("## Exp4 scaling: log-det time (ms) and covariance memory (floats)\n")
        print(md_table(df, "{:.2f}"))
        print()

    if _has("exp4_ablation_structure.csv"):
        print("## Exp4 ablation: structured fit (rel Frobenius to real GGN)\n")
        print(md_table(pd.read_csv(f"{R}/exp4_ablation_structure.csv"), "{:.4f}"))
        print()
    if _has("exp4b_ablation_controlled.csv"):
        print("## Exp4b ablation: controlled scrambled-AR(1) target (recoverable)\n")
        print(md_table(pd.read_csv(f"{R}/exp4b_ablation_controlled.csv"), "{:.4f}"))
        print()
    if _has("exp4_ablation_steps.csv"):
        print("## Exp4 ablation: fit vs steps\n")
        print(md_table(pd.read_csv(f"{R}/exp4_ablation_steps.csv"), "{:.4f}"))


if __name__ == "__main__":
    report()
