"""Exp 16 — the ordering catastrophe was a FITTING ARTIFACT.

Exp 13(c) found that scale-normalising the Frobenius target (fit to ``T/s``, then
multiply by ``s`` -- exact, since the LCM family is positively homogeneous) raised
the headline joint-UQ result from 54% to 87% on identical seeds. This experiment
asks how far that reaches, because Exp 10 / 10b -- the project's central NEGATIVE
findings -- used the same unnormalised fit.

They reach all the way. Three published conclusions do not survive:

  * "with an unknown order the learned coordinate overfits into a confidently
    wrong chain (-84%, i.e. worse than diagonal)"
  * "spectral seriation is insufficient (-79%)"
  * "the method needs |Spearman| >= 0.999"

Part A runs the Exp 10 protocol at its exact settings (d=800, n=3000, permuted,
seeds 0-2) and changes ONE thing at a time, so the cause is unambiguous:

  A  exp10 as published -- uninformed init, unnormalised fit
  B  the same, scale-normalised                 <- isolates normalisation
  C  exp10 seriation, FROZEN, unnormalised (as published)
  D  seriation as INITIALISATION, then refined, normalised
  E  seriation FROZEN, normalised               <- isolates frozen-vs-refined

Part B re-runs the Exp 10b fragility curve with and without normalisation, to see
whether the threshold itself is real or was an artifact of the same deficiency.

Writes results/exp16_reconcile.csv and results/exp16_fragility.csv.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

import exp10_ordering as E
import common as G  # GLM helpers live in the shared harness
from exp15_ordering_fixes import joint_ll, spearman

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


def _problem(d, seed, n, rho, ntest, noise, prior, permute=True):
    """Data, references, and a gap() closure -- all posteriors via `laplace-torch`."""
    X, y, Xt, yt = E.simulate(n, d, rho, seed, ntest, noise)
    if permute:
        g = torch.Generator().manual_seed(1000 + seed)
        perm = torch.randperm(d, generator=g)
        X, Xt = X[:, perm], Xt[:, perm]
        true_c = perm.double()          # column j carries latent coordinate perm[j]
    else:
        true_c = torch.arange(d, dtype=torch.float64)

    w_map = G.glm_map_weights(X, y, prior, noise)
    GGN = X.T @ X / noise**2            # only used to seed the seriation estimate

    def jl(la):
        mu, Cj = G.glm_joint(la, Xt, noise)
        return joint_ll(mu, 0.5 * (Cj + Cj.T), yt)

    jf = jl(G.fit_glm(X, y, "full", prior, noise, w_map=w_map))
    jd = jl(G.fit_glm(X, y, "diag", prior, noise, w_map=w_map))

    def gap(la):
        return 100 * (jl(la) - jd) / (jf - jd)

    return X, y, w_map, GGN, true_c, gap


def _fit(d, X, y, w_map, prior, sigma, normalise, a_init=None, freeze_a=False,
         steps=800, lr=5e-2):
    """One LCM posterior through the library.

    `normalise=False` reproduces the pre-fix behaviour through the library's own
    `lcm_normalize` switch, so the "as published" arms stay reproducible now that
    `LCMCovariance` normalises by default. Orderings go through `lcm_init_a` /
    `lcm_learn_a`, added to `LCMLaplace` for exactly this.
    """
    return G.fit_glm(
        X, y, "lcm", prior, sigma, w_map=w_map,
        lcm_steps=steps, lcm_lr=lr, lcm_normalize=normalise,
        lcm_init_a=a_init if a_init is not None else torch.linspace(0, d * 0.1, d),
        lcm_learn_a=not freeze_a,
    )


def part_a(d=800, n=3000, rho=0.9, ntest=400, noise=0.5, prior=1.0, seeds=(0, 1, 2)):
    print(f"=== A: Exp 10 protocol, one variable changed at a time (d={d}, permuted) ===")
    rows = []
    for seed in seeds:
        X, y, w_map, GGN, true_c, gap = _problem(d, seed, n, rho, ntest, noise, prior)
        seri = G.seriation_coord(GGN) * 0.1
        variants = {
            "A_exp10_unnorm":       dict(normalise=False, a_init=None, freeze_a=False),
            "B_normalised":         dict(normalise=True,  a_init=None, freeze_a=False),
            "C_seri_frozen_unnorm": dict(normalise=False, a_init=seri, freeze_a=True),
            "D_seri_init_refined":  dict(normalise=True,  a_init=seri, freeze_a=False),
            "E_seri_frozen_norm":   dict(normalise=True,  a_init=seri, freeze_a=True),
            "F_oracle_refined":     dict(normalise=True,  a_init=true_c * 0.1, freeze_a=False),
        }
        for name, kw in variants.items():
            torch.manual_seed(seed)
            la = _fit(d, X, y, w_map, prior, noise, **kw)
            rows.append(dict(seed=seed, variant=name, gap=gap(la),
                             recovery=abs(spearman(la.lcm.a.detach(), true_c))))
        print(f"  seed {seed}: " + "  ".join(
            f"{r['variant'].split('_')[0]}={r['gap']:.0f}%" for r in rows[-len(variants):]))
    df = pd.DataFrame(rows)
    print(f"\n  {'variant':22} {'joint gap closed':>20}   {'recovery':>9}")
    for v in df.variant.unique():
        g = df[df.variant == v]
        print(f"  {v:22} {g.gap.mean():12.1f}% +/- {1.96*g.gap.std()/np.sqrt(len(g)):5.1f}   "
              f"{g.recovery.mean():9.3f}")
    return df


def part_b(d=400, n=3000, rho=0.9, ntest=400, noise=0.5, prior=1.0, seeds=(0, 1, 2),
           noises=(0.0, 0.3, 1.0, 3.0, 10.0)):
    """Exp 10b fragility curve, with and without normalisation. Order FROZEN."""
    print(f"\n=== B: fragility curve, unnormalised vs normalised (d={d}, order frozen) ===")
    rows = []
    print(f"  {'pos.noise':>9} {'|rho_s|':>9} {'gap unnorm':>12} {'gap norm':>12}")
    for nz in noises:
        acc, gu, gn = [], [], []
        for seed in seeds:
            X, y, w_map, _, _, gap = _problem(d, seed, n, rho, ntest, noise, prior,
                                              permute=False)
            g = torch.Generator().manual_seed(seed)
            c = torch.arange(d, dtype=torch.float64) + nz * torch.randn(d, generator=g)
            c = torch.argsort(torch.argsort(c)).double()
            acc.append(abs(spearman(c, torch.arange(d, dtype=torch.float64))))
            for norm, store in ((False, gu), (True, gn)):
                torch.manual_seed(seed)
                la = _fit(d, X, y, w_map, prior, noise, normalise=norm,
                          a_init=c * 0.1, freeze_a=True, steps=700)
                store.append(gap(la))
        rows.append(dict(pos_noise=nz, spearman=np.mean(acc),
                         gap_unnorm=np.mean(gu), gap_norm=np.mean(gn)))
        print(f"  {nz:>9.1f} {np.mean(acc):>9.4f} {np.mean(gu):>11.1f}% {np.mean(gn):>11.1f}%")
    print("  NOTE: |Spearman| saturates at 4 d.p. -- rows 2 and 3 both read 1.0000 while the")
    print("        gap moves by >30 points, so it is a poor diagnostic near 1.")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a = part_a(); a.to_csv(f"{RESULTS}/exp16_reconcile.csv", index=False)
    b = part_b(); b.to_csv(f"{RESULTS}/exp16_fragility.csv", index=False)
    print(f"\nsaved {RESULTS}/exp16_reconcile.csv, exp16_fragility.csv")
