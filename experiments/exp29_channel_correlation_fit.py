"""Exp 29 — the fitting question in the setting that produces the paper's accuracy numbers.

WHY THIS EXISTS, AND WHY IT IS NOT A REPEAT OF EXP 25. Reading the authors' released
code shows they use their covariance family in two unrelated ways, and every result this
project has produced so far addresses only the first.

  1. `efficient_inversion/{resnet34,vit}` builds the O(dim log dim) filtered likelihood over
     a *flattened feature map* -- 200 704 numbers for one ResNet-34 layer -- and uses it as a
     loss term while optimising images. This is what their likelihood and scalability
     figures report, and what Exp 25 and Exp 27 compare against.
  2. `resnet32/continual_runner/task_contrastive_cl_runner.py`, the entry point behind the
     continual-learning accuracy tables, imports a *different and much smaller* copy of the
     model (`resnet32/inversion/lcm.py`, dense, no likelihood at all). It hooks the input of
     every `BatchNorm2d`, accumulates a covariance over **channels only** -- `LCM(m.num_features)`,
     so 16 to 512 variables rather than 200 704 -- converts it to a **correlation** matrix,
     blends it across tasks, and fits it by mean-squared error with Adam for 200 epochs
     (`utils.fit_lcm_correlation`).

So the published accuracy gains rest on fitting a few-hundred-variable correlation matrix,
where Exp 25's two headline advantages both evaporate by construction: at 512 variables the
cost of fitting is irrelevant, and the object fitted is a correlation, not a covariance.
What does *not* evaporate is the ordering question. Channels have no natural one-dimensional
order, and this project has found five times over that the coordinates of this family are
recoverable by seriation but not learnable by gradient descent. Their rule learns them by
gradient descent from a `linspace` initialisation.

HYPOTHESIS. In this channel-wise setting their fitting rule is limited by the coordinate
ordering rather than by the loss, so seriating the channels first and then fitting in closed
form reconstructs the held-out channel correlation markedly better, at a fraction of the
cost -- while simply swapping the loss, without fixing the ordering, changes little.

THE CLAIM A PAPER WOULD MAKE. "The published accuracy gains are obtained with a covariance
that is badly fitted; seriating the channels and fitting in closed form recovers much more of
the real correlation structure for the same parameter budget and less compute."

CHEAPEST FALSIFIER, AND WHAT WOULD MAKE US DROP IT. Fit on one set of images, score on a
held-out set. **If their gradient rule matches the closed form on held-out correlation error,
the Exp 25 result does not transfer to the configuration behind their accuracy tables**, and
the honest claim shrinks to the large flattened-feature setting only. Equally, if seriation
does not separate from the natural channel order, the ordering story does not apply here.

Every arm is scored by relative error against the held-out empirical correlation, which is
scale-free, so the oracle-scale problem that invalidated this project's earlier marginal
likelihood comparisons cannot arise. Two reference points keep the numbers honest: storing
the full empirical matrix from the fitting split, which is what the method claims to replace
and costs O(C^2) memory instead of O(C), and the identity, which is what you get for no
parameters at all. A structured fit is only interesting between those two.

Runs on the laptop: the largest matrix here is 512 x 512.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common  # noqa: E402
from exp25_paper_scale_baselines import WideCNN, load_cifar, train  # noqa: E402
from laplace.lcm import LCMCovariance  # noqa: E402


def load_published_fit(ref_dir):
    """Their dense model and their correlation fitting rule, from the clone.

    Appended to `sys.path`, never inserted: their directory ships a `utils.py`, and
    putting it first would shadow any later `import utils` elsewhere in the process.
    """
    d = os.path.join(ref_dir, "resnet32")
    if d not in sys.path:
        sys.path.append(d)
    sys.path.append(os.path.join(d, "inversion"))
    from inversion.lcm import LCM  # noqa: PLC0415
    from utils import fit_lcm_correlation  # noqa: PLC0415
    return LCM, fit_lcm_correlation


# ------------------------------------------------------------------ statistics
@torch.no_grad()
def channel_correlations(model, X, batch=64):
    """Per-layer channel correlation at the authors' hook point.

    They register a forward hook on every `BatchNorm2d` and take `inp[0]`, reshaped so
    every spatial position of every image is one observation of a `C`-vector. This
    reproduces that exactly, streaming the count, sum and `C x C` scatter so no
    activations are retained.
    """
    mods = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    n = [0] * len(mods)
    s1 = [None] * len(mods)
    s2 = [None] * len(mods)

    def make_hook(i):
        def hook(_module, inp, _out):
            x = inp[0].detach().double()
            x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            if s1[i] is None:
                s1[i] = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
                s2[i] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64,
                                    device=x.device)
            n[i] += x.shape[0]
            s1[i] += x.sum(0)
            s2[i] += x.T @ x
        return hook

    handles = [m.register_forward_hook(make_hook(i)) for i, m in enumerate(mods)]
    model.eval()
    for i in range(0, len(X), batch):
        model(X[i:i + batch])
    for h in handles:
        h.remove()

    out = []
    for i, m in enumerate(mods):
        mu = s1[i] / n[i]
        cov = (s2[i] / n[i] - torch.outer(mu, mu)).cpu()
        out.append((m.num_features, to_correlation(cov)))
    return out


def to_correlation(cov):
    sd = cov.diagonal().clamp(min=1e-12).sqrt()
    R = cov / sd[:, None] / sd[None, :]
    R = 0.5 * (R + R.T)
    R.fill_diagonal_(1.0)
    return R


def seriation_order(R):
    """Channels in chain order, by the Fiedler vector of the |correlation| graph.

    `common.seriation_coord` returns each channel's rank; `fit_moments` wants the
    permutation that lists channels along the chain, which is the argsort instead.
    """
    A = R.abs()
    lap = torch.diag(A.sum(1)) - A
    fiedler = torch.linalg.eigh(lap).eigenvectors[:, 1]
    return torch.argsort(fiedler)


def bands_along(R, order):
    """The diagonal and first two off-diagonal bands of `R`, in chain order."""
    P = R[order][:, order]
    return P.diagonal().clone(), P.diagonal(1).clone(), P.diagonal(2).clone()


# ----------------------------------------------------------------------- arms
def lcm_correlation(dim, a, w, d):
    """Correlation implied by `diag(d) + diag(w) K(a) diag(w)`, formed densely."""
    K = torch.exp(-(a[:, None] - a[None, :]).abs())
    return to_correlation(torch.diag(d) + K * torch.outer(w, w))


def fit_published(LCM, fit_fn, R, order=None, epochs=200, lr=0.01):
    """Their rule: Adam on mean-squared error between correlations.

    With `order`, only the initialisation of the coordinates changes -- the loss, the
    optimiser, the learning rate and the 200 epochs are theirs untouched -- so the
    comparison isolates the ordering from the fitting objective.
    """
    dim = R.shape[0]
    mod = LCM(dim).double()
    if order is not None:
        pos = torch.empty(dim, dtype=torch.float64)
        pos[order] = torch.linspace(-1.0, 1.0, dim, dtype=torch.float64)
        with torch.no_grad():
            mod.a.copy_(pos)
    fit_fn(mod, R, epochs=epochs, lr=lr)
    with torch.no_grad():
        return mod.correlation().double()


def fit_family_best_effort(LCM, R, order, epochs=5000, lr=0.05):
    """The same family and the same loss, fitted far past the authors' budget.

    This arm exists to stop us mistaking one thing for another. If their 200-epoch rule
    fits poorly, the cause is either the budget or the family, and only a deliberately
    over-trained fit tells us which. Seriated initialisation, a cosine-decayed learning
    rate and 25x their epochs; at these widths it still costs under a second, so there is
    no excuse for leaving the question open.
    """
    dim = R.shape[0]
    mod = LCM(dim).double()
    pos = torch.empty(dim, dtype=torch.float64)
    pos[order] = torch.linspace(-1.0, 1.0, dim, dtype=torch.float64)
    with torch.no_grad():
        mod.a.copy_(pos)
    opt = torch.optim.Adam([mod.a, mod.w, mod.u_raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(mod.correlation(), R).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return mod.correlation().double()


def fit_moments_arm(R, order):
    est = LCMCovariance(R.shape[0], dtype=torch.float64)
    s0, s1, s2 = bands_along(R, order)
    est.fit_moments(moments=(s0, s1, s2), order=order)
    with torch.no_grad():
        d = nn.functional.softplus(est.u_raw) + est.eps
        return lcm_correlation(R.shape[0], est.a, est.w, d)


def fit_moments_signed(R, order, clip=0.999):
    """The closed-form estimator with the coupling signs restored.

    The library's estimator solves `w_{p+1}^2 = A[p,p+1] A[p+1,p+2] / A[p,p+2]` and then
    takes the positive square root, so every coupling it produces is positive. That is
    harmless on a flattened feature map, where neighbouring positions are almost always
    positively correlated, but roughly half of the off-diagonal entries of a channel
    correlation matrix are negative, and those come out as a negative `w^2` that is
    clamped to zero -- the fit collapses towards the identity.

    The sign is recoverable from the same bands at no cost. The decay factor is positive,
    so `sign(A[p,p+1]) = sign(w_p w_{p+1})`, and fixing `sign(w_0) = +1` propagates the
    signs along the chain. The magnitudes, the decay and the diagonal are unchanged, so
    this adds no parameters -- it only stops discarding information already in hand.
    """
    dim = R.shape[0]
    s0, s1, s2 = bands_along(R, order)
    tol = 1e-6 * float(s0.mean().clamp(min=1e-6))

    w2 = torch.zeros(dim, dtype=torch.float64)
    if dim > 2:
        safe = torch.where(s2.abs() > tol, s2, torch.full_like(s2, tol))
        w2[1:-1] = torch.where(s2.abs() > tol, s1[:-1] * s1[1:] / safe, torch.zeros_like(s2))
        w2[0], w2[-1] = w2[1], w2[-2]
    w2 = w2.clamp(min=0.0).minimum(clip * s0)
    w = w2.sqrt()

    signs = torch.ones(dim, dtype=torch.float64)
    step = torch.where(s1 < 0, -torch.ones_like(s1), torch.ones_like(s1))
    signs[1:] = torch.cumprod(step, dim=0)
    w = w * signs

    denom = (w[:-1] * w[1:]).abs().clamp(min=tol)
    phi = (s1.abs() / denom).clamp(min=tol, max=1.0 - 1e-12)
    a_chain = torch.cat([torch.zeros(1, dtype=torch.float64), (-phi.log()).cumsum(0)])

    a = torch.empty(dim, dtype=torch.float64)
    wv = torch.empty(dim, dtype=torch.float64)
    a[order], wv[order] = a_chain, w
    d = (s0 - w2).clamp(min=tol)
    dv = torch.empty(dim, dtype=torch.float64)
    dv[order] = d
    return lcm_correlation(dim, a, wv, dv)


def structure_stats(R, order):
    """Why a chain fit succeeds or fails here, measured rather than assumed.

    Three things decide it. How much of the correlation is negative, which the positive
    square root in the closed-form estimator cannot represent. How often that estimator's
    implied `w^2` comes out negative and is clamped to zero, which is the collapse towards
    the identity made countable. And how much of the eigenvalue mass sits in the leading
    two directions, which says whether the structure is a chain of local couplings at all
    or a few global modes that a low-rank model captures directly.
    """
    dim = R.shape[0]
    off = ~torch.eye(dim, dtype=torch.bool)
    s0, s1, s2 = bands_along(R, order)

    tol = 1e-6 * float(s0.mean().clamp(min=1e-6))
    w2 = torch.zeros(dim, dtype=torch.float64)
    if dim > 2:
        safe = torch.where(s2.abs() > tol, s2, torch.full_like(s2, tol))
        w2[1:-1] = torch.where(s2.abs() > tol, s1[:-1] * s1[1:] / safe, torch.zeros_like(s2))
        w2[0], w2[-1] = w2[1], w2[-2]

    evals = torch.linalg.eigvalsh(R).clamp(min=0.0)
    return {
        "frac_offdiag_negative": float((R[off] < 0).double().mean()),
        "frac_band1_negative": float((s1 < 0).double().mean()),
        "frac_w2_negative": float((w2 < 0).double().mean()),
        "eig_top2_share": float(evals.sort(descending=True).values[:2].sum() / evals.sum()),
    }


def fit_factor(R, rank):
    """Best rank-`rank` eigen-approximation, renormalised to unit diagonal.

    A matched-budget reference: `rank * C` loadings plus `C` diagonal entries, against the
    chain's `2C` free parameters once its diagonal is fixed by the normalisation.
    """
    evals, evecs = torch.linalg.eigh(R)
    idx = torch.argsort(evals, descending=True)[:rank]
    V = evecs[:, idx] * evals[idx].clamp(min=0).sqrt()
    low = V @ V.T
    resid = (R.diagonal() - low.diagonal()).clamp(min=1e-8)
    return to_correlation(low + torch.diag(resid))


def offdiag_error(Rhat, Rtest):
    """Relative Frobenius error on the off-diagonal only.

    Every arm here has a unit diagonal by construction, so including it would credit
    them all with the same free match and shrink every difference.
    """
    m = ~torch.eye(Rtest.shape[0], dtype=torch.bool)
    return float((Rhat[m] - Rtest[m]).norm() / Rtest[m].norm())


# ----------------------------------------------------------------------- main
def run_seed(args, seed, LCM, fit_fn):
    common.set_seed(seed)
    dev = common.DEVICE
    X, y = load_cifar(args.data_root, args.dataset, args.n_images, seed)
    X, y = X.float().to(dev), y.to(dev)
    n_cls = 100 if args.dataset == "cifar100" else 10
    model = WideCNN([int(c) for c in args.channels.split(",")], n_cls).float().to(dev)
    train(model, X, y, args.epochs, seed=seed)

    n_fit = int(len(X) * args.fit_frac)
    fit_R = channel_correlations(model, X[:n_fit], args.batch)
    test_R = channel_correlations(model, X[n_fit:], args.batch)

    rows = []
    for li, ((C, Rf), (_, Rt)) in enumerate(zip(fit_R, test_R)):
        order = seriation_order(Rf)
        natural = torch.arange(C)
        mean_abs = float(Rf[~torch.eye(C, dtype=torch.bool)].abs().mean())
        stats = structure_stats(Rf, order)

        def add(name, Rhat, secs, n_par):
            rows.append({"seed": seed, "layer": f"bn{li + 1}", "channels": C,
                         "arm": name, "offdiag_err": offdiag_error(Rhat, Rt),
                         "full_err": common.frob_rel(Rhat, Rt), "seconds": secs,
                         "params_per_dim": n_par, "mean_abs_offdiag": mean_abs, **stats})

        add("store_full_matrix", Rf, 0.0, C)          # what the method replaces
        add("identity", torch.eye(C, dtype=torch.float64), 0.0, 0)

        t = time.perf_counter()
        add("published_mse", fit_published(LCM, fit_fn, Rf, None, args.fit_epochs),
            time.perf_counter() - t, 2)
        t = time.perf_counter()
        add("published_mse_seriated", fit_published(LCM, fit_fn, Rf, order, args.fit_epochs),
            time.perf_counter() - t, 2)
        t = time.perf_counter()
        add("family_best_effort", fit_family_best_effort(LCM, Rf, order, args.best_effort_epochs),
            time.perf_counter() - t, 2)
        t = time.perf_counter()
        add("moments_natural_order", fit_moments_arm(Rf, natural), time.perf_counter() - t, 2)
        t = time.perf_counter()
        add("moments_seriated", fit_moments_arm(Rf, order), time.perf_counter() - t, 2)
        t = time.perf_counter()
        add("moments_signed_natural", fit_moments_signed(Rf, natural), time.perf_counter() - t, 2)
        t = time.perf_counter()
        add("moments_signed_seriated", fit_moments_signed(Rf, order), time.perf_counter() - t, 2)
        for r in (1, 2):
            t = time.perf_counter()
            add(f"factor_rank{r}", fit_factor(Rf, r), time.perf_counter() - t, r + 1)

        print(f"  seed {seed}  {rows[-1]['layer']}  C={C}  "
              f"mean |off-diagonal correlation| {mean_abs:.3f}  "
              f"negative off-diagonals {stats['frac_offdiag_negative']:.0%}  "
              f"negative couplings on the chain {stats['frac_band1_negative']:.0%}  "
              f"clamped w^2 {stats['frac_w2_negative']:.0%}  "
              f"top-2 eigenvalue share {stats['eig_top2_share']:.0%}", flush=True)
        for r in rows[-11:]:
            print(f"    {r['arm']:24} off-diagonal error {r['offdiag_err']:.4f}"
                  f"   {r['seconds']:7.2f}s", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref-dir", default="reference/REMIX")
    p.add_argument("--data-root", default=os.path.expanduser("~/data"))
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--channels", default="32,128,256,512",
                   help="channel widths, spanning the authors' 16-512 range")
    p.add_argument("--n-images", type=int, default=4000)
    p.add_argument("--fit-frac", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--fit-epochs", type=int, default=200, help="theirs: 200")
    p.add_argument("--best-effort-epochs", type=int, default=5000,
                   help="budget for the over-trained reference fit of the same family")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--tag", default="exp29_channel_correlation")
    args = p.parse_args()

    LCM, fit_fn = load_published_fit(args.ref_dir)
    rows = []
    for seed in range(args.seeds):
        rows += run_seed(args, seed, LCM, fit_fn)

    df = pd.DataFrame(rows)
    path = common.save_results(df, args.tag, config=vars(args))
    print(f"\n  Channel correlation at every batch-norm layer, {args.seeds} seed(s), "
          f"{args.dataset}. Fitted on one half of the images, scored on the other.")
    print(f"\n  {'arm':24}{'off-diag err':>14}{'s.d.':>8}{'seconds':>9}")
    for arm in df.arm.unique():
        g = df[df.arm == arm]
        print(f"  {arm:24}{g.offdiag_err.mean():14.4f}{g.offdiag_err.std():8.4f}"
              f"{g.seconds.mean():9.2f}")
    print("\n  Off-diagonal error is relative Frobenius distance to the held-out empirical"
          "\n  correlation, so 0 is perfect and 1 is no better than predicting zero"
          "\n  correlation everywhere (lower is better). 'store_full_matrix' keeps the whole"
          "\n  fitting-split matrix, the O(C^2) memory this family exists to avoid, and is"
          "\n  the floor a structured fit is trying to approach.")
    print_paired(df)
    print(f"\nsaved {path}")


def print_paired(df):
    """The contrasts that actually answer the question, paired and scaled.

    The spread of each arm's mean above is dominated by differences between layers, which
    cancel in a comparison: every arm in a cell was fitted to the same matrix, so the
    contrasts belong paired by (seed, layer). Each one is also reported as a share of the
    distance from their published rule to the stored-matrix floor, because a difference can
    be statistically clear and still buy almost none of the available headroom -- which is
    the question when deciding whether the fitting budget or the family is the constraint.
    """
    wide = df.pivot_table(index=["seed", "layer"], columns="arm", values="offdiag_err")
    headroom = float(wide["published_mse"].mean() - wide["store_full_matrix"].mean())
    contrasts = [
        ("25x fit budget, ordering held fixed", "family_best_effort", "published_mse_seriated"),
        ("seriated ordering, at their budget", "published_mse_seriated", "published_mse"),
        ("rank-2 factor vs the over-trained chain", "factor_rank2", "family_best_effort"),
        ("rank-2 factor vs their published rule", "factor_rank2", "published_mse"),
    ]
    print(f"\n  Paired contrasts, n={len(wide)} cells (seed x layer). Negative favours the"
          f"\n  first arm named. 'Headroom' is the {headroom:.4f} between their published rule"
          f"\n  and keeping the whole matrix.\n")
    print(f"  {'contrast':42}{'mean':>9}{'s.e.':>8}{'s.e. away':>11}"
          f"{'same sign':>11}{'headroom':>10}")
    for label, a, b in contrasts:
        d = wide[a] - wide[b]
        se = float(d.std() / len(d) ** 0.5)
        mean = float(d.mean())
        same = int((d < 0).sum() if mean < 0 else (d > 0).sum())
        print(f"  {label:42}{mean:+9.4f}{se:8.4f}{abs(mean) / max(se, 1e-12):11.1f}"
              f"{f'{same}/{len(d)}':>11}{abs(mean) / headroom:9.1%}")


if __name__ == "__main__":
    main()
