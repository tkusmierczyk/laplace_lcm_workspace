"""Exp 30 -- how much does the closed-form fit depend on being given the right order?

The closed-form moment fit identifies the chain's parameters from three bands of the
empirical second moment, which are only the chain's neighbours if the coordinates are
already in chain order. So `order` is an input, not something the estimator discovers --
unlike the published method, which learns the coordinate and sorts by it.

That raises two questions this experiment answers on data drawn from a known chain, where
the true order is available as a reference no real dataset provides:

  1. How badly does the fit degrade when the order is wrong? Gracefully, or not at all?
  2. Can spectral seriation recover the order well enough to stand in for it, at the
     dimensions this method targets?

Four arms per setting, scored by held-out negative log-likelihood (lower is better):
the true order, a seriated order recovered from the data alone, a random permutation, and
an independent-coordinates baseline that marks the "no structure found" floor.

Swept over the adjacent correlation of the generating chain (how strong the local coupling
is) and over dimension, because seriation on a long path graph with sample noise is a
harder problem than on a short one.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Laplace"))
from laplace.lcm import LCMCovariance, seriation_order  # noqa: E402


def make_chain(dim, gap, seed, dtype=torch.float64):
    """A ground-truth LCM whose coordinates are evenly spaced in the natural order."""
    torch.manual_seed(seed)
    m = LCMCovariance(dim, dtype=dtype)
    with torch.no_grad():
        m.a.copy_(torch.arange(dim, dtype=dtype) * gap)
        m.w.copy_(torch.rand(dim, dtype=dtype) * 0.5 + 0.8)
    return m


@torch.no_grad()
def score(X, held, order, dim):
    est = LCMCovariance(dim, dtype=torch.float64)
    est.fit_moments(X, order=order)
    return float(est.nll(held))


def run_setting(dim, gap, n, seed):
    true = make_chain(dim, gap, seed)
    X = true.sample(n, generator=torch.Generator().manual_seed(seed + 1)).double()
    held = true.sample(n, generator=torch.Generator().manual_seed(seed + 2)).double()

    C = torch.corrcoef(X.T)
    adj = float(C.diagonal(1).abs().mean())
    ser = seriation_order(C)
    natural = torch.arange(dim)
    # |correlation| with the true index, up to reflection: 1.0 is perfect recovery,
    # 0 is chance. Monotonicity is what the chain needs, not the exact permutation.
    agree = abs(
        float(torch.corrcoef(torch.stack([ser.double(), natural.double()]))[0, 1])
    )
    with torch.no_grad():
        diag = float(
            -torch.distributions.Normal(0.0, X.std(0)).log_prob(held).sum(1).mean()
        )
    return {
        "dim": dim,
        "gap": gap,
        "seed": seed,
        "adj_corr": adj,
        "seriation_agreement": agree,
        "nll_true_order": score(X, held, None, dim),
        "nll_seriated": score(X, held, ser, dim),
        "nll_random": score(X, held, torch.randperm(dim), dim),
        "nll_diagonal": diag,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", default="256,1024,2048")
    p.add_argument("--gaps", default="0.05,0.223,0.8",
                   help="coordinate spacing; smaller means stronger adjacent correlation")
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--tag", default="exp30_ordering_dependence")
    args = p.parse_args()

    rows = []
    for dim in (int(d) for d in args.dims.split(",")):
        for gap in (float(g) for g in args.gaps.split(",")):
            for seed in range(args.seeds):
                rows.append(run_setting(dim, gap, args.n_samples, seed))
                r = rows[-1]
                print(f"  dim {dim:>5}  adjacent |corr| {r['adj_corr']:.3f}  "
                      f"order recovery {r['seriation_agreement']:.3f}  "
                      f"NLL true {r['nll_true_order']:.1f} / seriated "
                      f"{r['nll_seriated']:.1f} / random {r['nll_random']:.1f} / "
                      f"diagonal {r['nll_diagonal']:.1f}", flush=True)

    df = pd.DataFrame(rows)
    path = common.save_results(df, args.tag, config=vars(args))

    print("\n  Closed-form fit on data drawn from a known chain, "
          f"{args.seeds} seed(s). Held-out negative log-likelihood, lower is better.")
    print(f"\n  {'dim':>6}{'adj |corr|':>12}{'order rec.':>12}"
          f"{'true':>10}{'seriated':>10}{'random':>10}{'diagonal':>10}")
    g = df.groupby(["dim", "gap"], sort=True).mean(numeric_only=True).reset_index()
    for _, r in g.iterrows():
        print(f"  {int(r['dim']):>6}{r['adj_corr']:>12.3f}{r['seriation_agreement']:>12.3f}"
              f"{r['nll_true_order']:>10.1f}{r['nll_seriated']:>10.1f}"
              f"{r['nll_random']:>10.1f}{r['nll_diagonal']:>10.1f}")
    print("\n  'order rec.' is |correlation| between the seriated order and the true one,"
          "\n  so 1.0 is perfect recovery and 0 is chance. The diagonal column is the"
          "\n  no-structure floor: a fit that reaches it has found nothing.")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
