"""Numerical checks for the note "KFAC x LCM: Structured Kronecker Factors for Laplace
Posteriors". Every figure the note reports is produced here.

  part 1  O(m) LCM primitives in precision form (tridiagonal K^-1, Woodbury solve,
          determinant-lemma log-det, perturb-and-solve sampling, closure)
  part 2  the one-factor eigen reduction for A (x) G + delta*I, and its 3-way form
  part 3  factored damping: Loewner bound, overconfidence, scale-split invariance
  part 4  construction cost: KFAC form+eigendecompose vs a matrix-free LCM fit
  part 5  are real KFAC factors chain-like or low-rank?

Run:  PYTHONPATH=experiments:Laplace python experiments/kron_lcm_checks.py
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import cholesky_banded, solve_banded

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- O(m) LCM ops
def _prep(t, w, d):
    idx = np.argsort(t); ts, ws, ds = t[idx], w[idx], d[idx]
    m = len(t)
    phi = np.exp(-np.diff(ts)); q = 1.0 - phi**2
    Pd = np.empty(m); Pe = -phi / q
    Pd[0] = 1.0 + phi[0]**2 / q[0]
    Pd[1:-1] = 1.0 / q[:-1] + phi[1:]**2 / q[1:]
    Pd[-1] = 1.0 / q[-1]
    return idx, ws, ds, q, Pd + ws**2 / ds, Pe


def lcm_logdet(t, w, d):
    """log det M via the matrix determinant lemma: log|D| + log|K| + log|T|."""
    _, ws, ds, q, Td, Pe = _prep(t, w, d)
    L = cholesky_banded(np.vstack([np.r_[0.0, Pe], Td]))
    return np.log(ds).sum() + np.log(q).sum() + 2 * np.log(L[-1]).sum()


def lcm_solve(t, w, d, b):
    """M^{-1} b via Woodbury, with T = K^{-1} + W D^{-1} W tridiagonal."""
    idx, ws, ds, q, Td, Pe = _prep(t, w, d)
    bs = b[idx]
    y = solve_banded((1, 1), np.vstack([np.r_[0.0, Pe], Td, np.r_[Pe, 0.0]]), (ws / ds) * bs)
    xs = bs / ds - (ws / ds) * y
    x = np.empty_like(xs); x[idx] = xs
    return x


def lcm_sample_cov(t, w, d, rng):
    idx = np.argsort(t); ts = t[idx]; m = len(t)
    phi = np.exp(-np.diff(ts)); z = np.empty(m); xi = rng.standard_normal(m)
    z[0] = xi[0]
    for k in range(1, m):
        z[k] = phi[k - 1] * z[k - 1] + math.sqrt(1 - phi[k - 1]**2) * xi[k]
    x = np.empty(m); x[idx] = w[idx] * z + np.sqrt(d[idx]) * rng.standard_normal(m)
    return x


def lcm_sample_prec(t, w, d, rng):
    """x ~ N(0, M^{-1}) by perturb-and-solve."""
    return lcm_solve(t, w, d, lcm_sample_cov(t, w, d, rng))


def lcm_dense(t, w, d):
    return np.diag(d) + np.outer(w, w) * np.exp(-np.abs(t[:, None] - t[None, :]))


# --------------------------------------------------------------------- part 1
def part1():
    import sys; sys.path.insert(0, "Laplace")
    from laplace.lcm import logdet_parallel
    print("== part 1: O(m) LCM primitives ==")
    rng = np.random.default_rng(0); m = 400
    t = rng.permutation(np.cumsum(rng.uniform(0.02, 0.3, m)))
    w = rng.uniform(0.3, 2.0, m); d = rng.uniform(0.05, 1.0, m)
    M = lcm_dense(t, w, d); b = rng.standard_normal(m)
    idx = np.argsort(t); K = np.exp(-np.abs(t[idx][:, None] - t[idx][None, :]))
    Ki = np.linalg.inv(K)
    print(f"  K^-1 beyond first off-diagonal (rel.): {np.abs(np.triu(Ki, 2)).max()/np.abs(Ki).max():.1e}")
    lib = float(logdet_parallel(*[torch.tensor(v) for v in (t, w, d)]))
    print(f"  log det: lemma {lcm_logdet(t, w, d):.10f}  dense {np.linalg.slogdet(M)[1]:.10f}  library {lib:.10f}")
    x = lcm_solve(t, w, d, b)
    print(f"  solve residual {np.linalg.norm(M@x-b)/np.linalg.norm(b):.1e}")
    S = np.array([lcm_sample_prec(t, w, d, rng) for _ in range(40000)])
    Mi = np.linalg.inv(M)
    print(f"  perturb-and-solve cov vs M^-1: {np.linalg.norm(S.T@S/len(S)-Mi)/np.linalg.norm(Mi):.3f}")
    s, c = 3.7, rng.uniform(0, 1, m)
    print(f"  closure: {np.abs(s*M+np.diag(c)-lcm_dense(t, math.sqrt(s)*w, s*d+c)).max():.1e}")
    for mm in (10_000, 100_000, 1_000_000):
        tt = np.cumsum(rng.uniform(.02, .3, mm)); ww = rng.uniform(.3, 2, mm); dd = rng.uniform(.05, 1, mm)
        t0 = time.perf_counter(); lcm_logdet(tt, ww, dd); lcm_solve(tt, ww, dd, rng.standard_normal(mm))
        print(f"  m={mm:>9,}: log-det + solve {1e3*(time.perf_counter()-t0):7.1f} ms")


# --------------------------------------------------------------------- part 2
def part2():
    print("== part 2: one-factor eigen reduction ==")
    rng = np.random.default_rng(1); i, o, delta = 300, 7, 0.3
    t = rng.permutation(np.cumsum(rng.uniform(.02, .3, i))); w = rng.uniform(.3, 2, i); d = rng.uniform(.05, 1, i)
    A = lcm_dense(t, w, d)
    B = rng.standard_normal((o, o)); G = B@B.T/o + np.diag(np.r_[np.zeros(2), rng.uniform(0, .1, o-2)])
    gam, Q = np.linalg.eigh(G); gam = np.clip(gam, 0, None)
    Lam = np.kron(A, G) + delta*np.eye(i*o)
    vec = lambda X: X.flatten(order="F")
    V = rng.standard_normal((o, i))
    ld = sum(lcm_logdet(t, math.sqrt(g)*w, g*d+delta) for g in gam)
    print(f"  log det: reduction {ld:.10f}  dense {np.linalg.slogdet(Lam)[1]:.10f}")
    Xt = np.stack([lcm_solve(t, math.sqrt(g)*w, g*d+delta, (Q.T@V)[j]) for j, g in enumerate(gam)])
    print(f"  solve residual {np.linalg.norm(Lam@vec(Q@Xt)-vec(V))/np.linalg.norm(vec(V)):.1e}")
    phi = rng.standard_normal(i); J = np.kron(phi[None, :], np.eye(o))
    Pd = J @ np.linalg.solve(Lam, J.T)
    Pr = Q @ np.diag([phi @ lcm_solve(t, math.sqrt(g)*w, g*d+delta, phi) for g in gam]) @ Q.T
    print(f"  last-layer predictive covariance error {np.linalg.norm(Pr-Pd)/np.linalg.norm(Pd):.1e}")
    C_, T_ = 5, 60
    tT = np.cumsum(rng.uniform(.05, .3, T_)); wT = rng.uniform(.3, 2, T_); dT = rng.uniform(.05, 1, T_)
    Bc = rng.standard_normal((C_, C_)); AC = Bc@Bc.T/C_; kap = np.clip(np.linalg.eigvalsh(AC), 0, None)
    Lam3 = np.kron(np.kron(AC, lcm_dense(tT, wT, dT)), G) + delta*np.eye(C_*T_*o)
    ld3 = sum(lcm_logdet(tT, math.sqrt(k*g)*wT, k*g*dT+delta) for k in kap for g in gam)
    print(f"  3-way log det: reduction {ld3:.10f}  dense {np.linalg.slogdet(Lam3)[1]:.10f}")


# --------------------------------------------------------------------- part 3
def _softmax_head_factors(seed=0):
    torch.manual_seed(seed)
    N, n_in, H, C = 3000, 20, 60, 6
    X = torch.randn(N, n_in); y = (X @ torch.randn(n_in, C) + .5*torch.randn(N, C)).argmax(1)
    net = nn.Sequential(nn.Linear(n_in, H), nn.Tanh(), nn.Linear(H, C))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad(); F.cross_entropy(net(X), y).backward(); opt.step()
    with torch.no_grad():
        phi = torch.cat([torch.tanh(net[0](X)), torch.ones(N, 1)], 1).double()
        p = torch.softmax(net(X), 1).double()
    A = (phi.T @ phi / N).numpy()
    G = (torch.diag_embed(p) - torch.einsum("nc,nd->ncd", p, p)).mean(0).numpy()
    return A, G, N


def part3():
    print("== part 3: factored damping ==")
    A, G, N = _softmax_head_factors()
    C = len(G); delta = 1.0 / N          # one data point's worth of prior curvature
    print(f"  softmax output factor: min eigenvalue {np.linalg.eigvalsh(G).min():.1e}")
    ex = lambda a, g: np.outer(a, g) + delta
    for label, balanced in (("pi = 1", False), ("pi trace-balanced", True)):
        pi = math.sqrt((np.trace(A)/len(A)) / (np.trace(G)/C)) if balanced else 1.0
        a, g = np.linalg.eigvalsh(A), np.clip(np.linalg.eigvalsh(G), 0, None); sd = math.sqrt(delta)
        r = np.outer(a + pi*sd, g + sd/pi) / ex(a, g)
        shifts = []
        for AA, GG in ((A, G), (25*A, G/25)):
            p2 = math.sqrt((np.trace(AA)/len(AA)) / (np.trace(GG)/C)) if balanced else 1.0
            aa, gg = np.linalg.eigvalsh(AA), np.clip(np.linalg.eigvalsh(GG), 0, None)
            shifts.append(np.log(np.outer(aa + p2*sd, gg + sd/p2)).sum())
        print(f"  {label:18} inflation {r.min():.1f}x..{r.max():.0f}x  evidence gap {0.5*np.log(r).sum():.0f} nats"
              f"  split (A,G)->(25A,G/25) moves log det by {shifts[1]-shifts[0]:+.4f}")
    pi = math.sqrt((np.trace(A)/len(A)) / (np.trace(G)/C)); sd = math.sqrt(delta)
    Lx = np.kron(A, G) + delta*np.eye(len(A)*C)
    Lf = np.kron(A + pi*sd*np.eye(len(A)), G + sd/pi*np.eye(C))
    print(f"  min eig(Lambda_fd - Lambda_exact) = {np.linalg.eigvalsh(Lf-Lx).min():.1e}")
    rat = np.diag(np.linalg.inv(Lf)) / np.diag(np.linalg.inv(Lx))
    print(f"  marginal variance ratio fd/exact: {rat.min():.3f}..{rat.max():.3f}")


# --------------------------------------------------------------------- part 4
def _ar1(N, d, rho=0.95):
    Z = torch.randn(N, d); X = torch.empty(N, d); X[:, 0] = Z[:, 0]
    for j in range(1, d):
        X[:, j] = rho*X[:, j-1] + math.sqrt(1-rho**2)*Z[:, j]
    return X


def part4():
    import sys; sys.path.insert(0, "Laplace")
    from laplace.lcm import LCMCovariance
    print("== part 4: construction cost (N=2000, S=100 steps, float64, CPU) ==")
    torch.set_default_dtype(torch.float64); torch.manual_seed(0)
    N, S, rows = 2000, 100, []
    for i in (512, 1024, 2048, 4096):
        Phi = _ar1(N, i)
        t0 = time.perf_counter(); A = Phi.T@Phi/N; torch.linalg.eigh(A); tk = time.perf_counter()-t0
        m = LCMCovariance(i, dtype=torch.float64)
        with torch.no_grad():
            m.a.copy_(torch.arange(i)*-math.log(0.95))
        t0 = time.perf_counter(); m.fit_frobenius(Phi, n_steps=S, lr=5e-2); tl = time.perf_counter()-t0
        with torch.no_grad():
            err = float((m.dense()-A).norm()/A.norm())
        rows.append((i, tk, tl))
        print(f"  i={i:5}: KFAC {tk:6.2f}s   LCM fit {tl:6.2f}s   fit error {err:.3f}", flush=True)
    r = np.array(rows)
    print(f"  scaling: KFAC ~ i^{np.polyfit(np.log(r[:,0]), np.log(r[:,1]), 1)[0]:.2f}, "
          f"LCM ~ i^{np.polyfit(np.log(r[:,0]), np.log(r[:,2]), 1)[0]:.2f}")


# --------------------------------------------------------------------- part 5
def part5():
    import sys; sys.path.insert(0, "experiments"); sys.path.insert(0, "Laplace")
    import common as C
    import exp13_sameclass as X13
    print("== part 5: are real KFAC factors chain-like or low-rank? ==")
    torch.set_default_dtype(torch.float64); torch.manual_seed(0)

    def fit(kind, A, a0=None, k=None):
        n = A.shape[0]
        ctor = {"diag": lambda T: X13.Diag(n, T), "lr": lambda T: X13.DiagLowRank(n, k, T)}.get(kind)
        return X13.fit_any("lcm" if kind == "lcm" else kind, ctor, n, A, a_init=a0, n_steps=600)[1]

    def row(label, A, order=None):
        ev = torch.linalg.eigvalsh(A).clamp(min=0); er = float(ev.sum()**2/(ev**2).sum())
        a0 = order if order is not None else C.seriation_coord(A)*0.1
        r = dict(diag=fit("diag", A), lcm=fit("lcm", A, a0), lr2=fit("lr", A, k=2), lr8=fit("lr", A, k=8))
        print(f"  {label:46} eff.rank {er:5.1f} | diag {r['diag']:.3f} LCM {r['lcm']:.3f} "
              f"lowrank2 {r['lr2']:.3f} lowrank8 {r['lr8']:.3f}", flush=True)

    def cov_b(Phi):
        P = torch.cat([Phi, torch.ones(len(Phi), 1)], 1); return P.T@P/len(P)

    N, d = 4000, 100; Xs = _ar1(N, d)
    y = torch.tanh(Xs @ torch.sin(torch.linspace(0, 6, d))/5) + .1*torch.randn(N)
    net = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(600):
        opt.zero_grad(); F.mse_loss(net(Xs).squeeze(), y).backward(); opt.step()
    with torch.no_grad():
        h1 = torch.tanh(net[0](Xs)); h2 = torch.tanh(net[2](h1))
    row("MLP layer 1, ordered inputs (known order)", cov_b(Xs),
        order=torch.cat([torch.arange(d)*0.05, torch.tensor([d*0.05 + 50.0])]))
    row("MLP layer 2, hidden (seriated)", cov_b(h1))
    row("MLP layer 3, hidden (seriated)", cov_b(h2))

    N, T = 3000, 100; Sx = _ar1(N, T).unsqueeze(1)
    yt = torch.tanh(Sx.squeeze(1) @ torch.sin(torch.linspace(0, 6, T))/5) + .1*torch.randn(N)
    cnn = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.Tanh(), nn.Conv1d(16, 16, 7, padding=3),
                        nn.Tanh(), nn.Flatten(), nn.Linear(16*T, 1))
    opt = torch.optim.Adam(cnn.parameters(), lr=3e-3)
    for _ in range(400):
        opt.zero_grad(); F.mse_loss(cnn(Sx).squeeze(), yt).backward(); opt.step()
    with torch.no_grad():
        fmap = cnn[:4](Sx)
    Cc = 16; Phi = fmap.reshape(N, Cc*T); A = Phi.T@Phi/N
    R = A.reshape(Cc, T, Cc, T).permute(0, 2, 1, 3).reshape(Cc*Cc, T*T)
    sv = torch.linalg.svdvals(R)
    print(f"  1-D CNN feature map: channel (x) position separability error {float(torch.sqrt((sv[1:]**2).sum())/sv.norm()):.3f}")
    _, s, Vh = torch.linalg.svd(R, full_matrices=False)
    AT = (Vh[0]*math.sqrt(s[0])).reshape(T, T); AT = AT*torch.sign(AT.trace()); AT = 0.5*(AT + AT.T)
    row("1-D CNN position factor (known order)", AT, order=torch.arange(T)*0.1)

    from sklearn.datasets import fetch_openml
    df = fetch_openml("tecator", version=1, as_frame=True, parser="auto").frame
    Xt = torch.tensor(df[[f"absorbance_{i}" for i in range(1, 101)]].to_numpy(), dtype=torch.float64)
    Xt = (Xt - Xt.mean(0))/Xt.std(0)
    row("Tecator NIR input covariance (known order)", Xt.T@Xt/len(Xt), order=torch.arange(100)*0.1)


if __name__ == "__main__":
    part1(); part2(); part3(); part4(); part5()
