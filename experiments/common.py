"""Shared harness for evaluating the LCM Laplace approximation against baselines.

Provides: reproducible seeding, data utilities, MAP training, a uniform
Laplace-fitting + marginal-likelihood tuning protocol for every structure
(MAP / diag / kron / lowrank / full / lcm), and the evaluation metrics used
across experiments (regression test-LL & RMSE, classification NLL/ECE/OOD-AUROC,
and posterior-fidelity KL / Frobenius to the exact full Laplace).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import laplace
from laplace import Laplace

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Human-readable names and (subset, structure) for the Laplace factory.
METHODS = {
    "MAP": None,
    "Diag": ("diag",),
    "KFAC": ("kron",),
    "Full": ("full",),
    "LCM": ("lcm",),
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ provenance
def _git(path: str) -> dict | None:
    """Short HEAD and dirty flag of the repository containing ``path``."""
    try:
        run = lambda *a: subprocess.run(  # noqa: E731
            ["git", "-C", path, *a], capture_output=True, text=True, timeout=15
        )
        head = run("rev-parse", "--short", "HEAD")
        if head.returncode:
            return None
        return {
            "commit": head.stdout.strip(),
            "dirty": bool(run("status", "--porcelain").stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(config: dict | None = None) -> dict:
    """Everything needed to re-run this: both commits, versions, host, config.

    The experiments and the laplace-torch fork are separate repositories, so both
    hashes are recorded -- a result is only reproducible given the pair.
    """
    lib = os.path.dirname(os.path.dirname(os.path.abspath(laplace.__file__)))
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiments_git": _git(HERE),
        "library_git": _git(lib),
        "library_path": lib,
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads": torch.get_num_threads(),
        "dtype": str(torch.get_default_dtype()),
        "config": dict(config or {}),
    }


def save_results(df, name: str, config: dict | None = None, results_dir: str | None = None) -> str:
    """Write ``<name>.csv`` plus ``<name>.meta.json`` holding :func:`provenance`."""
    d = results_dir or RESULTS
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.csv")
    df.to_csv(path, index=False)
    with open(os.path.join(d, f"{name}.meta.json"), "w") as f:
        json.dump(provenance(config), f, indent=2, default=str)
    return path


# ------------------------------------------------------------------ data helpers
def standardize(train: np.ndarray, *others: np.ndarray):
    mu, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True) + 1e-8
    return [( (a - mu) / sd ) for a in (train, *others)], (mu, sd)


def loader(X, y, batch_size=128, shuffle=False):
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ------------------------------------------------------------------ MAP training
def train_map(model, train_loader, epochs, lr, weight_decay, likelihood):
    model = model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss() if likelihood == "regression" else nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    return model.eval()


# ------------------------------------------- uniform Laplace fit + marglik tuning
def fit_laplace(model, train_loader, structure, likelihood, subset="all",
                tune_steps=200, tune_lr=1e-1, tune_sigma=True, common_prior=None,
                lcm_steps=600, lcm_lr=3e-2, backend=None):
    """Fit one Laplace variant and (unless `common_prior` is given) tune its
    prior precision -- and, for regression, its observation noise -- by maximising
    the Laplace marginal likelihood. Returns (la, fit_seconds)."""
    kw = {}
    if structure == "lcm":
        kw.update(lcm_steps=lcm_steps, lcm_lr=lcm_lr)
    if backend is not None:
        kw["backend"] = backend
    t0 = time.perf_counter()
    la = Laplace(model, likelihood, subset_of_weights=subset,
                 hessian_structure=structure, **kw)
    la.fit(train_loader)
    fit_s = time.perf_counter() - t0

    if common_prior is not None:
        la.prior_precision = common_prior
        return la, fit_s

    # Joint marglik optimisation of log prior-precision (+ log sigma for regression),
    # identical across all structures for a fair comparison.
    log_prior = torch.zeros(1, device=DEVICE, requires_grad=True)
    params = [log_prior]
    if likelihood == "regression" and tune_sigma:
        log_sigma = torch.zeros(1, device=DEVICE, requires_grad=True)
        params.append(log_sigma)
    else:
        log_sigma = None
    opt = torch.optim.Adam(params, lr=tune_lr)
    for _ in range(tune_steps):
        opt.zero_grad()
        pp = log_prior.exp()
        sn = log_sigma.exp() if log_sigma is not None else None
        nmll = -la.log_marginal_likelihood(prior_precision=pp, sigma_noise=sn)
        nmll.backward()
        opt.step()
    with torch.no_grad():
        la.prior_precision = log_prior.exp().detach()
        if log_sigma is not None:
            la.sigma_noise = log_sigma.exp().detach()
    return la, fit_s


# ------------------------------------------------------------ regression metrics
@torch.no_grad()
def regression_predictive(la, X):
    """GLM predictive mean and total variance (functional + observation noise)."""
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=DEVICE)
    f_mu, f_var = la(X, pred_type="glm")
    f_mu = f_mu.squeeze(-1)
    f_var = f_var.reshape(f_mu.shape[0], -1)[:, 0]
    total_var = f_var + la.sigma_noise.squeeze() ** 2
    return f_mu.cpu().numpy(), total_var.cpu().numpy()


def map_regression_predictive(model, X, sigma):
    with torch.no_grad():
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=DEVICE)
        mu = model(X).squeeze(-1).cpu().numpy()
    return mu, np.full_like(mu, float(sigma) ** 2)


def gaussian_test_ll(mu, var, y, y_sd):
    """Mean test log-likelihood in the *original* (unstandardised) target scale."""
    y = np.asarray(y).reshape(-1)
    # standardised-target LL, then correct for the y-standardisation Jacobian.
    ll = -0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)
    return float(ll.mean() - np.log(y_sd))


def rmse(mu, y, y_sd):
    y = np.asarray(y).reshape(-1)
    return float(np.sqrt(np.mean((mu - y) ** 2)) * y_sd)


# -------------------------------------------------------- classification metrics
@torch.no_grad()
def classification_probs(la, X, link="probit", batch=512):
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    outs = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(DEVICE)
        outs.append(la(xb, pred_type="glm", link_approx=link).cpu())
    return torch.cat(outs).numpy()


@torch.no_grad()
def map_probs(model, X, batch=512):
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    outs = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(DEVICE)
        outs.append(torch.softmax(model(xb), -1).cpu())
    return torch.cat(outs).numpy()


def accuracy(probs, y):
    return float((probs.argmax(1) == np.asarray(y)).mean())


def nll_classification(probs, y):
    y = np.asarray(y)
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


def ece(probs, y, n_bins=15):
    y = np.asarray(y)
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def ood_auroc(probs_in, probs_out):
    """AUROC separating in- vs out-of-distribution by predictive confidence
    (max prob). Higher = better OOD detection."""
    from sklearn.metrics import roc_auc_score
    s_in, s_out = probs_in.max(1), probs_out.max(1)
    # in-dist should be MORE confident -> use negative confidence as OOD score
    scores = np.concatenate([-s_in, -s_out])
    labels = np.concatenate([np.zeros(len(s_in)), np.ones(len(s_out))])
    return float(roc_auc_score(labels, scores))


# ------------------------------------------------------------- posterior fidelity
@torch.no_grad()
def dense_posterior_cov(la):
    """Dense posterior covariance for *any* structure (diag/kron/full/lcm),
    obtained uniformly as ``J Cov J^T`` with ``J = I`` (identity Jacobians)."""
    P = la.n_params
    eye = torch.eye(P, device=la._device, dtype=la._dtype).reshape(P, 1, P)
    return la.functional_covariance(eye).double().cpu()


def kl_gaussian(cov_a, cov_f):
    """KL( N(m, cov_a) || N(m, cov_f) ) for shared mean m."""
    cov_a, cov_f = cov_a.double(), cov_f.double()
    d = cov_f.shape[0]
    sign, logdet_f = torch.linalg.slogdet(cov_f)
    _, logdet_a = torch.linalg.slogdet(cov_a)
    sol = torch.linalg.solve(cov_f, cov_a)
    return float(0.5 * (torch.trace(sol) - d + logdet_f - logdet_a))


def frob_rel(A, B):
    A, B = A.double(), B.double()
    return float((A - B).norm() / B.norm())


# ---------------------------------------------------------------- GLM helpers
# Experiments 7-11 and 13-16 study Bayesian linear models. A GLM is just a
# one-layer network, so `Laplace(nn.Linear(d, 1, bias=False), "regression", ...)`
# handles it end to end -- there is no reason to hand-roll the posterior, and the
# only reason those experiments originally did is that the first synthetic probe
# was written that way and the rest copied it.
#
# These helpers exist only to (a) supply the exact MAP, which for a Gaussian GLM is
# the closed-form ridge solution rather than something to reach by SGD, and (b)
# expose the JOINT predictive, which the regime-A helpers above did not need.
# Everything else routes straight into the library.


def glm_map_weights(X, y, prior, sigma):
    """Ridge MAP — the exact Laplace expansion point for a Gaussian GLM."""
    d = X.shape[1]
    GGN = X.T @ X / sigma**2
    return torch.linalg.solve(GGN + prior * torch.eye(d, dtype=X.dtype), X.T @ y / sigma**2)


def seriation_coord(GGN):
    """Spectral (Fiedler) seriation of the |correlation| graph — an ordering
    estimate when the 1-D order is unknown. Pass as `lcm_init_a`, then refine."""
    dg = torch.sqrt(GGN.diagonal().clamp(min=1e-12))
    Cr = (GGN / dg[:, None] / dg[None, :]).abs()
    lap = torch.diag(Cr.sum(1)) - Cr
    fied = torch.linalg.eigh(lap).eigenvectors[:, 1]
    return torch.argsort(torch.argsort(fied)).to(GGN.dtype)


def fit_glm(X, y, structure, prior, sigma, model=None, batch_size=256, w_map=None, **kw):
    """Fit a `laplace-torch` posterior for a linear model.

    Pass `model` to use an existing network (the usual case: train it, then hand it
    over). If omitted, a `Linear(d, 1, bias=False)` is built and set to the exact
    ridge MAP — the right expansion point for a GLM, and the reason these
    experiments do not train anything.

    `structure` is any `hessian_structure` the library accepts ("diag", "full",
    "kron", "lcm"); extra kwargs (`lcm_steps`, `lcm_init_a`, `lcm_learn_a`, ...)
    pass through untouched.
    """
    if model is None:
        d = X.shape[1]
        model = nn.Linear(d, 1, bias=False).to(X.dtype)
        with torch.no_grad():
            w = glm_map_weights(X, y, prior, sigma) if w_map is None else w_map
            model.weight.copy_(w.unsqueeze(0))
    ds = TensorDataset(X, y.unsqueeze(-1) if y.ndim == 1 else y)
    la = Laplace(model, "regression", subset_of_weights="all",
                 hessian_structure=structure, **kw)
    la.fit(DataLoader(ds, batch_size=batch_size))
    la.prior_precision = torch.as_tensor(prior, dtype=X.dtype)
    la.sigma_noise = torch.as_tensor(sigma, dtype=X.dtype)
    return la


@torch.no_grad()
def glm_marginal(la, Xt, sigma):
    """Per-point predictive mean and total variance (library GLM predictive)."""
    f_mu, f_var = la(Xt, pred_type="glm")
    f_mu = f_mu.squeeze(-1)
    return f_mu, f_var.reshape(f_mu.shape[0], -1)[:, 0] + sigma**2


@torch.no_grad()
def glm_joint(la, Xt, sigma):
    """Joint predictive mean and covariance over a batch, via the library's own
    `functional_covariance` (`Js P^-1 Js^T`) — the quantity meant for joint
    prediction."""
    f_mu, _ = la(Xt, pred_type="glm")
    f_mu = f_mu.squeeze(-1)
    # NB: `backend.jacobians` differentiates w.r.t. the whole model and returns all
    # zeros for a last-layer Laplace; `last_layer_jacobians` is the right call there.
    Js = (la.backend.last_layer_jacobians(Xt) if la._key[0] == "last_layer"
          else la.backend.jacobians(Xt))[0]
    Cov = la.functional_covariance(Js)
    eye = torch.eye(Cov.shape[0], dtype=Cov.dtype, device=Cov.device)
    return f_mu, Cov + sigma**2 * eye
