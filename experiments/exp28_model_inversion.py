"""Exp 28 — does the covariance model change the *images* the method invents?

WHY THIS IS THE RIGHT TEST. Everything this project measured so far asked the structured
covariance to predict. The method it comes from (Krukowski et al., arXiv:2605.11804) does
not predict with it: it **samples** from it, using the covariance's log-likelihood as a loss
term while optimising images that a frozen network believes in, so those images can replace
a stored dataset. Exp 27 showed the chain is the only cheap family whose samples carry real
spatial correlation. This asks whether that translates into better invented images.

It is also the one place where the authors' own ablation is exactly the control this project
has spent weeks constructing by hand. With their weight `W_LCM = 0` the loop *still* matches
each layer's feature mean and variance (their `synthetic_samples.py`, the two lines added to
`loss_bn` inside the layer loop). So their own baseline is a **diagonal** feature match, and
the covariance term's entire contribution is off-diagonal structure on top of it.

HYPOTHESIS. The off-diagonal structure improves the invented images, and fitting the same
covariance family by our closed-form moment estimator improves them further, because Exp 25
showed that estimator fits the family far better than the Frobenius rule the authors use
(3-25x better held-out feature likelihood) and Exp 27 showed the Frobenius fit flattens the
correlation profile that the term is supposed to impose.

THE CLAIM A PAPER WOULD MAKE. "Fitting the Laplace chain by moments rather than by a
Frobenius objective produces measurably more realistic inverted data at no extra cost."

CHEAPEST FALSIFIER, AND WHAT WOULD MAKE US DROP IT. Run the authors' loop unchanged except
for the covariance term, and score the resulting images on a network that was *not* inverted.
**If the three arms are indistinguishable there, the covariance term does not matter for the
method's own application**, and Exp 27's sampling result has no downstream consequence --
which would be the most important negative this project could report.

Arms (identical loop, identical weights, identical schedule -- only the fit differs):
  none       `W_LCM = 0`: the authors' own ablation, a diagonal mean/variance match
  frobenius  their fit, `utils.frobenius_dist_reduced` under Adam, as in their script
  moments    the same family fitted by `LCMCovariance.fit_moments` (ours), one pass, no
             optimiser, then copied into *their* module so the loop is untouched

Scored three ways, none of them circular:
  confidence       mean probability the inverted network assigns to the target class
  profile error    gap between the correlation-vs-distance curve of the generated features
                   and of real images of that class (Exp 27's metric; correlations are
                   normalised, so a variance scale cannot fake it)
  Frechet distance between generated and real features in a **held-out** network
                   (ResNet-18), which was never inverted -- the standard realism measure

The authors' code is imported from a local clone, not copied: their repository carries no
LICENSE file, so nothing of theirs is vendored into this one. `--ref-dir` points at it.

MEMORY, WHICH THEIR PUBLISHED SETTINGS DO NOT FIT ON A 24 GB CARD. Their `log_likelihood`
builds a `(batch, dim, 2, 2)` float64 tensor per prefix-scan step: 321 MB per step at
`layer1` (200 704 dims) and their batch of 50, about 18 steps, roughly three such tensors
alive per step. Measured on a 4090: all four layers at once needs ~32 GB (job 541487 died at
22.7 GB allocated), and even one layer at a time needs ~20 GB, which still does not fit
beside the ResNet's own activations (job 541566 died at 22.9 GB). Two changes bring it down,
neither of which touches the experiment: each layer's log-likelihood is recomputed in the
backward pass rather than stored, and the batch is split *inside* the likelihood, which
`layer_nll` shows is exact. The batch that the images, the cross-entropy and the
mean/variance match see is still their 50, so the baseline is unchanged -- simply shrinking
the batch would have weakened the mean/variance match that every arm depends on, their
`W_LCM = 0` ablation included. The laptop's 8 GB cannot hold it regardless.

Writes results/exp28_inversion.csv and a PNG grid per arm.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore")

import common  # noqa: E402
from laplace.lcm import LCMCovariance, _inverse_softplus  # noqa: E402

STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# Their weights and schedule, unchanged.
W_CE, W_BN, W_LCM, W_TV = 1.0, 1.0, 5.0, 0.1
CONFIGS = [{"size": 112, "iters": 1000, "lr": 0.05}, {"size": 224, "iters": 2000, "lr": 0.02}]
POLISH_ITERS = 400


class FeatureHook:
    def __init__(self, module):
        self.features = None
        module.register_forward_hook(lambda m, i, o: setattr(self, "features", o.clone()))


class BNHook:
    def __init__(self, module):
        self.module, self.mean, self.var = module, None, None
        module.register_forward_hook(self._fn)

    def _fn(self, module, inp, out):
        t = inp[0]
        self.mean, self.var = t.mean(dim=(0, 2, 3)), t.var(dim=(0, 2, 3), unbiased=False)


def load_reference(ref_dir):
    """Import the authors' LCM module and Frobenius objective from their clone."""
    d = os.path.join(ref_dir, "efficient_inversion", "resnet34")
    if d not in sys.path:
        # append, never insert: their directory ships a `utils.py`, and putting it first
        # on sys.path would let it shadow any later `import utils` in the process.
        sys.path.append(d)
    from lcm import LCM  # noqa: PLC0415
    from utils import frobenius_dist_reduced  # noqa: PLC0415
    return LCM, frobenius_dist_reduced


def load_class_images(data_dir, limit, device):
    from PIL import Image  # noqa: PLC0415
    from torchvision import transforms  # noqa: PLC0415
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(*STATS)])
    paths = sorted(
        p for p in (os.path.join(data_dir, f) for f in os.listdir(data_dir))
        if p.lower().endswith((".jpeg", ".jpg", ".png")))[:limit]
    if not paths:
        raise FileNotFoundError(f"no images under {data_dir}")
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])


def build_model(device, layers):
    from torchvision.models import ResNet34_Weights, resnet34  # noqa: PLC0415
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device).eval()
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    for p in model.parameters():
        p.requires_grad = False
    hooks = {l: FeatureHook(getattr(model, l)) for l in layers}
    bns = [BNHook(m) for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224, device=device))
    dims = {l: hooks[l].features.numel() for l in layers}
    shapes = {l: tuple(hooks[l].features.shape[1:]) for l in layers}
    return model, hooks, bns, dims, shapes


@torch.no_grad()
def layer_stats(model, hooks, X, layers, device, batch=16):
    """Mean, and the three bands the moment estimator needs, in one streaming pass."""
    acc = {l: None for l in layers}
    n = 0
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        model(xb)
        n += len(xb)
        for l in layers:
            f = hooks[l].features.reshape(len(xb), -1).double()
            s = acc[l]
            if s is None:
                s = acc[l] = {k: torch.zeros(f.shape[1], dtype=torch.float64, device=device)
                              for k in ("mu", "s0", "s1", "s2")}
            s["mu"] += f.sum(0)
            s["s0"] += (f * f).sum(0)
            s["s1"][:-1] += (f[:, :-1] * f[:, 1:]).sum(0)
            s["s2"][:-2] += (f[:, :-2] * f[:, 2:]).sum(0)
    out = {}
    for l in layers:
        s = acc[l]
        mu = s["mu"] / n
        # Second moments about the mean, E[xy] - mu_x mu_y, in the band lengths
        # fit_moments expects: s0 has `dim` entries, s1 has `dim - 1`, s2 has `dim - 2`.
        s0 = s["s0"] / n - mu * mu
        s1 = s["s1"][:-1] / n - mu[:-1] * mu[1:]
        s2 = s["s2"][:-2] / n - mu[:-2] * mu[2:]
        out[l] = {"mu": mu, "s0": s0, "s1": s1, "s2": s2}
    return out


def fit_frobenius(lcms, model, hooks, X, layers, dims, device, epochs, batch=32):
    """The authors' fit, with their learning rates and their objective."""
    _, frob_reduced = load_reference(ARGS.ref_dir)
    opts = {l: optim.Adam([
        {"params": lcms[l].u_raw, "lr": 1e-2},
        {"params": lcms[l].w, "lr": 1e-2},
        {"params": lcms[l].a, "lr": 0.12, "weight_decay": 1e-5},
    ], betas=(0.9, 0.999)) for l in layers}
    for ep in range(epochs):
        tot = {l: 0.0 for l in layers}
        nb = 0
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            xb = X[perm[i:i + batch]].to(device)
            with torch.no_grad():
                model(xb)
            for l in layers:
                f = hooks[l].features.reshape(len(xb), -1)
                V = f - lcms[l].mu
                opts[l].zero_grad()
                u = F.softplus(lcms[l].u_raw) + 1e-5
                loss = frob_reduced(u, lcms[l].a, lcms[l].w, V) / (dims[l] ** 2)
                loss.backward()
                opts[l].step()
                tot[l] += float(loss)
            nb += 1
        if ep % 10 == 0 or ep == epochs - 1:
            print("    fit epoch %3d/%d  " % (ep + 1, epochs)
                  + "  ".join(f"{l}: {tot[l] / nb:.6f}" for l in layers), flush=True)


@torch.no_grad()
def fit_moments(lcms, stats, layers):
    """Our closed-form estimator, then copied into the authors' module.

    Their module stores the diagonal as ``softplus(u_raw) + 1e-5``, so the diagonal our
    estimator returns is inverted through that to leave their forward pass untouched.
    """
    for l in layers:
        s = stats[l]
        est = LCMCovariance(s["s0"].shape[0], dtype=torch.float64).to(s["s0"].device)
        est.fit_moments(moments=(s["s0"], s["s1"], s["s2"]))
        lcms[l].a.copy_(est.a.float())
        lcms[l].w.copy_(est.w.float())
        d = (est.d - 1e-5).clamp(min=1e-8)
        lcms[l].u_raw.copy_(_inverse_softplus(d).float())
        print(f"    moments fit {l}: coordinate span {float(est.a.max() - est.a.min()):.1f}, "
              f"median diagonal {float(est.d.median()):.3e}", flush=True)
        del est


def layer_nll(module, f, chunk):
    """Their per-layer negative log-likelihood, with the batch split to bound memory.

    Why this is exact rather than an approximation: their `log_likelihood` ends in
    `ll_elements.sum(dim=1).mean()`, a mean over images of each image's total, and every
    other quantity it computes (the sorted coordinates, the innovation variances, the Kalman
    gains, the unbatched prefix scan) depends only on the parameters, not on the batch. So a
    size-weighted mean over chunks equals the whole-batch value, and the gradients add the
    same way.

    Checked directly against their module, chunk sizes 1/3/4/7 against the whole batch of
    10: the value agrees to 3e-14 and the gradient with respect to the images -- the only
    thing this loop optimises -- to 2e-16, i.e. exactly. The gradient with respect to the
    covariance parameters differs by ~3e-7, which is their own float32 downcasts at lines 69
    and 99 of `lcm.py` showing up under reassociation, and is irrelevant here because those
    parameters are frozen during inversion (only the Frobenius *fit* trains them, and that
    uses their Frobenius distance, not this likelihood, and is not chunked).

    Why it is needed: the batched scan keeps a `(chunk, dim, 2, 2)` float64 tensor per step,
    about three of them alive across ~18 steps, which is ~20 GB for `layer1` alone at their
    batch of 50 -- more than a 24 GB card holds even when only one layer's graph is live
    (measured: job 541566 died at 22.9 GB inside a single layer's recompute). Chunking is
    the only lever here that leaves the experiment itself untouched: the batch the images,
    the cross-entropy and the mean/variance match see is still 50.
    """
    n = f.shape[0]
    if chunk <= 0 or chunk >= n:
        return checkpoint(module.nll, f, use_reentrant=False)
    out = 0.0
    for i in range(0, n, chunk):
        part = f[i:i + chunk]
        out = out + checkpoint(module.nll, part, use_reentrant=False) * (part.shape[0] / n)
    return out


def invert(model, hooks, bns, lcms, dims, target_idx, args, device, use_lcm):
    """The authors' inversion loop, verbatim in structure, weights and schedule."""
    from torchvision import transforms  # noqa: PLC0415
    mean = torch.tensor(STATS[0], device=device).view(1, 3, 1, 1)
    std = torch.tensor(STATS[1], device=device).view(1, 3, 1, 1)
    label = torch.full((args.batch,), target_idx, dtype=torch.long, device=device)
    aug = nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)))
    img = None

    def feature_losses(batch_n):
        lbn, llcm = 0.0, 0.0
        for l in lcms:
            f = hooks[l].features.reshape(batch_n, -1)
            lbn = lbn + (f.mean(0) - STAT_MU[l]).pow(2).mean()
            lbn = lbn + (f.var(0) - STAT_VAR[l]).pow(2).mean()
            if use_lcm:
                llcm = llcm + layer_nll(lcms[l], f, args.nll_chunk) / dims[l]
        return lbn, llcm

    for cfg in CONFIGS:
        iters = max(1, int(cfg["iters"] * args.iters_scale))
        size = cfg["size"]
        img = (torch.rand(args.batch, 3, size, size, device=device) if img is None
               else F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False))
        img = img.detach().requires_grad_(True)
        opt = optim.Adam([img], lr=cfg["lr"])
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
        for step in range(iters):
            opt.zero_grad()
            if step < iters // 2 and step % 15 == 0:
                img.data = F.avg_pool2d(img.data, kernel_size=3, stride=1, padding=1)
            sx, sy = np.random.randint(-3, 4, 2)
            aug_img = aug(torch.roll(img, shifts=(int(sx), int(sy)), dims=(2, 3)))
            inp = F.interpolate(aug_img, size=(224, 224), mode="bilinear", align_corners=False)
            logits = model((inp - mean) / std)
            loss_ce = F.cross_entropy(logits, label)
            loss_bn = sum((h.mean - h.module.running_mean).pow(2).mean()
                          + (h.var - h.module.running_var).pow(2).mean() for h in bns)
            extra_bn, loss_lcm = feature_losses(args.batch)
            tv = (torch.abs(aug_img[:, :, :, 1:] - aug_img[:, :, :, :-1]).mean()
                  + torch.abs(aug_img[:, :, 1:, :] - aug_img[:, :, :-1, :]).mean())
            total = (W_CE * loss_ce + W_BN * (loss_bn + extra_bn)
                     + (W_LCM * loss_lcm if use_lcm else 0.0) + W_TV * tv)
            total.backward()
            torch.nn.utils.clip_grad_norm_([img], max_norm=1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                img.clamp_(0.0, 1.0)
            if step % 250 == 0:
                msg = f"    {size}px step {step:4d}/{iters}  CE {float(loss_ce):.2f}"
                if use_lcm:
                    msg += f"  LCM-NLL {float(loss_lcm):.2f}"
                print(msg, flush=True)

    polish = max(1, int(POLISH_ITERS * args.iters_scale))
    opt = optim.Adam([img], lr=0.005)
    for step in range(polish):
        opt.zero_grad()
        inp = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        logits = model((inp - mean) / std)
        loss_ce = F.cross_entropy(logits, label)
        loss_bn = sum((h.mean - h.module.running_mean).pow(2).mean()
                      + (h.var - h.module.running_var).pow(2).mean() for h in bns)
        extra_bn, loss_lcm = feature_losses(args.batch)
        tv = (torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
              + torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean())
        total = (W_CE * 20.0 * loss_ce + W_BN * (loss_bn + extra_bn)
                 + (W_LCM * loss_lcm if use_lcm else 0.0) + W_TV * 0.5 * tv)
        total.backward()
        torch.nn.utils.clip_grad_norm_([img], max_norm=1.0)
        opt.step()
        with torch.no_grad():
            img.clamp_(0.0, 1.0)
    return img.detach()


@torch.no_grad()
def lag_profile(F_mat, lags):
    Z = F_mat - F_mat.mean(0)
    Z = Z / Z.std(0).clamp(min=1e-8)
    return {k: float((Z[:, :-k] * Z[:, k:]).mean()) for k in lags}


@torch.no_grad()
def frechet(A, B):
    """Frechet distance between two Gaussians fitted to feature matrices."""
    from scipy.linalg import sqrtm  # noqa: PLC0415
    mu_a, mu_b = A.mean(0).cpu().numpy(), B.mean(0).cpu().numpy()
    Ca = np.cov(A.cpu().numpy(), rowvar=False)
    Cb = np.cov(B.cpu().numpy(), rowvar=False)
    cov_sqrt = sqrtm(Ca @ Cb)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(Ca + Cb - 2 * cov_sqrt))


@torch.no_grad()
def heldout_features(X, device, batch=16):
    """Pooled features from a network that was never inverted."""
    from torchvision.models import ResNet18_Weights, resnet18  # noqa: PLC0415
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    body = nn.Sequential(*list(net.children())[:-1])
    out = []
    for i in range(0, len(X), batch):
        out.append(body(X[i:i + batch].to(device)).flatten(1).cpu())
    del net, body
    return torch.cat(out)


def main():
    global ARGS, STAT_MU, STAT_VAR
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref-dir", default="reference/REMIX", help="clone of the authors' repo")
    p.add_argument("--data-root", default="/shared/sets/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train")
    p.add_argument("--wnid", default="n03457902", help="ImageNet class folder (default: greenhouse)")
    p.add_argument("--class-index", type=int, default=580)
    p.add_argument("--layers", default="layer1,layer2,layer3,layer4")
    p.add_argument("--arms", default="none,frobenius,moments")
    p.add_argument("--batch", type=int, default=50, help="images invented at once (theirs: 50)")
    p.add_argument("--n-real", type=int, default=256, help="real images used to fit and to score")
    p.add_argument("--lcm-epochs", type=int, default=50, help="theirs: 50")
    p.add_argument("--iters-scale", type=float, default=1.0, help="<1 shortens every stage")
    p.add_argument("--nll-chunk", type=int, default=25,
                   help="images per chunk inside the covariance term; 0 uses the whole batch. "
                        "Exact either way -- see layer_nll")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tag", default="exp28_inversion")
    ARGS = p.parse_args()
    layers = ARGS.layers.split(",")
    arms = ARGS.arms.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ARGS.out_dir or os.path.join(common.RESULTS, "exp28_images")
    os.makedirs(out_dir, exist_ok=True)

    LCM_cls, _ = load_reference(ARGS.ref_dir)
    X = load_class_images(os.path.join(ARGS.data_root, ARGS.wnid), ARGS.n_real, device)
    print(f"  {len(X)} real images of class {ARGS.wnid} (index {ARGS.class_index}), device {device}")
    model, hooks, bns, dims, shapes = build_model(device, layers)
    print("  layer dimensions: " + ", ".join(f"{l}={dims[l]} {shapes[l]}" for l in layers))
    stats = layer_stats(model, hooks, X, layers, device)
    STAT_MU = {l: stats[l]["mu"].float() for l in layers}
    STAT_VAR = {l: stats[l]["s0"].float() for l in layers}
    real_heldout = heldout_features(X, device)
    score_layer = layers[-1]
    lags = [k for k in (1, 2, 4, shapes[score_layer][2], shapes[score_layer][1] *
                        shapes[score_layer][2]) if k < dims[score_layer]]
    with torch.no_grad():
        model(X[: min(len(X), 32)].to(device))
        real_prof = lag_profile(hooks[score_layer].features.reshape(-1, dims[score_layer]).double(),
                                lags)
    print(f"  scoring on {score_layer}; real correlation profile "
          + ", ".join(f"lag{k}={v:+.3f}" for k, v in real_prof.items()))

    rows = []
    for seed in range(ARGS.seeds):
        for arm in arms:
            t0 = time.perf_counter()
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"\n  === seed {seed}, arm '{arm}' ===", flush=True)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            lcms = {l: LCM_cls(dims[l]).to(device) for l in layers}
            for l in layers:
                lcms[l].mu = STAT_MU[l].unsqueeze(0)
            if arm == "frobenius":
                fit_frobenius(lcms, model, hooks, X, layers, dims, device, ARGS.lcm_epochs)
            elif arm == "moments":
                fit_moments(lcms, stats, layers)
            img = invert(model, hooks, bns, lcms, dims, ARGS.class_index, ARGS, device,
                         use_lcm=(arm != "none"))
            with torch.no_grad():
                mean = torch.tensor(STATS[0], device=device).view(1, 3, 1, 1)
                std = torch.tensor(STATS[1], device=device).view(1, 3, 1, 1)
                inp = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
                norm = (inp - mean) / std
                logits = model(norm)
                conf = float(F.softmax(logits, 1)[:, ARGS.class_index].mean())
                gen_prof = lag_profile(
                    hooks[score_layer].features.reshape(-1, dims[score_layer]).double(), lags)
                gen_heldout = heldout_features(norm.cpu(), device)
            prof_err = float(np.mean([abs(gen_prof[k] - real_prof[k]) for k in real_prof]))
            fd = frechet(gen_heldout, real_heldout)
            peak_gb = (torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0)
            rows.append({"seed": seed, "arm": arm, "confidence": conf,
                         "profile_err": prof_err, "frechet_heldout": fd,
                         "minutes": (time.perf_counter() - t0) / 60, "peak_gb": peak_gb,
                         **{f"gen_lag{k}": v for k, v in gen_prof.items()},
                         **{f"real_lag{k}": v for k, v in real_prof.items()}})
            print(f"    -> confidence {conf * 100:.1f}%  profile error {prof_err:.4f}  "
                  f"Frechet(held-out) {fd:.1f}  [{rows[-1]['minutes']:.1f} min, "
                  f"peak GPU {peak_gb:.1f} GB]", flush=True)
            try:
                from torchvision.utils import save_image  # noqa: PLC0415
                save_image(img[: min(16, len(img))],
                           os.path.join(out_dir, f"{ARGS.tag}_{ARGS.wnid}_{arm}_seed{seed}.png"),
                           nrow=4)
            except Exception as e:                                     # noqa: BLE001
                print(f"    (could not save images: {e})")
            del lcms, img
            torch.cuda.empty_cache() if device == "cuda" else None

    df = pd.DataFrame(rows)
    path = common.save_results(df, ARGS.tag, config=vars(ARGS))
    print(f"\n  Inverted images of one ImageNet class, {ARGS.seeds} seed(s). "
          f"The loop, its weights and its schedule are the authors'; only the covariance "
          f"fit differs between arms.")
    print(f"\n  {'arm':12}{'confidence':>12}{'profile err':>13}{'Frechet held-out':>18}"
          f"{'min':>7}{'peak GB':>9}")
    for arm in arms:
        g = df[df.arm == arm]
        print(f"  {arm:12}{g.confidence.mean() * 100:11.1f}%{g.profile_err.mean():13.4f}"
              f"{g.frechet_heldout.mean():18.1f}{g.minutes.mean():7.1f}{g.peak_gb.max():9.1f}")
    print("\n  profile error: 0 would mean the generated features carry exactly the real"
          "\n  correlation-vs-distance curve (lower is better). Frechet distance is measured in"
          "\n  a ResNet-18 that was never inverted, so it cannot be gamed by the loop (lower is"
          "\n  better). Confidence is the inverted network's own belief (higher is better).")
    print(f"\nsaved {path}; images in {out_dir}")


if __name__ == "__main__":
    main()
