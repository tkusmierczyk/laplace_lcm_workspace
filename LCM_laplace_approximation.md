# LCM for structured-covariance Laplace approximation of BNN posteriors

Deep dive on idea #1. Central claim under test: **the last equation of Appendix A is exactly the
right primitive for the Laplace approximation**, because the matrix that *defines* a Laplace
posterior is natively a sum of rank-1 outer products — the same object App A fits.

---

## 1. Why Appendix A's last equation *is* a Laplace-approximation primitive

**The equation (App A, final line).** For centered vectors `{v_i}` with second-moment
`Ŝ = (1/N)Σ_i v_i v_iᵀ`, it minimizes `‖M − Ŝ‖²_F` (dropping the `Ŝ`-only constant), where
`M = diag(d) + diag(w)K(a)diag(w)`:

```
L_F(a,w,d) = ‖d‖² + 2·dᵀw² + (w²)ᵀK(2a)(w²)            ← model term  ⟨M,M⟩_F, O(P log P)
             − (2/N) Σ_i [ dᵀv_i² + (w⊙v_i)ᵀK(a)(w⊙v_i) ] ← data term  (2/N)Σ v_iᵀM v_i
```

The property that matters: **it fits a structured full-covariance Gaussian to a `P×P`
second-moment matrix using only matrix-free contractions `v_iᵀ M v_i` (each `O(P log P)` via
prefix sums), never forming, inverting, or taking the log-det of a dense `P×P` matrix.**

**The Laplace posterior is built from exactly such a matrix.** For a BNN with parameters
`θ∈ℝ^P`, loss `L(θ) = −Σ_n log p(y_n|x_n,θ) − log p(θ)`, MAP `θ*`:

```
p(θ|D) ≈ N(θ*, Σ_post),   Σ_post = H⁻¹,   H = ∇²L(θ*).
```

In practice `H` is replaced by the (PSD) **Generalized Gauss–Newton / Fisher**, which is a
**sum of rank-1 outer products of per-example gradients**:

```
H ≈ F + σ₀⁻²I,     F = (1/N) Σ_n g_n g_nᵀ,     g_n = ∇_θ log p(y_n|x_n,θ*).
```

`F` has *exactly the form* `Ŝ = (1/N)Σ v v ᵀ` with `v_n = g_n`. So App A's last equation, with the
per-example gradients substituted for the "samples," fits an LCM matrix to the Fisher in
`O(P log P)` per datum and `O(P)` memory — the one operation that otherwise blocks full-covariance
Laplace at `P ≥ 10⁴`. **That is the concrete relationship: the Fisher = App A's target.**

There are two clean instantiations, differing in whether the *sum of outer products* we feed in is
the Fisher (precision) or a weight-sample covariance.

---

## 2. The math — two formulations

Structural fact used throughout: `A(a,w,d)=diag(d)+diag(w)K(a)diag(w)` is **dense** with `3P`
params; its inverse `A⁻¹` is **tridiagonal**; log-det, quadratic forms, and sampling are all
`O(P log P)` / `O(P)` (App B–D).

### Formulation A — Fisher fit (proper GGN/Fisher Laplace)
Set `v_n = g_n`, target `Ŝ = F`. Fit `A ≈ F` with the App-A objective; fold the prior by adding
`σ₀⁻²` to `d`. **Interpret `A` as the posterior precision:**

```
Σ_post⁻¹ ≈ A = diag(d) + diag(w)K(a)diag(w),      Σ_post = A⁻¹  (tridiagonal, closed form, App C).
```

- Because `A` is *dense*, the fit sees the **off-diagonal Fisher entries** — it captures curvature
  correlations that diagonal Laplace discards, using `3P` parameters.
- Everything downstream is cheap: sample `θ ~ N(θ*, A⁻¹)` in `O(P)`; `log|Σ_post| = −log|A|` in
  `O(P)`; predictive quadratic forms in `O(P log P)`.
- **Honest subtlety.** This makes the *covariance* `A⁻¹` tridiagonal, i.e. `Cov(θ_i,θ_j)=0` for
  non-neighbours in the ordering. That is a real modeling constraint. If you instead want a
  tridiagonal *precision* (a GMRF: dense covariance, conditional independence given neighbours),
  set the precision to `A⁻¹` and fit `‖A⁻¹ − F‖²_F` — but a tridiagonal target only matches `F` on
  its 3 diagonals, collapsing to plain "tridiagonal-precision Laplace" and not using the kernel's
  dense expressiveness. **The interesting, App-A-native use is fitting the dense `A` to `F`.**
- Diagonal Laplace is the special case `w=0` (`A=diag(d)`, `d=diag(F)+σ₀⁻²`). LCM is the strict,
  same-cost generalization — this is the cleanest way to frame the contribution.

### Formulation B — weight-covariance fit (SWAG-style stochastic-weight Laplace)
Collect weight samples `{θ_t}` (SGD/SGLD iterates near `θ*`, or short-run posterior draws), set
`v_t = θ_t − θ̄`, target `Ŝ = Ĉov(θ) ≈ Σ_post`. Fit `A ≈ Ŝ` and **interpret `A` as the covariance:**

```
Σ_post ≈ A = diag(d) + diag(w)K(a)diag(w),      Σ_post⁻¹ = A⁻¹  (tridiagonal → GMRF over weights).
```

- This is *literally the paper's algorithm* (fit LCM to a sample covariance) applied to weights
  instead of features — the most direct transplant of App A, and `A` is a bona-fide covariance,
  giving the natural GMRF posterior (tridiagonal precision = weights conditionally independent
  given their two neighbours in the learned order).
- Sample via the AR(1) recursion in `O(P)`. Baseline here is **SWAG** (diagonal + low-rank
  covariance of the same iterates): swap SWAG's diag+low-rank for LCM, same iterates.
- Optionally fit `a,w,d` by the *exact* Gaussian NLL of the weight samples via the Kalman filter
  (App D) instead of Frobenius — App A is just the cheapest fit.

**Recommendation.** Lead with **Formulation A** (it is "the Laplace approximation" proper, and it
directly instantiates the paper's diagonal→structured thesis in weight space). Use Formulation B /
SWAG as a second, equally-App-A-faithful variant and a strong baseline family.

**Per-layer / block structure.** A single 1-D ordering of all `P` weights is the weakest link
(see the limitations in `LCM_bayesian_uses.md`). The natural fix, mirroring KFAC, is a **separate
LCM chain per layer/tensor** — block-diagonal across layers, LCM within a layer. Weights within a
filter/layer are far more plausibly 1-D-orderable, and it keeps everything `O(P)`.

---

## 3. First experiment — baselines and what to show

**Setting.** Start where an exact reference exists: **last-layer Laplace** (P = features×classes)
or a **small MLP/CNN**, where the *full* Fisher `F` and exact posterior `N(θ*, H⁻¹)` are
computable as gold standard. This isolates the covariance-modeling question with a ground truth to
compare against — the direct analog of the paper's Fig. 4/10 (feature log-likelihood: diagonal vs
LCM vs full). Then scale to `P ≥ 10⁴–10⁶` (full-net, per-layer LCM) where full Laplace is
infeasible and only the structured methods survive.

**Baselines (ranked).**
1. **Diagonal Laplace** — *the* "does structure help?" control; the direct analog of the paper's
   diagonal-covariance baseline, and the `w=0` special case of LCM. **Primary.**
2. **Exact / full Laplace (`H⁻¹`)** — gold-standard reference; feasible for last-layer/small nets.
   Gives the approximation-gap denominator. **Primary where feasible.**
3. **KFAC Laplace** (Ritter et al. 2018) — the standard *structured* Laplace; tests whether LCM's
   cheaper `3P` chain is competitive with Kronecker blocks.
4. **Last-layer Laplace** — the common practical default (`laplace-torch`).
5. **Low-rank+diagonal / SWAG** — the other structured-covariance family (natural pair for
   Formulation B).
6. **MAP, MC-Dropout, Deep Ensembles** — non-Laplace UQ references for the downstream metrics.

**Tooling.** `laplace-torch` (Daxberger et al. 2021, *Laplace Redux*) already implements
diagonal/KFAC/last-layer/full Laplace plus the predictive and marginal-likelihood machinery. Drop
LCM in as a new covariance backend (fit-to-Fisher + `O(P)` sample + `O(P)` log-det) and reuse their
benchmarks — makes the first experiment low-effort.

**What to show.**
1. **Posterior-approximation quality vs. the exact Gaussian** (where computable) — the paper's
   Fig-4 analog in weight space:
   - `KL(N(θ*,Σ_LCM) ‖ N(θ*,H⁻¹))` and `‖Σ_LCM − H⁻¹‖` vs. the diagonal counterparts.
   - log-likelihood of held-out posterior/weight samples under LCM vs diagonal. **Expect LCM to
     dominate diagonal exactly when the Fisher has off-diagonal structure.**
2. **Bayesian predictive quality** (the real payoff, on real benchmarks): test **NLL** and
   **calibration (ECE)** of the MC predictive, and **OOD detection** (predictive-entropy /
   mutual-information AUROC), LCM vs diagonal vs KFAC vs full.
3. **Marginal likelihood** `log p(D) ≈ −L(θ*) + ½log|2πΣ_post|` for **prior-precision (`σ₀`)
   selection / model comparison** — a concrete place where LCM's `O(P)` log-det is a capability
   diagonal-plus-something methods lack cheaply.
4. **Cost & scaling**: parameters (`3P` vs `P` diag vs `P²` full vs KFAC block sizes), fit and
   per-sample wall-clock/memory, demonstrated up to `P ≥ 10⁴–10⁶` — the claim is *diagonal-like
   cost, better-than-diagonal quality*.
5. **When structure helps vs. doesn't** (honest, mirrors the paper's ViT early-layer result): show
   LCM ≈ diagonal when curvature is near-diagonal, and LCM ≫ diagonal when it is correlated
   (deeper layers, correlated features). This delineates *when to use it* rather than overclaiming.

**One-line thesis to test.** *A tridiagonal-structured Laplace posterior, fit to the Fisher by the
Appendix-A Frobenius objective at diagonal-like cost, is closer to the true posterior and better
calibrated than diagonal Laplace whenever the network's curvature carries real off-diagonal
correlation — and it scales to `P ≥ 10⁴` where full/KFAC Laplace do not.*
