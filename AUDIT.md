# Adversarial audit: are the negative results real?

An attempt to *break* the project's conclusions — by finding an error in how the paper's code was
incorporated, or a choice of formulation / architecture / data / settings that manufactured the
negative results. Five independent checks. **All five failed to overturn the conclusions**, and two
of them strengthen the central claim. Written 2026-08-09.

---

## 1. Is the port of the paper's code faithful? — VERIFIED EXACT

Ran the paper's original `lcm.py` (fetched from the authors' repo) and my implementation on
*identical* parameters, against a dense `MultivariateNormal` ground truth:

| implementation | mean log-likelihood | error vs dense truth |
|---|---|---|
| paper's `LCM.log_likelihood` | −92.89584 | 8.7e-06 |
| mine, default ε=1e-6 | −92.89607 | 2.3e-04 |
| mine, ε matched to paper (1e-5) | — | **0.00e+00** |

The entire residual was the nugget epsilon convention: the paper hardcodes `1e-5`, my default is
`1e-6` (configurable). With matching ε the agreement is exact. The 2×2 associative-scan Riccati
recursion, the kernel, and the state-space formulation all match the source.

## 2. Is the library integration correct? — VERIFIED EXACT

LCM strictly generalises a diagonal covariance (`w = 0`), so an LCM posterior with `w` forced to
zero *must* reproduce `DiagLaplace` exactly. It does:

| check | result |
|---|---|
| diagonal of LCM precision == diagonal Laplace | ✔ true |
| max abs off-diagonal of LCM precision | 0.00e+00 |
| log-determinant | 100.048805 vs 100.048805 |
| predictive probabilities, max abs difference | 8.9e-08 (float32 precision) |
| marginal likelihood | −128.898438 vs −128.898438 |

Together with the 15 unit/integration tests (every fast primitive checked against a dense
reference), the integration is sound. The negative results are not an implementation artifact.

## 3. Did the *formulation* choice manufacture the negatives? — NO; the choice was correct

The one substantive methodological decision never tested during the project: LCM was fit to the
**precision** (the GGN) throughout, whereas the paper fits it to a **covariance**. If that were
wrong, every negative result would be suspect. Tested directly on the 1-D-correlated GLM:

| | KL to exact posterior | marginal LL | joint LL |
|---|---|---|---|
| diagonal | 352.6 | −0.8144 | −0.8213 |
| **A: fit precision (what was done)** | **75.3** | **−0.8135** | **−0.8090** |
| B: fit covariance (the alternative) | 1752.8 | −0.9423 | −0.8770 |
| full (exact) | 0 | −0.8114 | −0.8001 |

Formulation A is better on *every* metric, and B is *worse than diagonal* on both predictive
metrics. (B does win on raw covariance Frobenius error, 0.595 vs 0.785 — it fits the large-variance
directions better — but that does not translate into posterior or predictive quality.) The choice
made was the right one; the alternative would have produced a *more* negative result.

## 4. Would a larger model / more compute help? — NO; the mismatch is structural

The decisive question for "should we run this on a cluster". A linear layer's GGN is *exactly*
Kronecker (`Λ ⊗ ΦΦᵀ`) — which is KFAC's home and which a bandwidth-1 chain cannot represent except
degenerately. Measured on real trained last-layer GGNs:

| relative Frobenius error | Linear(8→3), P=24 | Linear(12→4), P=48 |
|---|---|---|
| nearest Kronecker product | **0.530** | **0.451** |
| best LCM chain | 0.733 | 0.709 |
| diagonal | 0.906 | 0.931 |

**The gap widens with scale.** Growing the layer improved the Kronecker fit (0.530 → 0.451) while
the chain stayed flat (0.733 → 0.709). Scaling up makes KFAC *relatively better*, not LCM. No
amount of compute changes a structural representational mismatch — this is the one intervention
that provably cannot work.

Corroborating evidence already in hand: the Exp 4 ablation showed the chain fit to a real GGN
plateaus at ~0.66 regardless of learned ordering, nugget, or optimisation budget (0.6585 at 800
steps → 0.6582 at 1600), while the *same* fitter recovers a genuine chain to 1.4e-4. The
limitation is the target, not the model, the optimiser, or the scale.

## 5. Was LCM applied to the wrong *target* (weights instead of features)? — NO evidence for it

The paper applies LCM to conv **feature maps** (spatially correlated, naturally raster-orderable),
whereas this project applied it to **weight curvature**. Fitting LCM to an actual conv feature-map
covariance (8×14×14, D=1568):

| target | LCM | diagonal |
|---|---|---|
| conv feature-map covariance (paper's target) | 0.924 | 0.967 |
| last-layer weight GGN (this project's target) | 0.66 | 0.97 |

LCM did *worse relative to diagonal* on the paper's own target than on weight curvature. So "wrong
target" does not explain the negative results.

*Caveat, stated honestly:* this test is confounded — it used an untrained network and `n = 512 <
D = 1568`, so the empirical covariance is rank-deficient, a regime Exp 9 showed LCM fits poorly.
Note the same caveat applies to the paper, whose feature dimension reaches ~2·10⁵ with far smaller
batches: it is also in the `n ≪ D` regime, where absolute fit quality is poor and only the
comparison against diagonal is meaningful. This is consistent with, not contradictory to, the
project's findings.

---

## What the audit did *not* rule out

Stated so the audit is not over-read:

- Formulation B was tested at one setting (GLM, d=250–400), not across the full campaign.
- The habitat test (§5) is confounded by rank deficiency and an untrained network; a clean version
  would use a trained net with `n > D`.
- Some early experiments used 2–3 seeds (the headline joint-UQ result was re-run at 10).
- Only two real datasets.

None of these affect the two structural findings (§3, §4), which are the load-bearing ones.

---

## Recommendations

**1. Do not buy compute for this.** A larger model or a cluster is the single intervention that
provably will not help weight-space Laplace: the Kronecker-vs-chain mismatch is representational
and *widens* with scale. Redirect any compute budget to more real ordered-feature datasets.

**2. The publication verdict stands, and is now better supported.** The workshop → TMLR path as an
honest characterisation plus systems contribution is unchanged — but it can now state that the
negative results survived an adversarial audit of the two most plausible confounds (fit-domain
choice and scale). That is a genuinely stronger paper than one that simply reports a negative.

**3. Next experiments should add breadth, not size.** More real datasets with a known 1-D order and
`n ≳ d`: NIR/Raman spectroscopy beyond Tecator, genomics with linkage-disequilibrium structure,
climate/spatial transects. With marginal-likelihood-tuned noise and ≥10 seeds. This is what would
move the joint-UQ claim from "one dataset" to "a real regime".

**4. One genuinely untested Bayesian setting.** Functional / GP-space Laplace, where the correlated
variables are *data points* rather than weights, and the covariance over points is genuinely
Matérn-like — LCM there is a linear-time GP. Be aware this is classical state-space-GP territory,
so it would be a systems contribution, not novelty. Worth a scoping experiment before committing.

**5. Fix the remaining implementation weakness.** Wall-clock scales `O(d^1.44)` rather than
`O(d log d)` because the blocked kernel's sequential block count grows with the coordinate range.
An associative-scan block carry — the same trick already used for the log-determinant — should
recover the intended asymptotics. This is the weakest part of the systems story and is fixable.

**6. Do not pursue ordering recovery expecting a win.** The fragility threshold (≥0.999 required;
0.998 already fails) means even a very good seriation is insufficient. Unless a method can
essentially solve the permutation exactly, this direction has poor expected value.
