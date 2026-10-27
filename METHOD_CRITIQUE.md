# Two problems with the published method

About *"Stop Marginalizing My Dreams: Model Inversion via Laplace Kernel for Continual
Learning"* (arXiv:2605.11804, Krukowski et al.), whose covariance model this project started
from.

**The short version.** Their implementation is sound — three faults an earlier draft of this
file blamed on them turned out to be bugs in *our* port. The method has two real problems, and
they are opposite in kind. Over channels, the chain family cannot represent the covariance
structure that is present. Over flattened feature maps it can, and the fitting fails to find
it. Neither problem touches the paper's accuracy numbers, which we never tested.

---

## Background the problems depend on

The paper's claim is that modelling feature dependencies — rather than assuming a diagonal
covariance — is what makes its method work.

Its covariance family assigns every variable a coordinate `aᵢ` on a line and sets correlation
by distance, `K(a)ᵢⱼ = exp(-|aᵢ-aⱼ|)`, giving `Σ = diag(d) + diag(w) K(a) diag(w)`: three
parameters per variable instead of the `O(N²)` of a dense covariance. The coordinates are
*learned*, so the model is meant to discover its own ordering.

**It is applied in two places, and our two problems live in different ones:**

| | Over **channels** | Over **flattened feature maps** |
|---|---|---|
| A variable is | one channel, 16–512 per layer | one position in the map, `D = C×H×W`, up to ~10⁶ |
| Intrinsic ordering? | **None** — channels have no adjacency | **Yes** — neighbours are adjacent pixels |
| Their code | `resnet32/inversion/lcm.py` (dense, 36 lines) | `efficient_inversion/resnet34/lcm.py` (prefix scan, 112 lines) |
| Fitted by | mean-squared error to an empirical correlation matrix | not fitted; the NLL enters the inversion loss |
| Produces | **the accuracy tables** | the likelihood and scalability figures |

Two separate implementations with disjoint APIs. The paper documents both — Figure 2 for the
channel case, the appendix on architectural adaptation for the flattened one ("up to ∼10⁶") —
but reuses `C` for both the channel count and the flattened length, which hides the
distinction. Having two is **not** a problem; it is the right engineering choice at 512 and at
10⁶ variables.

## The two problems are different kinds of failure

Both end with a covariance that carries no useful off-diagonal structure, which is why they are
easy to confuse. They are distinguished by one question:

> **Does a better parameter setting exist inside the same family?**

| | Problem 1, over channels | Problem 2, over flattened maps |
|---|---|---|
| Is the target the family's shape? | **No** — dense, ~rank-2, half the entries negative | **Yes** — strongly local, neighbour correlation 0.80–0.94 against 0.13 for random pairs |
| Does a good fit exist in the family? | **Not one we can find.** Four fitting routes and two initialisations all land at 0.62–1.00, against a 0.02 floor | **Yes, and we exhibit it.** A closed-form estimator in the same family reaches 0.13 nats/dim where their rule reaches 0.60 |
| So the failure is | **representational** — the family is too small | **optimisation** — the family is right, the search fails |
| What would fix it | a different family; a rank-2 factor model already does better | better fitting; the closed form already does better |

The practical difference: Problem 1 says their *model choice* is wrong for the configuration
behind the accuracy tables, and no amount of engineering rescues it. Problem 2 says their
*model choice is fine* and their fitting procedure is leaving most of the available structure
on the table.

---

## Problem 1 — Over channels: no good fit exists in this family (a representational limit)

**This is the configuration behind the accuracy tables.**

### What we measured

Relative Frobenius error on the **off-diagonal** of a held-out channel correlation matrix —
scale-free, and giving no arm credit for the unit diagonal every one of them has by
construction. 0 is perfect; 1.0 equals predicting zero correlation everywhere. 3 seeds × 4
layers:

| Arm | Params per channel | Off-diagonal error |
|---|---|---|
| Store the full empirical matrix — the `O(C²)` memory the family exists to avoid | C | **0.0208** |
| **Rank-2 factor model**, renormalised to unit diagonal | 3 | **0.5268** |
| The chain, over-trained 25× past their budget | 2 | 0.6246 |
| The chain, exactly as published (200 epochs, lr 0.01) | 2 | 0.6700 |
| Identity | 0 | 1.0000 |

Contrasts are taken **paired by (seed, layer)**, because every arm in a cell is fitted to the
same matrix and the marginal spread across cells is dominated by between-layer variation.
Scaled against the 0.6492 of headroom between their rule and the stored matrix:

| Contrast | Effect | Standard errors | Same sign | Share of headroom |
|---|---|---|---|---|
| 25× fit budget, ordering held fixed | −0.0085 | 2.3 | 8/12 | **1.3%** |
| Seriated ordering, at their budget | −0.0370 | 3.1 | 9/12 | 5.7% |
| Rank-2 vs the over-trained chain | −0.0978 | 6.7 | **12/12** | 15.1% |
| Rank-2 vs their published rule | −0.1432 | 7.2 | **12/12** | 22.1% |

### Why this is a problem

**The family leaves most of the off-diagonal structure unexplained** — 0.67 against a floor of
0.02.

**Fitting is not the binding constraint.** With the ordering held fixed, 25× more optimisation
is worth 1.3% of the headroom, at 2.3 standard errors and not even consistent in sign.
Seriating the coordinates first is worth 5.7%. So this is not an under-tuned optimiser; the
ceiling belongs to the family.

**A low-rank alternative does several times better.** A rank-2 factor model — one more
parameter per channel, no learned ordering, no kernel — beats the chain's *over-trained*
ceiling by 0.0978 at 6.7 standard errors with the same sign in all 12 cells. Note the budget is
not matched: rank-1, which *is* matched at 2 parameters, is worse (0.7515). The win needs the
extra parameter — but one extra parameter per channel buys several times more than every
available fix to the chain's fitting put together.

### Why the family cannot do it — and the test that confirms it

**Where the family sits.** As the coordinate span goes to zero, `K(a) → 11ᵀ` and
`Σ → diag(d) + w wᵀ`. So the chain family *contains* diagonal-plus-rank-one as its degenerate
limit, and adds distance decay on top. It cannot reach rank-2: there is one `w`, hence one
global mode, whatever the coordinates do. Sign enters only through `sign(wᵢwⱼ)`, so the
achievable sign pattern is also rank-1 — a two-camp split.

**What the target is.** Measured in the same run:

- **47–49% of off-diagonal entries are negative.**
- **The top two eigenvalues carry 37–56% of the eigenvalue mass.**

Dense and essentially rank-2: a couple of global modes, with channels loading on them with
either sign.

**The prediction this makes, and it holds.** If the family is boxed in between rank-1 (its own
limit) and rank-2 (out of reach), its best fit must land in that interval. It does, exactly:

```
rank-1 factor  0.7482   >   chain, best effort  0.6246   >   rank-2 factor  0.5268
```

The chain beats rank-1 — so its decay structure is buying something real — and still cannot
approach rank-2, because a second global mode is not in the family at any parameter setting.
That is what "representational limit" means here, and it is why more fitting cannot help.

The intuition is a seating plan: if relationships are friendships and feuds spread across two
overlapping groups, no single ordering of seats satisfies them, however long you spend
rearranging.

### Then why do the accuracy numbers improve?

**We do not know, because we never tested their accuracy.** This is the largest gap in the
document. There is one candidate we did measure.

In the inversion setting, where the fitted object can be inspected, it degenerates to a
diagonal with variances 3–16× the real feature variances (Problem 2). An over-broad Gaussian
NLL is a weak, nearly inert penalty on feature magnitude — a regulariser, not a correlation
prior. Loose regularisers routinely improve results while their internal structure means
nothing.

Two things lift this above a guess. Their baseline is a diagonal with a *different* and worse
variance scale, so part of the reported gain may be nothing more than landing closer to the
right scale. And this project found independently, across several experiments, that apparent
wins from structured covariance reduce to getting the variance *scale* right rather than the
correlation *shape* — a tempered diagonal matches a full covariance once that scale is
oracle-set.

So the likely reading is that **the term helps, but not through the stated mechanism.** Settling
it needs one flag: their pipeline already skips the covariance entirely when `alpha_frob = 0`
(`task_contrastive_cl_runner.py:497`).

### How solid

**Strong — it survived a deliberate falsification attempt.** We suspected the published
initialisation was handicapping the chain: `a` starts at `linspace(-1, 1, dim)`, a span of 2
regardless of dimension, which puts adjacent coordinates at `exp(-2/(C-1))` = 0.996 at 512
channels. Expressing decay over a few neighbours needs a span of order `C`, so we re-ran with
the span scaled to the channel count. It got **much worse**: +0.3004 at 13.2 standard errors,
12/12 cells, at their budget, and +0.2778 over-trained. Their span is near-optimal here, and
the chain's ceiling is genuine.

*Caveat: a small convolutional network on CIFAR-10, not their ResNet-32 continual-learning
pipeline, and it scores correlation reconstruction rather than accuracy.*

### Re-run it, and how it was checked

Script: [`experiments/exp29_channel_correlation_fit.py`](experiments/exp29_channel_correlation_fit.py).
~15 min on the laptop GPU, or on CPU with `CUDA_VISIBLE_DEVICES=""`.

```bash
python -u experiments/exp29_channel_correlation_fit.py \
  --channels 32,128,256,512 --n-images 4000 --epochs 4 \
  --fit-epochs 200 --best-effort-epochs 5000 --seeds 3 \
  --tag exp29_channel_correlation
```

Writes `results/exp29_channel_correlation.csv` plus a `.meta.json` recording config, seeds,
experiment and library commits, and environment.

**The published arm is their code.** `load_published_fit()` (line 73) imports `LCM` and
`fit_lcm_correlation` directly from the clone, so 0.6700 is their model under their objective,
learning rate and epoch count. Each of our arms changes exactly one thing.

| Statement | Check |
|---|---|
| The contrasts | `print_paired()` — paired by (seed, layer), and reported both in standard errors and as a share of headroom, so an effect can be statistically clear and visibly negligible |
| Negative fraction, eigenvalue mass | `structure_stats()` — measured per layer in the same run, not argued from a separate diagnostic |
| Reproduction | re-ran end to end: their rule 0.6745 vs 0.6700, rank-2 0.5304 vs 0.5268 — inside the seed spread, not bit-identical because the convolutions are non-deterministic |
| Falsification | the `*_span_dim` arms; both remain in the script rather than being a one-off check |

---

## Problem 2 — Over flattened feature maps: a good fit exists and their fitting misses it (an optimisation limit)

**This is the model-inversion setting the paper is named after.** A milder complaint than
Problem 1, and the opposite kind.

Here the family is the right choice. The data is strongly local — neighbouring coordinates
correlate 0.80 / 0.85 / 0.94 in absolute value against 0.13 / 0.12 / 0.14 for random pairs, a
ratio near 6–7× at every width — exactly the structure a distance-decay chain exists to
express.

**And a good fit demonstrably exists inside the family.** A closed-form moment estimator on the
same parameterisation reaches **0.1325 nats per dimension** where their rule reaches **0.6000**
(held-out Gaussian cost of centred features, lower is better, 3 seeds). Same family, same data,
far better parameters. So nothing below is a claim about what the model can express — it is a
claim about what their fitting finds.

That estimator is not in the paper, and is written up separately in
[`MOMENT_ESTIMATOR.md`](MOMENT_ESTIMATOR.md) — including why it does **not** rescue Problem 1,
and why feeding it into their inversion loop makes the output worse rather than better.

### What we measured

After running the authors' fitting code with their settings and their learning rates:

| Layer | Coordinate span | Implied adjacent correlation | Fitted ÷ real feature variance |
|---|---|---|---|
| 1 | **0.0** | 0.0002 | 3.06× |
| 2 | **0.0** | 0.0000 | 6.81× |
| 3 | **0.0** | 0.0000 | 16.03× |
| 4 | 1.2 | 0.0002 | 0.63× |

The coordinates start at span 2 and **collapse to a single point** at three of four layers.
With all coordinates coincident `K(a)` is all-ones, so `Σ` can only be diagonal-plus-rank-one —
and the rank-one weights vanish too, which is what the 0.0002 measures.

### Why this is a problem

**What remains is not a correlation model.** It is a diagonal with variances 3–16× the real
feature variances. In the application the paper is named after, at the point the covariance
acts, there is no off-diagonal structure in it. Whatever the term achieves, it achieves as a
loosely scaled diagonal — which is the baseline the method claims to improve on.

**Their own ablation does better without it.** Their `W_LCM = 0` ablation still matches every
layer's feature mean and variance, so it is a diagonal feature match and the covariance term's
whole contribution is off-diagonal structure. With it off, the generated features track the
real correlation-vs-distance profile **better** than with either covariance: profile error
0.0250 (3 seeds, ±0.0012) against 0.0623 for their Frobenius fit and 0.2184 for a well-fitted
chain.

**It is invisible from the loss curve**, which falls smoothly (3.87 → 3.72 over 3000 steps) and
looks converged — consistent with an over-broad Gaussian whose gradients barely move anything.

### Why the fitting fails

Two further results confirm that the gap above is an optimisation gap rather than a quirk of
one estimator. **Exact maximum likelihood** on the same family — optimising precisely the
scored quantity, at 180× the fitting cost — also stalls, at 0.6811, so it is not their choice
of loss that is at fault but gradient descent on this parameterisation. And
*freezing* `a` at equal spacing uses one fewer parameter per dimension and fits **better** at
both large widths (0.5969 vs 0.6165 at `D`≈65k; 0.8396 vs 0.8472 at `D`=196 608). The learned
coordinate is not earning its keep, and both gradient routes stall in the same region.

**A likely cause, which is arithmetic rather than measurement.** `a` initialises at
`linspace(-1, 1, dim)` — span 2 whatever the dimension. At `D` = 200 704 that puts adjacent
coordinates at correlation 0.99999 and the most distant pair at `exp(-2)` = 0.135, so the model
starts almost structureless. Expressing decay over a few neighbours needs a span of order `D`;
the closed-form fit indeed finds 817 775, about 400 000× the initialisation. Gradient descent
starts that far away, with almost no signal pushing the coordinates apart, and it is cheaper to
reduce the loss by collapsing and letting the diagonal absorb the variance. The intuition:
shelving 200 000 books by topic, starting from a single stack, nudging each one a hair at a
time — abandoning the ordering is the easier move.

We tested this hypothesis over channels and it was **wrong there** — but channels are the wrong
place to test it, since that data wants global rather than local structure. Testing it properly
means repeating the span comparison at full width, which is fit-only and cheap.

### How solid

**The collapse measurement rests on a single diagnostic run that has not been reproduced**, and
should be before it is published. The results around it — the ablation comparison, the stalled
maximum likelihood, the frozen-coordinate comparison — are 3-seed results and firm.

*Caveat: one ImageNet class, one architecture, one seed per covariance arm, with the noise floor
borrowed from the 3-seed baseline.*

### Re-run it, and how it was checked

Scripts: [`experiments/exp28_model_inversion.py`](experiments/exp28_model_inversion.py) via
[`cluster/exp28.sbatch`](cluster/exp28.sbatch); the fitting comparisons are
[`experiments/exp25_paper_scale_baselines.py`](experiments/exp25_paper_scale_baselines.py) via
[`cluster/exp25.sbatch`](cluster/exp25.sbatch). Needs a 24 GB card and ImageNet.

```bash
# three arms, one job each (~12 h per covariance arm, 2 min for the ablation)
ARMS=none       SEEDS=3 TAG=exp28_full_none       sbatch cluster/exp28.sbatch   # job 541797
ARMS=frobenius  SEEDS=1 TAG=exp28_full_frobenius  sbatch cluster/exp28.sbatch   # job 541779
ARMS=moments    SEEDS=1 TAG=exp28_full_moments    sbatch cluster/exp28.sbatch   # job 541778

# fit diagnostic: identical fits, inversion truncated, so it costs minutes
ARMS=frobenius,moments ITERS_SCALE=0.003 TAG=exp28_fit_diag sbatch cluster/exp28.sbatch  # job 542802
```

Outputs `results/exp28_full_{none,frobenius,moments}.csv`,
`results/exp28_fit_diag_table.txt`, `results/exp25_paper_scale.csv`, each with a `.meta.json`
carrying job id, host, both commits and full argv.

**The loop and the covariance are theirs.** `load_reference()` (line 109) imports `LCM` and
`frobenius_dist_reduced` from the clone; the inversion loop, its loss weights and its schedule
are theirs, and the mean/variance match runs in all three arms. *One exception:* the Exp 25
fitting comparison uses **our** port of their Frobenius objective (`lcm.fit_frobenius`), not
their code — validated against their original and by Experiment 22 reproducing their published
figure, but it is a port.

| Statement | Check |
|---|---|
| The collapse itself | **Not independently reproduced.** `fit_report()` prints it. The moment arm's spans in that table match job 541778's own log to the decimal, proving the diagnostic reproduces *that* fit exactly — but the Frobenius arm has no such cross-check, because the full run never printed its spans |
| The coordinates really moved | `a` initialises at `linspace(-1, 1, dim)`, span 2, so 0.0 is movement rather than a parameter stuck at a stationary point — checked at `efficient_inversion/resnet34/lcm.py:14` and `resnet32/inversion/lcm.py:13` |
| The ablation comparison | ran 3 seeds rather than the approved 1 to establish a noise floor: run-to-run s.d. 0.0012 on profile error and 1.44 on Fréchet distance, so a one-seed arm must move 0.004 or 4.3 for three-sigma separation. Both covariance arms move far more |
| Not an artifact of the scored layer | re-scored the ablation on the two wide early layers, 3 seeds each: 0.0240 ± 0.0003 and 0.0241 ± 0.0006 against 0.0250 ± 0.0012 |
| Our memory fix is exact | [`experiments/exp28_check_chunk_exact.py`](experiments/exp28_check_chunk_exact.py) checks chunk sizes 1/3/4/7 against the whole batch through their unmodified `nll`: value agrees to 3e-14, gradient w.r.t. the images to 2e-16 |
| Gradient fitting stalls | Exp 25 adds an exact-maximum-likelihood arm optimising the scored quantity directly, and a frozen-coordinate arm at one fewer parameter per dimension. 3 seeds, job 541479 |

---

## The implementation is not a problem

An earlier draft of this file listed three defects in their code: an overflow in the Appendix-B
prefix sum above coordinate range ~709, an inverse-softplus overflow, and a diverging fit
initialisation. **All three are retracted.** Their code contains no inverse-softplus and no
init scaling, and their kernel matvec works in log space with positive/negative splitting
specifically to avoid that NaN (`efficient_inversion/resnet34/utils.py:42-78`). Those were bugs
in **our** port, fixed in our fork and recorded in
[`Laplace/laplace/LCM_PROVENANCE.md`](Laplace/laplace/LCM_PROVENANCE.md).

What genuinely remains is minor and affects none of their results: three float32 downcasts
inside an otherwise float64 prefix scan (`lcm.py:41,69,99`; measured as ~3e-7 on parameter
gradients under reassociation), no test suite, and no licence file. Separately, the ResNet-34
inversion configuration needs ~32 GB, so it does not fit the 24 GB RTX 4090 the paper lists
among its hardware (paper line 1693) — awkward rather than wrong, since the paper does not say
which experiment ran on which card. The covariance term dominates runtime there, 355× the rest
of the loop, but **this does not carry to their accuracy experiments**, for which the paper
reports 7.28 GB for both REMIX and the baseline and a 7% time increase (Table 6) — which
nothing we measured contradicts.

## Why this is not us holding it wrong

- **Our port was checked against their original** on identical parameters against a dense
  `MultivariateNormal` reference. The only discrepancy was their hard-coded nugget `eps`; with
  it matched the agreement is exact ([`AUDIT.md`](AUDIT.md) §1).
- **We reproduced their positive result before criticising anything** — their objective on CNN
  feature maps reproduces the paper's Fig-4 claim (Experiment 22).
- **Problem 2 runs their code, their loop, their weights and their schedule**, imported rather
  than reimplemented. Only the covariance differs between arms.
- **Our memory fix was verified exact** against their unmodified function.

## What this does not establish

- **Their accuracy claims are untested by us.** Everything here measures fit quality and
  generated-feature structure. Their numbers may be real and caused by something other than
  correlation structure. Their `alpha_frob = 0` path makes this cheap to settle, and it is the
  single most informative experiment left.
- **The family is not useless.** Over flattened feature maps it is the right shape, and fitted
  well it beats every same-budget alternative we tried. Problem 2 is about the fitting.
- Problem 1 is a CIFAR-10 proxy; Problem 2 is one class, one architecture, single seeds.

## Corrections made to this document

| Earlier claim | Outcome |
|---|---|
| "Two completely unrelated uses… the paper does not distinguish them" | **Retracted** — the paper documents both; now background, not a defect |
| Prefix-sum overflow, inverse-softplus, init divergence as their bugs | **Retracted** — all three are ours |
| "In both settings their ablation already does the work" | **Narrowed** to the inversion setting |
| "Variances 3×, 7×, 16× too large" | **Qualified** — true at three layers, 0.63× at the fourth |
| "The covariance term costs 355×" | **Scoped** to the inversion path; their accuracy runs report negligible overhead |
| "A chain cannot represent it" | **Weakened** to the measured bound, since `sign(wᵢwⱼ)` admits a rank-1 sign pattern |
| "Exact likelihood lands close to their rule" | **Corrected** — 0.6811 is slightly worse than 0.6000 |
| One root cause behind both problems | **Retracted** — they are opposites: collapsing toward all-ones is family-appropriate over channels and a fitting failure over feature maps |
| "18–20× faster to fit" | **Corrected** to 17.5–20× |

## Everything referenced here

| What | Where |
|---|---|
| Channel-covariance experiment | [`experiments/exp29_channel_correlation_fit.py`](experiments/exp29_channel_correlation_fit.py) → `results/exp29_channel_correlation.csv`, `results/exp29_span_check.csv` |
| Model-inversion experiment | [`experiments/exp28_model_inversion.py`](experiments/exp28_model_inversion.py), [`cluster/exp28.sbatch`](cluster/exp28.sbatch) → `results/exp28_full_*.csv`, `results/exp28_fit_diag_table.txt` |
| Exactness proof for the memory fix | [`experiments/exp28_check_chunk_exact.py`](experiments/exp28_check_chunk_exact.py) |
| Fitting comparison at their scale | [`experiments/exp25_paper_scale_baselines.py`](experiments/exp25_paper_scale_baselines.py), [`cluster/exp25.sbatch`](cluster/exp25.sbatch) → `results/exp25_paper_scale.csv` |
| Their code (cloned, never vendored, no licence) | `reference/REMIX/` — channel model `resnet32/inversion/lcm.py`, flattened model `efficient_inversion/resnet34/lcm.py`, channel fitting `resnet32/utils.py`, channel pipeline `resnet32/continual_runner/task_contrastive_cl_runner.py` |
| Full write-ups, every number and caveat | [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md) — Experiment 29 at line 2331, 28 at 2134, 27 at 2064, 25 at 1950, 22 at 1696, gap register at 2487 |
| Line-by-line boundary against their code | [`Laplace/laplace/LCM_PROVENANCE.md`](Laplace/laplace/LCM_PROVENANCE.md) |
| Adversarial check on our own conclusions | [`AUDIT.md`](AUDIT.md) |

Every result file has a `.meta.json` beside it with config, seeds, both git commits, host and —
for cluster runs — the SLURM job id, so any number here traces to the code that produced it.
