# Searching for a compelling use of LCM: findings and honest assessment

Following the three EXPERIMENTS.md leads — (ii) per-layer LCM, (iii) associative-scan
log-det, (iv) distributional OOD scores — plus a targeted search for LCM's structural
home. The result is one genuine positive finding, one concrete systems win, and an
honest boundary on where the method helps. It is **not** a slam-dunk publishable
application; this note states exactly what holds and what does not.

## 1. The leads, resolved

* **(ii) Per-layer (block-diagonal) LCM.** Reduces KL-to-full by ~12–22% vs a single
  global chain on all-weights, but does **not** close the gap to KFAC; identical to the
  global chain for last-layer (a single block). Modest.
* **(iv) Epistemic OOD (mutual information / BALD).** Correctly exposes epistemic
  uncertainty (MAP → 0.50 AUROC), and KFAC/Full benefit (0.72), but **LCM (0.62) still
  trails diagonal (0.64)** on CIFAR last-layer. Does not rescue LCM.
* **(iii) Associative-scan log-determinant.** ✅ **A real win** — see §4.

## 2. The one genuine positive: *joint* predictive uncertainty

Every prior negative used a **marginal** metric (NLL, ECE, single-point OOD), and
marginal predictive uncertainty is nearly insensitive to off-diagonal covariance. The
value of a structured covariance appears only in **joint** predictive uncertainty —
the correlations between several predictions. On a 1-D-correlated GLM (LCM's structural
home: ordered, correlated features as in genomics/spectra/time-series; KFAC N/A; full
is `O(d^3)`), with `n=3000` training points:

| test-LL (mean/3 seeds) | d=500 | d=1000 | d=1500 |
|---|---|---|---|
| MARGINAL: diag → **lcm** → full | −0.838→**−0.839**→−0.838 | −0.963→**−0.964**→−0.957 | −1.047→**−1.046**→−1.035 |
| JOINT: diag → **lcm** → full | −0.840→**−0.834**→−0.831 | −0.991→**−0.951**→−0.941 | −1.084→**−1.033**→−1.014 |
| **joint gap-to-full closed by LCM** | **70%** | **81%** | **74%** |

LCM ties diagonal on marginal LL but recovers **70–81%** of the diagonal→full gap on
*joint* LL, and its joint predictive-covariance error is 2–5× lower than diagonal's
(e.g. 0.076 vs 0.367 at d=500). This is robust across seeds and dimensions (weakening
only as `d → n`, where the fit degrades).

## 3. The honest boundary: it needs `n ≳ d`, so active learning does NOT benefit

The natural application of joint uncertainty is **batch active learning / experimental
design** (choosing a diverse, jointly-informative batch needs candidate correlations).
We tested it — and LCM **fails** here:

| % of oracle info-gain above random | d=500 | d=1000 | d=1500 |
|---|---|---|---|
| greedy(diagonal) | 83% | 87% | 90% |
| top-k marginal (diagonal) | 69% | 76% | 87% |
| **greedy(LCM)** | **55%** | **43%** | **46%** |

LCM-guided selection is worse than diagonal-guided *and* than naive top-k. The cause is
a genuine regime mismatch: active learning is **low-data** (n ≪ d), where the GGN is
low-rank and prior-dominated, and LCM's full-rank chain fits it poorly. Measured fit
residual to the GGN (d=500) as data grows:

| n (training points) | 40 | 200 | 1000 | 3000 |
|---|---|---|---|---|
| GGN rank | 40 | 200 | 500 | 500 |
| **LCM fit residual** | **0.738** | 0.455 | 0.214 | 0.129 |

So the joint-UQ advantage holds only when there is enough data to fit the covariance
(`n ≳ d`) — exactly the regime active learning is not in.

## 4. The concrete systems win: associative-scan log-determinant

The Kalman log-det via a 2×2-matrix prefix (associative) scan, `logdet_parallel`:
matches the sequential version to machine precision, is differentiable, and is
**15–64× faster** (4.8 ms vs 305 ms at d=2000; completes at d=20k where it was 0.6 s).
Now the default in `LCMLaplace`. This removes the earlier wall-clock weakness
unconditionally.

## 5. Assessment: is there a compelling, publishable use?

**Honest answer: a narrow one, not yet a killer application.**

* **What is real and publishable-as-a-focused-contribution:** LCM delivers calibrated
  **joint** predictive uncertainty for well-determined (`n ≳ d`) models with ordered,
  correlated features, at `3d` memory and `O(d log d)` computation — a regime where
  KFAC does not apply and full Laplace is `O(d^3)`. The right framing is a GP/GMRF-
  adjacent "scalable joint predictive uncertainty via learned Gauss–Markov structure,"
  demonstrated on **joint forecasting / simultaneous prediction intervals / coherent
  sampling** (all well-determined), *not* active learning.
* **What does not work (report honestly):** weight-space NN Laplace (Kronecker
  curvature → KFAC wins); marginal calibration/OOD (structure-insensitive → ties
  diagonal); low-data tasks including active learning and early-stage BO (fit
  collapses). 
* **The blocker to a broader claim:** the matrix-free/low-rank fit. Making LCM fit a
  low-rank GGN well (natural-gradient / damped / rank-aware fitting) is the single
  change that could open the low-data applications — this is the concrete next step, not
  a guaranteed win.

**Bottom line for a paper.** The strongest *honest* paper here is a focused systems +
UQ contribution: the near-linear structured Gaussian and its joint-UQ calibration in
its home regime, with a clear-eyed characterization (this document) of where it does
and does not help. A splashier "LCM beats X on task Y" claim is not supported by the
evidence gathered.
