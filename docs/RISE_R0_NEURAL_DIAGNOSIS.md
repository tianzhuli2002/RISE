# RISE R0 neural diagnosis

## Scope and decision

R0 is the existing (p_\theta=\mathcal N((a,b,a+b),I_3)) likelihood ratio,
with the original 3-by-3 training theta grid, reference sample, sidecar,
held-out points, unsigned loss and pseudo-data likelihood evaluator. The
existing `RatioHead` has 8,769 parameters; the reference-anchored bilinear
alternative has 8,245. The exact bilinear-control maximum absolute log-ratio
error is `5.55e-7`.

The neural qualification remains **RISE_R0_BLOCKED**. The dominant failure
class is **STOCHASTIC_OPTIMIZATION**: a passing full-batch solution loses
qualification after continued minibatch optimization from the same weights.
The exact function is representable, and residual dominance is not supported.
This conclusion is specific to R0 and makes no collider-sensitivity claim.

## Direct representability and stochastic comparisons

Ordinary log-ratio regression on the bilinear head used the nine training
theta points and an 80% reference-event training split for 4,000 Adam steps.
Training-theta RMSE is `0.000960`; held-out-theta RMSEs are `0.03153`,
`0.02424`, and `0.10333` for `(0.35,-0.4)`, `(-0.3,0.45)`, and boundary
diagnostic `(1,-0.5)`. The exact basis control and small interior errors show
that the required function is representable; the boundary regression error
also shows that representation alone does not guarantee useful generalization.

| Training path | Maximum held-out RMSE | Held-out (E_{p_0}[\hat r]) | Surface RMSE | Profile a / b RMSE | Hessian error | Grid best-fit matches |
|---|---:|---|---:|---:|---:|---:|
| Existing head, stochastic 4k | 0.36545 | 0.9865, 0.9654, 0.9868 | 42.39 | 27.91 / 8.16 | 0.202 | 0/8 |
| Existing head, stochastic 16k | 0.21436 | 0.9406, 0.9958, 0.8492 | 24.56 | 11.93 / 17.38 | 0.099 | 0/8 |
| Bilinear, stochastic 16k | 0.09011 | 0.9868, 1.0059, 0.9796 | 8.81 | 2.00 / 5.45 | 0.035 | 5/8 |
| Bilinear, zero-residual start, stochastic 16k | 0.08788 | 0.9862, 1.0060, 0.9826 | 9.14 | 2.17 / 5.96 | 0.048 | 5/8 |
| Bilinear explicit-only, stochastic 16k | 0.07918 | 0.9759, 1.0037, 0.9821 | 8.93 | 5.40 / 8.11 | 0.081 | 2/8 |
| Bilinear, exact-risk theta/class-balanced stochastic 16k | 0.17415 | 1.0173, 1.0177, 0.9798 | 7.09 | 4.21 / 3.18 | 0.046 | 5/8 |

The bilinear interaction improves event-level and surface agreement over the
existing head, but no stochastic fit meets all predeclared R0 gates. Removing
the residual branch does not fix the stochastic problem. The extra
theta/class-balanced run has 28 or 29 rows in each of the 18 theta-by-class
cells per 512-row batch. Per-cell inverse-count weights make its empirical
loss exactly estimate the same uniform-theta, equal-class unsigned risk. It
also fails R0; held-out event ratios remain poor, particularly at the boundary.
This rejects theta/class count fluctuation as a sufficient correction.

## Deterministic BCE and branch controls

Full-data LBFGS with the unchanged unsigned objective from two initializations
reached losses `0.528538823` and `0.528540671` (difference `1.85e-6`). The
first passes R0 with held-out RMSEs `0.00904`, `0.00859`, `0.02107`,
normalization means `1.0047`, `1.0039`, `1.0108`, centered likelihood-surface
RMSE `2.07`, and 6/8 exact grid best-fit matches (continuous-likelihood
qualification passes). The second fails with held-out RMSEs `0.03065`,
`0.02486`, `0.09349`, surface RMSE `3.61`, and 5/8 grid matches. Thus almost
equal empirical BCE does not select a unique held-out-calibrated ratio.

Deterministic LBFGS with the residual branch disabled passes at loss
`0.528537393` (held-out RMSEs `0.01040`, `0.00957`, `0.01223`; normalization
`1.0058`, `0.9924`, `1.0132`; surface RMSE `2.28`). This demonstrates a
successful deterministic solution in the explicit branch, but the matching
stochastic explicit-only fit still fails.

At the stored full stochastic checkpoint, the explicit/residual output RMS is
`2.406 / 0.296`; parameter-gradient norms are `0.01097 / 0.00111`. The
residual branch is about 12% of output RMS and 10% of explicit-gradient norm,
so it is not dominating or starving the explicit interaction. In the
explicit-only stochastic run, residual output is exactly zero and only
1,396/8,245 parameters are trainable. Its logged gradients with respect to
the two additive branch-output leaves are equal by the sum structure of the
logit; those are not parameter-gradient norms.

The exact-risk theta/class-balanced repeat also failed. Fixed snapshots at
4k/8k/16k on the same balanced 16,384-row diagnostic population show that the
explicit branch output RMS grows `2.112 -> 2.165 -> 2.406`, while residual RMS
falls `0.766 -> 0.462 -> 0.296`. Explicit/residual parameter-gradient norms
are `0.02780 -> 0.01279 -> 0.01097` and `0.00824 -> 0.00392 -> 0.00111`;
event-representation gradient norm stays near `0.004`. The branch output
cosine is positive (`0.664 -> 0.900`), so cancellation is not indicated.

At the passing deterministic checkpoint, 128 maintained-objective minibatches
of 512 rows have mean loss `0.52847 +/- 0.01753` and gradient-norm
`0.07535 +/- 0.01654`. The mean-gradient/noise-norm ratio is `0.0648`.
Theta-cell counts average 56.2–57.6 of the expected 56.9; positive/reference
class counts average 256.4/255.6. Thus the batch means are balanced, but the
minibatch gradient fluctuates strongly compared with its mean direction.

Starting the unchanged stochastic qualification from the passing deterministic
state at LR `0.002` still fails after 16k updates. The sole later correction
tested lowered LR tenfold to `0.0002`, holding architecture, row sampling,
theta/class sampling, batch size and unsigned objective fixed. This improves
the confidence-region surface RMSEs to `0.13–0.34` and gives 7/8 grid
best-fit matches, but R0 still fails the predeclared normalization gate at
training theta `(-0.8,0.8)`: `E_p0[rhat]=0.9491 +/- 0.0118`, 4.3 combined
Monte Carlo standard errors below one. Held-out normalization passes at
`1.0007, 0.9965, 1.0014`; held-out log-ratio RMSEs are
`0.0200, 0.0095, 0.0358`. No acceptance threshold was changed.

These results support stochastic-optimization sensitivity rather than a
missing x-theta interaction or residual-branch interference. The residual
branch is not dominant, and theta/class counts are balanced on average. R0
remains blocked because one training-grid normalization point fails.

## Readiness and next action

`RISE_NEURAL_R0 = FAIL`. R1, the detector-level interference toy and real
2HDM training were not run. RISE is **not ready for detector-level or signed
interference physics**. The single next action is four-batch gradient
accumulation (effective batch 2,048) from the passing deterministic bilinear
checkpoint at LR `0.0002`, with the same sampler and unsigned objective. The
measured gradient noise motivates this one correction; qualification gates
remain unchanged.

The full result record is
`results/experiments/rise/parameter-conditioning-v1/R0/decision.json`.
Detailed checkpoints and metrics are under
`results/experiments/rise/parameter-conditioning-v1/R0/`;
the exact-risk stratified comparison is under
`results/experiments/rise/parameter-conditioning-v1/R0-theta-balanced-exact-risk/`.
The architecture/optimization comparison plot is
`results/experiments/rise/parameter-conditioning-v1/R0-qualification/RISE_R0_architecture_and_optimization_diagnostic.png`.
Branch scales and gradients are in
`results/experiments/rise/parameter-conditioning-v1/R0/bilinear/branch_diagnostics.json`;
the minibatch-gradient measurement is in
`results/experiments/rise/r0-neural-diagnosis-v1/complete-02/minibatch_gradient_noise.json`.
The branch figure is
`results/experiments/rise/parameter-conditioning-v1/R0-qualification/RISE_R0_branch_diagnostics.png`.
