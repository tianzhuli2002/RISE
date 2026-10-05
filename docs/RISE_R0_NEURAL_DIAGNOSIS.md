# RISE R0 neural diagnosis

## Scope and decision

R0 is the existing (p_\theta=\mathcal N((a,b,a+b),I_3)) likelihood ratio,
with the original 3-by-3 training theta grid, reference sample, sidecar,
held-out points, unsigned loss and pseudo-data likelihood evaluator. The
existing `RatioHead` has 8,769 parameters; the reference-anchored bilinear
alternative has 8,245. The exact bilinear-control maximum absolute log-ratio
error is `5.55e-7`.

The neural qualification remains **RISE_R0_BLOCKED**. The evidence isolates
the dominant failure to **optimization-path-sensitive solution selection**:
the same bilinear function class can represent the exact ratio, but the
validated BCE has multiple numerically similar solutions with markedly
different held-out ratio closure. Residual-branch interference and random
theta/class composition alone do not explain the failure. This conclusion is
specific to R0 and makes no collider-sensitivity claim.

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

The exact-risk theta/class-balanced repeat also failed. Taken together, the
deterministic same-loss/different-generalization result and stochastic
failures isolate the issue to optimization-path-dependent selection among
poorly distinguished BCE solutions, rather than an impossible x-theta
representation, residual dominance, or simple theta/class minibatch
imbalance. A residual optimizer-solution ambiguity remains; no R0 correction
has been adopted.

## Readiness and next action

`RISE_NEURAL_R0 = FAIL`. R1, the detector-level interference toy and real
2HDM training were not run. RISE is **not ready for detector-level or signed
interference physics**. The single next action is to initialize the unchanged
stochastic BCE qualification from the passing deterministic bilinear
checkpoint and measure whether minibatch optimization preserves its verified
held-out solution. This directly tests the remaining path-sensitivity without
adding an architecture or changing the objective.

The full result record is
`results/experiments/rise/parameter-conditioning-v1/R0/decision.json`.
Detailed checkpoints and metrics are under
`results/experiments/rise/parameter-conditioning-v1/R0/`;
the exact-risk stratified comparison is under
`results/experiments/rise/parameter-conditioning-v1/R0-theta-balanced-exact-risk/`.
The compact comparison plot is
`results/experiments/rise/parameter-conditioning-v1/R0-qualification/RISE_R0_architecture_and_optimization_diagnostic.png`.
