# R0 architecture qualification result

Starting commit: `44264aee110ed2f3a49103ac6e119bf7dc56898b`.
The corrected loss and signed algebra remain unchanged. Their existing successful
validations are reused; no loss-scaling, signed-algebra or fixed-theta closure
job was repeated. Detector-level RISE and real 2HDM training remain prohibited.

One alternative head was implemented: `ReferenceAnchoredBilinearRatioHead`.
Its 8,245 parameters are comparable to the existing head's 8,769. Delta-theta
linear/quadratic features, a 32-wide embedding, explicit bilinear interaction
and a small residual MLP are anchored by g(h,delta)-g(h,0). Exact reference
anchoring, finite gradients, exact R0 representability and the maintained
RISE unsigned forward path pass targeted checks. The default head is unchanged.

R0 uses the exact existing 48,000 reference coordinates, regenerated from the
original RNG stream and checked against every immutable sidecar weight. Both
heads use Adam LR 0.002, batch 512, seed 20261002, identical row/theta/class RNG
streams and the corrected unsigned loss. Fixed diagnostics use independent
reference/target samples and the same eight 200-event pseudo-datasets as the
previous likelihood comparison. No independent normalization correction is
applied. Criteria in `r0.json` were written before training.

| Head / steps | Mean training-theta log-ratio RMSE | Held-out RMSEs | Exact grid fits | Surface RMSE | Profile a / b RMSE | Hessian relative error |
|---|---:|---|---:|---:|---|---:|
| Existing / 4k | 0.2394 | 0.1504, 0.1481, 0.3654 | 0/8 | 42.39 | 27.91 / 8.16 | 0.202 |
| Existing / 8k | 0.1414 | 0.1694, 0.1343, 0.1830 | 1/8 | 21.41 | 17.41 / 13.25 | 0.101 |
| Existing / 16k | 0.1349 | 0.1281, 0.0595, 0.2144 | 0/8 | 24.56 | 11.93 / 17.38 | 0.099 |
| Bilinear / 4k | 0.1346 | 0.0541, 0.0518, 0.2878 | 3/8 | 12.80 | 6.41 / 4.44 | 0.048 |
| Bilinear / 8k | 0.0817 | 0.0411, 0.0341, 0.0982 | 1/8 | 10.67 | 6.19 / 5.50 | 0.061 |
| Bilinear / 16k | 0.0766 | 0.0339, 0.0269, 0.0901 | 5/8 | 8.81 | 2.00 / 5.45 | 0.035 |

Held-out points are (0.35,-0.4), (-0.3,0.45), and boundary diagnostic (1,-0.5).
Bilinear 16k reference means are 0.98684, 1.00592, 0.97962 there. Its local
likelihood offsets are 0.43-0.63 exact statistical sigma, and confidence-region
surface RMSEs are 0.54-0.79 log-likelihood units. Those likelihood gates pass.
However event-level and normalization gates fail at several training/held-out
points. For example theta=(0,-0.8) has E_p0[rhat]=0.93260 with combined
training-reference/evaluation MC SE 0.01222. Boundary log-ratio RMSE 0.09011
exceeds its predeclared 3-sigma MC allowance 0.07228. Reference anchoring itself
is exact but does not normalize other theta points.

Verdict: **RISE_R0_BLOCKED**. Explicit interaction is a useful inductive bias,
but neither extended training nor this architecture establishes trustworthy
parameterized closure. The remaining objective/optimization error is unresolved;
no additional architecture is tested or adopted as default. R1 is not trained.
Its exact mean/covariance definition is fixed in `r1.json` and independently
cross-checked against SciPy Gaussian log densities. R2 design is not reached.

The single next experiment is an exact Gaussian bilinear/quadratic-basis
optimization control through the unchanged parameterized unsigned objective.
This isolates the remaining error without another flexible architecture search.

All checkpoint states, fixed-checkpoint metrics, training histories, matched
comparison and decisions are retained under
`results/experiments/rise/parameter-conditioning-v1`. The raw-x event encoder is
identity in this qualification. These results imply no real collider sensitivity.
