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

### Follow-up objective and initialization controls

The stochastic 16k-step bilinear run was repeated with the residual output
layer zero-initialized. This leaves the reference anchoring, explicit
event--parameter interaction, data, theta grid, seed, Adam settings and
unsigned objective unchanged. It did not close the remaining gates:

| Bilinear initialization / steps | Held-out RMSEs | Held-out `E_p0[rhat]` | Grid best-fit matches | Surface RMSE | Profile a / b RMSE | Hessian error |
|---|---|---|---:|---:|---|---:|
| Standard / 16k | 0.0339, 0.0269, 0.0901 | 0.98684, 1.00592, 0.97962 | 5/8 | 8.81 | 2.00 / 5.45 | 0.035 |
| Zero residual / 16k | 0.0352, 0.0269, 0.0879 | 0.98615, 1.00595, 0.98255 | 5/8 | 9.14 | 2.17 / 5.96 | 0.048 |

The follow-up run records equal-theta-weighted training RMSEs of `0.2778`
(existing / 4k), `0.1502` (existing / 16k), `0.0936` (bilinear / 16k), and
`0.0946` (zero-residual bilinear / 16k). The earlier table above retains the
aggregate metric from the initial matched comparison.

The exact bilinear control is representable to maximum absolute log-ratio
error `5.55e-7`. On the 48k-event population, deterministic full-data BCE
optimization from two starts reached nearly identical losses (`0.528538823`
and `0.528540671`), but only one start passed all held-out ratio gates. A
deterministic fit with the residual branch disabled passed at loss
`0.528537393`. These controls show that explicit interaction substantially
improves the stochastic fit, while the stochastic qualification remains
sensitive to the optimization path and residual branch. The cause is not yet
isolated to a single implementation component.

The explicit-only stochastic run and an exactly risk-preserving
theta/class-balanced repeat have now both completed and failed R0. Full-data
LBFGS passes from one initialization and fails from another at nearly the same
objective value. The remaining diagnosis is optimization-path-sensitive
solution selection; residual-branch dominance and theta/class batch-count
fluctuation are not sufficient explanations. See
[`RISE_R0_NEURAL_DIAGNOSIS.md`](RISE_R0_NEURAL_DIAGNOSIS.md) for the final
metrics, machine-readable decision, and single next experiment.

R1, detector-level RISE, the interference toy, and real 2HDM training remain
gated.

All checkpoint states, fixed-checkpoint metrics, training histories, matched
comparison and decisions are retained under
`results/experiments/rise/parameter-conditioning-v1`. The raw-x event encoder is
identity in this qualification. These results imply no real collider sensitivity.
