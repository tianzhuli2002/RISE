# RISE scientific closure: source and semantic audit

This is a source-level audit of the implementation at `eec042f` and its
follow-up branch. It does not certify the estimator until the exact toys pass.

## Unsigned ratio

The single output is a binary classifier logit `ell(x,theta)` with membership
one for target and zero for reference. The two classes are sampled with equal
frequency. `unsigned_loss` weights each event by its nonnegative event weight,
normalizes by the complete target/reference training mass when that is supplied,
and averages the binary cross entropy with equal class prior. At the population
optimum,

```text
ell(x,theta) = log p_target(x|theta) - log p_reference(x|theta0)
```

up to numerical/model error. Prediction exponentiates this log ratio.

## Signed ratio

The four softmax classes are target-positive, target-negative,
reference-positive, and reference-negative. `signed_loss` assigns each row by
membership and sign, weights its cross entropy by `abs(weight) / component_mass`,
and uses component masses computed from the complete training EVENT_ID
population. This learns the four normalized nonnegative component densities
with equal class priors. A common softmax gauge cancels from the ratio.

For masses `(T+,T-,R+,R-)` and normalized component densities
`(fT+,fT-,fR+,fR-)`, the implemented signed measures are

```text
qT(x) = T+ fT+(x) - T- fT-(x)
qR(x) = R+ fR+(x) - R- fR-(x)
Zt = T+ - T-
Zr = R+ - R-
ratio_shape(x) = [qT(x) / qR(x)] * [Zr / Zt]
```

The network outputs class logits; `signed_ratio` reconstructs each component
density as `exp(logit_j - max(logits)) * mass_j`, subtracts negative components,
and applies `Zr/Zt`. It fails on nonpositive signed normalizations or reference
density. This is a normalized shape ratio. Total rate information is removed by
`Zr/Zt`; a separate rate term is not implemented.

This is the ratio-of-signed-mixtures decomposition discussed in
[Drnevich et al., arXiv:2410.10216](https://arxiv.org/abs/2410.10216). The
implementation uses one four-class normalized-density classifier rather than
separate binary sub-ratio heads; exact algebra closure is required to establish
that this parameterization reconstructs the same signed ratio.

## Provenance gap found during the audit

Before this follow-up, direct parameterized sidecar writes did not require a
weight convention, and ROOT conversion left the top-level normalization as
`unknown` even when explicit POI metadata was supplied. This allowed an
ambiguous parameterized sidecar to reach training. The follow-up now requires
`absolute`, `relative_to_nominal`, `shape_only`, or `rate_bearing` for native
and reweighted sidecars, preserves the declared definition and normalization,
and marks metadata-only ROOT audits as `unresolved`. RISE training still
estimates normalized shapes; rate-bearing inputs do not create a rate
likelihood term.

## Analytic closure status

The corrected reweighted-loss denominators are complete training component
masses divided by the uniformly sampled training EVENT_ID population. In the
controlled `N=4096` versus `4N=16384` check, the corrected loss and shared and
ratio-head gradient norms are identical; the legacy formula scales each by
`1/4`. Exact two-sided and one-sided signed-ratio algebra close to below
`7e-16` maximum absolute error.

The analytic run2 checkpoint does not yet give likelihood closure. Independent
reference normalization of the learned ratio improves some profile metrics
but leaves substantial likelihood-surface and profile errors; the normalized
best fit still misses the exact grid point in all eight reweighted pseudo
datasets. The same parameterized-shape problem appears in the native-point
control. At the fixed interior point `(0.8, 0.0)`, the reweighted loss path
recovers the exactly linear Gaussian log-ratio coefficients with `0.0107`
log-ratio RMSE and selects the correct likelihood hypothesis in all 40
pseudo-datasets. The failure is therefore localized to joint x-theta conditioning,
not the reweighted sampling/loss path or the signed-mixture algebra.

RISE analytic closure remains blocked. The next implementation is one explicit
bilinear x-theta interaction in the RatioHead, followed by the existing
analytic likelihood test. No detector-level or 2HDM training is justified by
these results.
