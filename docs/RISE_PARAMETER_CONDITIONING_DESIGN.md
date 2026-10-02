# Parameter conditioning qualification

R0 compares the existing head and exactly one reference-anchored bilinear head.
The existing 48,000 reference events and their sidecar are retained. The original
Gaussian RNG stream regenerates the event coordinates; exact weights must agree
with the immutable sidecar before optimization. Both heads see the same rows,
theta draws, class construction, corrected unsigned loss, Adam LR, seed and
step counts. Existing-head states at 4k and 16k distinguish slow optimization
from an interaction inductive bias; both heads are recorded at 4k/8k/16k.

The new head uses delta theta, its degree-two monomials, a small learned parameter
embedding, an explicit event–parameter bilinear interaction and a small residual
MLP. Subtracting the value at delta theta zero anchors log r exactly at reference.
The event encoder remains identity for these analytic tests. There is no
architecture or hyperparameter search and no added theta conditioning in L-GATr.

Acceptance is fixed in `r0.json` before training. Normalization includes finite
training-reference importance-weight uncertainty, not only evaluation MC error.
Event ratio accuracy is compared with the delta-method uncertainty in an
importance-weighted target mean/intercept. Likelihood errors are judged in units
of exact local statistical precision and within the exact DeltaNLL<=3 region;
whole-domain surfaces, profiles and Hessians remain visible in the output.
A boundary diagnostic can veto qualification. A failed gate remains a failed
gate; no normalization correction is silently applied to qualification ratios.

If R0 qualifies, R1 is fixed in advance as a 3D Gaussian with
mu=(a,b,ab), and lower-triangular L rows
(exp(.15a),0,0), (.10b,exp(-.10b),0), (.05a,.08b,1), Sigma=L L^T.
It uses raw x, the same coarse training grid and interpolation/boundary points,
reference theta=(0,0), and the selected R0 head. Exact log densities use a
separate double-precision Gaussian calculation. The covariance's eigenvalues
must be positive throughout the domain. R1 receives no additional event-feature
engineering, head or hyperparameter scan.

No detector-level or real 2HDM work is authorized by R0 alone. R1 must also pass.
