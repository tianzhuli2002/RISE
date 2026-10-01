# RISE implementation surface

Source: `ec54f6b13817009b5ec0472e084a034f044980a4`; isolated branch
`research/rise-implementation-v1`. RISE has no KNOT dependency. It uses the
maintained measured-event L-GATr encoder and query machinery, data adapter,
Lightning checkpoint/scaler metadata, component-loading checks and optimizer.

## Detector representation and training

`RiseModule` encodes the measured event into the existing pooled representation.
`RatioHead` embeds explicit ordered theta coordinates and conditions at the
ratio head. No assignment, neutrino or reconstructed parent is required as a
head input.

- **0:** SBI loss only. Reconstruction objectives and samplers do not execute.
- **P:** load `visible`/`queries` components using the maintained component loader;
  optionally freeze them or fine-tune. A reconstruction Lightning checkpoint
  is required. Prediction checks the declared initialization provenance, but reloads the full
  self-contained RISE checkpoint without reopening the pretraining checkpoint.
- **A:** SBI plus the existing configuration-driven assignment, invisible and
  parent auxiliary losses. Auxiliary objectives share the single measured
  encoder pass and obey the maintained dependency/gradient policies.

Positive-only mode emits a scalar log ratio using a class-balanced weighted
BCE. Negative weights are rejected by this loss. If weights vary by parameter
point, the adapter supplies fixed TRAIN-population class normalisations so
changing cross sections do not silently alter class priors.

Signed-mixture mode learns four nonnegative components: target+, target-,
reference+, reference-. A four-way classifier uses absolute weights divided by
fixed per-parameter TRAIN component masses. Zero-mass components are masked;
zero-weight events are excluded from loss evaluation. The resulting logits
encode relative component densities under equal component priors. At inference:

```
r = (M_t+ exp(z_t+) - M_t- exp(z_t-))
    / (M_r+ exp(z_r+) - M_r- exp(z_r-)) * Z_r / Z_t
Z_t = M_t+ - M_t-; Z_r = M_r+ - M_r-
```

This supports negative reference weights as well as negative target weights.
Prediction returns ratio, sign and log absolute ratio, since a signed density
ratio may itself be negative. Nonpositive total normalisations or nonpositive
reference densities fail; they are not clipped into a physical likelihood.
The approach uses the signed-mixture decomposition described in
[arXiv:2410.10216](https://arxiv.org/abs/2410.10216), with a shared multiclass
component classifier. No physics-specific Big-Dipper assumptions are made.

## Sidecar contract

`write_sidecar`/`SBISidecar` use HDF5 `schema_version=rise_sbi_sidecar_v1`:

| Dataset | Meaning |
|---|---|
| `EVENT_ID [N]` | Unique authoritative measured-event identity, in input order |
| `nominal/signed_weight [N]` | Original signed nominal weight |
| `process_id [N]` | Explicit process/sample identity |
| `parameters/names [D]` | Ordered semantic parameter names |
| `parameters/theta [N,D]` | Native generated parameter points |
| `parameters/grid [K,D]` | Reweighted parameter points |
| `parameters/signed_weights [N,K]` | Full signed weight for each point |
| `nuisance/<name> [N] or [N,L]` | Separate optional extra weights; definitions are in metadata |

Native mode uses `theta`; reweighted mode uses `grid` and `signed_weights`.
`metadata_only` explicitly represents a ROOT audit without established POI
coordinates. It cannot supply parameter-point training samples. The metadata
attribute records weight definitions, normalisation (including `unknown`),
source identity, source branches and ROOT schema/histogram metadata. Negative
weights are stored unchanged. Writes validate a temporary HDF5 before installing
it and refuse existing output paths.

`validate_join(measured_path)` requires identical EVENT_ID arrays **in the same
order**. Batch lookup subsequently gathers by exact validated identities, in
requested batch order; missing IDs, duplicates and reordered whole-file joins
fail. `sample_points` exposes theta, grid index, signed weight, sign, magnitude,
nominal weight and process identity for both modes.
`ReweightedRatioBatch` adds explicit target/reference membership against a
configured reference grid point and fits component masses on TRAIN IDs only.
The bounded runner saves those IDs, grid and masses in the existing Lightning
checkpoint and checks them on reload. Native samples expose the same theta and
weight interface; the study must define how its separate target/reference
populations are drawn before using the loss. The provided automatic runner
currently selects reweighted mode only.

## Actual ROOT findings

Audited directly:
`/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/user.leet.50321401._000001.output.root`.

- One TTree `reco`, 25,452 entries. Unique `eventNumber` (uint64), `runNumber`
  (uint32, 410000), `mcChannelNumber` (uint32, 601230).
- Scalar `weight_mc_NOSYS`: negative in 98 events, range [-164.816, 290.691].
  It is the nominal MC-weight branch; no combined generator x detector weight
  is silently constructed.
- Scalar nominal detector correction fields: `weight_pileup_NOSYS`,
  `weight_leptonSF_tight_NOSYS`, `weight_jvt_effSF_NOSYS`,
  `weight_ftag_effSF_GN2v01_Continuous_NOSYS`; also scalar
  `globalTriggerEffSF_NOSYS` and per-object lepton efficiency vectors.
  These are not a POI grid or an exported systematic variation ensemble.
- Scalar `weight_max_NOSYS`: 1,507 negatives, range [-1,1]. Its physics definition
  is unresolved. No reweight meaning is inferred from its name.
- `listOfSystematics` labels only `NOSYS`. `metadata` labels
  `fullsim`, `mc23a`, `601230`, `e8514`. The cut bookkeeper contains initial
  event count 210000, sum of weights 17963908, sum of squared weights 1566717568.
  Their relation to the selected event population/normalisation is not assumed.
- No generator weight vector, scale/PDF variation grid, alternative-generator
  weight set, weight-name metadata or established ME/BSM/POI reweights found.

The smoke sidecar stores the first 512 rows and the selected scalar extras;
extras remain metadata-labelled unresolved/correction fields in `nuisance/`,
not parameter weights. Its audit records scalar weight statistics for all
25,452 ROOT rows. This ROOT-only sidecar uses eventNumber as identity and joins
a new identity-only fixture. Existing converted JER datasets may instead use
source-partition/row EVENT_ID: supply an explicit ROOT-to-HDF5 identity mapping
before joining those datasets. No existing reconstruction HDF5 is mutated.

## Bounded commands

The ROOT adapter uses the optional `sbi` dependency group (`uproot`, `awkward`).
The estimator itself uses the existing PyTorch dependency.

From this worktree:

```bash
PYTHONPATH=src:. python scripts/validate_rise.py
PYTHONPATH=src python scripts/validate_sbi_weights.py
PYTHONPATH=src python scripts/root_to_sbi.py /path/to/input.root \
  --output results/experiments/rise/root_weights.h5 --events 512
PYTHONPATH=src python scripts/run_rise.py train \
  --config configs/research/rise_0_jer.yaml \
  --research configs/research/rise_0.yaml \
  --sidecar results/experiments/rise/development/toy_grid.h5 \
  --reference-point 0 --output results/experiments/rise/toy_train --steps 2
```

Set `signed_mixture: true` in a copy of the research YAML for the signed toy
sidecar. Use `predict` with `--checkpoint`, a fresh output directory and
`--batches 1` to export every theta in the sidecar grid for each validation
event. Development runners require explicit train/validation-only measured
HDF5, preserve production outputs and never submit jobs. The ROOT converter
accepts externally documented parameter metadata (including `weight_convention`
set to `absolute` or `relative_to_nominal`) and explicit identity mappings;
a vector branch alone never establishes parameter semantics.

Validation artifacts live in `results/experiments/rise/development/`: one-step
synthetic reconstruction pretraining, two steps per RISE mode, reload/prediction,
native and grid joins, ROOT smoke/audit, positive Gaussian toy (250 tiny head
steps) and signed toy (80 tiny head steps). Analytic signed-mixture algebra is
checked with negatives in **both** populations. These are software checks and
carry no collider sensitivity claim. Re-running validation refreshes its own
bounded checkpoints/reports.

## Limits before a physics study

Real POI parameters and corresponding signed weights are missing. The ROOT
nominal/correction weights cannot supply them. A generation/reweight audit and
canonical identity mapping are required. Signed head calibration is only
checked with toys: finite-sample reference positivity and component cancellation
must be evaluated on the real TRAIN/validation populations. All masses must
come from the training population with the same weight and sampling convention.
Native target/reference sampling for a selected real process remains a study
choice, not a guessed converter feature.
