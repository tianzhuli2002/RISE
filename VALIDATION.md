# Bounded implementation checks

Source commit: `ec54f6b13817009b5ec0472e084a034f044980a4`. CPU only; sealed_test_used=false.

```json
{
  "toy": {
    "gaussian_log_ratio_correlation": 0.9992618560791016,
    "gaussian_log_ratio_rmse": 0.20456326007843018,
    "trained_theta_response": 1.4650508165359497,
    "signed_target_and_reference_algebra": "PASS",
    "signed_training_steps": 80,
    "unsigned_training_steps": 250
  },
  "real_root_smoke": "PASS",
  "0": {
    "finite_loss": 0.20863330364227295,
    "encoder_gradient": 0.21666253915494357,
    "auxiliary_encoder_gradient": null,
    "truth_free_prediction": true,
    "train_steps": 2,
    "reload_prediction": true
  },
  "P": {
    "finite_loss": 0.11397908627986908,
    "encoder_gradient": 0.05326828470035075,
    "auxiliary_encoder_gradient": null,
    "truth_free_prediction": true,
    "train_steps": 2,
    "reload_prediction": true
  },
  "A": {
    "finite_loss": 5.380236625671387,
    "encoder_gradient": 1.1865798987911544,
    "auxiliary_encoder_gradient": 1.1349529229876516,
    "truth_free_prediction": true,
    "train_steps": 2,
    "reload_prediction": true
  },
  "source_commit": "ec54f6b13817009b5ec0472e084a034f044980a4",
  "branch": "research/rise-implementation-v1",
  "worktree": "/net/scratch/w00238tl/JointEventReconstruction_RISE",
  "sealed_test_used": false,
  "population": "32 synthetic events, 24 train / 8 validation; ROOT metadata audit only",
  "rise_end_to_end": "PASS",
  "configs": {
    "configs/research/rise_a.yaml": "2767b79221ea5bb8cf06b75bdab46184960fdaac54bfb553c401408989240d8d",
    "configs/research/rise_0.yaml": "7488f56dc94bdc43d41c1e88ab9bec8b4a9e4472b13fcbe6d44c5f66cfd1791f",
    "configs/research/rise_p_jer.yaml": "29ab801a701815954a6b7117c0f18e3bfda4c1cf995f291798c3c233642bc30e",
    "configs/research/rise_p.yaml": "37e2aa791fb9dc145bcf6ea4581730a5b6a859ec4ad262c054a954d3229dd0e8",
    "configs/research/rise_a_jer.yaml": "f3ab377cf12032562b259f42559e03be24a802b16d1190a838f23ce96bec6e5f",
    "configs/research/rise_0_jer.yaml": "900d9be90b77965639f80a8b940a01d1854982c96c7a95fd2ee795f7bf6be458"
  },
  "inputs": {
    "/net/scratch/w00238tl/JointEventReconstruction_RISE/results/experiments/rise/development/measured.h5": "b8d5db2be095236f601ce32c0fe7a01a312c656476b64fed60aef27cc5478b99",
    "/net/scratch/w00238tl/JointEventReconstruction_RISE/results/experiments/rise/development/parents.h5": "abe0872e3baa6a8800bdf5dbdaecc9b4ebf13db36abd1c8abf404d83699186da"
  },
  "targeted_existing_scaler_tests": "maintained loader exercised by RISE-P",
  "cli_smoke": "PASS: bounded train and measured-only reload prediction",
  "pretrained_full_reload_self_contained": "PASS",
  "syntax_import_config": "PASS",
  "ruff": "PASS",
  "scalar_vector_weight_adapter": "PASS"
}
```

No collider performance or parameter-constraint comparison was run.
