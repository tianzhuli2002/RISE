"""Rerun the frozen stochastic bilinear R0 check with a zero residual output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import scripts.validation.qualify_rise_parameter_conditioning as qualification
from joint_event_reconstruction.sbi_data import SBISidecar
from scripts.validation.qualify_rise_parameter_conditioning import exact_log_ratio

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/experiments/rise_parameter_conditioning/r0.json"
OUTPUT = ROOT / "results/experiments/rise/r0-neural-diagnosis-v1/zero-residual-output"


def main():
    config = json.loads(CONFIG_PATH.read_text())
    sidecar = SBISidecar(ROOT / config["training_sidecar"])
    rng = np.random.default_rng(config["seed"])
    rng.normal(size=(9 * 6000, 3))
    reference = rng.normal(size=(config["training_reference_events"], 3))
    grid = np.asarray(
        [(a, b) for a in config["training_axis"] for b in config["training_axis"]]
    )
    np.testing.assert_array_equal(sidecar.grid, grid)
    exact_weights = np.stack(
        [np.exp(exact_log_ratio(reference, theta, "R0")) for theta in grid], axis=1
    )
    np.testing.assert_allclose(sidecar.weights, exact_weights, rtol=1e-12, atol=1e-12)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite RISE zero-residual R0 result: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    qualification.OUT = OUTPUT
    torch.set_num_threads(1)
    result = qualification.fit("bilinear", reference, sidecar, config, "R0")
    final = result[str(max(config["checkpoints"]))]
    decision = {
        "head": "ReferenceAnchoredBilinearRatioHead",
        "correction": "zero-initialize the residual output linear layer; objective, data, theta grid, seed, and Adam settings unchanged",
        "reference_population_sha256": hashlib.sha256(
            reference.astype("<f8").tobytes()
        ).hexdigest(),
        "training_sidecar_sha256": hashlib.sha256(
            (ROOT / config["training_sidecar"]).read_bytes()
        ).hexdigest(),
        "training_checkpoints": result,
        "RISE_NEURAL_R0": "PASS" if final["qualification_pass"] else "FAIL",
        "RISE_R1_GATE_MET": bool(final["qualification_pass"]),
    }
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n"
    )
    heldout = [
        row for row in final["ratio_points"] if row["parameter_set"] == "heldout"
    ]
    training = [
        row
        for row in final["ratio_points"]
        if row["parameter_set"] == "training" and row["theta"] != config["reference_theta"]
    ]
    print(
        json.dumps(
            {
                "RISE_NEURAL_R0": decision["RISE_NEURAL_R0"],
                "16k_qualification": final["qualification_pass"],
                "training_theta_log_ratio_RMSE": {
                    str(row["theta"]): row["log_ratio_RMSE"] for row in training
                },
                "heldout_theta_log_ratio_RMSE": {
                    str(row["theta"]): row["log_ratio_RMSE"] for row in heldout
                },
                "heldout_E_p0_rhat": {
                    str(row["theta"]): row["E_p0_rhat"] for row in heldout
                },
                "likelihood_summary": final["likelihood"]["summary"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
