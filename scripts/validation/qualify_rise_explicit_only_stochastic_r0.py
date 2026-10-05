"""Qualify the reference-anchored explicit branch on the maintained R0 SBI path."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from joint_event_reconstruction.sbi_data import SBISidecar
from scripts.validation import qualify_rise_parameter_conditioning as qualification

ROOT = Path(__file__).resolve().parents[2]
BASE_OUTPUT = ROOT / "results/experiments/rise/parameter-conditioning-v1"
OUTPUT = BASE_OUTPUT / "R0-explicit-only-stochastic"


def main():
    torch.set_num_threads(1)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite explicit-only R0 result: {OUTPUT}")
    config = json.loads(qualification.CONFIG.read_text())
    if config["toy"] != "R0: Normal((a,b,a+b), I3)":
        raise ValueError("R0 config changed from the frozen Gaussian mean-shift toy")
    grid = np.array(
        [(a, b) for a in config["training_axis"] for b in config["training_axis"]]
    )
    rng = np.random.default_rng(config["seed"])
    rng.normal(size=(9 * 6000, 3))  # Keep the established native/reference stream.
    reference = rng.normal(size=(config["training_reference_events"], 3))
    sidecar = SBISidecar(ROOT / config["training_sidecar"])
    exact_weights = np.stack(
        [np.exp(qualification.exact_log_ratio(reference, theta, "R0")) for theta in grid],
        axis=1,
    )
    np.testing.assert_allclose(sidecar.weights, exact_weights, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(sidecar.grid, grid)

    qualification.OUT = OUTPUT
    results = qualification.fit(
        "bilinear_explicit_only",
        reference,
        sidecar,
        config,
        "R0",
        disable_residual=True,
    )
    selected_step = str(max(config["checkpoints"]))
    result = results[selected_step]
    decision = {
        "status": "completed",
        "verdict": "RISE_R0_PASS_BILINEAR_EXPLICIT_ONLY"
        if result["qualification_pass"]
        else "RISE_R0_BLOCKED",
        "toy": config["toy"],
        "head": "ReferenceAnchoredBilinearRatioHead",
        "residual_branch_enabled": False,
        "training_objective": "maintained ReweightedRatioBatch and unsigned_loss, unchanged",
        "training_steps": max(config["checkpoints"]),
        "training_seed": config["seed"],
        "theta_grid": grid.tolist(),
        "reference_data_sha256": hashlib.sha256(
            reference.astype("<f8").tobytes()
        ).hexdigest(),
        "sidecar_weights_verified_against_exact_ratio": True,
        "qualification_result": result,
        "result_directory": str(OUTPUT / "R0/bilinear_explicit_only"),
    }
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({key: decision[key] for key in (
        "verdict", "training_steps", "qualification_result"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
