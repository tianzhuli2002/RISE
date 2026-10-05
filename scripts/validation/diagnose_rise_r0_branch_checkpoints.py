"""Snapshot bilinear/residual branch scales on the fixed maintained R0 BCE."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from joint_event_reconstruction.sbi_data import ReweightedRatioBatch, SBISidecar
from scripts.validation.diagnose_rise_r0_neural import (
    branch_diagnostics,
    make_objective_population,
    model,
    reference_sample,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/experiments/rise_parameter_conditioning/r0.json"
CHECKPOINT_DIR = ROOT / "results/experiments/rise/parameter-conditioning-v1/R0/bilinear"
OUTPUT = ROOT / "results/experiments/rise/parameter-conditioning-v1/R0/bilinear/branch_diagnostics.json"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    sidecar = SBISidecar(ROOT / config["training_sidecar"])
    reference, grid, _weights = reference_sample(config, sidecar)
    adapter = ReweightedRatioBatch(
        sidecar,
        sidecar.event_id,
        reference_point=4,
        seed=int(config["seed"]) + 1,
    )
    population = make_objective_population(sidecar, adapter, reference.astype("float32"))
    rows = {}
    for step in (4000, 8000, 16000):
        checkpoint_path = CHECKPOINT_DIR / f"step_{step}.pt"
        checkpoint_bytes = checkpoint_path.read_bytes()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["step"] != step:
            raise ValueError(f"Checkpoint step metadata mismatch: {checkpoint_path}")
        head = model(config, int(config["seed"]))
        head.load_state_dict(checkpoint["state_dict"])
        values = branch_diagnostics(head, population)
        rows[str(step)] = {
            **values,
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "checkpoint_step": int(checkpoint["step"]),
            "diagnostic_population": "fixed exact R0 reference population, sidecar weights, balanced theta/class rows, maintained unsigned_loss",
        }
    write_json(
        OUTPUT,
        {
            "status": "completed",
            "training_performed": False,
            "theta_grid": grid.tolist(),
            "population_events": len(reference),
            "diagnostic_limit": 16384,
            "checkpoints": rows,
        },
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
