"""Measure one fixed R0 minibatch-gradient ensemble at the passing BCE fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from joint_event_reconstruction.rise import (
    ReferenceAnchoredBilinearRatioHead,
    unsigned_loss,
)
from joint_event_reconstruction.sbi_data import ReweightedRatioBatch, SBISidecar
from scripts.validation.diagnose_rise_r0_neural import reference_sample, write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/experiments/rise_parameter_conditioning/r0.json"
INITIAL_STATE = ROOT / "results/experiments/rise/r0-neural-diagnosis-v1/complete-02/deterministic_seed_20261403.pt"
OUTPUT = ROOT / "results/experiments/rise/r0-neural-diagnosis-v1/complete-02/minibatch_gradient_noise.json"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    torch.set_num_threads(1)
    sidecar = SBISidecar(ROOT / config["training_sidecar"])
    rng_data = np.random.default_rng(int(config["seed"]))
    rng_data.normal(size=(9 * 6000, 3))
    reference = rng_data.normal(size=(int(config["training_reference_events"]), 3)).astype(np.float32)
    np.testing.assert_array_equal(
        sidecar.grid,
        np.asarray([(a, b) for a in config["training_axis"] for b in config["training_axis"]]),
    )
    reference_sample(config, sidecar)
    head = ReferenceAnchoredBilinearRatioHead(3, 2, int(config["hidden"]), config["reference_theta"])
    head.load_state_dict(torch.load(INITIAL_STATE, map_location="cpu", weights_only=False))
    adapter = ReweightedRatioBatch(
        sidecar, sidecar.event_id, reference_point=4, seed=int(config["seed"]) + 1
    )
    rng = np.random.default_rng(int(config["seed"]) + 9901)
    vectors, rows, losses, theta_counts, class_counts = [], [], [], [], []
    for _ in range(128):
        selected = rng.choice(len(reference), size=int(config["batch_size"]), replace=False)
        batch = adapter(
            {
                "tokens": torch.ones((len(selected), 1)),
                "event_index": torch.as_tensor(sidecar.event_id[selected]),
            }
        )
        x = torch.as_tensor(reference[selected], dtype=torch.float32)
        loss = unsigned_loss(
            head(x, batch["theta"]),
            batch["membership"],
            batch["signed_weight"],
            batch["class_normalisation"],
        )
        head.zero_grad(set_to_none=True)
        loss.backward()
        flat = torch.cat(
            [parameter.grad.detach().flatten() for parameter in head.parameters() if parameter.grad is not None]
        )
        vectors.append(flat.numpy().copy())
        rows.append(float(torch.linalg.vector_norm(flat)))
        losses.append(float(loss.detach()))
        theta_counts.append(np.bincount(batch["parameter_point"].numpy(), minlength=len(sidecar.grid)))
        class_counts.append(np.bincount(batch["membership"].to(torch.int64).numpy(), minlength=2))
    gradients = np.stack(vectors)
    gradient_mean = gradients.mean(axis=0)
    component_variance = gradients.var(axis=0, ddof=1)
    write_json(
        OUTPUT,
        {
            "status": "completed",
            "training_performed": False,
            "objective": "maintained reweighted unsigned_loss at the saved passing deterministic bilinear state",
            "gradient_batches": len(gradients),
            "batch_size": int(config["batch_size"]),
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses, ddof=1)),
            "gradient_norm_mean": float(np.mean(rows)),
            "gradient_norm_std": float(np.std(rows, ddof=1)),
            "gradient_norm_coefficient_of_variation": float(np.std(rows, ddof=1) / np.mean(rows)),
            "mean_gradient_norm": float(np.linalg.norm(gradient_mean)),
            "rms_noise_norm": float(np.sqrt(component_variance.sum())),
            "mean_to_noise_norm_ratio": float(np.linalg.norm(gradient_mean) / np.sqrt(component_variance.sum())),
            "theta_count_mean_by_grid_point": np.stack(theta_counts).mean(0).tolist(),
            "theta_count_sd_by_grid_point": np.stack(theta_counts).std(0, ddof=1).tolist(),
            "positive_class_count_mean": float(np.stack(class_counts)[:, 1].mean()),
            "positive_class_count_sd": float(np.stack(class_counts)[:, 1].std(ddof=1)),
            "reference_class_count_mean": float(np.stack(class_counts)[:, 0].mean()),
            "reference_class_count_sd": float(np.stack(class_counts)[:, 0].std(ddof=1)),
            "learning_rate_used_by_failed_warmstart": float(config["learning_rate"]),
            "checkpoint": str(INITIAL_STATE.relative_to(ROOT)),
        },
    )
    print(OUTPUT.read_text(), flush=True)


if __name__ == "__main__":
    main()
