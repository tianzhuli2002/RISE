"""Check RISE weighted-loss and gradient scales against population size."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from joint_event_reconstruction.rise import RatioHead, unsigned_loss
from joint_event_reconstruction.sbi_data import (
    ReweightedRatioBatch,
    SBISidecar,
    write_sidecar,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261002
POPULATION_SIZE = 4096
BATCH_SIZE = 1024
GRID = np.asarray([[0.0, 0.0], [0.4, -0.2]], dtype=np.float64)


def make_sidecar(path, event_ids, x, weights):
    write_sidecar(
        path,
        event_id=event_ids,
        nominal_weight=np.ones(len(event_ids)),
        process_id=np.full(len(event_ids), "loss_scale_toy", dtype="U20"),
        metadata={
            "weight_definition": "exact Gaussian importance ratio",
            "source_identity": "deterministically tiled Gaussian toy",
            "normalisation": "unit target and reference area",
            "weight_convention": "shape_only",
        },
        parameter_grid=GRID,
        parameter_names=("a", "b"),
        signed_weights=weights,
    )


def gradient_norm(module):
    squares = [
        parameter.grad.detach().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.sqrt(torch.stack(squares).sum()))


def evaluate(sidecar_path, event_ids, x, batch_ids, head, shared):
    sidecar = SBISidecar(sidecar_path)
    adapter = ReweightedRatioBatch(
        sidecar, event_ids, reference_point=0, seed=SEED + 1
    )
    batch = {
        "tokens": torch.ones((len(batch_ids), 1), dtype=torch.float32),
        "event_index": torch.as_tensor(batch_ids, dtype=torch.long),
    }
    adapted = adapter(batch)
    rows = sidecar.positions(batch_ids)
    features = torch.as_tensor(x[rows], dtype=torch.float32)

    def get_loss(normalisation):
        head.zero_grad(set_to_none=True)
        shared.zero_grad(set_to_none=True)
        logits = head(shared(features), adapted["theta"])
        loss = unsigned_loss(
            logits,
            adapted["membership"],
            adapted["signed_weight"],
            normalisation,
        )
        loss.backward()
        return {
            "loss": float(loss.detach()),
            "shared_gradient_norm": gradient_norm(shared),
            "ratio_head_gradient_norm": gradient_norm(head),
        }

    corrected = get_loss(adapted["class_normalisation"])
    legacy_mass = adapter.masses[
        adapted["parameter_point"].cpu().numpy()
    ][:, [0, 2]]
    legacy = get_loss(torch.as_tensor(legacy_mass, dtype=torch.float32))
    return corrected, legacy


def main():
    output = Path(
        "results/experiments/rise/preproduction-v1/analytic-run2/loss_scaling.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    x_base = rng.normal(size=(POPULATION_SIZE, 3))
    mu = np.asarray([GRID[1, 0], GRID[1, 1], GRID[1].sum()])
    target_weights = np.exp(x_base @ mu - 0.5 * mu @ mu)
    weights_base = np.column_stack((np.ones(POPULATION_SIZE), target_weights))
    x_large = np.tile(x_base, (4, 1))
    weights_large = np.tile(weights_base, (4, 1))
    ids_small = np.arange(1, POPULATION_SIZE + 1, dtype=np.int64)
    ids_large = np.arange(1, 4 * POPULATION_SIZE + 1, dtype=np.int64)
    small_path = output.parent / "loss_scale_n.h5"
    large_path = output.parent / "loss_scale_4n.h5"
    make_sidecar(small_path, ids_small, x_base, weights_base)
    make_sidecar(large_path, ids_large, x_large, weights_large)

    shared = nn.Linear(3, 8)
    head = RatioHead(8, 2, 16)
    batch_ids = ids_small[:BATCH_SIZE]
    corrected_n, legacy_n = evaluate(
        small_path, ids_small, x_base, batch_ids, head, shared
    )
    corrected_4n, legacy_4n = evaluate(
        large_path, ids_large, x_large, batch_ids, head, shared
    )
    result = {
        "population_sizes": [POPULATION_SIZE, 4 * POPULATION_SIZE],
        "batch_size": BATCH_SIZE,
        "same_underlying_population": True,
        "same_initial_model_state": True,
        "corrected": {"N": corrected_n, "4N": corrected_4n},
        "corrected_4n_over_n": {
            key: corrected_4n[key] / corrected_n[key]
            for key in corrected_n
        },
        "legacy_full_total_denominator": {"N": legacy_n, "4N": legacy_4n},
        "legacy_4n_over_n": {
            key: legacy_4n[key] / legacy_n[key] for key in legacy_n
        },
        "shared_gradient_represents": "trainable event projection before RatioHead",
        "reconstruction_gradient_comparison": "not available without a measured/reconstruction batch",
    }
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
