"""One theta/class-stratified stochastic control for the existing R0 bilinear head."""

import hashlib
import json
import time

import numpy as np
import torch

from joint_event_reconstruction.rise import (
    ReferenceAnchoredBilinearRatioHead,
    unsigned_loss,
)
from joint_event_reconstruction.sbi_data import ReweightedRatioBatch, SBISidecar
from scripts.validation.qualify_rise_parameter_conditioning import (
    CONFIG,
    OUT,
    ROOT,
    branch_snapshot,
    diagnostics,
    exact_log_ratio,
    save,
)


def balanced_training_batch(sidecar, adapter, event_rows, grid, config, rng):
    """Keep the same population risk while fixing theta/class counts per batch."""
    batch_size = config["batch_size"]
    cells = np.asarray(
        [(point, membership) for point in range(len(grid)) for membership in (0, 1)],
        dtype=np.int64,
    )
    quotient, remainder = divmod(batch_size, len(cells))
    labels = np.repeat(cells, quotient, axis=0)
    if remainder:
        labels = np.concatenate((labels, cells[:remainder]), axis=0)
    labels = labels[rng.permutation(batch_size)]
    points, membership = labels.T
    cell_index = 2 * points + membership
    cell_counts = np.bincount(cell_index, minlength=len(cells))
    if cell_counts.max() - cell_counts.min() > 1 or np.count_nonzero(membership) * 2 != batch_size:
        raise AssertionError("Theta/class batch strata are not balanced")

    ids = sidecar.event_id[event_rows]
    rows = sidecar.positions(ids)
    signed_weight = sidecar.weights[
        rows, np.where(membership == 1, points, adapter.reference_point)
    ]
    population_masses = adapter.masses[points]
    per_event_masses = population_masses / adapter.population_size
    # Reweight 28/29-row cells so the minibatch mean is exactly the same
    # uniform-theta, equal-class empirical risk as independent row sampling.
    sampling_correction = batch_size / (len(cells) * cell_counts[cell_index])
    np.testing.assert_allclose(
        np.bincount(cell_index, weights=sampling_correction, minlength=len(cells)),
        batch_size / len(cells),
        rtol=0,
        atol=1e-12,
    )

    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32)

    return {
        "theta": tensor(grid[points]),
        "membership": tensor(membership),
        "signed_weight": tensor(signed_weight * sampling_correction),
        "class_normalisation": tensor(per_event_masses[:, [0, 2]]),
    }


def main():
    torch.set_num_threads(1)
    config = json.loads(CONFIG.read_text())
    config["theta_class_sampling"] = (
        "each batch is stratified across all 9 training theta points and both "
        "membership classes, with 28 or 29 rows per cell and inverse-cell-count "
        "correction preserving the unchanged uniform-theta, equal-class risk"
    )
    grid = np.asarray(
        [(a, b) for a in config["training_axis"] for b in config["training_axis"]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(config["seed"])
    # Preserve the established analytic-run2 RNG sequence and reference sample.
    rng.normal(size=(9 * 6000, 3))
    reference = rng.normal(size=(config["training_reference_events"], 3))
    reference_hash = hashlib.sha256(reference.astype("<f8").tobytes()).hexdigest()

    sidecar = SBISidecar(ROOT / config["training_sidecar"])
    exact_weights = np.stack(
        [np.exp(exact_log_ratio(reference, theta, "R0")) for theta in grid], axis=1
    )
    np.testing.assert_allclose(sidecar.weights, exact_weights, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(sidecar.grid, grid)

    torch.manual_seed(config["seed"])
    head = ReferenceAnchoredBilinearRatioHead(
        3, 2, config["hidden"], config["reference_theta"]
    )
    adapter = ReweightedRatioBatch(
        sidecar,
        sidecar.event_id,
        reference_point=4,
        seed=config["seed"] + 1,
    )
    event_rng = np.random.default_rng(config["seed"] + 3)
    cell_rng = np.random.default_rng(config["seed"] + 7001)
    optimizer = torch.optim.Adam(head.parameters(), lr=config["learning_rate"])

    output = OUT / "R0-theta-balanced-exact-risk"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_steps = set(config["checkpoints"])
    history, checkpoints = [], {}
    start = time.perf_counter()
    for step in range(1, max(checkpoint_steps) + 1):
        event_rows = event_rng.choice(len(reference), size=config["batch_size"], replace=False)
        batch = balanced_training_batch(
            sidecar, adapter, event_rows, grid, config, cell_rng
        )
        log_ratio = head(torch.as_tensor(reference[event_rows], dtype=torch.float32), batch["theta"])
        loss = unsigned_loss(
            log_ratio,
            batch["membership"],
            batch["signed_weight"],
            batch["class_normalisation"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss) or any(
            not torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
            if parameter.grad is not None
        ):
            raise FloatingPointError(f"Nonfinite theta-balanced R0 optimization at step {step}")
        optimizer.step()
        if step % 500 == 0:
            history.append(
                {"step": step, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - start}
            )
        if step in checkpoint_steps:
            torch.save(
                {"state_dict": head.state_dict(), "step": step, "config": config},
                output / f"step_{step}.pt",
            )
            result = diagnostics(head, reference, config, "R0")
            result.update(
                {
                    "step": step,
                    "parameter_count": sum(p.numel() for p in head.parameters()),
                    "trainable_parameter_count": sum(
                        p.numel() for p in head.parameters() if p.requires_grad
                    ),
                    "theta_class_sampling": config["theta_class_sampling"],
                    "wall_seconds": time.perf_counter() - start,
                }
            )
            branch = branch_snapshot(head, reference, sidecar, config)
            if branch is not None:
                result["branch_snapshot"] = branch
            save(output / f"step_{step}.json", result)
            checkpoints[str(step)] = result
            save(output / "history.json", history)
            print(f"R0 theta-balanced step={step} pass={result['qualification_pass']}", flush=True)

    passed = checkpoints[str(max(checkpoint_steps))]["qualification_pass"]
    save(output / "comparison.json", checkpoints)
    save(
        output / "decision.json",
        {
            "verdict": "RISE_R0_PASS_BILINEAR_HEAD" if passed else "RISE_R0_BLOCKED",
            "selected_head": "bilinear" if passed else None,
            "training_reference_sha256": reference_hash,
            "sidecar_sha256_verified_against_exact_ratio": True,
            "seed": config["seed"],
            "optimizer": config["optimizer"],
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "training_steps": max(checkpoint_steps),
            "theta_grid": grid.tolist(),
            "theta_class_sampling": config["theta_class_sampling"],
            "qualification": config["qualification"],
            "qualification_pass": bool(passed),
            "checkpoints": {step: f"step_{step}.json" for step in sorted(checkpoints, key=int)},
        },
    )
    print(f"Finished R0 theta-balanced pass={passed} wall={time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
