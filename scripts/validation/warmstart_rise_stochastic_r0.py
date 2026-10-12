"""One unchanged-objective stochastic R0 run from the passing deterministic fit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from joint_event_reconstruction.rise import (
    ReferenceAnchoredBilinearRatioHead,
    unsigned_loss,
)
from joint_event_reconstruction.sbi_data import ReweightedRatioBatch, SBISidecar
from scripts.validation.diagnose_rise_r0_neural import reference_sample, write_json
from scripts.validation.qualify_rise_parameter_conditioning import diagnostics

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/experiments/rise_parameter_conditioning/r0.json"
DETERMINISTIC = ROOT / "results/experiments/rise/r0-neural-diagnosis-v1/complete-02/deterministic_seed_20261403.pt"
OUTPUT = ROOT / "results/experiments/rise/r0-neural-diagnosis-v1/complete-02/warmstart_stochastic"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite warm-start R0 result: {output}")
    if not DETERMINISTIC.is_file():
        raise FileNotFoundError(f"Passing deterministic checkpoint is missing: {DETERMINISTIC}")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    seed = int(config["seed"])
    sidecar = SBISidecar(ROOT / config["training_sidecar"])
    rng_data = np.random.default_rng(seed)
    rng_data.normal(size=(9 * 6000, 3))
    reference = rng_data.normal(size=(int(config["training_reference_events"]), 3))
    grid = np.asarray(
        [(a, b) for a in config["training_axis"] for b in config["training_axis"]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(sidecar.grid, grid)
    _, expected_grid, _ = reference_sample(config, sidecar)
    np.testing.assert_array_equal(expected_grid, grid)

    head = ReferenceAnchoredBilinearRatioHead(
        3, 2, int(config["hidden"]), config["reference_theta"]
    )
    checkpoint = torch.load(DETERMINISTIC, map_location="cpu", weights_only=False)
    head.load_state_dict(checkpoint)
    adapter = ReweightedRatioBatch(
        sidecar, sidecar.event_id, reference_point=4, seed=seed + 1
    )
    learning_rate = float(config["learning_rate"] if args.learning_rate is None else args.learning_rate)
    if learning_rate <= 0:
        parser.error("learning rate must be positive")
    batch_size = int(config["batch_size"] if args.batch_size is None else args.batch_size)
    if batch_size <= 0 or batch_size > len(reference):
        parser.error(f"batch size must be in [1, {len(reference)}]")
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed + 3)
    checkpoints = set(map(int, config["checkpoints"]))
    history, results = [], {}
    start = time.perf_counter()
    resolved_config = {
        **config,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "initialization": str(DETERMINISTIC.relative_to(ROOT)),
        "objective": "maintained ReweightedRatioBatch and unsigned_loss, unchanged",
    }
    write_json(output / "config.json", resolved_config)
    for step in range(1, max(checkpoints) + 1):
        rows = rng.choice(len(reference), size=batch_size, replace=False)
        batch = adapter(
            {
                "tokens": torch.ones((len(rows), 1)),
                "event_index": torch.as_tensor(sidecar.event_id[rows]),
            }
        )
        logits = head(torch.as_tensor(reference[rows], dtype=torch.float32), batch["theta"])
        loss = unsigned_loss(
            logits,
            batch["membership"],
            batch["signed_weight"],
            batch["class_normalisation"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss) or any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
        ):
            raise FloatingPointError(f"Nonfinite warm-start stochastic BCE at step {step}")
        optimizer.step()
        if step % 1000 == 0:
            history.append({"step": step, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - start})
        if step in checkpoints:
            record = diagnostics(head, reference, config, "R0")
            record.update(
                {
                    "step": step,
                    "initialization": str(DETERMINISTIC.relative_to(ROOT)),
                    "optimizer": "Adam minibatch; maintained ReweightedRatioBatch and unsigned_loss unchanged",
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": 1,
                    "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
                    "wall_seconds": time.perf_counter() - start,
                }
            )
            results[str(step)] = record
            record["learning_rate"] = learning_rate
            torch.save(head.state_dict(), output / f"step_{step}.pt")
            write_json(output / f"step_{step}.json", record)
            write_json(output / "history.json", history)
            print(
                json.dumps(
                    {"step": step, "qualification_pass": record["qualification_pass"], "wall_seconds": record["wall_seconds"]}
                ),
                flush=True,
            )
    write_json(
        output / "comparison.json",
        {
            "status": "completed",
            "purpose": "test whether standard minibatch optimization preserves the passing deterministic full-batch solution",
            "data_sidecar": str(Path(config["training_sidecar"])),
            "initialization": str(DETERMINISTIC.relative_to(ROOT)),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "results": results,
            "training_history": history,
        },
    )


if __name__ == "__main__":
    main()
