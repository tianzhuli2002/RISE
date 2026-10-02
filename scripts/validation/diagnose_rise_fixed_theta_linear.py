"""Test the RISE reweighted SBI path with an exactly representable fixed-theta ratio."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from joint_event_reconstruction.rise import unsigned_loss
from joint_event_reconstruction.sbi_data import ReweightedRatioBatch, SBISidecar
from scripts.validation.validate_rise_analytic_closure import (
    BATCH_SIZE,
    LEARNING_RATE,
    NATIVE_EVENTS_PER_POINT,
    REWEIGHTED_EVENTS,
    SEED,
    TRAIN_STEPS,
    exact_log_ratio,
    make_grid,
    mean,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/experiments/rise/preproduction-v1/analytic-run2"
OUTPUT = RUN / "fixed_theta_linear_closure.json"
THETA_INDEX = 7  # Interior training-grid point (0.8, 0.0).
REFERENCE_POINT = 4


def training_reference_sample():
    rng = np.random.default_rng(SEED)
    grid = make_grid()
    for theta in grid:
        rng.normal(size=(NATIVE_EVENTS_PER_POINT, 3)) + mean(theta)
    return rng.normal(size=(REWEIGHTED_EVENTS, 3))


def torch_float(value):
    return torch.as_tensor(value, dtype=torch.float32)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    torch.set_num_threads(1)
    sidecar = SBISidecar(RUN / "reweighted_reference.h5")
    grid = sidecar.grid
    theta = np.asarray(grid[THETA_INDEX], dtype=np.float64)
    mu = mean(theta)
    exact_w = mu
    exact_b = -0.5 * float(np.dot(mu, mu))
    x_reference = training_reference_sample()
    if len(x_reference) != len(sidecar.event_id):
        raise ValueError("Recreated Gaussian reference events do not match sidecar EVENT_ID rows")

    torch.manual_seed(SEED + 80)
    adapter = ReweightedRatioBatch(
        sidecar,
        sidecar.event_id,
        reference_point=REFERENCE_POINT,
        seed=SEED + 81,
    )
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sample_rng = np.random.default_rng(SEED + 82)
    last_loss = None
    for _ in range(TRAIN_STEPS):
        rows = sample_rng.choice(len(sidecar.event_id), size=BATCH_SIZE, replace=False)
        ids = sidecar.event_id[rows]
        batch = {
            "tokens": torch.ones((BATCH_SIZE, 1), dtype=torch.float32),
            "event_index": torch.as_tensor(ids, dtype=torch.long),
        }
        adapted = adapter(batch)
        membership = adapted["membership"]
        positions = sidecar.positions(ids)
        selected_points = np.where(
            membership.cpu().numpy().astype(np.int64) == 1,
            THETA_INDEX,
            REFERENCE_POINT,
        )
        signed_weight = torch_float(sidecar.weights[positions, selected_points])
        class_norm = torch_float(
            np.broadcast_to(
                adapter.masses[THETA_INDEX, [0, 2]] / adapter.population_size,
                (BATCH_SIZE, 2),
            ).copy()
        )
        logits = model(torch_float(x_reference[positions]))[:, 0]
        loss = unsigned_loss(logits, membership, signed_weight, class_norm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())

    learned_w = model.weight.detach().cpu().numpy()[0].astype(np.float64)
    learned_b = float(model.bias.detach().cpu())
    rng = np.random.default_rng(SEED + 83)
    reference_eval = rng.normal(size=(24000, 3))
    target_eval = rng.normal(size=(24000, 3)) + mu
    evaluation = np.concatenate((reference_eval, target_eval))
    exact = exact_log_ratio(evaluation, theta)
    learned = evaluation @ learned_w + learned_b
    log_ratio_rmse = float(np.sqrt(np.mean((learned - exact) ** 2)))
    log_ratio_bias = float(np.mean(learned - exact))
    reference_weights = np.exp(reference_eval @ learned_w + learned_b)
    normalization_mean = float(reference_weights.mean())
    normalization_se = float(reference_weights.std(ddof=1) / np.sqrt(len(reference_weights)))

    # At this fixed theta, the likelihood scan has the two known hypotheses
    # {reference, theta*}; the fixed-theta classifier is not extrapolated in theta.
    dataset_rng = np.random.default_rng(SEED + 84)
    exact_wins = 0
    learned_wins = 0
    exact_scores = []
    learned_scores = []
    datasets = []
    for index in range(40):
        x = dataset_rng.normal(size=(200, 3)) + mu
        exact_score = float(exact_log_ratio(x, theta).sum())
        learned_score = float((x @ learned_w + learned_b).sum())
        exact_scores.append(exact_score)
        learned_scores.append(learned_score)
        exact_fit = theta.tolist() if exact_score > 0.0 else [0.0, 0.0]
        learned_fit = theta.tolist() if learned_score > 0.0 else [0.0, 0.0]
        exact_wins += exact_score > 0.0
        learned_wins += learned_score > 0.0
        datasets.append(
            {
                "experiment": index,
                "exact_log_likelihood_ratio": exact_score,
                "learned_log_likelihood_ratio": learned_score,
                "exact_best_fit_theta": exact_fit,
                "learned_best_fit_theta": learned_fit,
            }
        )

    result = {
        "diagnostic": "fixed_theta_linear_logistic_using_ReweightedRatioBatch_and_unsigned_loss",
        "theta_star": theta.tolist(),
        "exact_coefficients": {"w": exact_w.tolist(), "b": exact_b},
        "learned_coefficients": {"w": learned_w.tolist(), "b": learned_b},
        "optimization": {
            "steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "final_loss": last_loss,
            "reference_point_index": REFERENCE_POINT,
            "fixed_target_grid_index": THETA_INDEX,
            "training_population_size": int(adapter.population_size),
        },
        "ratio_closure": {
            "log_ratio_rmse": log_ratio_rmse,
            "log_ratio_bias": log_ratio_bias,
            "E_p0_r_hat": normalization_mean,
            "E_p0_r_hat_mc_se": normalization_se,
            "normalization_pull_from_one": (normalization_mean - 1.0) / normalization_se,
            "evaluation_reference_and_target_events_each": 24000,
        },
        "dataset_likelihood": {
            "scan": "two hypotheses: theta0 and fixed theta_star",
            "pseudo_datasets": datasets,
            "exact_best_fit_theta_star_fraction": float(exact_wins / len(datasets)),
            "learned_best_fit_theta_star_fraction": float(learned_wins / len(datasets)),
            "exact_mean_log_likelihood_ratio": float(np.mean(exact_scores)),
            "learned_mean_log_likelihood_ratio": float(np.mean(learned_scores)),
        },
        "sealed_test_used": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(OUTPUT),
        "theta_star": result["theta_star"],
        "exact_coefficients": result["exact_coefficients"],
        "learned_coefficients": result["learned_coefficients"],
        "ratio_closure": result["ratio_closure"],
        "dataset_likelihood_summary": {
            key: value
            for key, value in result["dataset_likelihood"].items()
            if key != "pseudo_datasets"
        },
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
