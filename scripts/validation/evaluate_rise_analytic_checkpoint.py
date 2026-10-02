"""Add normalization uncertainty and exact likelihood comparisons to a RISE toy run."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from joint_event_reconstruction.rise import RatioHead

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261002
TRAIN_GRID_AXIS = (-0.8, 0.0, 0.8)
HELDOUT_THETA = ((0.35, -0.4), (-0.3, 0.45), (1.0, -0.5))
THETA_DOMAIN = (-1.0, 1.0)
HIDDEN = 64
EVALUATION_EVENTS = 12000


def make_grid():
    return np.asarray(
        [(a, b) for a in TRAIN_GRID_AXIS for b in TRAIN_GRID_AXIS], dtype=np.float64
    )


def mean(theta):
    theta = np.asarray(theta, dtype=np.float64)
    return np.stack((theta[..., 0], theta[..., 1], theta[..., 0] + theta[..., 1]), axis=-1)


def exact_log_ratio(x, theta):
    mu = mean(theta)
    return np.sum(mu * x, axis=-1) - 0.5 * np.sum(mu * mu, axis=-1)


def predict(head, x, theta, chunk_size=8192):
    values = []
    theta = np.asarray(theta, dtype=np.float64)
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        values.append(
            head(
                torch.as_tensor(x[start:stop], dtype=torch.float32),
                torch.as_tensor(theta[start:stop], dtype=torch.float32),
            )
            .detach()
            .cpu()
            .numpy()
        )
    return np.concatenate(values)


def ratio_metrics(head, points, rng, label):
    rows = []
    for theta in points:
        theta = np.asarray(theta, dtype=np.float64)
        reference = rng.normal(size=(EVALUATION_EVENTS, 3))
        target = rng.normal(size=(EVALUATION_EVENTS, 3)) + mean(theta)
        x = np.concatenate((reference, target))
        truth = exact_log_ratio(x, theta)
        prediction = predict(
            head,
            x,
            np.broadcast_to(theta, x.shape[:1] + theta.shape),
        )
        if np.var(truth) == 0:
            correlation = "N/A"
        else:
            correlation = float(np.corrcoef(truth, prediction)[0, 1])

        exact = exact_log_ratio(reference, theta)
        predicted = predict(
            head,
            reference,
            np.broadcast_to(theta, (len(reference), len(theta))),
        )
        exact_weights = np.exp(exact)
        predicted_weights = np.exp(predicted)
        exact_mean = float(exact_weights.mean())
        exact_se = float(exact_weights.std(ddof=1) / np.sqrt(len(exact_weights)))
        predicted_mean = float(predicted_weights.mean())
        predicted_se = float(
            predicted_weights.std(ddof=1) / np.sqrt(len(predicted_weights))
        )
        weighted_observable = np.average(reference, axis=0, weights=predicted_weights)
        target_mean = mean(theta)
        rows.append(
            {
                "theta": theta.tolist(),
                "parameter_set": label,
                "log_ratio_bias": float(np.mean(prediction - truth)),
                "log_ratio_rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                "log_ratio_correlation": correlation,
                "E_reference_r_hat": predicted_mean,
                "E_reference_r_hat_mc_se": predicted_se,
                "normalization_pull_from_one": (predicted_mean - 1.0) / predicted_se,
                "E_reference_r_exact_mc": exact_mean,
                "E_reference_r_exact_mc_se": exact_se,
                "reweighted_observable_mean": weighted_observable.tolist(),
                "exact_observable_mean": target_mean.tolist(),
                "reweighted_observable_mean_error": float(
                    np.linalg.norm(weighted_observable - target_mean)
                ),
            }
        )
    return rows


def surface(head, x, scan):
    n_events = len(x)
    repeated_x = np.tile(x, (len(scan), 1))
    repeated_theta = np.repeat(scan, n_events, axis=0)
    learned = predict(head, repeated_x, repeated_theta).reshape(len(scan), n_events).sum(1)
    exact = np.asarray([exact_log_ratio(x, theta).sum() for theta in scan])
    return exact, learned


def local_hessian(values, i, j, step):
    if i == 0 or j == 0 or i == values.shape[0] - 1 or j == values.shape[1] - 1:
        return None
    return np.asarray(
        [
            [
                (values[i + 1, j] - 2 * values[i, j] + values[i - 1, j])
                / step**2,
                (
                    values[i + 1, j + 1]
                    - values[i + 1, j - 1]
                    - values[i - 1, j + 1]
                    + values[i - 1, j - 1]
                )
                / (4 * step**2),
            ],
            [
                (
                    values[i + 1, j + 1]
                    - values[i + 1, j - 1]
                    - values[i - 1, j + 1]
                    + values[i - 1, j - 1]
                )
                / (4 * step**2),
                (values[i, j + 1] - 2 * values[i, j] + values[i, j - 1])
                / step**2,
            ],
        ],
        dtype=np.float64,
    )


def likelihood_comparison(head, rng, experiments=8):
    axis = np.linspace(*THETA_DOMAIN, 31)
    scan = np.asarray([(a, b) for a in axis for b in axis], dtype=np.float64)
    spacing = float(axis[1] - axis[0])
    theta_true = np.asarray(HELDOUT_THETA[0], dtype=np.float64)
    results = []
    for experiment in range(experiments):
        x = rng.normal(size=(200, 3)) + mean(theta_true)
        exact_flat, learned_flat = surface(head, x, scan)
        exact_surface = exact_flat.reshape(len(axis), len(axis))
        learned_surface = learned_flat.reshape(len(axis), len(axis))
        exact_peak = np.unravel_index(np.argmax(exact_surface), exact_surface.shape)
        learned_peak = np.unravel_index(np.argmax(learned_surface), learned_surface.shape)
        exact_centered = exact_surface - exact_surface.max()
        learned_centered = learned_surface - learned_surface.max()
        exact_profile_a = exact_surface.max(axis=1)
        learned_profile_a = learned_surface.max(axis=1)
        exact_profile_b = exact_surface.max(axis=0)
        learned_profile_b = learned_surface.max(axis=0)
        exact_profile_a -= exact_profile_a.max()
        learned_profile_a -= learned_profile_a.max()
        exact_profile_b -= exact_profile_b.max()
        learned_profile_b -= learned_profile_b.max()
        exact_hessian = local_hessian(exact_surface, *exact_peak, spacing)
        learned_hessian = local_hessian(learned_surface, *learned_peak, spacing)
        hessian_relative_error = None
        hessian_eigenvalues = None
        if exact_hessian is not None and learned_hessian is not None:
            hessian_relative_error = float(
                np.linalg.norm(learned_hessian - exact_hessian)
                / np.linalg.norm(exact_hessian)
            )
            hessian_eigenvalues = np.linalg.eigvalsh(learned_hessian).tolist()
        results.append(
            {
                "experiment": experiment,
                "true_theta": theta_true.tolist(),
                "exact_best_fit_theta": [float(axis[exact_peak[0]]), float(axis[exact_peak[1]])],
                "learned_best_fit_theta": [float(axis[learned_peak[0]]), float(axis[learned_peak[1]])],
                "best_fit_theta_difference": [
                    float(axis[learned_peak[0]] - axis[exact_peak[0]]),
                    float(axis[learned_peak[1]] - axis[exact_peak[1]]),
                ],
                "centered_surface_rmse": float(
                    np.sqrt(np.mean((learned_centered - exact_centered) ** 2))
                ),
                "centered_surface_correlation": float(
                    np.corrcoef(exact_centered.ravel(), learned_centered.ravel())[0, 1]
                ),
                "profile_a_rmse": float(np.sqrt(np.mean((learned_profile_a - exact_profile_a) ** 2))),
                "profile_b_rmse": float(np.sqrt(np.mean((learned_profile_b - exact_profile_b) ** 2))),
                "profile_a_exact": exact_profile_a.tolist(),
                "profile_a_learned": learned_profile_a.tolist(),
                "profile_b_exact": exact_profile_b.tolist(),
                "profile_b_learned": learned_profile_b.tolist(),
                "exact_hessian_at_fit": None if exact_hessian is None else exact_hessian.tolist(),
                "learned_hessian_at_fit": None if learned_hessian is None else learned_hessian.tolist(),
                "learned_hessian_eigenvalues": hessian_eigenvalues,
                "hessian_relative_error": hessian_relative_error,
                "hessian_stable": bool(
                    hessian_eigenvalues is not None
                    and np.all(np.asarray(hessian_eigenvalues) < 0)
                ),
            }
        )
    numeric = (
        "centered_surface_rmse",
        "profile_a_rmse",
        "profile_b_rmse",
        "hessian_relative_error",
    )
    summary = {
        key: float(np.nanmedian([row[key] for row in results if row[key] is not None]))
        for key in numeric
    }
    summary["fraction_exact_best_fit_matched"] = float(
        np.mean([row["exact_best_fit_theta"] == row["learned_best_fit_theta"] for row in results])
    )
    summary["fraction_learned_hessian_negative_definite"] = float(
        np.mean([row["hessian_stable"] for row in results])
    )
    return {
        "events_per_dataset": 200,
        "pseudo_datasets": experiments,
        "heldout_truth_theta": theta_true.tolist(),
        "scan_axis": axis.tolist(),
        "datasets": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    if not run.is_relative_to(ROOT / "results/experiments/rise"):
        raise ValueError("Checkpoint path must stay under the RISE result root")
    if not output.is_relative_to(ROOT / "results/experiments/rise"):
        raise ValueError("Output path must stay under the RISE result root")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    torch.set_num_threads(1)
    grid = make_grid()
    heads = {}
    for name in ("native_point", "reweighted_reference"):
        head = RatioHead(3, 2, HIDDEN)
        head.load_state_dict(torch.load(run / f"{name}.pt", map_location="cpu", weights_only=True))
        head.eval()
        heads[name] = head
    rows = {}
    for name, seed in (("native_point", SEED + 4), ("reweighted_reference", SEED + 5)):
        points = [*grid.tolist(), *HELDOUT_THETA]
        rows[name] = ratio_metrics(heads[name], points, np.random.default_rng(seed), "training_grid_or_heldout")
    likelihood = likelihood_comparison(heads["reweighted_reference"], np.random.default_rng(SEED + 6))
    result = {
        "source_run": str(run),
        "ratio_metrics": rows,
        "likelihood_surface_comparison": likelihood,
        "theta0_note": "Correlation is N/A because the exact log-ratio has zero variance.",
        "sealed_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
