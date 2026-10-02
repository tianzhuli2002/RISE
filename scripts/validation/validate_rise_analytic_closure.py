"""Exact two-parameter Gaussian NRE closure for native and weighted RISE inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from joint_event_reconstruction.rise import RatioHead, unsigned_loss
from joint_event_reconstruction.sbi_data import (
    ReweightedRatioBatch,
    SBISidecar,
    write_sidecar,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261002
TRAIN_GRID_AXIS = (-0.8, 0.0, 0.8)
HELDOUT_THETA = ((0.35, -0.4), (-0.3, 0.45), (1.0, -0.5))
THETA_DOMAIN = (-1.0, 1.0)
HIDDEN = 64
TRAIN_STEPS = 4000
BATCH_SIZE = 512
LEARNING_RATE = 0.002
NATIVE_EVENTS_PER_POINT = 6000
REWEIGHTED_EVENTS = 48000
EVALUATION_EVENTS = 12000


def make_grid():
    return np.asarray(
        [(a, b) for a in TRAIN_GRID_AXIS for b in TRAIN_GRID_AXIS], dtype=np.float64
    )


def mean(theta):
    theta = np.asarray(theta, dtype=np.float64)
    return np.stack((theta[..., 0], theta[..., 1], theta[..., 0] + theta[..., 1]), axis=-1)


def exact_log_ratio(x, theta):
    x = np.asarray(x, dtype=np.float64)
    mu = mean(theta)
    return np.sum(mu * x, axis=-1) - 0.5 * np.sum(mu * mu, axis=-1)


def torch_tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def fit_native(head, grid, native_x, native_ids, sidecar, rng, generator):
    lookup = {int(event_id): index for index, event_id in enumerate(native_ids)}
    groups = []
    sampled = sidecar.sample_points(native_ids, np.random.default_rng(SEED))
    if not np.array_equal(sampled["theta"], np.repeat(grid, NATIVE_EVENTS_PER_POINT, axis=0)):
        raise AssertionError("Native sidecar theta rows do not match generated samples")
    for point in range(len(grid)):
        start = point * NATIVE_EVENTS_PER_POINT
        groups.append(native_ids[start : start + NATIVE_EVENTS_PER_POINT])
    reference_point = int(np.flatnonzero(np.all(grid == 0.0, axis=1))[0])
    reference_ids = groups[reference_point]
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
    final_loss = None
    for _ in range(TRAIN_STEPS):
        n = BATCH_SIZE // 2
        points = rng.integers(0, len(grid), size=n)
        target_ids = np.asarray(
            [rng.choice(groups[int(point)]) for point in points], dtype=np.int64
        )
        reference_ids_batch = rng.choice(reference_ids, size=n, replace=True)
        target_rows = np.fromiter(
            (lookup[int(event_id)] for event_id in target_ids), dtype=np.int64
        )
        reference_rows = np.fromiter(
            (lookup[int(event_id)] for event_id in reference_ids_batch), dtype=np.int64
        )
        x = torch_tensor(np.concatenate((native_x[target_rows], native_x[reference_rows])))
        theta = torch_tensor(np.concatenate((grid[points], grid[points])))
        membership = torch.cat((torch.ones(n), torch.zeros(n)))
        loss = unsigned_loss(head(x, theta), membership, torch.ones(BATCH_SIZE))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss


def fit_reweighted(head, grid, sidecar, x_reference, rng):
    ids = sidecar.event_id
    adapter = ReweightedRatioBatch(
        sidecar,
        ids,
        reference_point=int(np.flatnonzero(np.all(grid == 0.0, axis=1))[0]),
        seed=SEED + 1,
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
    final_loss = None
    for _ in range(TRAIN_STEPS):
        rows = rng.choice(len(ids), size=BATCH_SIZE, replace=False)
        batch = {
            "tokens": torch.ones((BATCH_SIZE, 1), dtype=torch.float32),
            "event_index": torch_tensor(ids[rows]).long(),
        }
        adapted = adapter(batch)
        x = torch_tensor(x_reference[rows])
        loss = unsigned_loss(
            head(x, adapted["theta"]),
            adapted["membership"],
            adapted["signed_weight"],
            adapted["class_normalisation"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss, adapter


@torch.no_grad()
def predict_log_ratio(head, x, theta, *, batch_size=8192):
    result = []
    theta = np.asarray(theta, dtype=np.float64)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        condition = np.broadcast_to(theta, (stop - start, len(theta))).copy()
        result.append(
            head(torch_tensor(x[start:stop]), torch_tensor(condition)).cpu().numpy()
        )
    return np.concatenate(result)


@torch.no_grad()
def predict_rows(head, x, theta, *, batch_size=8192):
    result = []
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        result.append(
            head(
                torch_tensor(x[start:stop]),
                torch_tensor(theta[start:stop]),
            ).cpu().numpy()
        )
    return np.concatenate(result)


def parameter_metrics(head, grid, rng):
    rows = []
    for theta in (*grid.tolist(), *HELDOUT_THETA):
        theta = np.asarray(theta, dtype=np.float64)
        reference = rng.normal(size=(EVALUATION_EVENTS, 3))
        target = rng.normal(size=(EVALUATION_EVENTS, 3)) + mean(theta)
        x = np.concatenate((reference, target))
        truth = exact_log_ratio(x, theta)
        prediction = predict_log_ratio(head, x, theta)
        corr = float(np.corrcoef(truth, prediction)[0, 1])
        predicted_weights = np.exp(predict_log_ratio(head, reference, theta))
        expected_ratio = float(predicted_weights.mean())
        exact_weights = np.exp(exact_log_ratio(reference, theta))
        exact_mean = float(exact_weights.mean())
        weighted_mean = np.average(reference, axis=0, weights=predicted_weights)
        rows.append(
            {
                "theta": theta.tolist(),
                "log_ratio_rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                "log_ratio_bias": float(np.mean(prediction - truth)),
                "log_ratio_correlation": corr,
                "reference_mean_predicted_ratio": expected_ratio,
                "reference_mean_exact_ratio_mc": exact_mean,
                "reweighted_x_mean": weighted_mean.tolist(),
                "truth_x_mean": mean(theta).tolist(),
                "x_mean_error_norm": float(np.linalg.norm(weighted_mean - mean(theta))),
            }
        )
    return rows


def native_vs_reweighted(native_x, grid, x_reference, weights):
    results = []
    for point, theta in enumerate(grid):
        start = point * NATIVE_EVENTS_PER_POINT
        sample = native_x[start : start + NATIVE_EVENTS_PER_POINT]
        weight = weights[:, point]
        reweighted_mean = np.average(x_reference, axis=0, weights=weight)
        native_mean = sample.mean(axis=0)
        native_se = sample.std(axis=0, ddof=1) / np.sqrt(len(sample))
        effective = weight.sum() ** 2 / np.square(weight).sum()
        centered = x_reference - reweighted_mean
        reweighted_var = np.average(centered**2, axis=0, weights=weight)
        reweighted_se = np.sqrt(reweighted_var / effective)
        denominator = np.sqrt(native_se**2 + reweighted_se**2)
        z = (native_mean - reweighted_mean) / denominator
        results.append(
            {
                "theta": theta.tolist(),
                "native_mean": native_mean.tolist(),
                "reweighted_mean": reweighted_mean.tolist(),
                "mean_difference_standardized": z.tolist(),
                "max_abs_standardized_mean_difference": float(np.max(np.abs(z))),
                "native_covariance_diagonal": np.var(sample, axis=0, ddof=1).tolist(),
                "reweighted_covariance_diagonal": reweighted_var.tolist(),
                "reweighted_effective_sample_size": float(effective),
            }
        )
    return results


@torch.no_grad()
def pseudo_data_recovery(head, rng):
    theta_true = np.asarray(HELDOUT_THETA[0], dtype=np.float64)
    scan_axis = np.linspace(*THETA_DOMAIN, 31)
    scan = np.asarray([(a, b) for a in scan_axis for b in scan_axis])
    n_events = 200
    experiments = 40
    estimates = []
    covered = []
    a_coverage = []
    b_coverage = []
    for _ in range(experiments):
        x = rng.normal(size=(n_events, 3)) + mean(theta_true)
        scores = np.empty(len(scan), dtype=np.float64)
        for point_start in range(0, len(scan), 31):
            theta_block = scan[point_start : point_start + 31]
            repeated_x = np.tile(x, (len(theta_block), 1))
            repeated_theta = np.repeat(theta_block, n_events, axis=0)
            scores[point_start : point_start + len(theta_block)] = (
                predict_rows(head, repeated_x, repeated_theta)
                .reshape(len(theta_block), n_events)
                .sum(axis=1)
            )
        surface = scores.reshape(len(scan_axis), len(scan_axis))
        best = np.unravel_index(np.argmax(surface), surface.shape)
        estimates.append((scan_axis[best[0]], scan_axis[best[1]]))
        maximum = float(surface[best])
        profile_a = surface.max(axis=1)
        profile_b = surface.max(axis=0)
        interval_a = scan_axis[profile_a >= maximum - 0.5]
        interval_b = scan_axis[profile_b >= maximum - 0.5]
        a_coverage.append(bool(interval_a.min() <= theta_true[0] <= interval_a.max()))
        b_coverage.append(bool(interval_b.min() <= theta_true[1] <= interval_b.max()))
        covered.append(a_coverage[-1] and b_coverage[-1])
    estimates = np.asarray(estimates)
    return {
        "theta_true": theta_true.tolist(),
        "pseudo_experiments": experiments,
        "events_per_experiment": n_events,
        "scan_axis": scan_axis.tolist(),
        "mean_estimate": estimates.mean(axis=0).tolist(),
        "bias": (estimates.mean(axis=0) - theta_true).tolist(),
        "estimate_standard_deviation": estimates.std(axis=0, ddof=1).tolist(),
        "profile_68_coverage_a": float(np.mean(a_coverage)),
        "profile_68_coverage_b": float(np.mean(b_coverage)),
        "profile_68_joint_coverage": float(np.mean(covered)),
        "interval_definition": "profile log-likelihood within 0.5 of maximum; diagnostic only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=ROOT / "results/experiments/rise/preproduction-v1/analytic-run1",
        type=Path,
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results/experiments/rise"):
        raise ValueError("Analytic closure output must stay under the RISE result root")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    grid = make_grid()
    reference_point = int(np.flatnonzero(np.all(grid == 0.0, axis=1))[0])

    native_ids = np.arange(1, len(grid) * NATIVE_EVENTS_PER_POINT + 1, dtype=np.int64)
    native_x = np.concatenate(
        [
            rng.normal(size=(NATIVE_EVENTS_PER_POINT, 3)) + mean(theta)
            for theta in grid
        ]
    )
    native_theta = np.repeat(grid, NATIVE_EVENTS_PER_POINT, axis=0)
    native_path = output / "native_points.h5"
    write_sidecar(
        native_path,
        event_id=native_ids,
        nominal_weight=np.ones(len(native_ids)),
        process_id=np.full(len(native_ids), "analytic_gaussian", dtype="U20"),
        metadata={
            "weight_definition": "unit signed event weight for direct normalized Gaussian samples",
            "source_identity": f"in-memory native Gaussian draws, seed={SEED}",
            "normalisation": "unit area at each native parameter point",
            "weight_convention": "shape_only",
        },
        theta=native_theta,
        parameter_names=("a", "b"),
    )
    native_sidecar = SBISidecar(native_path)

    reweighted_ids = np.arange(1, REWEIGHTED_EVENTS + 1, dtype=np.int64) + native_ids[-1]
    x_reference = rng.normal(size=(REWEIGHTED_EVENTS, 3))
    reweighted = np.stack(
        [np.exp(exact_log_ratio(x_reference, theta)) for theta in grid], axis=-1
    )
    reweighted_path = output / "reweighted_reference.h5"
    write_sidecar(
        reweighted_path,
        event_id=reweighted_ids,
        nominal_weight=np.ones(REWEIGHTED_EVENTS),
        process_id=np.full(REWEIGHTED_EVENTS, "analytic_gaussian_reference", dtype="U32"),
        metadata={
            "weight_definition": "exact p(x|theta)/p(x|theta0) importance ratios",
            "source_identity": f"in-memory p(x|theta0) Gaussian draws, seed={SEED}",
            "normalisation": "unit target area; per-point train mass estimated from the fixed reference sample",
            "weight_convention": "shape_only",
        },
        parameter_grid=grid,
        parameter_names=("a", "b"),
        signed_weights=reweighted,
    )
    reweighted_sidecar = SBISidecar(reweighted_path)

    native_head = RatioHead(3, 2, HIDDEN)
    native_loss = fit_native(
        native_head,
        grid,
        native_x,
        native_ids,
        native_sidecar,
        np.random.default_rng(SEED + 2),
        generator,
    )
    torch.manual_seed(SEED)
    reweighted_head = RatioHead(3, 2, HIDDEN)
    reweighted_loss, adapter = fit_reweighted(
        reweighted_head,
        grid,
        reweighted_sidecar,
        x_reference,
        np.random.default_rng(SEED + 3),
    )
    # The four held-out points include three interior interpolation probes and
    # one declared-domain boundary diagnostic; none enters either optimizer.
    heldout_metrics = {
        "native_point": parameter_metrics(
            native_head, grid, np.random.default_rng(SEED + 4)
        ),
        "reweighted_reference": parameter_metrics(
            reweighted_head,
            grid,
            np.random.default_rng(SEED + 5),
        ),
    }
    # Parameter response is tested independently of any average score.
    response = {
        name: [
            float(
                np.mean(
                    predict_log_ratio(
                        head,
                        np.zeros((1, 3)),
                        theta,
                    )
                )
            )
            for theta in ((-0.6, 0.4), (0.6, -0.4))
        ]
        for name, head in (("native_point", native_head), ("reweighted_reference", reweighted_head))
    }
    pseudo = pseudo_data_recovery(
        reweighted_head, np.random.default_rng(SEED + 6)
    )
    equivalence = native_vs_reweighted(
        native_x, grid, x_reference, reweighted
    )
    result = {
        "toy": {
            "model": "x|theta ~ Normal((a,b,a+b), I3)",
            "reference_theta": [0.0, 0.0],
            "exact_log_ratio": "mu(theta) dot x - 0.5 * ||mu(theta)||^2",
            "training_grid": grid.tolist(),
            "heldout_theta": [list(point) for point in HELDOUT_THETA],
            "parameter_domain": list(THETA_DOMAIN),
            "boundary_diagnostic": list(HELDOUT_THETA[-1]),
            "seed": SEED,
        },
        "optimizer": {
            "head": "RISE RatioHead(event_width=3, parameter_dim=2, hidden=64)",
            "loss": "RISE unsigned_loss",
            "steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "native_final_loss": native_loss,
            "reweighted_final_loss": reweighted_loss,
        },
        "data_representations": {
            "native_sidecar": str(native_path),
            "native_rows": len(native_ids),
            "native_theta_verified_from_sidecar": True,
            "reweighted_sidecar": str(reweighted_path),
            "reweighted_rows": int(REWEIGHTED_EVENTS),
            "reweighted_grid": reweighted_sidecar.grid.tolist(),
            "reweighted_reference_point": reference_point,
            "reweighted_adapter": "ReweightedRatioBatch",
            "reweighted_training_population_masses": adapter.masses.tolist(),
        },
        "native_vs_reweighted_distribution": equivalence,
        "heldout_exact_vs_learned": heldout_metrics,
        "parameter_response_at_x_zero": response,
        "pseudo_data_recovery": pseudo,
        "sealed_test_used": False,
    }
    for name, head in (("native_point", native_head), ("reweighted_reference", reweighted_head)):
        torch.save(head.state_dict(), output / f"{name}.pt")
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
