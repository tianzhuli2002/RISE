"""One matched ratio-head comparison; R1 runs only after R0 qualification."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from joint_event_reconstruction.rise import (
    RatioHead,
    ReferenceAnchoredBilinearRatioHead,
    unsigned_loss,
)
from joint_event_reconstruction.sbi_data import (
    ReweightedRatioBatch,
    SBISidecar,
    write_sidecar,
)
from scripts.validation.diagnose_rise_ratio_calibration import surface_stats
from scripts.validation.evaluate_rise_analytic_checkpoint import local_hessian, predict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/experiments/rise/parameter-conditioning-v1"
CONFIG = ROOT / "configs/experiments/rise_parameter_conditioning/r0.json"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gaussian_parameters(theta, toy):
    a, b = np.asarray(theta, dtype=np.float64)
    if toy == "R0":
        return np.array([a, b, a + b]), np.eye(3)
    if toy != "R1":
        raise ValueError(f"Unknown exact toy {toy}")
    lower = np.array(
        [
            [np.exp(0.15 * a), 0, 0],
            [0.10 * b, np.exp(-0.10 * b), 0],
            [0.05 * a, 0.08 * b, 1.0],
        ]
    )
    return np.array([a, b, a * b]), lower @ lower.T


def exact_log_ratio(x, theta, toy):
    mu, covariance = gaussian_parameters(theta, toy)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign != 1 or not np.isfinite(logdet):
        raise ValueError(f"Invalid exact covariance at {toy} theta={theta}")
    residual = np.asarray(x, dtype=np.float64) - mu
    quadratic = np.sum(residual * np.linalg.solve(covariance, residual.T).T, axis=-1)
    return -0.5 * (quadratic + logdet - np.sum(np.asarray(x) ** 2, axis=-1))


def sample(rng, events, theta, toy):
    mu, covariance = gaussian_parameters(theta, toy)
    return rng.normal(size=(events, 3)) @ np.linalg.cholesky(covariance).T + mu


def importance_log_ratio_uncertainty(reference, theta, evaluation, toy):
    """Delta method for weighted Gaussian mean/covariance and log normalization.

    R0 covariance is known. R1 covariance is estimated in the statistical
    allowance using weighted first/second moments; no such moments supervise NN.
    """
    mu, covariance = gaussian_parameters(theta, toy)
    weight = np.exp(exact_log_ratio(reference, theta, toy))
    logz_variance = weight.var(ddof=1) / len(weight) / weight.mean() ** 2
    centered = reference - mu
    if toy == "R0":
        mean_covariance = (
            (centered * weight[:, None]).T
            @ (centered * weight[:, None])
            / weight.sum() ** 2
        )
        sensitivity = evaluation - mu
        variance = np.einsum("ni,ij,nj->n", sensitivity, mean_covariance, sensitivity)
    else:
        rows, cols = np.triu_indices(3)
        stats = np.concatenate(
            (reference, reference[:, rows] * reference[:, cols]), axis=1
        )
        exact_stats = np.r_[mu, (covariance + np.outer(mu, mu))[rows, cols]]
        weighted = (stats - exact_stats) * weight[:, None]
        moment_covariance = weighted.T @ weighted / weight.sum() ** 2
        alpha = np.linalg.solve(covariance, (evaluation - mu).T).T
        gradient_covariance = 0.5 * (
            alpha[:, :, None] * alpha[:, None, :] - np.linalg.inv(covariance)
        )
        gradient_mean = alpha - 2 * (gradient_covariance @ mu)
        gradient_second = gradient_covariance[:, rows, cols] * np.where(
            rows == cols, 1, 2
        )
        gradient = np.concatenate((gradient_mean, gradient_second), axis=1)
        variance = np.einsum("ni,ij,nj->n", gradient, moment_covariance, gradient)
    return float(np.sqrt(variance.mean() + logz_variance)), float(
        np.sqrt(logz_variance)
    )


def continuous_peak(surface, axis):
    peak = np.unravel_index(surface.argmax(), surface.shape)
    hessian = local_hessian(surface, *peak, axis[1] - axis[0])
    if hessian is None or not (np.linalg.eigvalsh(hessian) < 0).all():
        return None, hessian
    i, j = peak
    spacing = axis[1] - axis[0]
    gradient = np.array(
        [surface[i + 1, j] - surface[i - 1, j], surface[i, j + 1] - surface[i, j - 1]]
    ) / (2 * spacing)
    return np.array([axis[i], axis[j]]) - np.linalg.solve(hessian, gradient), hessian


@torch.no_grad()
def diagnostics(head, reference, config, toy):
    rng = np.random.default_rng(config["seed"] + 101)
    grid = np.array(
        [(a, b) for a in config["training_axis"] for b in config["training_axis"]]
    )
    rows = []
    points = [(theta, "training") for theta in grid] + [
        (np.array(theta), "heldout") for theta in config["heldout_theta"]
    ]
    for theta, label in points:
        evaluation_reference = rng.normal(
            size=(config["evaluation_reference_events"], 3)
        )
        target = sample(rng, len(evaluation_reference), theta, toy)
        x = np.concatenate((evaluation_reference, target))
        truth = exact_log_ratio(x, theta, toy)
        learned = predict(head, x, np.broadcast_to(theta, (len(x), 2)))
        ratio = np.exp(learned[: len(evaluation_reference)].astype(np.float64))
        uncertainty, logz_se = importance_log_ratio_uncertainty(
            reference, theta, x, toy
        )
        evaluation_se = ratio.std(ddof=1) / np.sqrt(len(ratio))
        combined_se = np.hypot(evaluation_se, logz_se)
        rmse = float(np.sqrt(np.mean((learned - truth) ** 2)))
        is_reference = bool(np.array_equal(theta, config["reference_theta"]))
        norm_ok = (
            bool(abs(ratio.mean() - 1) <= 3 * combined_se) if not is_reference else True
        )
        ratio_ok = bool(rmse <= 3 * uncertainty) if not is_reference else True
        rows.append(
            {
                "theta": theta.tolist(),
                "parameter_set": label,
                "log_ratio_bias": float((learned - truth).mean()),
                "log_ratio_RMSE": rmse,
                "correlation": float(np.corrcoef(truth, learned)[0, 1])
                if truth.var() > 1e-20
                else None,
                "E_p0_rhat": float(ratio.mean()),
                "evaluation_MC_SE": float(evaluation_se),
                "training_log_normalization_MC_SE": logz_se,
                "combined_normalization_SE": float(combined_se),
                "event_ratio_MC_allowance_3sigma": 3 * uncertainty,
                "normalization_pass": norm_ok,
                "event_ratio_pass": ratio_ok,
                "reference_point_not_used_as_statistical_gate": is_reference,
            }
        )
    axis = np.linspace(-1, 1, config["likelihood_axis_points"])
    scan = np.array([(a, b) for a in axis for b in axis])
    rng = np.random.default_rng(config["seed"] + 6)
    exact_all, learned_all = [], []
    for _ in range(config["pseudo_datasets"]):
        x = sample(
            rng, config["pseudo_dataset_events"], config["heldout_theta"][0], toy
        )
        exact_all.append([exact_log_ratio(x, theta, toy).sum() for theta in scan])
        learned_all.append(
            predict(head, np.tile(x, (len(scan), 1)), np.repeat(scan, len(x), axis=0))
            .reshape(len(scan), len(x))
            .astype(np.float64)
            .sum(1)
        )
    exact_all, learned_all = np.array(exact_all), np.array(learned_all)
    likelihood = surface_stats(
        exact_all.ravel(), learned_all.ravel(), axis, config["pseudo_datasets"]
    )
    for row, exact_flat, learned_flat in zip(
        likelihood["datasets"], exact_all, learned_all
    ):
        exact = exact_flat.reshape(len(axis), len(axis))
        learned = learned_flat.reshape(len(axis), len(axis))
        centered_difference = (learned - learned.max()) - (exact - exact.max())
        region = exact >= exact.max() - 3
        region_rmse = float(np.sqrt(np.mean(centered_difference[region] ** 2)))
        exact_peak, exact_hessian = continuous_peak(exact, axis)
        learned_peak, learned_hessian = continuous_peak(learned, axis)
        distance = None
        if exact_peak is not None and learned_peak is not None:
            delta = learned_peak - exact_peak
            distance = float(np.sqrt(delta @ (-exact_hessian) @ delta))
        row.update(
            {
                "confidence_region_surface_RMSE": region_rmse,
                "continuous_exact_best_fit": None
                if exact_peak is None
                else exact_peak.tolist(),
                "continuous_learned_best_fit": None
                if learned_peak is None
                else learned_peak.tolist(),
                "best_fit_statistical_distance": distance,
                "location_pass": bool(distance is not None and distance <= 1),
                "confidence_region_shape_pass": bool(region_rmse <= 1),
                "learned_hessian_negative_definite": bool(
                    learned_hessian is not None
                    and (np.linalg.eigvalsh(learned_hessian) < 0).all()
                ),
            }
        )
    qualified = all(
        row["normalization_pass"] and row["event_ratio_pass"] for row in rows
    ) and all(
        row["location_pass"] and row["confidence_region_shape_pass"]
        for row in likelihood["datasets"]
    )
    return {
        "ratio_points": rows,
        "likelihood": likelihood,
        "qualification_pass": qualified,
        "reference_anchor_max_absolute_log_ratio": float(
            np.abs(predict(head, reference[:1024], np.zeros((1024, 2)))).max()
        ),
    }


def fit(name, reference, sidecar, config, toy):
    torch.manual_seed(config["seed"])
    head = (
        RatioHead(3, 2, config["hidden"])
        if name == "existing"
        else ReferenceAnchoredBilinearRatioHead(
            3, 2, config["hidden"], config["reference_theta"]
        )
    )
    adapter = ReweightedRatioBatch(
        sidecar, sidecar.event_id, reference_point=4, seed=config["seed"] + 1
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(config["seed"] + 3)
    directory = OUT / toy / name
    directory.mkdir(parents=True, exist_ok=False)
    history, checkpoints = [], {}
    start = time.perf_counter()
    for step in range(1, max(config["checkpoints"]) + 1):
        rows = rng.choice(len(reference), size=config["batch_size"], replace=False)
        batch = adapter(
            {
                "tokens": torch.ones((len(rows), 1)),
                "event_index": torch.as_tensor(sidecar.event_id[rows]),
            }
        )
        logits = head(
            torch.as_tensor(reference[rows], dtype=torch.float32), batch["theta"]
        )
        loss = unsigned_loss(
            logits,
            batch["membership"],
            batch["signed_weight"],
            batch["class_normalisation"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss) or any(
            not torch.isfinite(p.grad).all()
            for p in head.parameters()
            if p.grad is not None
        ):
            raise FloatingPointError(f"{toy}/{name} nonfinite optimization at {step}")
        optimizer.step()
        if step % 500 == 0:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "wall_seconds": time.perf_counter() - start,
                }
            )
            print(toy, name, history[-1], flush=True)
        if step in config["checkpoints"]:
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "config": config,
                    "head": type(head).__name__,
                },
                directory / f"step_{step}.pt",
            )
            result = diagnostics(head, reference, config, toy)
            result.update(
                {
                    "step": step,
                    "parameter_count": sum(p.numel() for p in head.parameters()),
                    "wall_seconds": time.perf_counter() - start,
                }
            )
            checkpoints[str(step)] = result
            save(directory / f"step_{step}.json", result)
            save(directory / "history.json", history)
            print(
                toy, name, step, "qualified", result["qualification_pass"], flush=True
            )
    save(directory / "comparison.json", checkpoints)
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("toy", choices=["R0", "R1"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    config_path = CONFIG if args.toy == "R0" else CONFIG.with_name("r1.json")
    config = json.loads(config_path.read_text())
    grid = np.array(
        [(a, b) for a in config["training_axis"] for b in config["training_axis"]]
    )
    rng = np.random.default_rng(config["seed"])
    if args.toy == "R0":
        rng.normal(
            size=(9 * 6000, 3)
        )  # Preserve the established native-before-reference RNG stream.
        reference = rng.normal(size=(config["training_reference_events"], 3))
        sidecar = SBISidecar(ROOT / config["training_sidecar"])
        exact_weights = np.stack(
            [np.exp(exact_log_ratio(reference, theta, "R0")) for theta in grid], axis=1
        )
        np.testing.assert_allclose(
            sidecar.weights, exact_weights, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_array_equal(sidecar.grid, grid)
        results = {
            name: fit(name, reference, sidecar, config, "R0")
            for name in ["existing", "bilinear"]
        }
        extended = str(max(config["checkpoints"]))
        selected = (
            "existing"
            if results["existing"][extended]["qualification_pass"]
            else (
                "bilinear"
                if results["bilinear"][extended]["qualification_pass"]
                else None
            )
        )
        verdict = (
            "RISE_R0_BLOCKED"
            if selected is None
            else f"RISE_R0_PASS_{selected.upper()}_HEAD"
        )
        save(
            OUT / "R0/decision.json",
            {
                "verdict": verdict,
                "selected_head": selected,
                "qualification": config["qualification"],
                "reference_data_sha256": hashlib.sha256(
                    reference.astype("<f8").tobytes()
                ).hexdigest(),
                "existing_sidecar_unchanged": True,
                "checkpoint_results": {
                    name: {step: f"{name}/step_{step}.json" for step in steps}
                    for name, steps in results.items()
                },
            },
        )
    else:
        decision = json.loads((OUT / "R0/decision.json").read_text())
        if decision["selected_head"] is None:
            raise ValueError("R1 prohibited: R0 did not qualify")
        directory = OUT / "R1"
        directory.mkdir(parents=True, exist_ok=False)
        reference = rng.normal(size=(config["training_reference_events"], 3))
        path = directory / "reference_weights.h5"
        weights = np.stack(
            [np.exp(exact_log_ratio(reference, theta, "R1")) for theta in grid], axis=1
        )
        write_sidecar(
            path,
            event_id=np.arange(1, len(reference) + 1),
            nominal_weight=np.ones(len(reference)),
            process_id=np.full(len(reference), "R1_exact_Gaussian"),
            metadata={
                "weight_definition": "exact parameter-dependent Gaussian mean/covariance ptheta/p0",
                "source_identity": f"R1 p0 Gaussian seed {config['seed']}",
                "normalisation": "shape only, fixed TRAIN reference-population component totals",
                "weight_convention": "shape_only",
            },
            parameter_names=["a", "b"],
            parameter_grid=grid,
            signed_weights=weights,
        )
        result = fit(
            decision["selected_head"], reference, SBISidecar(path), config, "R1"
        )
        passed = result[str(max(config["checkpoints"]))]["qualification_pass"]
        save(
            directory / "decision.json",
            {
                "RISE_R1_ANALYTIC_QUALIFICATION": "PASS" if passed else "FAIL",
                "head": decision["selected_head"],
                "results": result,
            },
        )
    print("Finished", args.toy, flush=True)


if __name__ == "__main__":
    main()
