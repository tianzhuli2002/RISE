"""Classify RISE analytic ratio error with independent-reference calibration."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from joint_event_reconstruction.rise import RatioHead
from scripts.validation.evaluate_rise_analytic_checkpoint import (
    HIDDEN,
    ROOT,
    SEED,
    THETA_DOMAIN,
    exact_log_ratio,
    local_hessian,
    mean,
    predict,
)

REFERENCE_EVENTS = 24000
EVALUATION_EVENTS = 24000
CHUNK = 2048


def surface_stats(exact_flat, learned_flat, axis, experiments=8):
    exact_surfaces = exact_flat.reshape(experiments, len(axis), len(axis))
    learned_surfaces = learned_flat.reshape(experiments, len(axis), len(axis))
    spacing = float(axis[1] - axis[0])
    rows = []
    for index, (exact, learned) in enumerate(zip(exact_surfaces, learned_surfaces)):
        exact_peak = np.unravel_index(np.argmax(exact), exact.shape)
        learned_peak = np.unravel_index(np.argmax(learned), learned.shape)
        exact_centered = exact - exact.max()
        learned_centered = learned - learned.max()
        exact_a, learned_a = exact.max(axis=1), learned.max(axis=1)
        exact_b, learned_b = exact.max(axis=0), learned.max(axis=0)
        exact_a -= exact_a.max()
        learned_a -= learned_a.max()
        exact_b -= exact_b.max()
        learned_b -= learned_b.max()
        exact_hessian = local_hessian(exact, *exact_peak, spacing)
        learned_hessian = local_hessian(learned, *learned_peak, spacing)
        hessian_error = None
        if exact_hessian is not None and learned_hessian is not None:
            hessian_error = float(
                np.linalg.norm(learned_hessian - exact_hessian)
                / np.linalg.norm(exact_hessian)
            )
        rows.append(
            {
                "experiment": index,
                "exact_best_fit_theta": [float(axis[exact_peak[0]]), float(axis[exact_peak[1]])],
                "learned_best_fit_theta": [float(axis[learned_peak[0]]), float(axis[learned_peak[1]])],
                "best_fit_offset": [
                    float(axis[learned_peak[0]] - axis[exact_peak[0]]),
                    float(axis[learned_peak[1]] - axis[exact_peak[1]]),
                ],
                "centered_surface_rmse": float(np.sqrt(np.mean((learned_centered - exact_centered) ** 2))),
                "profile_a_rmse": float(np.sqrt(np.mean((learned_a - exact_a) ** 2))),
                "profile_b_rmse": float(np.sqrt(np.mean((learned_b - exact_b) ** 2))),
                "hessian_relative_error": hessian_error,
            }
        )
    summary = {}
    for key in ("centered_surface_rmse", "profile_a_rmse", "profile_b_rmse", "hessian_relative_error"):
        values = [row[key] for row in rows if row[key] is not None]
        summary[key] = float(np.median(values)) if values else None
    summary["best_fit_match_fraction"] = float(
        np.mean([row["best_fit_offset"] == [0.0, 0.0] for row in rows])
    )
    summary["mean_absolute_best_fit_offset"] = np.mean(
        np.abs(np.asarray([row["best_fit_offset"] for row in rows])), axis=0
    ).tolist()
    return {"datasets": rows, "summary": summary}


@torch.no_grad()
def reference_log_normalizers(head, thetas, reference, chunk_size=CHUNK):
    values = np.empty(len(thetas), dtype=np.float64)
    for index, theta in enumerate(thetas):
        theta_rows = np.broadcast_to(theta, (len(reference), len(theta)))
        logs = predict(head, reference, theta_rows, chunk_size=chunk_size).astype(np.float64)
        if not np.isfinite(logs).all():
            raise ValueError(f"Nonfinite ratio output while calibrating theta={theta.tolist()}")
        log_z = float(np.logaddexp.reduce(logs) - math.log(len(logs)))
        if not math.isfinite(log_z):
            raise ValueError(f"Nonfinite log Z while calibrating theta={theta.tolist()}")
        values[index] = log_z
    return values


@torch.no_grad()
def check_independent_normalization(head, thetas, log_z, reference, chunk_size=CHUNK):
    means = np.empty(len(thetas), dtype=np.float64)
    standard_errors = np.empty(len(thetas), dtype=np.float64)
    for index, (theta, offset) in enumerate(zip(thetas, log_z)):
        theta_rows = np.broadcast_to(theta, (len(reference), len(theta)))
        logs = predict(head, reference, theta_rows, chunk_size=chunk_size).astype(np.float64)
        weights = np.exp(logs - offset)
        if not np.isfinite(weights).all():
            raise ValueError(f"Nonfinite calibrated weights at theta={theta.tolist()}")
        means[index] = weights.mean()
        standard_errors[index] = weights.std(ddof=1) / np.sqrt(len(weights))
    return {
        "mean": float(means.mean()),
        "maximum_absolute_deviation": float(np.max(np.abs(means - 1.0))),
        "rms_deviation": float(np.sqrt(np.mean((means - 1.0) ** 2))),
        "maximum_mc_pull": float(np.max(np.abs(means - 1.0) / standard_errors)),
        "theta_count": len(thetas),
        "sample_size": len(reference),
    }


@torch.no_grad()
def likelihood_surfaces(head, axis, theta_grid, log_z, rng, experiments=8, n_events=200):
    theta_true = np.asarray((0.35, -0.4), dtype=np.float64)
    scan = theta_grid
    exact_flat = np.empty(experiments * len(scan), dtype=np.float64)
    raw_flat = np.empty_like(exact_flat)
    normalized_flat = np.empty_like(exact_flat)
    for experiment in range(experiments):
        x = rng.normal(size=(n_events, 3)) + mean(theta_true)
        exact_flat[experiment * len(scan) : (experiment + 1) * len(scan)] = [
            exact_log_ratio(x, theta).sum() for theta in scan
        ]
        for start in range(0, len(scan), 16):
            theta_block = scan[start : start + 16]
            repeated_x = np.tile(x, (len(theta_block), 1))
            repeated_theta = np.repeat(theta_block, n_events, axis=0)
            predicted = predict(head, repeated_x, repeated_theta).reshape(len(theta_block), n_events).sum(1)
            idx = experiment * len(scan) + start
            raw_flat[idx : idx + len(theta_block)] = predicted
            normalized_flat[idx : idx + len(theta_block)] = (
                predicted - n_events * log_z[start : start + len(theta_block)]
            )
    return {
        "raw": surface_stats(exact_flat, raw_flat, axis, experiments),
        "normalized": surface_stats(exact_flat, normalized_flat, axis, experiments),
        "events_per_dataset": int(n_events),
        "pseudo_datasets": int(experiments),
        "truth_theta": theta_true.tolist(),
        "scan_axis": axis.tolist(),
    }


def compact_run_summary(root):
    run1 = json.loads((root / "analytic-run1/metrics.json").read_text())
    run2 = json.loads((root / "analytic-run2/metrics.json").read_text())
    def ratio_quality(metrics):
        result = {}
        for mode, rows in metrics["heldout_exact_vs_learned"].items():
            training = rows[:9]
            heldout = rows[9:]
            result[mode] = {
                "training_grid": {
                    "log_ratio_rmse_mean": float(np.mean([r["log_ratio_rmse"] for r in training])),
                    "E_p0_rhat_mean": float(np.mean([r["reference_mean_predicted_ratio"] for r in training])),
                    "E_p0_rhat_min": float(np.min([r["reference_mean_predicted_ratio"] for r in training])),
                    "E_p0_rhat_max": float(np.max([r["reference_mean_predicted_ratio"] for r in training])),
                },
                "heldout_theta": {
                    "log_ratio_rmse": [float(r["log_ratio_rmse"]) for r in heldout],
                    "E_p0_rhat": [float(r["reference_mean_predicted_ratio"]) for r in heldout],
                },
            }
        return result
    return {
        "run1_pre_fix": {"final_losses": run1["optimizer"], "ratio_quality": ratio_quality(run1)},
        "run2_post_fix": {"final_losses": run2["optimizer"], "ratio_quality": ratio_quality(run2)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/experiments/rise/preproduction-v1/analytic-run2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-events", type=int, default=REFERENCE_EVENTS)
    parser.add_argument("--evaluation-events", type=int, default=EVALUATION_EVENTS)
    parser.add_argument("--experiments", type=int, default=8)
    args = parser.parse_args()
    run = args.run.resolve()
    output = (args.output or run / "calibration_diagnostic.json").resolve()
    result_root = ROOT / "results/experiments/rise"
    if not run.is_relative_to(result_root) or not output.is_relative_to(result_root):
        raise ValueError("Run and output must stay under the RISE result root")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.reference_events <= 0 or args.evaluation_events <= 0 or args.experiments <= 0:
        parser.error("Event counts and experiment count must be positive")

    torch.set_num_threads(1)
    axis = np.linspace(*THETA_DOMAIN, 31)
    scan = np.asarray([(a, b) for a in axis for b in axis], dtype=np.float64)
    calibration_rng = np.random.default_rng(SEED + 70)
    evaluation_rng = np.random.default_rng(SEED + 71)
    reference_calibration = calibration_rng.normal(size=(args.reference_events, 3))
    reference_check = evaluation_rng.normal(size=(args.evaluation_events, 3))
    heads = {}
    for name in ("native_point", "reweighted_reference"):
        head = RatioHead(3, 2, HIDDEN)
        head.load_state_dict(torch.load(run / f"{name}.pt", map_location="cpu", weights_only=True))
        head.eval()
        heads[name] = head

    results = {}
    for index, (name, head) in enumerate(heads.items()):
        log_z = reference_log_normalizers(head, scan, reference_calibration)
        normalization_check = check_independent_normalization(head, scan, log_z, reference_check)
        surfaces = likelihood_surfaces(
            head,
            axis,
            scan,
            log_z,
            np.random.default_rng(SEED + 6),
            experiments=args.experiments,
        )
        results[name] = {
            "normalization_calibration": {
                "reference_population": "independent p0 Gaussian sample; not used for training, validation, or pseudo-datasets",
                "reference_seed": SEED + 70,
                "reference_events": int(args.reference_events),
                "theta_grid": scan.tolist(),
            "log_Z_by_theta": log_z.tolist(),
            "Z_by_theta": np.exp(log_z).tolist(),
            "log_Z_min": float(log_z.min()),
                "log_Z_max": float(log_z.max()),
                "Z_min": float(np.exp(log_z.min())),
                "Z_max": float(np.exp(log_z.max())),
                "independent_check_seed": SEED + 71,
                "independent_check": normalization_check,
            },
            "likelihood_surfaces": surfaces,
        }

    rw_raw = results["reweighted_reference"]["likelihood_surfaces"]["raw"]["summary"]
    rw_norm = results["reweighted_reference"]["likelihood_surfaces"]["normalized"]["summary"]
    def ratio(key):
        return rw_norm[key] / rw_raw[key] if rw_raw[key] else None

    surface_ratio = ratio("centered_surface_rmse")
    profile_ratios = [ratio("profile_a_rmse"), ratio("profile_b_rmse")]
    matched_gain = (
        rw_norm["best_fit_match_fraction"] > rw_raw["best_fit_match_fraction"]
    )
    if surface_ratio is not None and surface_ratio <= 0.5 and all(
        value is not None and value <= 0.5 for value in profile_ratios
    ) and matched_gain:
        diagnosis = "THETA_DEPENDENT_NORMALIZATION_CALIBRATION"
    else:
        diagnosis = "GENUINE_PARAMETERIZED_RATIO_APPROXIMATION_ERROR"

    previous = json.loads((run.parent / "analytic-run2/expanded_closure.json").read_text())
    loss_scaling = json.loads((run / "loss_scaling.json").read_text())
    signed = json.loads((run.parent / "signed_algebra.json").read_text())
    summary = {
        "existing_run_comparison": compact_run_summary(run.parent),
        "corrected_loss_scaling": loss_scaling,
        "signed_algebra": {
            "two_sided_maximum_absolute_ratio_error": signed["two_sided"][
                "maximum_absolute_ratio_error"
            ],
            "single_sided_maximum_absolute_ratio_error": signed["single_sided"][
                "maximum_absolute_ratio_error"
            ],
            "valid": (
                signed["two_sided"]["maximum_absolute_ratio_error"] < 1e-12
                and signed["single_sided"]["maximum_absolute_ratio_error"] < 1e-12
            ),
        },
        "existing_raw_likelihood_surface": previous["likelihood_surface_comparison"]["summary"],
        "normalization_calibration": results,
        "failure_mode": diagnosis,
        "sealed_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "failure_mode": diagnosis,
        "modes": {
            name: {
                "Z_min": row["normalization_calibration"]["Z_min"],
                "Z_max": row["normalization_calibration"]["Z_max"],
                "independent_E_p0_rnorm": row["normalization_calibration"]["independent_check"],
                "raw": row["likelihood_surfaces"]["raw"]["summary"],
                "normalized": row["likelihood_surfaces"]["normalized"]["summary"],
            }
            for name, row in results.items()
        },
        "signed_algebra_valid": signed["two_sided"]["maximum_absolute_ratio_error"] < 1e-12
        and signed["single_sided"]["maximum_absolute_ratio_error"] < 1e-12,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
