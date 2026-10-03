"""Test the maintained RISE R0 loss with its exact seven-term log-ratio basis."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from joint_event_reconstruction.rise import unsigned_loss
from joint_event_reconstruction.sbi_data import ReweightedRatioBatch, SBISidecar
from scripts.validation.diagnose_rise_ratio_calibration import surface_stats
from scripts.validation.evaluate_rise_analytic_checkpoint import exact_log_ratio, mean

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "configs/experiments/rise_parameter_conditioning/r0.json").read_text()
)
SOURCE = ROOT / CONFIG["training_sidecar"]
OUT = ROOT / "results/experiments/rise/exact-objective-r0"
BETA_TRUTH = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])


def feature_map(x, theta):
    """Return (...,7) in order ax1,bx2,ax3,bx3,a2,b2,ab."""
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    a, b = theta[..., 0], theta[..., 1]
    x1, x2, x3 = np.moveaxis(x, -1, 0)
    event_parameter_shape = np.broadcast_shapes(x.shape[:-1], theta.shape[:-1])
    return np.stack(
        (
            a * x1,
            b * x2,
            a * x3,
            b * x3,
            np.broadcast_to(a * a, event_parameter_shape),
            np.broadcast_to(b * b, event_parameter_shape),
            np.broadcast_to(a * b, event_parameter_shape),
        ),
        axis=-1,
    )


def make_full_population(sidecar, adapter, x):
    """Deterministically enumerate the adapter's theta/class sampling law.

    Every train EVENT_ID is used once at every uniform theta point in both
    membership classes. The rows are fed to the maintained unsigned_loss with
    exactly the adapter's absolute weights and complete-population masses.
    This removes only stochastic minibatch/theta/member subsampling.
    """
    n, g = len(x), len(sidecar.grid)
    rows = sidecar.positions(sidecar.event_id)
    if len(rows) != adapter.population_size:
        raise ValueError("Sidecar population differs from RISE training adapter")
    x = np.asarray(x, dtype=np.float64)
    phi = feature_map(x[:, None, :], sidecar.grid[None, :, :])
    target_phi = phi.transpose(1, 0, 2).reshape(g * n, 7)
    reference_phi = target_phi.copy()
    features = np.concatenate((target_phi, reference_phi), axis=0)

    weight_grid = sidecar.weights[rows]
    reference_point = adapter.reference_point
    target_weight = weight_grid.T.reshape(-1)
    reference_weight = np.tile(weight_grid[:, reference_point], g)
    weights = np.concatenate((target_weight, reference_weight))
    membership = np.concatenate((np.ones(g * n), np.zeros(g * n)))
    target_mass = adapter.masses[:, 0] / adapter.population_size
    reference_mass = adapter.masses[:, 2] / adapter.population_size
    target_norm = np.repeat(target_mass, n)
    reference_norm = np.full(g * n, reference_mass[0])
    normalisation = np.column_stack(
        (
            np.concatenate((target_norm, np.ones(g * n))),
            np.concatenate((np.ones(g * n), reference_norm)),
        )
    )
    tensors = {
        "features": torch.as_tensor(features, dtype=torch.float64),
        "membership": torch.as_tensor(membership, dtype=torch.float64),
        "signed_weight": torch.as_tensor(weights, dtype=torch.float64),
        "class_normalisation": torch.as_tensor(normalisation, dtype=torch.float64),
    }
    return tensors, phi, weight_grid, target_mass, float(reference_mass[0])


def torch_loss(beta, population):
    logits = population["features"] @ beta
    return unsigned_loss(
        logits,
        population["membership"],
        population["signed_weight"],
        population["class_normalisation"],
    )


def analytic_gradient_influence(beta, phi, weights):
    """Full-grid gradient and event-cluster sandwich influence contributions."""
    n, grid_count, _ = phi.shape
    logits = np.einsum("ngk,k->ng", phi, beta)
    probability = expit(logits)
    totals = weights.sum(axis=0)
    if (totals <= 0).any():
        raise ValueError("Nonpositive target importance-weight population total")
    target_derivative = (probability - 1.0)[..., None] * phi
    reference_derivative = probability[..., None] * phi
    target_mean = np.einsum("ng,ngk->gk", weights, target_derivative) / totals[:, None]
    reference_mean = reference_derivative.mean(axis=0)
    grid_gradient = 0.5 * (target_mean + reference_mean)
    gradient = grid_gradient.mean(axis=0)

    influence = np.zeros((n, 7), dtype=np.float64)
    for point in range(grid_count):
        target_influence = (
            weights[:, point, None]
            / totals[point]
            * (target_derivative[:, point] - target_mean[point])
        )
        reference_influence = (
            reference_derivative[:, point] - reference_mean[point]
        ) / n
        influence += 0.5 / grid_count * (target_influence + reference_influence)
    covariance_gradient = (n / (n - 1)) * influence.T @ influence
    standard_error = np.sqrt(np.diag(covariance_gradient))
    return gradient, standard_error, covariance_gradient


def hessian_and_coefficient_covariance(beta, phi, weights):
    n, grid_count, _ = phi.shape
    probability = expit(np.einsum("ngk,k->ng", phi, beta))
    hessian = np.zeros((7, 7), dtype=np.float64)
    for point in range(grid_count):
        curvature = probability[:, point] * (1 - probability[:, point])
        target_probability = weights[:, point] / weights[:, point].sum()
        target_hessian = np.einsum(
            "n,ni,nj->ij", target_probability * curvature, phi[:, point], phi[:, point]
        )
        reference_hessian = np.einsum(
            "n,ni,nj->ij", curvature / n, phi[:, point], phi[:, point]
        )
        hessian += 0.5 / grid_count * (target_hessian + reference_hessian)
    if np.linalg.eigvalsh(hessian).min() <= 0:
        raise ValueError("Exact-basis empirical optimum has nonpositive Hessian")
    return hessian


def optimize(population, start):
    beta = torch.nn.Parameter(torch.as_tensor(start, dtype=torch.float64).clone())
    optimizer = torch.optim.LBFGS(
        [beta],
        lr=1.0,
        max_iter=500,
        max_eval=750,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = torch_loss(beta, population)
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(beta.grad).all():
            raise FloatingPointError("Nonfinite full-batch exact-basis objective")
        evaluations += 1
        return loss

    start_loss = float(torch_loss(beta.detach(), population))
    optimizer.step(closure)
    with torch.no_grad():
        final_loss = float(torch_loss(beta, population))
    grad = torch.autograd.grad(torch_loss(beta, population), beta)[0].detach().numpy()
    return beta.detach().numpy(), {
        "start": np.asarray(start).tolist(),
        "start_loss": start_loss,
        "final_loss": final_loss,
        "final_gradient_norm": float(np.linalg.norm(grad)),
        "closure_evaluations": evaluations,
        "optimizer": "torch.optim.LBFGS full batch, strong Wolfe line search",
    }


def independent_population(beta_values, coefficient_covariance, *, n_events=1_000_000):
    """Stream an independent million-event p0 and matching target population."""
    rng = np.random.default_rng(CONFIG["seed"] + 909)
    grid = np.asarray(
        [(a, b) for a in CONFIG["training_axis"] for b in CONFIG["training_axis"]]
    )
    points = [(theta, "training") for theta in grid] + [
        (np.asarray(theta), "heldout") for theta in CONFIG["heldout_theta"]
    ]
    rows = {name: [] for name in beta_values}
    for theta, label in points:
        mu = mean(theta)
        ref_sums = {name: np.zeros(7, dtype=np.float64) for name in beta_values}
        ref_log_weights = {name: [] for name in beta_values}
        ratio_gradient_sums = {
            name: np.zeros(7, dtype=np.float64) for name in beta_values
        }
        exact_weights = []
        for start in range(0, n_events, 50_000):
            size = min(50_000, n_events - start)
            reference = rng.normal(size=(size, 3))
            target = rng.normal(size=(size, 3)) + mu
            exact_ref = exact_log_ratio(reference, theta)
            exact_target = exact_log_ratio(target, theta)
            exact_weights.append(np.exp(exact_ref))
            for name, beta in beta_values.items():
                predicted_ref = feature_map(reference, theta) @ beta
                predicted_target = feature_map(target, theta) @ beta
                error = np.concatenate(
                    (predicted_ref - exact_ref, predicted_target - exact_target)
                )
                truth_pair = np.concatenate((exact_ref, exact_target))
                pred_pair = np.concatenate((predicted_ref, predicted_target))
                ref_ratio = np.exp(predicted_ref)
                ref_sums[name] += np.array(
                    [
                        error.sum(),
                        np.square(error).sum(),
                        np.sum(truth_pair),
                        np.sum(pred_pair),
                        np.square(truth_pair).sum(),
                        np.square(pred_pair).sum(),
                        np.sum(truth_pair * pred_pair),
                    ]
                )
                ref_log_weights[name].append(ref_ratio)
                ratio_gradient_sums[name] += (
                    ref_ratio[:, None] * feature_map(reference, theta)
                ).sum(0)
        exact_weight = np.concatenate(exact_weights)
        for name in beta_values:
            sums = ref_sums[name]
            count = 2 * n_events
            ratios = np.concatenate(ref_log_weights[name])
            ratio_mean = float(ratios.mean())
            ratio_se = float(ratios.std(ddof=1) / np.sqrt(n_events))
            truth_mean = float(exact_weight.mean())
            truth_se = float(exact_weight.std(ddof=1) / np.sqrt(n_events))
            mean_phi_weight = ratio_gradient_sums[name] / n_events
            coefficient_variance = 0.0
            if name == "beta_hat":
                coefficient_variance = float(
                    mean_phi_weight @ coefficient_covariance @ mean_phi_weight
                )
            total_ratio_se = float(np.sqrt(ratio_se**2 + coefficient_variance))
            covariance = sums[6] / count - sums[2] * sums[3] / count**2
            var_truth = sums[4] / count - (sums[2] / count) ** 2
            var_pred = sums[5] / count - (sums[3] / count) ** 2
            normalization_pull = (
                (ratio_mean - 1.0) / total_ratio_se
                if total_ratio_se > 0
                else (0.0 if np.isclose(ratio_mean, 1.0, atol=1e-14) else None)
            )
            ratio_only_pull = (
                (ratio_mean - 1.0) / ratio_se
                if ratio_se > 0
                else (0.0 if np.isclose(ratio_mean, 1.0, atol=1e-14) else None)
            )
            truth_only_pull = (
                (truth_mean - 1.0) / truth_se
                if truth_se > 0
                else (0.0 if np.isclose(truth_mean, 1.0, atol=1e-14) else None)
            )
            rows[name].append(
                {
                    "theta": theta.tolist(),
                    "parameter_set": label,
                    "log_ratio_bias": float(sums[0] / count),
                    "log_ratio_RMSE": float(np.sqrt(sums[1] / count)),
                    "correlation": (
                        float(covariance / np.sqrt(var_truth * var_pred))
                        if var_truth * var_pred > 0
                        else None
                    ),
                    "E_p0_r_hat": ratio_mean,
                    "E_p0_r_hat_MC_SE": ratio_se,
                    "E_p0_r_hat_training_plus_evaluation_SE": total_ratio_se,
                    "normalization_pull_with_training_uncertainty": normalization_pull,
                    "E_p0_r_truth": truth_mean,
                    "truth_normalization_MC_SE": truth_se,
                    "normalization_pull_from_one": ratio_only_pull,
                    "truth_normalization_pull_from_one": truth_only_pull,
                }
            )
    return rows


def likelihood_surfaces(beta, *, experiments=8, events=200):
    axis = np.linspace(-1.0, 1.0, CONFIG["likelihood_axis_points"])
    scan = np.asarray([(a, b) for a in axis for b in axis])
    rng = np.random.default_rng(CONFIG["seed"] + 6)
    exact_all, learned_all = [], []
    for _ in range(experiments):
        x = rng.normal(size=(events, 3)) + mean(CONFIG["heldout_theta"][0])
        exact_all.append([exact_log_ratio(x, theta).sum() for theta in scan])
        learned_all.append(
            [float(feature_map(x, theta).sum(axis=0) @ beta) for theta in scan]
        )
    exact_all, learned_all = np.asarray(exact_all), np.asarray(learned_all)
    return surface_stats(exact_all.ravel(), learned_all.ravel(), axis, experiments)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    torch.set_num_threads(1)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite RISE diagnostic output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sidecar = SBISidecar(SOURCE)
    grid = np.asarray(
        [(a, b) for a in CONFIG["training_axis"] for b in CONFIG["training_axis"]]
    )
    np.testing.assert_array_equal(sidecar.grid, grid)
    adapter = ReweightedRatioBatch(
        sidecar, sidecar.event_id, reference_point=4, seed=CONFIG["seed"] + 1
    )
    rows = sidecar.positions(sidecar.event_id)
    if len(rows) != CONFIG["training_reference_events"]:
        raise ValueError("Unexpected R0 sidecar population size")
    x = np.random.default_rng(CONFIG["seed"])
    # Preserve the original native-before-reference Gaussian RNG stream.
    x.normal(size=(9 * 6000, 3))
    reference = x.normal(size=(len(rows), 3))
    exact_weights = np.exp(
        np.stack([exact_log_ratio(reference, t) for t in grid], axis=1)
    )
    np.testing.assert_allclose(sidecar.weights, exact_weights, rtol=1e-12, atol=1e-12)

    # Confirm the maintained stochastic adapter's row semantics before the
    # complete grid/class enumeration below.
    probe_ids = sidecar.event_id[: CONFIG["batch_size"]]
    probe = adapter(
        {
            "tokens": torch.ones((len(probe_ids), 1), dtype=torch.float32),
            "event_index": torch.as_tensor(probe_ids),
        }
    )
    probe_rows = sidecar.positions(probe_ids)
    probe_points = probe["parameter_point"].numpy()
    probe_membership = probe["membership"].numpy().astype(int)
    expected_weights = sidecar.weights[
        probe_rows,
        np.where(probe_membership == 1, probe_points, adapter.reference_point),
    ]
    np.testing.assert_allclose(probe["signed_weight"].numpy(), expected_weights)
    np.testing.assert_allclose(
        probe["class_normalisation"].numpy(),
        adapter.masses[probe_points][:, [0, 2]] / adapter.population_size,
    )

    population, phi, weights, target_mass, reference_mass = make_full_population(
        sidecar, adapter, reference
    )
    exact_values = phi @ BETA_TRUTH
    analytic_values = np.stack([exact_log_ratio(reference, t) for t in grid], axis=1)
    identity_error = float(np.max(np.abs(exact_values - analytic_values)))
    if identity_error > 2e-14:
        raise AssertionError(f"Exact seven-feature identity error {identity_error}")
    beta_truth = torch.as_tensor(BETA_TRUTH, dtype=torch.float64).requires_grad_()
    loss_truth = torch_loss(beta_truth, population)
    loss_truth.backward()
    autodiff_gradient = beta_truth.grad.detach().numpy()
    analytic_gradient, gradient_se, gradient_covariance = analytic_gradient_influence(
        BETA_TRUTH, phi, weights
    )
    np.testing.assert_allclose(
        autodiff_gradient, analytic_gradient, rtol=2e-10, atol=2e-11
    )
    standardized_gradient = autodiff_gradient / gradient_se

    zero_beta, zero_start = optimize(population, np.zeros(7))
    second_beta, second_start = optimize(
        population, np.array([0.4, 1.4, 0.7, 0.8, -1.2, -0.7, -0.6])
    )
    np.testing.assert_allclose(zero_beta, second_beta, atol=3e-6, rtol=3e-6)
    beta_hat = zero_beta
    hessian = hessian_and_coefficient_covariance(beta_hat, phi, weights)
    inverse_hessian = np.linalg.inv(hessian)
    coefficient_covariance = inverse_hessian @ gradient_covariance @ inverse_hessian.T
    coefficient_errors = np.sqrt(np.diag(coefficient_covariance))
    standardized_coefficient_error = (beta_hat - BETA_TRUTH) / coefficient_errors
    gradient_at_optimum, optimum_gradient_se, _ = analytic_gradient_influence(
        beta_hat, phi, weights
    )

    # Also prove the adapter's theta-uniform/membership-balanced expected risk
    # equals the full categorical weighted population risk from unsigned_loss.
    logit = np.einsum("ngk,k->ng", phi, BETA_TRUTH)
    target_bce = np.logaddexp(0.0, -logit)
    reference_bce = np.logaddexp(0.0, logit)
    direct_risk = (
        0.5
        / len(grid)
        * sum(
            np.average(target_bce[:, j], weights=weights[:, j])
            + reference_bce[:, j].mean()
            for j in range(len(grid))
        )
    )
    torch_risk_difference = float(loss_truth.detach()) - float(direct_risk)
    if abs(torch_risk_difference) > 2e-12:
        raise AssertionError("Full unsigned_loss enumeration differs from derived risk")

    truth_surfaces = likelihood_surfaces(BETA_TRUTH)
    fitted_surfaces = likelihood_surfaces(beta_hat)
    population_metrics = independent_population(
        {"beta_truth": BETA_TRUTH, "beta_hat": beta_hat}, coefficient_covariance
    )
    report = {
        "status": "completed",
        "source": str(SOURCE),
        "seed": CONFIG["seed"],
        "event_population": len(rows),
        "theta_grid": grid.tolist(),
        "membership_probability": {"target": 0.5, "reference": 0.5},
        "parameter_probability": "uniform over all nine theta grid points",
        "exact_features": ["a*x1", "b*x2", "a*x3", "b*x3", "a^2", "b^2", "a*b"],
        "beta_truth": BETA_TRUTH.tolist(),
        "maximum_feature_identity_error": identity_error,
        "full_batch_loss_at_beta_truth": float(loss_truth.detach()),
        "analytic_population_risk_at_beta_truth": float(direct_risk),
        "full_batch_risk_difference": torch_risk_difference,
        "gradient_at_beta_truth": autodiff_gradient.tolist(),
        "gradient_SE": gradient_se.tolist(),
        "gradient_pull": standardized_gradient.tolist(),
        "gradient_max_abs_pull": float(np.max(np.abs(standardized_gradient))),
        "beta_hat": beta_hat.tolist(),
        "beta_hat_second_start": second_beta.tolist(),
        "beta_hat_minus_truth": (beta_hat - BETA_TRUTH).tolist(),
        "beta_hat_uncertainty": coefficient_errors.tolist(),
        "beta_hat_standardized_difference": standardized_coefficient_error.tolist(),
        "beta_truth_objective_stationary_within_3SE": bool(
            np.max(np.abs(standardized_gradient)) <= 3
        ),
        "full_batch_fit_converged_same_from_two_starts": bool(
            np.max(np.abs(zero_beta - second_beta)) <= 3e-6
            and np.max(np.abs(gradient_at_optimum / optimum_gradient_se)) <= 3
        ),
        "empirical_hessian": hessian.tolist(),
        "empirical_hessian_eigenvalues": np.linalg.eigvalsh(hessian).tolist(),
        "coefficient_covariance_sandwich": coefficient_covariance.tolist(),
        "optimizer_starts": {"zero": zero_start, "second": second_start},
        "exact_basis_likelihood_surfaces": {
            "beta_truth": truth_surfaces,
            "beta_hat": fitted_surfaces,
        },
        "independent_population_events_per_distribution": 1_000_000,
        "independent_population_metrics": population_metrics,
        "adapter_semantics": {
            "target_mass_over_population_by_theta": target_mass.tolist(),
            "reference_mass_over_population": reference_mass,
            "finite_population_logit_offset_log_W0_over_Wtheta": np.log(
                reference_mass / target_mass
            ).tolist(),
            "stochastic_adapter_probe_matches_sidecar_weights_and_masses": True,
            "full_grid_and_both_memberships_enumerated": True,
            "loss": "maintained unsigned_loss unchanged",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "exact_objective_r0.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (OUT / "exact_objective_derivation.md").write_text(
        derive_objective(target_mass, reference_mass)
    )
    print(
        json.dumps(
            {
                "loss_truth": report["full_batch_loss_at_beta_truth"],
                "gradient": report["gradient_at_beta_truth"],
                "gradient_SE": report["gradient_SE"],
                "gradient_pull": report["gradient_pull"],
                "beta_hat": report["beta_hat"],
                "beta_uncertainty": report["beta_hat_uncertainty"],
                "beta_pull": report["beta_hat_standardized_difference"],
                "full_batch_converged": report[
                    "full_batch_fit_converged_same_from_two_starts"
                ],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


def derive_objective(target_mass, reference_mass):
    rows = "\n".join(
        f"- theta grid index {i}: target mass/population = {mass:.12g}"
        for i, mass in enumerate(target_mass)
    )
    return f"""# RISE R0 empirical classifier objective

The reweighted adapter selects source EVENT_ID uniformly, theta uniformly over
the nine grid points, and membership independently with probability one half.
For membership 1 it supplies target importance weight w_i(theta); for membership
0 it supplies reference weight w_i(theta0). The BCE multiplier is that weight
divided by the complete training-population component mass divided by N. The
reference mass/population is {reference_mass:.12g}. The target masses/population are:

{rows}

Consequently the unchanged expected minibatch risk is

    L(f) = (1/(2G)) sum_theta [ sum_i w_i(theta)/W_theta BCE(f(x_i,theta),1)
                                + sum_i w_i(theta0)/W_0 BCE(f(x_i,theta),0) ].

The factor one half comes from equal membership sampling; 1/G comes from the
uniform theta sampler. The normalized component measures have equal prior mass.
For a continuous source measure q(x), its unrestricted Bayes logit is therefore

    log [ q(x) w_theta(x)/W_theta ] - log [ q(x) w_0(x)/W_0 ]
      = log r(x;theta,theta0) + log(W_0/W_theta).

Thus the finite-sidecar optimum includes c(theta)=log(W_0/W_theta). For the
analytic R0 density-ratio weights, W_theta/N estimates one and W_0/N=1, so this
offset vanishes in the infinite population. The reported exact-basis gradient
and coefficient uncertainty test whether finite-sidecar fluctuations explain
the observed empirical offset. There is no extra class-prior term because
membership priors are equal, and no theta-prior term survives because both
membership classes receive the same independently sampled theta. Signed
weights are nonnegative in this unsigned R0 sidecar; the code rejects negative
weights. In the full deterministic diagnostic, uniform theta and membership
sampling are integrated by enumerating every grid point and both classes once.
"""


if __name__ == "__main__":
    main()
