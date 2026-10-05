"""Separate R0 representation, deterministic BCE, and residual-branch effects."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from joint_event_reconstruction.rise import (
    ReferenceAnchoredBilinearRatioHead,
    unsigned_loss,
)
from joint_event_reconstruction.sbi_data import ReweightedRatioBatch, SBISidecar
from scripts.validation.evaluate_rise_analytic_checkpoint import predict
from scripts.validation.qualify_rise_parameter_conditioning import (
    diagnostics as standard_diagnostics,
)
from scripts.validation.qualify_rise_parameter_conditioning import (
    exact_log_ratio,
    sample,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/experiments/rise_parameter_conditioning/r0.json"
OUT = ROOT / "results/experiments/rise/r0-neural-diagnosis-v1"
BETA_TRUTH = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def model(config, seed: int) -> ReferenceAnchoredBilinearRatioHead:
    torch.manual_seed(seed)
    return ReferenceAnchoredBilinearRatioHead(
        3, 2, int(config["hidden"]), config["reference_theta"]
    )


def reference_sample(config, sidecar):
    rng = np.random.default_rng(config["seed"])
    rng.normal(size=(9 * 6000, 3))
    x = rng.normal(size=(int(config["training_reference_events"]), 3))
    grid = np.asarray(
        [(a, b) for a in config["training_axis"] for b in config["training_axis"]],
        dtype=np.float64,
    )
    weights = np.stack([np.exp(exact_log_ratio(x, theta, "R0")) for theta in grid], axis=1)
    np.testing.assert_array_equal(sidecar.grid, grid)
    np.testing.assert_allclose(sidecar.weights, weights, rtol=1e-12, atol=1e-12)
    return x, grid, weights


def set_exact_basis(head):
    """Set the existing head to the seven-term R0 likelihood ratio exactly."""
    with torch.no_grad():
        head.bilinear.zero_()
        head.offset.weight.zero_()
        for layer in head.residual.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
        # polynomial columns are [a, b, a^2, ab, b^2].
        head.bilinear[0, 0] = 1
        head.bilinear[1, 1] = 1
        head.bilinear[2, 0] = 1
        head.bilinear[2, 1] = 1
        head.offset.weight[0, 2:5] = -1


def direct_training_data(config, x, grid, weights):
    rng = np.random.default_rng(config["seed"] + 311)
    permutation = rng.permutation(len(x))
    split = int(0.8 * len(x))
    train_rows, validation_rows = permutation[:split], permutation[split:]
    xs, thetas, targets = [], [], []
    for point, theta in enumerate(grid):
        reference = x[train_rows]
        target_rows = rng.choice(
            train_rows,
            size=len(train_rows),
            replace=True,
            p=weights[train_rows, point] / weights[train_rows, point].sum(),
        )
        target = x[target_rows]
        for values in (reference, target):
            xs.append(values)
            thetas.append(np.broadcast_to(theta, (len(values), 2)))
            targets.append(exact_log_ratio(values, theta, "R0"))
    validation = {"reference": x[validation_rows], "grid": grid}
    return (
        np.concatenate(xs).astype(np.float32),
        np.concatenate(thetas).astype(np.float32),
        np.concatenate(targets).astype(np.float32),
        validation,
    )


def point_metrics(head, x, theta, toy="R0"):
    predicted = predict(head, x, np.broadcast_to(theta, (len(x), 2)))
    truth = exact_log_ratio(x, theta, toy)
    error = predicted - truth
    return {
        "events": len(x),
        "log_ratio_bias": float(error.mean()),
        "log_ratio_RMSE": float(np.sqrt(np.mean(error**2))),
        "absolute_error_q90": float(np.quantile(np.abs(error), 0.90)),
        "absolute_error_q99": float(np.quantile(np.abs(error), 0.99)),
        "correlation": float(np.corrcoef(predicted, truth)[0, 1])
        if np.std(truth) > 0
        else None,
    }


def regression_diagnostics(head, config, validation, seed):
    rng = np.random.default_rng(seed)
    train_grid = validation["grid"]
    train_rows = []
    heldout_rows = []
    for theta in train_grid:
        reference = rng.normal(size=(24_000, 3))
        target = sample(rng, len(reference), theta, "R0")
        row = point_metrics(head, np.concatenate((reference, target)), theta)
        row["theta"] = theta.tolist()
        train_rows.append(row)
    for theta in config["heldout_theta"]:
        theta = np.asarray(theta, dtype=np.float64)
        reference = rng.normal(size=(24_000, 3))
        target = sample(rng, len(reference), theta, "R0")
        row = point_metrics(head, np.concatenate((reference, target)), theta)
        row["theta"] = theta.tolist()
        heldout_rows.append(row)
    return {
        "training_theta_RMSE": float(
            np.sqrt(np.mean([row["log_ratio_RMSE"] ** 2 for row in train_rows]))
        ),
        "heldout_theta_RMSE": [row["log_ratio_RMSE"] for row in heldout_rows],
        "heldout_theta_bias": [row["log_ratio_bias"] for row in heldout_rows],
        "training_points": train_rows,
        "heldout_points": heldout_rows,
    }


def fit_direct_regression(config, sidecar, x, grid, weights, out):
    train_x, train_theta, train_target, validation = direct_training_data(
        config, x, grid, weights
    )
    device = torch.device("cpu")
    x_tensor = torch.as_tensor(train_x, device=device)
    theta_tensor = torch.as_tensor(train_theta, device=device)
    target_tensor = torch.as_tensor(train_target, device=device)
    head = model(config, int(config["seed"]) + 312)
    optimizer = torch.optim.Adam(head.parameters(), lr=float(config["learning_rate"]))
    rng = np.random.default_rng(config["seed"] + 313)
    history = []
    start = time.perf_counter()
    steps = 4000
    batch_size = int(config["batch_size"])
    for step in range(1, steps + 1):
        rows = torch.as_tensor(
            rng.integers(0, len(train_x), size=batch_size), dtype=torch.long
        )
        prediction = head(x_tensor[rows], theta_tensor[rows])
        loss = F.mse_loss(prediction, target_tensor[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss) or any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
        ):
            raise FloatingPointError(f"Direct R0 regression became nonfinite at step {step}")
        optimizer.step()
        if step in (1000, 2000, 4000):
            record = {
                "step": step,
                "minibatch_MSE": float(loss.detach()),
                "wall_seconds": float(time.perf_counter() - start),
            }
            history.append(record)
            torch.save(head.state_dict(), out / f"direct_step_{step}.pt")
            print("RISE direct regression", record, flush=True)
    diagnostics = regression_diagnostics(head, config, validation, int(config["seed"]) + 314)
    result = {
        "status": "completed",
        "method": "ordinary log-ratio MSE regression; same bilinear head; training theta grid only",
        "head_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "reference_training_events": len(x),
        "training_event_fraction": 0.8,
        "training_theta_grid": grid.tolist(),
        "optimizer": "Adam minibatch MSE",
        "steps": steps,
        "batch_size": batch_size,
        "history": history,
        **diagnostics,
        "wall_seconds": float(time.perf_counter() - start),
    }
    write_json(out / "direct_regression.json", result)
    return head, result


def make_objective_population(sidecar, adapter, x):
    n, grid_count = len(x), len(sidecar.grid)
    target_x = np.tile(x, (grid_count, 1))
    target_theta = np.repeat(sidecar.grid, n, axis=0)
    target_weight = sidecar.weights.T.reshape(-1)
    reference_x = np.tile(x, (grid_count, 1))
    reference_theta = target_theta.copy()
    reference_weight = np.tile(sidecar.weights[:, adapter.reference_point], grid_count)
    features = np.concatenate((target_x, reference_x)).astype(np.float32)
    theta = np.concatenate((target_theta, reference_theta)).astype(np.float32)
    membership = np.concatenate((np.ones(n * grid_count), np.zeros(n * grid_count))).astype(np.float32)
    weight = np.concatenate((target_weight, reference_weight)).astype(np.float32)
    target_mass = adapter.masses[:, 0] / adapter.population_size
    reference_mass = adapter.masses[:, 2] / adapter.population_size
    class_normalisation = np.column_stack(
        (
            np.concatenate((np.repeat(target_mass, n), np.ones(n * grid_count))),
            np.concatenate((np.ones(n * grid_count), np.full(n * grid_count, reference_mass[0]))),
        )
    ).astype(np.float32)
    if not np.isfinite(features).all() or not np.isfinite(weight).all():
        raise FloatingPointError("Nonfinite deterministic R0 objective population")
    return tuple(
        torch.as_tensor(value, dtype=torch.float32)
        for value in (features, theta, membership, weight, class_normalisation)
    )


def branch_components(head, h, theta):
    reference = head.reference_theta.to(device=h.device, dtype=h.dtype)
    delta = theta - reference
    left, right = head.quadratic_indices

    def evaluate(change):
        polynomial = torch.cat(
            (change, change[:, left] * change[:, right]), dim=-1
        )
        phi = torch.cat((polynomial, head.parameter_encoder(polynomial)), dim=-1)
        explicit = ((h @ head.bilinear) * phi).sum(-1) + head.offset(phi)[:, 0]
        residual = head.residual(torch.cat((h, phi), dim=-1))[:, 0]
        return explicit, residual

    explicit, residual = evaluate(delta)
    explicit_zero, residual_zero = evaluate(torch.zeros_like(delta))
    return explicit - explicit_zero, residual - residual_zero


def branch_diagnostics(head, population, limit=16_384):
    h, theta, membership, weight, normalisation = population
    theta_values = torch.unique(theta, dim=0)
    per_group = max(1, limit // (2 * len(theta_values)))
    selected = []
    for point in theta_values:
        point_mask = (theta == point).all(dim=1)
        for label in (0.0, 1.0):
            rows = torch.nonzero(point_mask & (membership == label), as_tuple=False)[:, 0]
            if len(rows) > per_group:
                offsets = torch.linspace(0, len(rows) - 1, per_group).long()
                rows = rows[offsets]
            selected.append(rows)
    rows = torch.cat(selected)
    h, theta = h[rows], theta[rows]
    h = h.detach().requires_grad_(True)
    membership, weight, normalisation = (
        membership[rows], weight[rows], normalisation[rows]
    )
    head.zero_grad(set_to_none=True)
    logits = head(h, theta)
    loss = unsigned_loss(logits, membership, weight, normalisation)
    loss.backward()
    explicit_parameters = [head.bilinear, *head.offset.parameters(), *head.parameter_encoder.parameters()]
    residual_parameters = list(head.residual.parameters())

    def norm(parameters):
        values = [p.grad.detach().norm().square() for p in parameters if p.grad is not None]
        return float(torch.sqrt(torch.stack(values).sum())) if values else 0.0

    with torch.no_grad():
        explicit, residual = branch_components(head, h[:2048], theta[:2048])
        explicit_flat, residual_flat = explicit.flatten(), residual.flatten()
        branch_cosine = torch.dot(explicit_flat, residual_flat) / (
            torch.linalg.vector_norm(explicit_flat)
            * torch.linalg.vector_norm(residual_flat)
        ).clamp_min(1e-20)
    return {
        "diagnostic_loss": float(loss.detach()),
        "explicit_output_RMS": float(explicit.square().mean().sqrt()),
        "residual_output_RMS": float(residual.square().mean().sqrt()),
        "explicit_residual_output_cosine": float(branch_cosine),
        "explicit_gradient_norm": norm(explicit_parameters),
        "residual_gradient_norm": norm(residual_parameters),
        "event_representation_gradient_norm": float(torch.linalg.vector_norm(h.grad)),
    }


def lbfgs_fit(head, population, *, max_iter=80):
    h, theta, membership, weight, normalisation = population
    optimizer = torch.optim.LBFGS(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=0.5,
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=1e-7,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = unsigned_loss(head(h, theta), membership, weight, normalisation)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite deterministic R0 unsigned BCE")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
        ):
            raise FloatingPointError("Nonfinite deterministic R0 gradient")
        evaluations += 1
        return loss

    initial_loss = float(
        unsigned_loss(head(h, theta), membership, weight, normalisation).detach()
    )
    start = time.perf_counter()
    optimizer.step(closure)
    end_loss = float(unsigned_loss(head(h, theta), membership, weight, normalisation))
    return {
        "initial_loss": initial_loss,
        "final_loss": end_loss,
        "closure_evaluations": evaluations,
        "wall_seconds": float(time.perf_counter() - start),
        "optimizer": "torch.optim.LBFGS, deterministic full batch, strong Wolfe",
    }


def load_stochastic_head(config, out):
    checkpoint = ROOT / "results/experiments/rise/parameter-conditioning-v1/R0/bilinear/step_16000.pt"
    head = model(config, int(config["seed"]))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(saved["state_dict"])
    return head


def main():
    parser_config = json.loads(CONFIG_PATH.read_text())
    parser_out = OUT
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=parser_out)
    parser.add_argument("--lbfgs-iterations", type=int, default=80)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite RISE R0 diagnosis: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    sidecar = SBISidecar(ROOT / parser_config["training_sidecar"])
    x, grid, weights = reference_sample(parser_config, sidecar)
    adapter = ReweightedRatioBatch(
        sidecar,
        sidecar.event_id,
        reference_point=4,
        seed=int(parser_config["seed"]) + 1,
    )

    # The functional family contains the exact R0 answer without regression.
    exact_control = model(parser_config, int(parser_config["seed"]) + 310)
    set_exact_basis(exact_control)
    rng = np.random.default_rng(int(parser_config["seed"]) + 310)
    x_control = rng.normal(size=(2048, 3)).astype(np.float32)
    theta_control = rng.uniform(-0.8, 0.8, size=(2048, 2)).astype(np.float32)
    exact_control_prediction = predict(exact_control, x_control, theta_control)
    exact_control_target = np.asarray(
        [exact_log_ratio(xx[None], tt, "R0")[0] for xx, tt in zip(x_control, theta_control)]
    )
    exact_control_error = float(np.max(np.abs(exact_control_prediction - exact_control_target)))
    if exact_control_error > 2e-6:
        raise AssertionError(f"Bilinear head exact R0 coefficient control failed: {exact_control_error}")

    _, direct = fit_direct_regression(
        parser_config, sidecar, x, grid, weights, out
    )

    # Use the complete 48k R0 training EVENT_ID population and unchanged sidecar
    # weights, masses, memberships, and unsigned_loss for the deterministic BCE.
    population = make_objective_population(sidecar, adapter, x.astype(np.float32))
    stochastic_head = load_stochastic_head(parser_config, out)
    stochastic_branch = branch_diagnostics(stochastic_head, population)
    deterministic_results = {}
    for start_seed in (int(parser_config["seed"]) + 401, int(parser_config["seed"]) + 402):
        head = model(parser_config, start_seed)
        branch_before = branch_diagnostics(head, population)
        fit = lbfgs_fit(head, population, max_iter=args.lbfgs_iterations)
        branch_after = branch_diagnostics(head, population)
        validation = standard_diagnostics(head, x, parser_config, "R0")
        name = f"seed_{start_seed}"
        torch.save(head.state_dict(), out / f"deterministic_{name}.pt")
        deterministic_results[name] = {
            "fit": fit,
            "branch_before": branch_before,
            "branch_after": branch_after,
            "qualification": validation,
        }
        print(
            "RISE deterministic BCE",
            name,
            fit["initial_loss"],
            fit["final_loss"],
            validation["qualification_pass"],
            flush=True,
        )

    explicit_only = model(parser_config, int(parser_config["seed"]) + 403)
    with torch.no_grad():
        for layer in explicit_only.residual.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
    for parameter in explicit_only.residual.parameters():
        parameter.requires_grad_(False)
    explicit_before = branch_diagnostics(explicit_only, population)
    explicit_fit = lbfgs_fit(explicit_only, population, max_iter=args.lbfgs_iterations)
    explicit_after = branch_diagnostics(explicit_only, population)
    explicit_qualification = standard_diagnostics(explicit_only, x, parser_config, "R0")
    torch.save(explicit_only.state_dict(), out / "explicit_only.pt")

    stochastic_validation = standard_diagnostics(
        stochastic_head, x, parser_config, "R0"
    )
    summary = {
        "status": "completed",
        "R0_objective": "maintained ReweightedRatioBatch sidecar and unsigned_loss unchanged",
        "population_events": len(x),
        "theta_grid": grid.tolist(),
        "exact_bilinear_control_max_abs_error": exact_control_error,
        "direct_regression": direct,
        "stochastic_16000_reference": {
            "checkpoint": str(
                ROOT / "results/experiments/rise/parameter-conditioning-v1/R0/bilinear/step_16000.pt"
            ),
            "branch_diagnostics": stochastic_branch,
            "qualification": stochastic_validation,
        },
        "deterministic_full_batch_bce": deterministic_results,
        "explicit_only": {
            "fit": explicit_fit,
            "branch_before": explicit_before,
            "branch_after": explicit_after,
            "qualification": explicit_qualification,
            "residual_requires_grad": any(
                parameter.requires_grad for parameter in explicit_only.residual.parameters()
            ),
        },
        "R1_run": False,
        "R1_reason": "R0 remains gated until this diagnosis establishes the unchanged head or one minimal correction passes the existing qualification.",
    }
    write_json(out / "diagnosis.json", summary)
    print(json.dumps({
        "exact_control_max_abs_error": exact_control_error,
        "direct_training_theta_RMSE": direct["training_theta_RMSE"],
        "direct_heldout_RMSE": direct["heldout_theta_RMSE"],
        "deterministic_bce": {k: v["qualification"]["qualification_pass"] for k,v in deterministic_results.items()},
        "explicit_only": explicit_qualification["qualification_pass"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
