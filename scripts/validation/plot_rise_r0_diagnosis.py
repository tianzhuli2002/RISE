"""Plot held-out R0 ratio and likelihood closure from the saved qualifications."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/experiments/rise"
OUT = RESULTS / "parameter-conditioning-v1/R0-qualification"


def read(path):
    return json.loads(path.read_text())


def max_heldout_rmse(points):
    values = [
        point["log_ratio_RMSE"]
        for point in points
        if point.get("parameter_set") == "heldout"
    ]
    if not values:
        values = [point["log_ratio_RMSE"] for point in points]
    return max(values)


def main():
    diagnosis = read(RESULTS / "r0-neural-diagnosis-v1/complete-02/diagnosis.json")
    comparison = read(
        RESULTS / "r0-neural-diagnosis-v1/zero-residual-output/R0_architecture_comparison.json"
    )
    rows = []

    for key, label in (("existing_4000", "RatioHead 4k"), ("existing_16000", "RatioHead 16k")):
        item = comparison["matched_stochastic_unsigned_loss_comparison"][key]
        rows.append((label, "stochastic", max(p["log_ratio_RMSE"] for p in item["heldout"]),
                     item["likelihood_summary"]["centered_surface_rmse"], item["qualification_pass"]))
    for key, label in (("bilinear_16000", "Bilinear 16k"),
                       ("bilinear_zero_residual_16000", "Bilinear zero-residual 16k")):
        item = comparison["matched_stochastic_unsigned_loss_comparison"][key]
        rows.append((label, "stochastic", max(p["log_ratio_RMSE"] for p in item["heldout"]),
                     item["likelihood_summary"]["centered_surface_rmse"], item["qualification_pass"]))

    explicit_stochastic = read(
        RESULTS / "parameter-conditioning-v1/R0-explicit-only-stochastic/decision.json"
    )["qualification_result"]
    rows.append(("Bilinear explicit-only 16k", "stochastic",
                 max_heldout_rmse(explicit_stochastic["ratio_points"]),
                 explicit_stochastic["likelihood"]["summary"]["centered_surface_rmse"],
                 explicit_stochastic["qualification_pass"]))

    deterministic = diagnosis["deterministic_full_batch_bce"]
    for key, label in (("seed_20261403", "Bilinear full-batch A"),
        ("seed_20261404", "Bilinear full-batch B")):
        item = deterministic[key]["qualification"]
        rows.append((label, "deterministic", max_heldout_rmse(item["ratio_points"]),
                     item["likelihood"]["summary"]["centered_surface_rmse"],
                     item["qualification_pass"]))
    explicit_det = diagnosis["explicit_only"]["qualification"]
    rows.append(("Bilinear explicit-only LBFGS", "deterministic",
                 max_heldout_rmse(explicit_det["ratio_points"]),
                 explicit_det["likelihood"]["summary"]["centered_surface_rmse"],
                 explicit_det["qualification_pass"]))

    balanced_path = RESULTS / "parameter-conditioning-v1/R0-theta-balanced-exact-risk/step_16000.json"
    if balanced_path.exists():
        item = read(balanced_path)
        rows.append(("Bilinear theta/class-balanced 16k", "balanced stochastic",
                     max_heldout_rmse(item["ratio_points"]),
                     item["likelihood"]["summary"]["centered_surface_rmse"],
                     item["qualification_pass"]))

    for directory, label in (
        ("warmstart_stochastic", "Warm-start Adam 0.002 / 16k"),
        ("warmstart_stochastic_lr_2e-4", "Warm-start Adam 0.0002 / 16k"),
    ):
        path = RESULTS / "r0-neural-diagnosis-v1/complete-02" / directory / "comparison.json"
        if path.exists():
            item = read(path)["results"]["16000"]
            rows.append((label, "stochastic", max_heldout_rmse(item["ratio_points"]),
                         item["likelihood"]["summary"]["centered_surface_rmse"],
                         item["qualification_pass"]))

    colors = {"stochastic": "#3B6FB6", "deterministic": "#C75B39", "balanced stochastic": "#248A68"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    y = list(range(len(rows)))
    for axis, metric_index, xlabel in (
        (axes[0], 2, "Maximum held-out log-ratio RMSE"),
        (axes[1], 3, "Centered likelihood-surface RMSE"),
    ):
        for index, row in enumerate(rows):
            mode, passed = row[1], row[4]
            value = row[metric_index]
            axis.barh(index, value, color=colors[mode], alpha=0.92)
            axis.text(value, index, "  PASS" if passed else "  fail", va="center", fontsize=8,
                      color="#176b48" if passed else "#777777")
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.22)
        axis.invert_yaxis()
    axes[0].set_yticks(y, [row[0] for row in rows])
    axes[0].set_title("Event-level ratio closure")
    axes[1].set_title("Pseudo-data likelihood closure")
    fig.suptitle("R0 neural qualification: solution depends on optimization path")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "RISE_R0_architecture_and_optimization_diagnostic.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {OUT / 'RISE_R0_architecture_and_optimization_diagnostic.png'}")

    branch_path = RESULTS / "parameter-conditioning-v1/R0/bilinear/branch_diagnostics.json"
    if branch_path.exists():
        branch = read(branch_path)["checkpoints"]
        steps = sorted(map(int, branch))
        explicit_rms = [branch[str(step)]["explicit_output_RMS"] for step in steps]
        residual_rms = [branch[str(step)]["residual_output_RMS"] for step in steps]
        explicit_grad = [branch[str(step)]["explicit_gradient_norm"] for step in steps]
        residual_grad = [branch[str(step)]["residual_gradient_norm"] for step in steps]
        event_grad = [branch[str(step)]["event_representation_gradient_norm"] for step in steps]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(steps, explicit_rms, "o-", color="#3B6FB6", label="explicit")
        axes[0].plot(steps, residual_rms, "s--", color="#C75B39", label="residual")
        axes[0].set_ylabel("anchored branch output RMS")
        axes[0].legend()
        axes[1].plot(steps, explicit_grad, "o-", color="#3B6FB6", label="explicit parameters")
        axes[1].plot(steps, residual_grad, "s--", color="#C75B39", label="residual parameters")
        axes[1].plot(steps, event_grad, "^-.", color="#248A68", label="event representation")
        axes[1].set_ylabel("unsigned BCE gradient L2")
        axes[1].legend(fontsize=8)
        for axis in axes:
            axis.set_xlabel("stochastic training step")
            axis.grid(alpha=0.22)
        fig.suptitle("Reference-anchored bilinear R0 branch diagnostics")
        fig.tight_layout()
        fig.savefig(OUT / "RISE_R0_branch_diagnostics.png", dpi=180)
        plt.close(fig)
        print(f"Wrote {OUT / 'RISE_R0_branch_diagnostics.png'}")


if __name__ == "__main__":
    main()
