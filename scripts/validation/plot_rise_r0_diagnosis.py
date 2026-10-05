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


if __name__ == "__main__":
    main()
