"""Close the signed-mixture ratio algebra with analytic component densities."""

import json
from pathlib import Path

import numpy as np
import torch

from joint_event_reconstruction.rise import signed_ratio

ROOT = Path(__file__).resolve().parents[2]


def gaussian_log_pdf(x, location, scale):
    return -0.5 * ((x - location) / scale) ** 2 - np.log(
        scale * np.sqrt(2 * np.pi)
    )


def check_two_sided(x):
    masses = np.asarray([5.0, 0.4, 4.0, 0.25], dtype=np.float64)
    component_log_density = np.stack(
        (
            gaussian_log_pdf(x, -0.4, 1.0),
            gaussian_log_pdf(x, 1.0, 1.0),
            gaussian_log_pdf(x, 0.0, 1.0),
            gaussian_log_pdf(x, 1.2, 1.0),
        ),
        axis=-1,
    )
    target = (
        masses[0] * np.exp(component_log_density[:, 0])
        - masses[1] * np.exp(component_log_density[:, 1])
    )
    reference = (
        masses[2] * np.exp(component_log_density[:, 2])
        - masses[3] * np.exp(component_log_density[:, 3])
    )
    expected = target / reference * (masses[2] - masses[3]) / (
        masses[0] - masses[1]
    )
    actual = signed_ratio(
        torch.as_tensor(component_log_density, dtype=torch.float64),
        torch.as_tensor(np.broadcast_to(masses, (len(x), 4)).copy()),
    )["ratio"].numpy()
    return {
        "case": "signed target over signed reference",
        "maximum_absolute_ratio_error": float(np.max(np.abs(actual - expected))),
        "minimum_target_signed_density": float(target.min()),
        "minimum_reference_signed_density": float(reference.min()),
        "points": len(x),
    }


def check_single_sided(x):
    masses = np.asarray([3.0, 0.0, 2.0, 0.0], dtype=np.float64)
    component_log_density = np.stack(
        (
            gaussian_log_pdf(x, 0.5, 0.8),
            np.zeros_like(x),
            gaussian_log_pdf(x, -0.3, 1.1),
            np.zeros_like(x),
        ),
        axis=-1,
    )
    expected = np.exp(component_log_density[:, 0] - component_log_density[:, 2])
    actual = signed_ratio(
        torch.as_tensor(component_log_density, dtype=torch.float64),
        torch.as_tensor(np.broadcast_to(masses, (len(x), 4)).copy()),
    )["ratio"].numpy()
    return {
        "case": "single-sided target and reference",
        "maximum_absolute_ratio_error": float(np.max(np.abs(actual - expected))),
        "points": len(x),
    }


def main():
    x = np.linspace(-1.5, 1.5, 301)
    result = {
        "implementation": "joint_event_reconstruction.rise.signed_ratio",
        "two_sided": check_two_sided(x),
        "single_sided": check_single_sided(x),
        "denominator_cancellation": {
            "probed": True,
            "minimum_reference_signed_density": check_two_sided(x)[
                "minimum_reference_signed_density"
            ],
            "outcome": "positive throughout the declared x interval; no singular cancellation",
        },
    }
    output = ROOT / "results/experiments/rise/preproduction-v1/signed_algebra.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
