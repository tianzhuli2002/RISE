"""Anchoring, analytic representability, gradients and independent R1 density."""

from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import multivariate_normal

from joint_event_reconstruction.rise import (
    RatioHead,
    ReferenceAnchoredBilinearRatioHead,
    RiseModule,
)
from scripts.validation.qualify_rise_parameter_conditioning import (
    exact_log_ratio,
    gaussian_parameters,
)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261002)
    head = ReferenceAnchoredBilinearRatioHead(3, 2, 64, [0.2, -0.1])
    x = torch.randn(32, 3, requires_grad=True)
    theta = torch.randn(32, 2, requires_grad=True)
    reference = torch.tensor([0.2, -0.1]).expand(32, -1)
    assert torch.equal(head(x, reference), torch.zeros(32))
    head(x, theta).square().mean().backward()
    assert x.grad.abs().sum() > 0 and theta.grad.abs().sum() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None
    )
    owner = SimpleNamespace(
        event_model=SimpleNamespace(
            encode_visible=lambda batch: {"pooled": batch["h"]}
        ),
        ratio_head=head,
        sbi_weight=1.0,
        mode="0",
    )
    batch = {
        "h": x.detach(),
        "theta": theta.detach(),
        "membership": torch.arange(32) % 2,
        "signed_weight": torch.ones(32),
        "class_normalisation": torch.ones(32, 2),
    }
    integrated = RiseModule.forward(owner, batch)
    assert torch.isfinite(integrated["loss"]) and integrated["log_ratio"].shape == (32,)
    model = ReferenceAnchoredBilinearRatioHead(3, 2, 64, [0, 0])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.bilinear[:, :2] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        model.offset.weight[0, 2:5] = torch.tensor([-1.0, -1.0, -1.0])
    theta = theta.detach()
    expected = exact_log_ratio(x.detach().numpy(), theta[0].numpy(), "R0")
    output = model(x, theta[0].expand(32, -1)).detach().numpy()
    np.testing.assert_allclose(output, expected, atol=2e-6, rtol=2e-6)
    rng = np.random.default_rng(310)
    values = rng.normal(size=(128, 3))
    for point in [[-0.8, -0.8], [0.35, -0.4], [1.0, -0.5], [1.0, 1.0]]:
        mu, covariance = gaussian_parameters(point, "R1")
        assert (np.linalg.eigvalsh(covariance) > 0).all()
        expected = multivariate_normal.logpdf(
            values, mean=mu, cov=covariance
        ) - multivariate_normal.logpdf(values, mean=np.zeros(3), cov=np.eye(3))
        np.testing.assert_allclose(
            exact_log_ratio(values, point, "R1"), expected, atol=1e-12
        )
    print("anchoring/gradients/R0 representability/independent R1 density: PASS")
    print(
        "parameter counts",
        sum(p.numel() for p in RatioHead(3, 2, 64).parameters()),
        sum(p.numel() for p in model.parameters()),
    )


if __name__ == "__main__":
    main()
