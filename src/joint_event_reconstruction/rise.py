"""Parameterized ratio estimation with optional JER reconstruction supervision.

Signed mode learns four normalized nonnegative component densities through
an equal-prior softmax classifier. Absolute weights enter its proper loss.
Physical target/reference masses enter the final signed-mixture ratio, never
negative BCE. Both target and reference may contain negative components.
See arXiv:2410.10216 for signed-mixture likelihood-ratio decomposition.
"""

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from joint_event_reconstruction.cli.run import _load_initial_components
from joint_event_reconstruction.training.joint import JointEventModule


class RatioHead(nn.Module):
    def __init__(self, event_width, parameter_dim, hidden, signed=False):
        super().__init__()
        self.parameter_dim = parameter_dim
        self.parameter_encoder = nn.Sequential(
            nn.Linear(parameter_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.network = nn.Sequential(
            nn.Linear(event_width + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 if signed else 1),
        )
        self.signed = signed

    def forward(self, h, theta):
        if (
            theta.shape != (len(h), self.parameter_dim)
            or not torch.isfinite(theta).all()
        ):
            raise ValueError("Theta must be finite [event,parameter]")
        result = self.network(
            torch.cat((h, self.parameter_encoder(theta.to(h))), dim=-1)
        )
        return result if self.signed else result[:, 0]


class ReferenceAnchoredBilinearRatioHead(nn.Module):
    """Unsigned log ratio with explicit h-theta interaction and exact anchoring.

    Degree-two delta-theta terms expose the mean-shift Gaussian's intercept.
    A small residual learns event dependence beyond the bilinear term. The
    subtraction uses the same event h, so r(h;theta0,theta0) is exactly one.
    """

    def __init__(self, event_width, parameter_dim, hidden, reference_theta):
        super().__init__()
        self.signed = False
        self.event_width, self.parameter_dim = event_width, parameter_dim
        reference = torch.as_tensor(reference_theta, dtype=torch.float32)
        if reference.shape != (parameter_dim,) or not torch.isfinite(reference).all():
            raise ValueError("Reference theta must be finite [parameter]")
        if min(event_width, parameter_dim, hidden) <= 0:
            raise ValueError("Positive event, parameter and hidden widths required")
        self.register_buffer("reference_theta", reference)
        indices = torch.triu_indices(parameter_dim, parameter_dim)
        self.register_buffer("quadratic_indices", indices)
        polynomial_width = parameter_dim + indices.shape[1]
        embedding_width = max(1, hidden // 2)
        self.parameter_encoder = nn.Sequential(
            nn.Linear(polynomial_width, embedding_width), nn.SiLU(),
            nn.Linear(embedding_width, embedding_width),
        )
        interaction_width = polynomial_width + embedding_width
        self.bilinear = nn.Parameter(torch.empty(event_width, interaction_width))
        nn.init.xavier_uniform_(self.bilinear)
        self.offset = nn.Linear(interaction_width, 1, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(event_width + interaction_width, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def _g(self, h, delta):
        left, right = self.quadratic_indices
        polynomial = torch.cat((delta, delta[:, left] * delta[:, right]), dim=-1)
        phi = torch.cat((polynomial, self.parameter_encoder(polynomial)), dim=-1)
        return (
            ((h @ self.bilinear) * phi).sum(-1)
            + self.offset(phi)[:, 0]
            + self.residual(torch.cat((h, phi), dim=-1))[:, 0]
        )

    def forward(self, h, theta, reference_theta=None):
        if (
            h.ndim != 2 or h.shape[1] != self.event_width
            or theta.shape != (len(h), self.parameter_dim)
            or not torch.isfinite(h).all() or not torch.isfinite(theta).all()
        ):
            raise ValueError("Finite event [B,H] and theta [B,P] required")
        reference = self.reference_theta if reference_theta is None else reference_theta
        reference = torch.as_tensor(reference, device=h.device, dtype=h.dtype)
        if reference.shape not in {(self.parameter_dim,), theta.shape} or not torch.isfinite(reference).all():
            raise ValueError("Reference theta must be finite [P] or [B,P]")
        delta = theta.to(h) - reference
        return self._g(h, delta) - self._g(h, torch.zeros_like(delta))


def unsigned_loss(logit, membership, signed_weight, normalisation=None):
    if (
        membership.shape != logit.shape
        or signed_weight.shape != logit.shape
        or not ((membership == 0) | (membership == 1)).all()
    ):
        raise ValueError("Invalid target/reference membership")
    if (signed_weight < 0).any() or not torch.isfinite(signed_weight).all():
        raise ValueError("Use signed-mixture mode for negative weights")
    if normalisation is not None:
        if (
            normalisation.shape != (len(logit), 2)
            or not torch.isfinite(normalisation).all()
            or (normalisation <= 0).any()
        ):
            raise ValueError("Invalid fixed class normalisations")
        # `normalisation` is the complete-population total divided by the
        # number of uniformly sampled training EVENT_IDs. This keeps the
        # minibatch risk on an O(1) scale as the population grows.
        mass = normalisation.gather(1, (1 - membership.long())[:, None])[:, 0]
        return (
            F.binary_cross_entropy_with_logits(
                logit, membership.to(logit), reduction="none"
            )
            * signed_weight
            / mass
        ).mean()
    # Class-balanced loss: equal target/reference priors, hence logit = log ratio.
    terms = []
    for label in (0, 1):
        selection = membership == label
        denominator = signed_weight[selection].sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"No nonzero weight for membership={label}")
        terms.append(
            (
                F.binary_cross_entropy_with_logits(
                    logit[selection], membership[selection].to(logit), reduction="none"
                )
                * signed_weight[selection]
            ).sum()
            / denominator
        )
    return torch.stack(terms).mean()


def component_labels(membership, signed_weight):
    if (
        membership.shape != signed_weight.shape
        or not ((membership == 0) | (membership == 1)).all()
        or not torch.isfinite(signed_weight).all()
    ):
        raise ValueError("Invalid signed SBI population")
    # 0 target+, 1 target-, 2 reference+, 3 reference-.
    return 2 * (1 - membership.long()) + (signed_weight < 0).long()


def signed_loss(logits, membership, signed_weight, component_masses):
    labels = component_labels(membership, signed_weight)
    if logits.shape != (len(labels), 4) or component_masses.shape != logits.shape:
        raise ValueError("Signed component logits/masses must be [B,4]")
    if not torch.isfinite(component_masses).all() or (component_masses < 0).any():
        raise ValueError("Component masses must be finite nonnegative totals")
    mass = component_masses.gather(1, labels[:, None])[:, 0]
    if ((mass <= 0) & (signed_weight != 0)).any():
        raise ValueError("Observed event in a zero-mass signed component")
    allowed = component_masses > 0
    if (~allowed.any(-1)).any():
        raise ValueError("Empty signed mixture")
    # Component masses are complete TRAIN totals divided by the number of
    # uniformly sampled training EVENT_IDs, never minibatch estimates. This
    # makes |w|/mass have unit expectation within each component and gives all
    # observed components equal classifier priors without an N-dependent loss.
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    supported = signed_weight != 0
    if not supported.any():
        raise ValueError("No nonzero SBI weight in batch")
    ce = F.cross_entropy(
        logits[supported].masked_fill(~allowed[supported], -torch.inf),
        labels[supported],
        reduction="none",
    )
    return (ce * signed_weight[supported].abs() / safe_mass[supported]).sum() / len(
        labels
    )


def signed_ratio(logits, masses):
    if masses.shape != logits.shape or logits.shape[-1] != 4:
        raise ValueError("Expected four component masses per prediction")
    if not torch.isfinite(masses).all() or (masses < 0).any():
        raise ValueError("Invalid signed masses")
    normalization = torch.stack(
        (masses[:, 0] - masses[:, 1], masses[:, 2] - masses[:, 3]), dim=-1
    )
    if (normalization <= 0).any():
        raise ValueError("Target/reference signed normalization must be positive")
    # Common shift cancels; this is stable under arbitrary softmax gauge.
    density = (logits - logits.max(-1, keepdim=True).values).exp() * masses
    target = density[:, 0] - density[:, 1]
    reference = density[:, 2] - density[:, 3]
    if not torch.isfinite(reference).all() or (reference <= 0).any():
        raise ValueError("Nonpositive reference density in signed-mixture prediction")
    ratio = target / reference * normalization[:, 1] / normalization[:, 0]
    if not torch.isfinite(ratio).all() or (ratio == 0).any():
        raise ValueError("Invalid/zero signed likelihood ratio")
    return {
        "ratio": ratio,
        "ratio_sign": ratio.sign(),
        "log_abs_ratio": ratio.abs().log(),
    }


class RiseModule(JointEventModule):
    def __init__(
        self,
        experiment,
        visible_config,
        *,
        research,
        initialize_pretrained=True,
        **kwargs,
    ):
        super().__init__(experiment, visible_config, **kwargs)
        self.research = OmegaConf.to_container(OmegaConf.create(research), resolve=True)
        self.mode = str(research["mode"])
        if self.mode not in {"0", "P", "A"}:
            raise ValueError("RISE mode must be 0/P/A")
        head = research.get("ratio_head", "existing")
        arguments = (
            int(experiment.model.latent.d_model),
            len(research["parameter_names"]), int(research["hidden"]),
        )
        if head == "existing":
            self.ratio_head = RatioHead(*arguments, signed=bool(research["signed_mixture"]))
        elif head == "reference_anchored_bilinear":
            if research["signed_mixture"]:
                raise ValueError("Reference-anchored bilinear head is unsigned only")
            self.ratio_head = ReferenceAnchoredBilinearRatioHead(
                *arguments, reference_theta=research["reference_theta"]
            )
        else:
            raise ValueError(f"Unknown RISE ratio_head={head}")
        if not research["parameter_names"] or len(
            set(research["parameter_names"])
        ) != len(research["parameter_names"]):
            raise ValueError("Explicit unique parameter names required")
        self.sbi_weight = float(research["sbi_loss_weight"])
        if self.sbi_weight <= 0:
            raise ValueError("Positive SBI loss weight required")
        if self.mode == "P":
            if not research.get("pretrained_checkpoint"):
                raise ValueError("RISE-P requires reconstruction checkpoint")
            if initialize_pretrained:
                _load_initial_components(
                    self,
                    research["pretrained_checkpoint"],
                    tuple(research["pretrained_components"]),
                )
        if self.mode in {"0", "P"}:
            for name, parameter in self.event_model.named_parameters():
                parameter.requires_grad_(
                    name.startswith(("visible.", "topology_queries."))
                )
        if bool(research["freeze_encoder"]):
            for parameter in self.event_model.visible.parameters():
                parameter.requires_grad_(False)
            if self.event_model.topology_queries is not None:
                for parameter in self.event_model.topology_queries.parameters():
                    parameter.requires_grad_(False)
        self.hparams["rise"] = self.research

    def validate_checkpoint_metadata(self, checkpoint):
        super().validate_checkpoint_metadata(checkpoint)
        if checkpoint["hyper_parameters"].get("rise") != self.hparams["rise"]:
            raise ValueError("RISE parameter/training configuration mismatch")

    def forward(self, batch, compute_objectives=True, **kwargs):
        base = self.event_model
        visible = base.encode_visible(batch)
        logits = self.ratio_head(visible["pooled"], batch["theta"])
        output = {"component_logits" if self.ratio_head.signed else "log_ratio": logits}
        if not compute_objectives:
            return output
        if self.ratio_head.signed:
            loss = signed_loss(
                logits,
                batch["membership"],
                batch["signed_weight"],
                batch["component_masses"],
            )
        else:
            loss = unsigned_loss(
                logits,
                batch["membership"],
                batch["signed_weight"],
                batch.get("class_normalisation"),
            )
        output["sbi_loss"] = loss
        output["loss"] = self.sbi_weight * loss
        if self.mode == "A":
            auxiliary = base._objective_forward(batch, visible)
            output.update({k: v for k, v in auxiliary.items() if k.endswith("_loss")})
            output["loss"] = output["loss"] + auxiliary["loss"]
        return output

    @torch.no_grad()
    def predict_ratio(self, batch):
        result = self(batch, compute_objectives=False)
        if self.ratio_head.signed:
            result.update(
                signed_ratio(result["component_logits"], batch["component_masses"])
            )
        else:
            result["ratio"] = result["log_ratio"].exp()
        result["EVENT_ID"] = batch["event_index"]
        result["theta"] = batch["theta"]
        return result
