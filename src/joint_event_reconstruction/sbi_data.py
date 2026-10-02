"""Event-aligned signed SBI sidecars. EVENT_ID is the authoritative JER key.

No ROOT eventNumber -> row-number identity guessing is permitted. Supply an
explicit mapping when the measured HDF5 uses source-partition/row identities.
"""

import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

WEIGHT_CONVENTIONS = {
    "absolute",
    "relative_to_nominal",
    "shape_only",
    "rate_bearing",
}


def _ids(values):
    a = np.asarray(values)
    if (
        a.ndim != 1
        or len(a) == 0
        or not np.issubdtype(a.dtype, np.integer)
        or len(np.unique(a)) != len(a)
    ):
        raise ValueError("EVENT_ID must be a unique integer vector")
    return a


def write_sidecar(
    path,
    *,
    event_id,
    nominal_weight,
    process_id,
    metadata,
    theta=None,
    parameter_grid=None,
    parameter_names=(),
    signed_weights=None,
    nuisance_weights=None,
):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    ids = _ids(event_id)
    n = len(ids)
    nominal = np.asarray(nominal_weight, dtype=np.float64)
    process = np.asarray(process_id)
    if nominal.shape != (n,) or not np.isfinite(nominal).all():
        raise ValueError("Nominal signed weights must be finite [N]")
    if process.shape != (n,):
        raise ValueError("Explicit process/sample identity must be [N]")
    mode = "metadata_only"
    if theta is not None:
        mode = "native"
        theta = np.asarray(theta, dtype=np.float64)
        if parameter_grid is not None or signed_weights is not None:
            raise ValueError("Select native or reweighted mode")
        if (
            theta.shape != (n, len(parameter_names))
            or not len(parameter_names)
            or not np.isfinite(theta).all()
        ):
            raise ValueError("Invalid native parameter coordinates")
    elif parameter_grid is not None:
        mode = "reweighted"
        parameter_grid = np.asarray(parameter_grid, dtype=np.float64)
        signed_weights = np.asarray(signed_weights, dtype=np.float64)
        if (
            parameter_grid.ndim != 2
            or len(parameter_grid) == 0
            or parameter_grid.shape[1] != len(parameter_names)
            or not len(parameter_names)
        ):
            raise ValueError("Invalid parameter grid")
        if (
            signed_weights.shape != (n, len(parameter_grid))
            or not np.isfinite(signed_weights).all()
            or not np.isfinite(parameter_grid).all()
        ):
            raise ValueError("Invalid parameter signed weights")
        if len(np.unique(parameter_grid, axis=0)) != len(parameter_grid):
            raise ValueError("Duplicate parameter points")
    elif signed_weights is not None or parameter_names:
        raise ValueError("Weights/names require known parameter coordinates")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("Duplicate parameter names")
    for key in ("weight_definition", "source_identity", "normalisation"):
        if key not in metadata:
            raise ValueError(f"Missing weight provenance: {key}")
    convention = metadata.get("weight_convention")
    if mode == "metadata_only":
        if convention != "unresolved" and convention not in WEIGHT_CONVENTIONS:
            raise ValueError(
                "Metadata-only weights must declare weight_convention=unresolved "
                "or an established convention"
            )
    else:
        if convention not in WEIGHT_CONVENTIONS:
            raise ValueError(
                "Parameterized sidecars require an established weight_convention"
            )
        if not isinstance(metadata["weight_definition"], str) or not metadata[
            "weight_definition"
        ].strip():
            raise ValueError("Parameterized sidecars require a nonempty weight_definition")
        if (
            not isinstance(metadata["normalisation"], str)
            or not metadata["normalisation"].strip()
            or metadata["normalisation"].strip().lower() in {"unknown", "unresolved"}
        ):
            raise ValueError(
                "Parameterized sidecars require an explicit weight normalisation"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".h5")
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = "rise_sbi_sidecar_v1"
            f.attrs["mode"] = mode
            f.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
            f.create_dataset("EVENT_ID", data=ids)
            f.create_dataset("nominal/signed_weight", data=nominal)
            f.create_dataset(
                "process_id",
                data=process.astype(h5py.string_dtype())
                if process.dtype.kind in "US"
                else process,
            )
            if mode != "metadata_only":
                f.create_dataset(
                    "parameters/names",
                    data=np.asarray(parameter_names, dtype=h5py.string_dtype()),
                )
                f.create_dataset(
                    "parameters/theta" if mode == "native" else "parameters/grid",
                    data=theta if mode == "native" else parameter_grid,
                )
            if mode == "reweighted":
                f.create_dataset("parameters/signed_weights", data=signed_weights)
            for name, values in (nuisance_weights or {}).items():
                a = np.asarray(values)
                if a.ndim not in (1, 2) or len(a) != n or not np.isfinite(a).all():
                    raise ValueError(f"Invalid nuisance weight {name}")
                f.create_dataset("nuisance/" + name, data=a)
        SBISidecar(tmp)  # Validate before installing the completed product.
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


class SBISidecar:
    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(path, "r") as f:
            if f.attrs.get("schema_version") != "rise_sbi_sidecar_v1":
                raise ValueError(f"{path}: unsupported SBI format")
            self.event_id = _ids(f["EVENT_ID"][:])
            self.mode = f.attrs["mode"]
            self.nominal = f["nominal/signed_weight"][:]
            self.process_id = f["process_id"][:]
            self.metadata = json.loads(f.attrs["metadata"])
            self.parameter_names = (
                tuple(f["parameters/names"].asstr()[:])
                if self.mode != "metadata_only"
                else ()
            )
            self.theta = f["parameters/theta"][:] if self.mode == "native" else None
            self.grid = f["parameters/grid"][:] if self.mode == "reweighted" else None
            self.weights = (
                f["parameters/signed_weights"][:] if self.mode == "reweighted" else None
            )
            self.nuisance = {k: v[:] for k, v in f.get("nuisance", {}).items()}
        n = len(self.event_id)
        if self.mode not in {"native", "reweighted", "metadata_only"}:
            raise ValueError("Unknown sidecar mode")
        if (
            self.nominal.shape != (n,)
            or self.process_id.shape != (n,)
            or not np.isfinite(self.nominal).all()
        ):
            raise ValueError("Malformed nominal data")
        if self.mode == "native" and (
            self.theta.shape != (n, len(self.parameter_names))
            or not np.isfinite(self.theta).all()
        ):
            raise ValueError("Malformed native theta")
        if self.mode == "reweighted" and (
            self.grid.ndim != 2
            or self.grid.shape[1] != len(self.parameter_names)
            or self.weights.shape != (n, len(self.grid))
            or not np.isfinite(self.weights).all()
            or not np.isfinite(self.grid).all()
        ):
            raise ValueError("Malformed parameter grid")
        if self.mode != "metadata_only" and (
            not self.parameter_names
            or len(set(self.parameter_names)) != len(self.parameter_names)
        ):
            raise ValueError("Missing/duplicate parameter names")
        if self.mode == "reweighted" and (
            len(self.grid) == 0 or len(np.unique(self.grid, axis=0)) != len(self.grid)
        ):
            raise ValueError("Empty/duplicate parameter grid")
        for key in ("weight_definition", "source_identity", "normalisation"):
            if key not in self.metadata:
                raise ValueError(f"{path}: missing weight provenance {key}")
        convention = self.metadata.get("weight_convention")
        if self.mode == "metadata_only":
            if convention != "unresolved" and convention not in WEIGHT_CONVENTIONS:
                raise ValueError(f"{path}: metadata-only weight convention is ambiguous")
        elif convention not in WEIGHT_CONVENTIONS:
            raise ValueError(f"{path}: parameter weight convention is absent or unresolved")
        elif (
            not isinstance(self.metadata["weight_definition"], str)
            or not self.metadata["weight_definition"].strip()
            or not isinstance(self.metadata["normalisation"], str)
            or not self.metadata["normalisation"].strip()
            or self.metadata["normalisation"].strip().lower()
            in {"unknown", "unresolved"}
        ):
            raise ValueError(f"{path}: parameter weight definition/normalisation is ambiguous")
        for name, values in self.nuisance.items():
            if (
                values.ndim not in (1, 2)
                or len(values) != n
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"{path}: malformed nuisance weight {name}")
        self.order = np.argsort(self.event_id)

    def validate_join(self, measured_path):
        with h5py.File(measured_path, "r") as f:
            ids = _ids(f["EVENT_ID"][:])
        if not np.array_equal(ids, self.event_id):
            raise ValueError(
                f"{self.path}: ordered EVENT_ID differs from {measured_path}"
            )

    def positions(self, event_id):
        requested = _ids(event_id)
        sorted_ids = self.event_id[self.order]
        pos = np.searchsorted(sorted_ids, requested)
        if (
            len(sorted_ids) == 0
            or (pos >= len(sorted_ids)).any()
            or not np.array_equal(sorted_ids[pos], requested)
        ):
            raise ValueError(f"{self.path}: missing requested EVENT_ID")
        return self.order[pos]

    def sample_points(self, event_id, rng):
        rows = self.positions(event_id)
        if self.mode == "metadata_only":
            raise ValueError(f"{self.path}: POI coordinates absent/unresolved")
        if self.mode == "native":
            theta = self.theta[rows]
            weight = self.nominal[rows]
            point = None
        else:
            point = rng.integers(0, len(self.grid), size=len(rows))
            theta = self.grid[point]
            weight = self.weights[rows, point]
        return {
            "theta": theta,
            "parameter_point": point,
            "signed_weight": weight,
            "weight_sign": np.sign(weight),
            "absolute_weight": np.abs(weight),
            "nominal_weight": self.nominal[rows],
            "process_id": self.process_id[rows],
        }


def inspect_root(path):
    import uproot

    report = {
        "source": str(Path(path).resolve()),
        "trees": {},
        "histogram_metadata": {},
        "poi_semantics": "absent_or_unresolved",
    }
    with uproot.open(path) as f:
        for key, kind in f.classnames().items():
            obj = f[key]
            if kind == "TTree":
                branches = obj.typenames()
                report["trees"][key] = {
                    "entries": obj.num_entries,
                    "branches": branches,
                    "weight_statistics": {},
                }
                for name in branches:
                    if "weight" in name.lower():
                        a = obj[name].array(library="ak")
                        import awkward as ak

                        flat = ak.to_numpy(ak.flatten(a, axis=None))
                        report["trees"][key]["weight_statistics"][name] = {
                            "type": branches[name],
                            "negative": int((flat < 0).sum()),
                            "min": float(flat.min()),
                            "max": float(flat.max()),
                        }
            elif kind.startswith("TH1"):
                report["histogram_metadata"][key] = {
                    "labels": obj.axis().labels(),
                    "values": obj.values().tolist(),
                }
    return report


def root_to_sidecar(
    path,
    output,
    *,
    tree,
    event_id_branch,
    nominal_branch,
    process_branch,
    selected_branches=(),
    entry_stop=None,
    event_id_mapping=None,
    parameter_metadata=None,
):
    """Selected extra weights are unresolved until an explicit definition is supplied.

    Parameter metadata: parameter_names, grid, weight_branch, weight_definition,
    normalisation. A vector branch alone is insufficient to define a POI grid.
    """
    import awkward as ak
    import uproot

    report = inspect_root(path)
    with uproot.open(path) as f:
        t = f[tree]
        n = min(t.num_entries, entry_stop) if entry_stop is not None else t.num_entries

        def read(name):
            return ak.to_numpy(t[name].array(entry_stop=n))

        source_ids = read(event_id_branch)
        ids = source_ids if event_id_mapping is None else np.asarray(event_id_mapping)
        if ids.shape != (n,):
            raise ValueError("Explicit ROOT to measured EVENT_ID mapping must be [N]")
        nominal = read(nominal_branch)
        if nominal.ndim != 1:
            raise ValueError("Nominal ROOT weight must be scalar")
        extras = {name: read(name) for name in selected_branches}
        meta = {
            "weight_definition": f"scalar branch {nominal_branch}; extra branches unresolved unless externally documented",
            "source_identity": {
                "path": str(Path(path).resolve()),
                "tree": tree,
                "event_id_branch": event_id_branch,
                "source_event_id": source_ids.tolist(),
                "event_id_mapping": "explicit"
                if event_id_mapping is not None
                else "branch_identity",
            },
            "source_branches": [nominal_branch, *selected_branches],
            "normalisation": "unknown",
            "weight_convention": "unresolved",
            "root_audit": report,
        }
        kwargs = {}
        if parameter_metadata is not None:
            required = {
                "parameter_names",
                "grid",
                "weight_branch",
                "weight_definition",
                "normalisation",
                "weight_convention",
            }
            if not required.issubset(parameter_metadata):
                raise ValueError("Incomplete externally established parameter metadata")
            raw_weights = read(parameter_metadata["weight_branch"])
            convention = parameter_metadata["weight_convention"]
            if convention not in WEIGHT_CONVENTIONS:
                raise ValueError(
                    "Specify absolute, relative_to_nominal, shape_only, or rate_bearing POI weights"
                )
            if convention == "relative_to_nominal":
                raw_weights = raw_weights * nominal[:, None]
            meta["source_branches"].append(parameter_metadata["weight_branch"])
            meta["weight_definition"] = parameter_metadata["weight_definition"]
            meta["normalisation"] = parameter_metadata["normalisation"]
            meta["weight_convention"] = convention
            kwargs = {
                "parameter_names": parameter_metadata["parameter_names"],
                "parameter_grid": parameter_metadata["grid"],
                "signed_weights": raw_weights,
            }
            meta["poi_definition"] = parameter_metadata
        write_sidecar(
            output,
            event_id=ids,
            nominal_weight=nominal,
            process_id=read(process_branch),
            metadata=meta,
            nuisance_weights=extras,
            **kwargs,
        )
    return report


class ReweightedRatioBatch:
    """Uniform event/POI sampling against one explicit reference grid point.

    Pass TRAIN event IDs to fit fixed per-point component masses. The same
    masses must be reused for validation/prediction. This adapter does not
    load measured data or reconstruct latent particles.
    """

    def __init__(self, sidecar, train_event_id, *, reference_point, seed):
        if sidecar.mode != "reweighted":
            raise ValueError(
                "Reweighted ratio adapter requires a genuine/synthetic parameter grid"
            )
        self.sidecar = sidecar
        self.reference_point = int(reference_point)
        self.rng = np.random.default_rng(seed)
        if not 0 <= self.reference_point < len(sidecar.grid):
            raise ValueError("Reference point outside grid")
        rows = sidecar.positions(train_event_id)
        self.population_size = len(rows)
        if self.population_size == 0:
            raise ValueError("SBI training population is empty")
        weights = sidecar.weights[rows]
        positive = np.maximum(weights, 0).sum(0)
        negative = -np.minimum(weights, 0).sum(0)
        reference = self.reference_point
        self.masses = np.stack(
            (
                positive,
                negative,
                np.full(len(positive), positive[reference]),
                np.full(len(negative), negative[reference]),
            ),
            axis=-1,
        )
        if (self.masses[:, 0] - self.masses[:, 1] <= 0).any() or positive[
            reference
        ] - negative[reference] <= 0:
            raise ValueError("Nonpositive signed TRAIN normalization")

    def __call__(self, batch):
        import torch

        ids = batch["event_index"].detach().cpu().numpy()
        rows = self.sidecar.positions(ids)
        points = self.rng.integers(0, len(self.sidecar.grid), size=len(rows))
        membership = self.rng.integers(0, 2, size=len(rows))
        weights = self.sidecar.weights[
            rows, np.where(membership == 1, points, self.reference_point)
        ]
        # Loss denominators are population totals per uniformly sampled source
        # event, not totals over the full population. This preserves the exact
        # normalized risk while keeping its scale independent of N.
        loss_masses = self.masses[points] / self.population_size
        dtype = batch["tokens"].dtype
        device = batch["tokens"].device

        def tensor(a):
            return torch.as_tensor(a, device=device, dtype=dtype)

        result = dict(batch)
        result.update(
            theta=tensor(self.sidecar.grid[points]),
            parameter_point=torch.as_tensor(points, device=device),
            membership=tensor(membership),
            signed_weight=tensor(weights),
            absolute_weight=tensor(np.abs(weights)),
            weight_sign=tensor(np.sign(weights)),
            nominal_weight=tensor(self.sidecar.nominal[rows]),
            component_masses=tensor(loss_masses),
            class_normalisation=tensor(loss_masses[:, [0, 2]]),
        )
        return result
