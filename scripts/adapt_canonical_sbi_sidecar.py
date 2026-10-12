"""Build a metadata-only RISE SBI sidecar from a canonical KNOT event HDF5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

RISE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RISE_ROOT / "src"))

from joint_event_reconstruction.sbi_data import SBISidecar, write_sidecar


def adapt(canonical_path: Path, output_path: Path) -> dict:
    canonical_path = canonical_path.resolve()
    output_path = output_path.resolve()
    if not canonical_path.is_file():
        raise FileNotFoundError(f"Canonical event HDF5 does not exist: {canonical_path}")

    with h5py.File(canonical_path, "r") as source:
        contract = source.attrs.get("canonical_contract_version")
        if contract != "kk_ttbar_roots_v2":
            raise ValueError(
                f"{canonical_path}: expected canonical_contract_version="
                f"'kk_ttbar_roots_v2', found {contract!r}"
            )
        required = (
            "EVENT_ID",
            "META/mcChannelNumber",
            "META/w_gen",
            "META/weight_available_w_gen",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"{canonical_path}: missing canonical SBI inputs {missing}")

        event_id = np.asarray(source["EVENT_ID"][:])
        process_id = np.asarray(source["META/mcChannelNumber"][:])
        nominal_weight = np.asarray(source["META/w_gen"][:], dtype=np.float64)
        weight_available = np.asarray(
            source["META/weight_available_w_gen"][:], dtype=bool
        )
        if (
            event_id.ndim != 1
            or not np.issubdtype(event_id.dtype, np.integer)
            or len(event_id) == 0
            or len(np.unique(event_id)) != len(event_id)
        ):
            raise ValueError(f"{canonical_path}: EVENT_ID must be unique integer values")
        if process_id.shape != event_id.shape or not np.issubdtype(
            process_id.dtype, np.integer
        ):
            raise ValueError(f"{canonical_path}: DSID values must be integer [N]")
        if weight_available.shape != event_id.shape or not weight_available.all():
            raise ValueError(
                f"{canonical_path}: raw generator weight availability is incomplete"
            )
        if nominal_weight.shape != event_id.shape or not np.isfinite(
            nominal_weight
        ).all():
            raise ValueError(
                f"{canonical_path}: META/w_gen must be finite for every selected event"
            )

        dsids = np.unique(process_id)
        dsid_attr = int(source.attrs.get("mcChannelNumber", -1))
        if len(dsids) != 1 or int(dsids[0]) != dsid_attr:
            raise ValueError(
                f"{canonical_path}: event DSIDs {dsids.tolist()} disagree with "
                f"canonical mcChannelNumber={dsid_attr}"
            )
        dsid = int(dsids[0])
        campaign = str(source.attrs.get("production_campaign", ""))
        sample_class = str(source.attrs.get("sample_class", ""))
        source_root = str(source.attrs.get("source_root_file", ""))
        source_root_sha256 = str(source.attrs.get("source_root_sha256", ""))

    metadata = {
        "weight_definition": (
            "raw generator weight from canonical META/w_gen; this is not a full "
            "analysis-level or cross-section-normalized event weight"
        ),
        "source_identity": {
            "canonical_hdf5": str(canonical_path),
            "canonical_contract_version": "kk_ttbar_roots_v2",
            "source_root_file": source_root,
            "source_root_sha256": source_root_sha256,
            "campaign": campaign,
            "mcChannelNumber": dsid,
            "sample_class_metadata": sample_class,
        },
        "normalisation": (
            "unresolved; cross section, filter efficiency, and full analysis "
            "weight are unavailable"
        ),
        "weight_convention": "unresolved",
        "parameter_coordinates": "unresolved; no theta or parameter weights supplied",
    }
    write_sidecar(
        output_path,
        event_id=event_id,
        nominal_weight=nominal_weight,
        process_id=process_id,
        metadata=metadata,
    )

    sidecar = SBISidecar(output_path)
    sidecar.validate_join(canonical_path)
    if sidecar.mode != "metadata_only" or sidecar.theta is not None:
        raise ValueError(f"{output_path}: unresolved inputs were not kept metadata-only")
    try:
        sidecar.sample_points(event_id[:1], np.random.default_rng(0))
    except ValueError as error:
        if "POI coordinates absent/unresolved" not in str(error):
            raise
    else:
        raise ValueError(f"{output_path}: RISE accepted missing parameter coordinates")

    return {
        "canonical_hdf5": str(canonical_path),
        "sidecar_hdf5": str(output_path),
        "events": len(event_id),
        "dsid": dsid,
        "campaign": campaign,
        "sample_class_metadata": sample_class,
        "mode": sidecar.mode,
        "ordered_event_id_join": "exact",
        "weight": "raw signed META/w_gen only",
        "normalisation": "unresolved",
        "theta": "unavailable; reader refuses parameter sampling",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(adapt(args.canonical, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
