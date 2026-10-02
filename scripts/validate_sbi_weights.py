"""Targeted scalar/vector ROOT and signed-weight conversion checks with toys."""

import tempfile
from pathlib import Path

import numpy as np
import uproot

from joint_event_reconstruction.sbi_data import SBISidecar, root_to_sidecar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/rise/preproduction-v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as temporary:
        root = Path(temporary) / "synthetic.root"
        nominal = np.array([1.0, -2.0, 3.0, -4.0, 5.0, 0.0])
        vector = np.tile([1.0, 2.0], (6, 1))
        with uproot.recreate(root) as f:
            tree = f.mktree(
                "toy",
                {
                    "event": "uint64",
                    "process": "uint32",
                    "nominal": "float64",
                    "toy_weight_vector": "2 * float64",
                },
            )
            tree.extend(
                {
                    "event": np.arange(6, dtype=np.uint64),
                    "process": np.ones(6, dtype=np.uint32),
                    "nominal": nominal,
                    "toy_weight_vector": vector,
                }
            )
        kwargs = {
            "tree": "toy",
            "event_id_branch": "event",
            "nominal_branch": "nominal",
            "process_branch": "process",
            "selected_branches": ["toy_weight_vector"],
            "entry_stop": 6,
        }
        mapping = 100 + np.arange(6)
        root_to_sidecar(
            root, Path(temporary) / "unresolved.h5", event_id_mapping=mapping, **kwargs
        )
        unresolved = SBISidecar(Path(temporary) / "unresolved.h5")
        assert unresolved.mode == "metadata_only" and np.array_equal(
            unresolved.nuisance["toy_weight_vector"], vector
        )
        assert np.array_equal(unresolved.event_id, mapping) and np.array_equal(
            unresolved.nominal, nominal
        )
        metadata = {
            "parameter_names": ["synthetic_theta"],
            "grid": [[0.0], [1.0]],
            "weight_branch": "toy_weight_vector",
            "weight_definition": "synthetic fixture, no physics meaning",
            "normalisation": "toy",
            "weight_convention": "relative_to_nominal",
        }
        root_to_sidecar(
            root, Path(temporary) / "relative.h5", parameter_metadata=metadata, **kwargs
        )
        relative = SBISidecar(Path(temporary) / "relative.h5")
        assert np.array_equal(relative.weights, vector * nominal[:, None])
        metadata["weight_convention"] = "absolute"
        root_to_sidecar(
            root, Path(temporary) / "absolute.h5", parameter_metadata=metadata, **kwargs
        )
        assert np.array_equal(
            SBISidecar(Path(temporary) / "absolute.h5").weights, vector
        )
        metadata.pop("weight_convention")
        try:
            root_to_sidecar(
                root,
                Path(temporary) / "missing_definition.h5",
                parameter_metadata=metadata,
                **kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Unspecified POI weight convention accepted")
    print(
        "PASS: scalar nominal, unresolved vector, explicit identity mapping, signed relative weights, absolute weights, missing definition rejection"
    )
    (OUT / "weight_adapter_checks.txt").write_text(
        "PASS: synthetic scalar/vector ROOT conversion, identity mapping, relative/absolute signed-weight conventions; no sealed test used\n"
    )


if __name__ == "__main__":
    main()
