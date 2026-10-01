"""Inspect ROOT or convert explicitly selected fields into an SBI sidecar."""

import argparse
import json
from pathlib import Path

from joint_event_reconstruction.sbi_data import inspect_root, root_to_sidecar


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--output")
    p.add_argument("--audit-output")
    p.add_argument("--tree", default="reco")
    p.add_argument("--event-id-branch", default="eventNumber")
    p.add_argument("--nominal-branch", default="weight_mc_NOSYS")
    p.add_argument("--process-branch", default="mcChannelNumber")
    p.add_argument("--extra-branch", action="append", default=[])
    p.add_argument("--events", type=int, default=512)
    p.add_argument(
        "--parameter-metadata",
        help="JSON with externally established names/grid/branch/definition/normalisation",
    )
    p.add_argument(
        "--event-id-mapping",
        help="NPY explicit ROOT-row to canonical measured EVENT_ID mapping",
    )
    a = p.parse_args()
    if a.events <= 0:
        p.error("Positive bounded event count required")
    if a.output:
        import numpy as np

        report = root_to_sidecar(
            a.input,
            a.output,
            tree=a.tree,
            event_id_branch=a.event_id_branch,
            nominal_branch=a.nominal_branch,
            process_branch=a.process_branch,
            selected_branches=a.extra_branch,
            entry_stop=a.events,
            event_id_mapping=None
            if a.event_id_mapping is None
            else np.load(a.event_id_mapping, allow_pickle=False),
            parameter_metadata=None
            if a.parameter_metadata is None
            else json.loads(Path(a.parameter_metadata).read_text()),
        )
    else:
        report = inspect_root(a.input)
    if a.audit_output:
        with Path(a.audit_output).open("x") as f:
            f.write(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
