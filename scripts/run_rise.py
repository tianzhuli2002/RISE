"""Bounded CPU RISE training/prediction on an explicitly joined POI sidecar."""

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.invisible import InvisibleScaler
from joint_event_reconstruction.data.partonic import PartonicScaler
from joint_event_reconstruction.rise import RiseModule
from joint_event_reconstruction.sbi_data import ReweightedRatioBatch, SBISidecar
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["train", "predict"])
    p.add_argument("--config", required=True)
    p.add_argument("--research", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--reference-point", required=True, type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batches", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.steps <= 0 or a.batches <= 0:
        p.error("Positive bounded steps/batches required")
    root = Path(__file__).resolve().parents[1]
    output = Path(a.output).resolve()
    if not output.is_relative_to(root / "results/experiments/rise"):
        raise ValueError("Use isolated results/experiments/rise output")
    output.mkdir(parents=True, exist_ok=False)
    config = load_experiment(a.config)
    visible = load_visible_config(config)
    research = OmegaConf.to_container(OmegaConf.load(a.research), resolve=True)
    torch.manual_seed(int(config.experiment.seed))
    sidecar = SBISidecar(a.sidecar)
    input_path = config.data.get("inference_input") if a.action == "predict" else None
    input_path = input_path or config.data.input
    sidecar.validate_join(input_path)
    if list(sidecar.parameter_names) != research["parameter_names"]:
        raise ValueError("Sidecar and head parameter names/order differ")
    with h5py.File(input_path, "r") as f:
        if "SPLIT" not in f or not np.isin(f["SPLIT"][:], (0, 1)).all():
            raise ValueError("Provide explicit train/validation-only development data")
    if a.action == "train":
        data = build_data(config, visible, device="cpu")
        data.setup("fit")
        kwargs = {
            "invisible_scaler": data.invisible_scaler,
            "partonic_scaler": data.partonic_scaler,
            "synthetic_data": bool(config.data.invisible.allow_synthetic),
        }
        module = RiseModule(config, visible, research=research, **kwargs)
        train_ids = data._selected_ids("train", np.arange(len(data.train_dataset)))
        adapter = ReweightedRatioBatch(
            sidecar,
            train_ids,
            reference_point=a.reference_point,
            seed=int(config.experiment.seed),
        )
        if not research["signed_mixture"] and (sidecar.weights < 0).any():
            raise ValueError("Negative weights require signed_mixture=true")

        def collate(items):
            return adapter(data.structured_collator(items))

        loader = DataLoader(
            data.train_dataset,
            batch_size=int(config.training.batch_size),
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
        module.hparams["sbi_training_population"] = {
            "event_ids": train_ids.tolist(),
            "reference_point": a.reference_point,
            "component_masses": adapter.masses.tolist(),
            "parameter_grid": sidecar.grid.tolist(),
            "sidecar_metadata": sidecar.metadata,
        }
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=a.steps,
            max_epochs=a.steps,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            default_root_dir=output,
        )
        trainer.fit(module, train_dataloaders=loader)
        trainer.save_checkpoint(output / "model.ckpt")
    else:
        if not a.checkpoint:
            p.error("--checkpoint required")
        saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        module = RiseModule(
            config,
            visible,
            research=research,
            initialize_pretrained=False,
            invisible_scaler=None
            if saved["invisible_scaler"] is None
            else InvisibleScaler.from_dict(saved["invisible_scaler"]),
            partonic_scaler=None
            if saved["partonic_scaler"] is None
            else PartonicScaler.from_dict(saved["partonic_scaler"]),
            synthetic_data=bool(config.data.invisible.allow_synthetic),
        )
        module.validate_checkpoint_metadata(saved)
        module.load_state_dict(saved["state_dict"])
        module.eval()
        population = saved["hyper_parameters"]["sbi_training_population"]
        if population["reference_point"] != a.reference_point or not np.array_equal(
            population["parameter_grid"], sidecar.grid
        ):
            raise ValueError("Prediction reference/grid differs from training")
        adapter = ReweightedRatioBatch(
            sidecar,
            np.asarray(population["event_ids"]),
            reference_point=a.reference_point,
            seed=int(config.experiment.seed),
        )
        if not np.array_equal(adapter.masses, population["component_masses"]):
            raise ValueError("Prediction weights differ from training population")
        data = build_data(config, visible, include_truth=False, device="cpu")
        data.setup("fit")
        for index, batch in enumerate(data.val_dataloader()):
            if index >= a.batches:
                break
            # Export every requested grid theta for every measured event.
            with h5py.File(output / f"ratio_batch_{index}.h5", "x") as f:
                f.create_dataset("EVENT_ID", data=batch["event_index"].numpy())
                f.create_dataset("parameter_grid", data=sidecar.grid)
                predictions = []
                for point, theta in enumerate(sidecar.grid):
                    query = dict(
                        batch,
                        theta=torch.tensor(theta, dtype=batch["tokens"].dtype).expand(
                            len(batch["tokens"]), -1
                        ),
                        component_masses=torch.tensor(
                            adapter.masses[point], dtype=batch["tokens"].dtype
                        ).expand(len(batch["tokens"]), -1),
                    )
                    predictions.append(
                        module.predict_ratio(query)["ratio"].cpu().numpy()
                    )
                f.create_dataset("ratio", data=np.stack(predictions, axis=-1))
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                "branch": subprocess.check_output(
                    ["git", "branch", "--show-current"], cwd=root, text=True
                ).strip(),
                "worktree": str(root),
                "config": str(Path(a.config).resolve()),
                "research": research,
                "sidecar": str(sidecar.path.resolve()),
                "reference_point": a.reference_point,
                "action": a.action,
                "bounded_steps": a.steps,
                "sealed_test_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
