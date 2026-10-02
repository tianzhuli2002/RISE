"""Bounded CPU RISE smoke; synthetic train/val and explicit real ROOT audit."""

import json
import subprocess
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.rise import (
    RatioHead,
    RiseModule,
    signed_loss,
    signed_ratio,
    unsigned_loss,
)
from joint_event_reconstruction.sbi_data import (
    ReweightedRatioBatch,
    SBISidecar,
    root_to_sidecar,
    write_sidecar,
)
from joint_event_reconstruction.training.joint import JointEventModule
from scripts.data.build_topology_query_synthetic import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/rise/development"
REAL = Path(
    "/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/user.leet.50321401._000001.output.root"
)


def norm(module, prefix):
    return (
        sum(
            float(p.grad.square().sum())
            for n, p in module.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        )
        ** 0.5
    )


def fixture():
    OUT.mkdir(parents=True, exist_ok=True)
    measured = OUT / "measured.h5"
    parents = OUT / "parents.h5"
    if not measured.exists():
        build(measured, parents, 32, 452)
        with h5py.File(measured, "r+") as f:
            f["SPLIT"][:] = np.where(np.arange(32) < 24, 0, 1)
        with h5py.File(parents, "r+") as f:
            support = f["PARTONIC/SUPPORT"][:, :2]
            del f["PARTONIC/SUPPORT"]
            f.create_dataset("PARTONIC/SUPPORT", data=support)
            del f["PARTONIC/ttbar"]
            f.attrs["partonic_roles"] = json.dumps(["top_e", "top_mu"])
            f.attrs["partonic_truth_bindings"] = json.dumps(
                {
                    "top_e": "synthetic_hard_process_top_e",
                    "top_mu": "synthetic_hard_process_top_mu",
                }
            )
    with h5py.File(measured) as f:
        ids = f["EVENT_ID"][:]
    meta = {
        "weight_definition": "synthetic toy only",
        "source_identity": "synthetic seed 452",
        "normalisation": "unit toy reference",
        "weight_convention": "shape_only",
    }
    grid = np.array([[0.0], [1.0]])
    weights = np.stack(
        (np.ones(len(ids)), np.exp(np.linspace(-1, 1, len(ids)))), axis=-1
    )
    weights[::5] *= -0.1
    if not (OUT / "toy_grid.h5").exists():
        write_sidecar(
            OUT / "toy_grid.h5",
            event_id=ids,
            nominal_weight=weights[:, 0],
            process_id=np.zeros(len(ids), dtype=np.int64),
            metadata=meta,
            parameter_grid=grid,
            parameter_names=["toy_mean"],
            signed_weights=weights,
            nuisance_weights={"toy_systematic": np.ones(len(ids))},
        )
    if not (OUT / "toy_native.h5").exists():
        write_sidecar(
            OUT / "toy_native.h5",
            event_id=ids,
            nominal_weight=weights[:, 0],
            process_id=np.zeros(len(ids), dtype=np.int64),
            metadata=meta,
            theta=np.repeat(grid, 16, axis=0),
            parameter_names=["toy_mean"],
        )
    sidecar = SBISidecar(OUT / "toy_grid.h5")
    sidecar.validate_join(measured)
    sampled = sidecar.sample_points(ids, np.random.default_rng(1))
    assert (sampled["weight_sign"] < 0).any()
    native = SBISidecar(OUT / "toy_native.h5")
    native.validate_join(measured)
    assert np.array_equal(
        native.sample_points(ids, np.random.default_rng(1))["signed_weight"],
        weights[:, 0],
    )
    mismatch = OUT / "mismatched.h5"
    if not mismatch.exists():
        with h5py.File(mismatch, "x") as f:
            f.create_dataset("EVENT_ID", data=ids[::-1])
    try:
        sidecar.validate_join(mismatch)
    except ValueError:
        pass
    else:
        raise AssertionError("Reordered event IDs accepted")
    try:
        sidecar.sample_points(np.array([999]), np.random.default_rng(1))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing event ID accepted")
    return sidecar


def toy():
    # Genuine parameterized Gaussian toy: log p(x|theta)/p(x|0)=theta*x-theta²/2.
    torch.manual_seed(453)
    head = RatioHead(1, 1, 32)
    optim = torch.optim.Adam(head.parameters(), lr=0.01)
    for step in range(250):
        theta = torch.rand(128, 1) * 2 - 1
        membership = torch.randint(0, 2, (128,)).float()
        x = torch.randn(128, 1) + membership[:, None] * theta
        loss = unsigned_loss(head(x, theta), membership, torch.ones(128))
        optim.zero_grad()
        loss.backward()
        optim.step()
    x = torch.linspace(-2, 2, 100)[:, None]
    theta = torch.full_like(x, 0.8)
    with torch.no_grad():
        pred = head(x, theta)
        known = 0.8 * x[:, 0] - 0.8**2 / 2
    correlation = float(torch.corrcoef(torch.stack((pred, known)))[0, 1])
    rmse = float((pred - known).square().mean().sqrt())
    assert correlation > 0.9 and rmse < 0.6
    theta_delta = float((head(x, theta) - head(x, -theta)).abs().mean().detach())
    assert theta_delta > 0.1
    # Both reference and target have negative components. Analytic component
    # logits exercise exact signed algebra independently of learned calibration.
    masses = torch.tensor([[1.2, 0.2, 1.3, 0.3]]).expand(100, -1)
    logits = torch.stack(
        (known, known, torch.zeros_like(known), torch.zeros_like(known)), dim=-1
    )
    ratio = signed_ratio(logits, masses)["ratio"]
    assert torch.allclose(ratio, known.exp(), atol=1e-5)
    signed_head = RatioHead(1, 1, 32, signed=True)
    signed_optim = torch.optim.Adam(signed_head.parameters(), lr=0.01)
    for step in range(80):
        labels = torch.arange(128) % 4
        membership = (labels < 2).float()
        weight = torch.where(labels % 2 == 0, torch.ones(128), -torch.ones(128) * 0.2)
        theta = torch.rand(128, 1) * 2 - 1
        x = torch.randn(128, 1) + membership[:, None] * theta
        loss = signed_loss(
            signed_head(x, theta),
            membership,
            weight,
            torch.tensor([[0.25, 0.05, 0.25, 0.05]]).expand(128, -1),
        )
        assert torch.isfinite(loss)
        signed_optim.zero_grad()
        loss.backward()
        signed_optim.step()
    # Zero-weight rows and absent sign components must be handled without NaN.
    assert torch.isfinite(
        signed_loss(
            torch.zeros(4, 4),
            torch.tensor([1, 1, 0, 0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.tensor([[1.0, 0.0, 1.0, 0.0]]).expand(4, -1),
        )
    )
    return {
        "gaussian_log_ratio_correlation": correlation,
        "gaussian_log_ratio_rmse": rmse,
        "trained_theta_response": theta_delta,
        "signed_target_and_reference_algebra": "PASS",
        "signed_training_steps": 80,
        "unsigned_training_steps": 250,
    }


def main():
    torch.set_num_threads(1)
    torch.manual_seed(452)
    sidecar = fixture()
    report = {"toy": toy()}
    real_out = OUT / "root_weights.h5"
    if not real_out.exists():
        audit = root_to_sidecar(
            REAL,
            real_out,
            tree="reco",
            event_id_branch="eventNumber",
            nominal_branch="weight_mc_NOSYS",
            process_branch="mcChannelNumber",
            selected_branches=[
                "weight_max_NOSYS",
                "weight_pileup_NOSYS",
                "weight_leptonSF_tight_NOSYS",
                "weight_jvt_effSF_NOSYS",
                "weight_ftag_effSF_GN2v01_Continuous_NOSYS",
            ],
            entry_stop=512,
        )
        (OUT / "root_audit.json").write_text(json.dumps(audit, indent=2))
    real = SBISidecar(real_out)
    assert real.mode == "metadata_only"
    import uproot

    with uproot.open(REAL) as f:
        assert np.array_equal(
            real.event_id, f["reco"]["eventNumber"].array(entry_stop=512, library="np")
        )
        assert np.array_equal(
            real.nominal,
            f["reco"]["weight_mc_NOSYS"].array(entry_stop=512, library="np"),
        )
    # This is a ROOT-only identity fixture; no existing measured HDF5 mapping is claimed.
    root_measured = OUT / "root_identity_fixture.h5"
    if not root_measured.exists():
        with h5py.File(root_measured, "x") as f:
            f.create_dataset("EVENT_ID", data=real.event_id)
    real.validate_join(root_measured)
    report["real_root_smoke"] = "PASS"
    for mode in ("0", "P", "A"):
        config = load_experiment(
            ROOT / f"configs/research/rise_{mode.lower()}_jer.yaml"
        )
        visible = load_visible_config(config)
        data = build_data(config, visible, device="cpu")
        data.setup("fit")
        batch = next(iter(data.train_dataloader()))
        kwargs = {
            "invisible_scaler": data.invisible_scaler,
            "partonic_scaler": data.partonic_scaler,
            "synthetic_data": True,
        }
        if mode == "0":
            preconfig = load_experiment(ROOT / "configs/research/rise_a_jer.yaml")
            previsible = load_visible_config(preconfig)
            predata = build_data(preconfig, previsible, device="cpu")
            predata.setup("fit")
            pre = JointEventModule(
                preconfig,
                previsible,
                invisible_scaler=predata.invisible_scaler,
                partonic_scaler=predata.partonic_scaler,
                synthetic_data=True,
            )
            trainer = L.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=1,
                limit_val_batches=0,
                num_sanity_val_steps=0,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                default_root_dir=OUT / "pretrain",
            )
            trainer.fit(pre, train_dataloaders=predata.train_dataloader())
            trainer.save_checkpoint(OUT / "reconstruction.ckpt")
        research = OmegaConf.to_container(
            OmegaConf.load(ROOT / f"configs/research/rise_{mode.lower()}.yaml"),
            resolve=True,
        )
        module = RiseModule(config, visible, research=research, **kwargs)
        if mode == "P":
            saved = torch.load(OUT / "reconstruction.ckpt", weights_only=False)[
                "state_dict"
            ]
            assert all(
                torch.equal(value, saved[name])
                for name, value in module.state_dict().items()
                if name.startswith(
                    ("event_model.visible.", "event_model.topology_queries.")
                )
            )
        train_ids = data._selected_ids("train", np.arange(len(data.train_dataset)))
        adapter = ReweightedRatioBatch(sidecar, train_ids, reference_point=0, seed=452)
        # Use signed mode for this negative-weight fixture in every RISE mode.
        research["signed_mixture"] = True
        module = RiseModule(config, visible, research=research, **kwargs)
        adapted = adapter(batch)
        output = module(adapted)
        assert torch.isfinite(output["loss"])
        auxiliary_gradient = None
        if mode == "A":
            auxiliary = sum(
                output[key]
                for key in ("assignment_loss", "invisible_loss", "partonic_loss")
            )
            auxiliary.backward(retain_graph=True)
            auxiliary_gradient = norm(module, "event_model.visible")
            assert auxiliary_gradient > 0
            module.zero_grad(set_to_none=True)
        output["loss"].backward()
        gradient = norm(module, "event_model.visible")
        assert gradient > 0
        if mode == "A":
            assert (
                norm(module, "event_model.reconstruction") > 0
                and norm(module, "event_model.invisible") > 0
            )
        if mode in {"0", "P"}:

            def forbidden(*args, **kwargs):
                raise AssertionError("SBI-only executed reconstruction")

            module.event_model.reconstruction.forward = forbidden
            if module.event_model.invisible is not None:
                module.event_model.invisible.sample = forbidden
                module.event_model.invisible.loss = forbidden

        def collate(items, adapter=adapter, data=data):
            return adapter(data.structured_collator(items))

        loader = DataLoader(
            data.train_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=2,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=OUT / mode,
        )
        trainer.fit(module, train_dataloaders=loader)
        trainer.save_checkpoint(OUT / f"{mode}.ckpt")
        restored = RiseModule(config, visible, research=research, **kwargs)
        saved = torch.load(OUT / f"{mode}.ckpt", weights_only=False)
        restored.validate_checkpoint_metadata(saved)
        restored.load_state_dict(saved["state_dict"])
        restored.eval()
        prediction = restored.predict_ratio(adapted)
        clean = {
            k: v
            for k, v in adapted.items()
            if "truth" not in k and "support" not in k and k != "structured"
        }
        truth_free = restored.predict_ratio(clean)
        assert torch.equal(prediction["ratio"], truth_free["ratio"])
        assert torch.isfinite(prediction["ratio"]).all() and torch.equal(
            prediction["EVENT_ID"], batch["event_index"]
        )
        with h5py.File(OUT / f"{mode}_prediction.h5", "w") as f:
            for key in ["EVENT_ID", "theta", "ratio", "ratio_sign", "log_abs_ratio"]:
                f.create_dataset(key, data=prediction[key].cpu().numpy())
        if mode == "P":
            frozen = dict(research, freeze_encoder=True)
            frozen_module = RiseModule(config, visible, research=frozen, **kwargs)
            frozen_module(adapted)["loss"].backward()
            assert (
                norm(frozen_module, "event_model.visible") == 0
                and norm(frozen_module, "ratio_head") > 0
            )
            assert not any(
                p.requires_grad for p in frozen_module.event_model.visible.parameters()
            )
        report[mode] = {
            "finite_loss": float(output["loss"].detach()),
            "encoder_gradient": gradient,
            "auxiliary_encoder_gradient": auxiliary_gradient,
            "truth_free_prediction": True,
            "train_steps": 2,
            "reload_prediction": True,
        }
    report.update(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        branch=subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        worktree=str(ROOT),
        sealed_test_used=False,
        population="32 synthetic events, 24 train / 8 validation; ROOT metadata audit only",
        rise_end_to_end="PASS",
    )
    (OUT / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
