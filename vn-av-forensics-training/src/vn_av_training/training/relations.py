"""Cached FATE relation training, resume, threshold selection and checkpoint loading."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from vn_av_training.common.runtime import fingerprint, read_json, sha, write_json
from vn_av_training.data.manifest import read_manifest, validate
from vn_av_training.data.relations import labels_for
from vn_av_training.models.relations import ACTIVITIES, RELATIONS, RelationHeads, relation_loss
from vn_av_training.training.trainer import save_checkpoint

FORMAT = "fate-relations-v1"


def validate_source(row, meta):
    """Verify original media when present; cache-only training remains supported."""
    if not Path(row["video"]).is_file():
        return
    assets = {"video": sha(row["video"])}
    if row.get("variant", {}).get("donor"):
        assets["donor"] = sha(row["variant"]["donor"])
    current = fingerprint({"assets": assets, "variant": row.get("variant", {"kind": "clean"})})
    if current != meta["source_fingerprint"]:
        raise ValueError(f"Source media/variant changed since extraction: {row['sample_id']}")


def load_sample(row, cache, radius, device="cpu"):
    path = Path(cache) / (row["sample_id"] + ".npz")
    meta = read_json(path.with_suffix(".json"))
    if meta.get("format") != "fate-window-v1" or sha(path) != meta["feature_sha256"]:
        raise ValueError(f"Invalid relation feature cache: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: archive[key].copy()
            for key in ("audio", "visual", "audio_valid", "visual_valid", "times_s")
        }
    n = len(arrays["times_s"])
    if (
        not n
        or not np.isfinite(arrays["times_s"]).all()
        or not np.allclose(np.diff(arrays["times_s"]), meta["step_s"], atol=1e-6)
    ):
        raise ValueError("Cache must have finite timestamps on a uniform grid")
    for name in ("audio", "visual"):
        if arrays[name].ndim != 2 or len(arrays[name]) != n or not np.isfinite(arrays[name]).all():
            raise ValueError("Invalid cached feature dimensions/values")
    batch = {
        key: torch.from_numpy(value)[None].to(device)
        for key, value in arrays.items()
        if key != "times_s"
    }
    batch["padding_valid"] = torch.ones((1, n), device=device, dtype=torch.bool)
    labels = labels_for(row, arrays["times_s"], meta["window_s"], meta["step_s"], radius)
    targets = {key: torch.from_numpy(value)[None].to(device) for key, value in labels.items()}
    return batch, targets, meta, arrays["times_s"]


def threshold_for_binary(labels, scores):
    """F1 threshold from validation only, with deterministic ties."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    candidates = np.unique(np.r_[0.5, scores]).clip(1e-6, 1 - 1e-6)
    best = (-1, 0.5)
    for threshold in candidates:
        predicted = scores >= threshold
        tp = np.sum(predicted & (labels == 1))
        fp = np.sum(predicted & (labels == 0))
        fn = np.sum(~predicted & (labels == 1))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best[1]


def collect_validation(model, rows, cache, device, active):
    model.eval()
    losses = []
    metrics = {name: {"labels": [], "scores": []} for name in active if name not in ACTIVITIES}
    with torch.no_grad():
        for row in rows:
            batch, targets, _, _ = load_sample(row, cache, model.config["radius"], device)
            output = model(batch, use_lag_alignment="timing" in active)
            selected = {
                key: value
                for key, value in targets.items()
                if ("timing" if key == "lag_class" else key) in active
            }
            try:
                loss, _ = relation_loss(output, selected)
            except ValueError as exc:
                if "No observed" in str(exc):
                    continue
                raise
            losses.append(float(loss))
            for name, bucket in metrics.items():
                if name == "timing":
                    target = targets["lag_class"]
                    mask = output["lag_valid"] & (target >= 0)
                    mask &= target < output["lag_logits"].shape[-1] - 1
                    # Report any nonzero lag; tolerance is explicitly zero grid steps.
                    labels = (target != model.config["radius"]).long()
                    probs = output["lag_logits"][..., :-1].softmax(-1)
                    scores = 1 - probs[..., model.config["radius"]]
                else:
                    labels = targets[name]
                    mask = output["valid"] & (labels >= 0)
                    scores = output["logits"][name].sigmoid()
                bucket["labels"].extend(labels[mask].cpu().tolist())
                bucket["scores"].extend(scores[mask].cpu().tolist())
    if not losses:
        raise ValueError("Validation has no observed usable labels")
    return float(np.mean(losses)), metrics


def fit_relations(cfg, resume=False):
    seed, device = int(cfg.get("seed", 42)), cfg.get("device", "cpu")
    options = cfg.get("training", {})
    torch.set_num_threads(int(options.get("torch_threads", 2)))
    torch.manual_seed(seed)
    rows = read_manifest(cfg["manifest"])
    validate(rows, check_files=False)
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "validation"]
    if not train or not val:
        raise ValueError("Training requires separate train and validation groups")
    radius = cfg.get("model", {}).get("radius", 4)
    seen = {
        split: {name: set() for name in (*RELATIONS, *ACTIVITIES, "timing")}
        for split in ("train", "validation")
    }
    signatures, content_hashes = set(), []
    first = None
    for row in train + val:
        batch, targets, meta, _ = load_sample(row, cfg["cache"], radius)
        validate_source(row, meta)
        first = batch if first is None else first
        signatures.add(meta["feature_signature"])
        content_hashes.append(
            (row["sample_id"], meta["feature_sha256"], meta["source_fingerprint"])
        )
        joint = batch["audio_valid"] & batch["visual_valid"]
        for name, target in targets.items():
            valid = (
                batch["audio_valid"]
                if name == "audio_speech"
                else (batch["visual_valid"] if name == "visual_speech" else joint)
            )
            values = target[valid & (target >= 0)].tolist()
            seen[row["split"]]["timing" if name == "lag_class" else name].update(values)
    if len(signatures) != 1:
        raise ValueError("All train/validation features must share one extraction signature")
    active = [
        name
        for name in (*RELATIONS, *ACTIVITIES)
        if all(seen[split][name] == {0, 1} for split in seen)
    ]
    if all(radius in seen[split]["timing"] and len(seen[split]["timing"]) >= 2 for split in seen):
        active.append("timing")
    if not set(active) & {"timing", *RELATIONS}:
        raise ValueError("Need positive and negative relation labels in both train and validation")
    model_cfg = {
        **cfg.get("model", {}),
        "audio_dim": first["audio"].shape[-1],
        "visual_dim": first["visual"].shape[-1],
    }
    model = RelationHeads(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.get("lr", 3e-4),
        weight_decay=options.get("weight_decay", 1e-4),
    )
    signature = next(iter(signatures))
    run_signature = fingerprint(
        {
            "rows": train + val,
            "cache": content_hashes,
            "model": model.config,
            "active": active,
            "seed": seed,
            "options": {k: v for k, v in options.items() if k != "epochs"},
        }
    )
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    last = out / "last.pt"
    start, best, stale = 0, float("inf"), 0
    if resume:
        state = torch.load(last, map_location=device, weights_only=True)
        if state.get("run_signature") != run_signature:
            raise ValueError("Resume config/data differ; use a new run directory")
        model.load_state_dict(state["state"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, stale = state["epoch"] + 1, state["best_loss"], state["stale"]
        torch.set_rng_state(state["rng_state"].cpu())
        if device.startswith("cuda") and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state"]])
    elif last.exists():
        raise FileExistsError("Run already exists; use --resume or change output")
    accumulation = int(options.get("accumulation", 8))
    epochs = int(options.get("epochs", 20))
    if accumulation < 1 or epochs < 1:
        raise ValueError("epochs and accumulation must be positive")
    history = read_json(out / "history.json") if resume else []
    for epoch in range(start, epochs):
        model.train()
        order = train.copy()
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        pending, losses = 0, []
        for row in order:
            batch, targets, _, _ = load_sample(row, cfg["cache"], radius, device)
            output = model(batch, use_lag_alignment="timing" in active)
            targets = {
                key: value
                for key, value in targets.items()
                if ("timing" if key == "lag_class" else key) in active
            }
            try:
                loss, _ = relation_loss(output, targets)
            except ValueError as exc:
                if "No observed" in str(exc):
                    continue
                raise
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            losses.append(float(loss.detach()))
            pending += 1
            if pending == accumulation:
                _step(model, optimizer, pending)
                pending = 0
        if pending:
            _step(model, optimizer, pending)
        if not losses:
            raise ValueError("No usable training labels after masking")
        val_loss, buckets = collect_validation(model, val, cfg["cache"], device, active)
        improved = val_loss < best
        best, stale = (val_loss, 0) if improved else (best, stale + 1)
        thresholds = {
            name: threshold_for_binary(bucket["labels"], bucket["scores"])
            for name, bucket in buckets.items()
            if set(bucket["labels"]) == {0, 1}
        }
        state = {
            "format": FORMAT,
            "state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model.config,
            "feature_signature": signature,
            "encoder_config": cfg["encoder"],
            "active_heads": active,
            "trained_heads": sorted(thresholds),
            "thresholds": thresholds,
            "lag_tolerance_steps": 0,
            "epoch": epoch,
            "best_loss": best,
            "stale": stale,
            "run_signature": run_signature,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
        }
        save_checkpoint(last, state)
        if improved:
            save_checkpoint(out / "best.pt", state)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": val_loss,
                "scorable_heads": sorted(thresholds),
            }
        )
        write_json(out / "history.json", history)
        print(history[-1], flush=True)
        if options.get("patience", 5) and stale >= options.get("patience", 5):
            break
    return {
        "checkpoint": str(out / "best.pt"),
        "trained_objectives": active,
        "note": "Only heads with usable positive/negative validation scores are exposed",
    }


def _step(model, optimizer, count):
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(count)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def load_relations(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError("Expected a trained FATE relation checkpoint")
    model = RelationHeads(**state["model_config"]).to(device)
    model.load_state_dict(state["state"])
    model.eval()
    return model, state
