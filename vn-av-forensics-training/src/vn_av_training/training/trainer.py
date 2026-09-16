"""Train, resume and adapt detectors on cached frozen AV-HuBERT features."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vn_av_training.common.runtime import atomic_bytes, fingerprint, read_json, sha, write_json
from vn_av_training.data.dataset import FeatureDataset, collate, interval_targets, to_device
from vn_av_training.data.manifest import read_manifest, validate
from vn_av_training.evaluation.metrics import predict, summarize, threshold_for
from vn_av_training.models.detector import Detector
from vn_av_training.models.losses import supervised_loss


def save_checkpoint(path, state):
    import io

    stream = io.BytesIO()
    torch.save(state, stream)
    atomic_bytes(path, stream.getvalue())


def load_checkpoint(path, device="cpu"):
    info = torch.load(path, map_location="cpu", weights_only=True)
    if info.get("format") != "vn-av-detector-v1":
        raise ValueError("Expected VN-AV detector checkpoint (not SyncNet/local.pt)")
    model = Detector(**info["model_config"]).to(device)
    model.load_state_dict(info["state"])
    model.eval()
    return model, info


def validate_cache(rows, cache):
    signatures = set()
    for r in rows:
        path = Path(cache) / (r["sample_id"] + ".npz")
        meta = read_json(path.with_suffix(".json"))
        if not path.is_file() or sha(path) != meta["feature_sha256"]:
            raise ValueError(f"Missing or corrupt features: {path}")
        if Path(r["video"]).is_file() and sha(r["video"]) != meta["source_sha256"]:
            raise ValueError(f"Changed source: {r['sample_id']}")
        signatures.add(meta["feature_signature"])
    if len(signatures) != 1:
        raise ValueError("Cache is empty or contains multiple encoder/preprocessing signatures")
    return next(iter(signatures))


def fit(cfg, resume=False, init_from=None):
    seed = int(cfg.get("seed", 42))
    device = cfg.get("device", "cpu")
    torch.set_num_threads(int(cfg.get("torch_threads", 2)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows = read_manifest(cfg["manifest"])
    validate(rows, check_files=False)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    if not train or not val:
        raise ValueError("Training requires train and validation splits")
    # Training does not load test features, labels or metrics.
    feature_signature = validate_cache(train + val, cfg["cache"])
    tcfg = cfg.get("training", {})
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not resume:
        raise FileExistsError("Run exists: use --resume or a new output directory")
    if resume and init_from:
        raise ValueError("Choose resume OR init-from")
    dataset = FeatureDataset(train, cfg["cache"], cfg.get("annotations"))
    first = dataset[0]
    model_cfg = dict(cfg.get("model", {}))
    model_cfg.setdefault("audio_dim", first["audio"].shape[-1])
    model_cfg.setdefault("visual_dim", first["visual"].shape[-1])
    model = Detector(**model_cfg).to(device)
    model_cfg = model.config
    phoneme_signature = None
    if model_cfg["phoneme"]:
        if not cfg.get("annotations"):
            raise ValueError("P3 requires an annotation file; run phonemes-auto or phonemes-import")
        events = read_json(cfg["annotations"])
        for split_rows in (train, val):
            selected = [
                e
                for r in split_rows
                for e in events.get(r["sample_id"], [])
                if e.get("phone") == "m"
            ]
            if not selected:
                raise ValueError("P3 needs /m/ events in both train and validation")
            if cfg.get("phoneme_mode", "manual_research") == "automatic" and any(
                e.get("source") != "automatic_mfa" for e in selected
            ):
                raise ValueError(
                    "Automatic P3 cannot consume manual annotations; use phonemes-auto"
                )
        if cfg.get("phoneme_mode") == "automatic":
            signatures = {
                e.get("pipeline_signature")
                for r in train + val
                for e in events.get(r["sample_id"], [])
            }
            if len(signatures) != 1 or None in signatures:
                raise ValueError(
                    "Automatic annotations need one matching alignment pipeline signature"
                )
            phoneme_signature = next(iter(signatures))
    if init_from:
        _, initial = load_checkpoint(init_from)
        if (
            initial["model_config"] != model_cfg
            or initial["feature_signature"] != feature_signature
        ):
            raise ValueError("init-from requires matching architecture and feature extraction")
        model.load_state_dict(initial["state"])
    labels = [r["clip_label"] for r in train if r.get("clip_label") is not None]
    if set(labels) != {0, 1}:
        raise ValueError(
            "Train clip labels must contain real and fake; controls alone are not fake data"
        )
    pos_weight = labels.count(0) / labels.count(1)
    weights = tcfg.get("loss_weights", {"clip": 1.0, "forgery": 1.0, "mismatch": 1.0})
    if weights.get("clip", 0) <= 0 or any(v < 0 for v in weights.values()):
        raise ValueError("Clip loss must be positive and loss weights nonnegative")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    stable_cfg = {k: v for k, v in tcfg.items() if k not in ("epochs", "patience")}
    signature = fingerprint(
        {
            "train_val": train + val,
            "features": feature_signature,
            "model": model_cfg,
            "training": stable_cfg,
            "seed": seed,
            "annotations": sha(cfg["annotations"]) if cfg.get("annotations") else None,
        }
    )
    start, best_loss, best_state, history, stale = 0, float("inf"), None, [], 0
    heads = {"clip": True, "forgery": False, "mismatch": False}
    if resume:
        state = torch.load(out / "last.pt", map_location=device, weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Resume config, data, labels or features changed")
        model.load_state_dict(state["state"])
        optimizer.load_state_dict(state["optimizer"])
        start, best_loss, best_state, history = (
            state["epoch"] + 1,
            state["best_loss"],
            state["best_state"],
            state["history"],
        )
        heads, stale = state["heads"], state["stale"]
        torch.set_rng_state(state["rng"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    batch_size = int(tcfg.get("batch_size", 4))
    validation = FeatureDataset(val, cfg["cache"], cfg.get("annotations"))
    for epoch in range(start, int(tcfg.get("epochs", 20))):
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collate,
        )
        model.train()
        total, count = 0.0, 0
        for raw in loader:
            batch = to_device(raw, device)
            output = model(batch)
            loss, components = supervised_loss(output, batch, weights, pos_weight)
            if not components:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 5.0)))
            optimizer.step()
            total += float(loss.detach())
            count += 1
            for h in components:
                heads[h] = True
        if not count:
            raise ValueError("No usable training samples")
        model.eval()
        val_total, val_count = 0.0, 0
        with torch.inference_mode():
            for raw in DataLoader(validation, batch_size=batch_size, collate_fn=collate):
                b = to_device(raw, device)
                loss, components = supervised_loss(model(b), b, weights, pos_weight)
                if components:
                    val_total += float(loss)
                    val_count += 1
        if not val_count:
            raise ValueError("No usable validation samples")
        val_loss = val_total / val_count
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append(
            {"epoch": epoch + 1, "train_loss": total / count, "validation_loss": val_loss}
        )
        print(history[-1], flush=True)
        save_checkpoint(
            out / "last.pt",
            {
                "signature": signature,
                "state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "best_state": best_state,
                "history": history,
                "heads": heads,
                "stale": stale,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        )
        write_json(out / "history.json", history)
        if tcfg.get("patience", 0) and stale >= tcfg["patience"]:
            break
    if best_state is None:
        raise ValueError("No trained state: epochs must be positive")
    model.load_state_dict(best_state)
    records = predict(model, validation, device, batch_size, heads=heads)
    eligible = [r for r in records if r["clip_label"] is not None and r["clip_score"] is not None]
    thresholds = {
        "clip": threshold_for(
            [r["clip_label"] for r in eligible], [r["clip_score"] for r in eligible]
        )
    }
    for head in ("forgery", "mismatch"):
        y, p = [], []
        for r in records:
            if head + "_scores" not in r:
                continue
            target, known = interval_targets(np.asarray(r["times_s"]), r.get(head + "_intervals"))
            valid = known & np.asarray(r["valid"])
            y.extend(target[valid].tolist())
            p.extend(np.asarray(r[head + "_scores"])[valid].tolist())
        if len(set(y)) == 2:
            thresholds[head] = threshold_for(y, p)
        else:
            heads[head] = False  # No calibrated temporal claim without both classes in validation.
    info = {
        "format": "vn-av-detector-v1",
        "model_config": model_cfg,
        "state": best_state,
        "feature_signature": feature_signature,
        "manifest_hash": sha(cfg["manifest"]),
        "config": cfg,
        "heads": heads,
        "thresholds": thresholds,
        "epochs": len(history),
        "best_validation_loss": best_loss,
        "signature": signature,
        "phoneme_signature": phoneme_signature,
        "phoneme_mode": cfg.get("phoneme_mode", "manual_research")
        if model_cfg["phoneme"]
        else "disabled",
    }
    save_checkpoint(out / "best.pt", info)
    write_json(out / "config.json", cfg)
    records = predict(model, validation, device, batch_size, thresholds, heads)
    write_json(out / "validation_predictions.json", records)
    write_json(out / "validation_metrics.json", summarize(records, thresholds))
    return out / "best.pt"
