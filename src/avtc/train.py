"""Train local detector and fine-tune pretrained FC layers; no test-set model selection."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .model import LocalModel
from .syncnet import METHOD, sha, write_json


def read_feature(root, row, device):
    with np.load(Path(root) / row["feature"], allow_pickle=False) as z:
        return {
            k: torch.tensor(
                z[k], device=device, dtype=torch.bool if k == "valid" else torch.float32
            )
            for k in ("visual", "audio", "label", "valid")
        }


def collect(model, root, rows, device):
    model.eval()
    tables = []
    with torch.inference_mode():
        for row in rows:
            x = read_feature(root, row, device)
            r = model.radius
            score = torch.sigmoid(model(x["visual"], x["audio"])).cpu().numpy()
            with np.load(Path(root) / row["feature"], allow_pickle=False) as z:
                tables.append(
                    pd.DataFrame(
                        {
                            "source_id": row["source_id"],
                            "speaker": row["speaker"],
                            "variant": row["variant"],
                            "time_s": z["times"][r:-r],
                            "score": score,
                            "label": z["label"][r:-r].astype(bool),
                            "valid": z["valid"][r:-r].astype(bool),
                        }
                    )
                )
    if not tables:
        raise ValueError("No usable samples in this split")
    return pd.concat(tables, ignore_index=True)


def select_threshold(table):
    valid = table[table.valid]
    if valid.label.nunique() != 2:
        raise ValueError("Validation must contain both aligned and locally inconsistent windows")
    best = (-1.0, 0.5)
    for t in np.unique(np.r_[0.0, np.quantile(valid.score, np.linspace(0, 1, 201)), 1.0]):
        pred = valid.score.to_numpy() > t
        gold = valid.label.to_numpy()
        tp = int((pred & gold).sum())
        fp = int((pred & ~gold).sum())
        fn = int((~pred & gold).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]


def fit(cache, output, epochs=10, fine_tune=True, resume=False, device=None, seed=42):
    root, out = Path(cache), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    index = json.loads((root / "index.json").read_text())
    splits = {
        s: [r for r in index["rows"] if r["split"] == s] for s in ("train", "validation", "test")
    }
    if any(not rows for rows in splits.values()):
        raise ValueError("Need train, validation, test")
    model = LocalModel(fine_tune=fine_tune).to(device)
    model.initialize(root / "projection_init.pt")
    projection_params = [
        p for name, p in model.named_parameters() if "projection" in name and p.requires_grad
    ]
    head_params = [
        p for name, p in model.named_parameters() if "projection" not in name and p.requires_grad
    ]
    groups = [{"params": head_params, "lr": 1e-3}]
    if projection_params:
        groups.append({"params": projection_params, "lr": 1e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    signature = {
        "cache_sha256": sha(root / "index.json"),
        "model_config": model.config,
        "seed": seed,
    }
    start, best_loss, best_state, history = 0, float("inf"), None, []
    last = out / "last.pt"
    if resume:
        if not last.exists():
            raise FileNotFoundError("No last.pt to resume")
        state = torch.load(last, map_location=device, weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Resume config/cache differs")
        model.load_state_dict(state["state"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["epoch"] + 1
        best_loss = state["best_loss"]
        best_state = state["best_state"]
        history = state["history"]
    elif last.exists() or (out / "local.pt").exists():
        raise FileExistsError("Use resume=True or a new training output directory")
    positive = negative = 0
    for row in splits["train"]:
        x = read_feature(root, row, "cpu")
        r = model.radius
        y = x["label"][r:-r][x["valid"][r:-r]]
        positive += int(y.sum())
        negative += len(y) - int(y.sum())
    if not positive or not negative:
        raise ValueError("Train split missing a class")
    weight = torch.tensor(negative / positive, device=device)
    for epoch in range(start, epochs):
        torch.manual_seed(seed + epoch)
        order = list(splits["train"])
        random.Random(seed + epoch).shuffle(order)
        model.train()
        total, count = 0.0, 0
        for row in order:
            x = read_feature(root, row, device)
            r = model.radius
            valid = x["valid"][r:-r]
            if valid.sum() < 3:
                continue
            logits = model(x["visual"], x["audio"])
            loss = F.binary_cross_entropy_with_logits(
                logits[valid], x["label"][r:-r][valid], pos_weight=weight
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            count += 1
        if count == 0:
            raise ValueError("No usable training windows")
        validation = collect(model, root, splits["validation"], device)
        g = validation[validation.valid]
        p = np.clip(g.score.to_numpy(), 1e-7, 1 - 1e-7)
        y = g.label.to_numpy(dtype=float)
        val_loss = float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))
        if not np.isfinite(val_loss):
            raise ValueError("No usable validation windows")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(
            {"epoch": epoch + 1, "train_loss": total / count, "validation_loss": val_loss}
        )
        print(history[-1], flush=True)
        torch.save(
            {
                "signature": signature,
                "state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "best_state": best_state,
                "history": history,
            },
            out / "last.tmp",
        )
        (out / "last.tmp").replace(last)
        pd.DataFrame(history).to_csv(out / "learning_curve.csv", index=False)
    if best_state is None:
        raise ValueError("No trained state")
    model.load_state_dict(best_state)
    validation = collect(model, root, splits["validation"], device)
    threshold, val_f1 = select_threshold(validation)
    validation.to_csv(out / "validation_timeline.csv", index=False)
    checkpoint = {
        "method": METHOD,
        "model_config": model.config,
        "state": best_state,
        "threshold": threshold,
        "validation_f1": val_f1,
        "signature": signature,
        "epochs_completed": len(history),
        "backbone_sha256": index["backbone_sha256"],
        "training": "Fine-tuned pretrained FC layers + temporal head; CNN frozen"
        if fine_tune
        else "Trained temporal head; SyncNet frozen",
    }
    torch.save(checkpoint, out / "local.tmp")
    (out / "local.tmp").replace(out / "local.pt")
    write_json(out / "training.json", {k: v for k, v in checkpoint.items() if k != "state"})
    return out / "local.pt"
