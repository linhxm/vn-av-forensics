"""Turn trained relation-head outputs into intervals on the original time grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vn_av_training.common.runtime import sha, write_json
from vn_av_training.models.relations import RELATIONS


def aggregate_relations(
    outputs, times_s, step_s, thresholds, trained_heads, lag_tolerance_steps=1, global_consensus=0.8
):
    """Aggregate a single clip. Scores are anomaly scores, not fake probabilities.

    times_s denotes cell starts on a verified uniform grid; step_s is supplied
    by preprocessing, never inferred from a model's token count. Missing cells
    break intervals, and an untrained head is never used in clip scoring.
    """
    times = np.asarray(times_s, dtype=np.float64)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or times[0] < 0
        or not np.isfinite(step_s)
        or step_s <= 0
        or not np.allclose(np.diff(times), step_s, atol=1e-6, rtol=1e-5)
    ):
        raise ValueError("Supply finite nonnegative cell starts and a verified uniform step_s")
    if outputs["valid"].shape != (1, len(times)):
        raise ValueError("Aggregation requires one clip matching the supplied time grid")
    allowed = {"timing", *RELATIONS}
    trained = set(trained_heads)
    if not trained <= allowed:
        raise ValueError("Unknown trained relation head")
    if any(
        name not in thresholds or not np.isfinite(thresholds[name]) or not 0 < thresholds[name] < 1
        for name in trained
    ):
        raise ValueError("Every trained relation head needs a threshold in (0, 1)")
    if lag_tolerance_steps < 0 or int(lag_tolerance_steps) != lag_tolerance_steps:
        raise ValueError("lag_tolerance_steps must be a nonnegative integer")
    if not 0.5 < global_consensus <= 1:
        raise ValueError("global_consensus must be in (0.5, 1]")

    def array(tensor):
        return tensor.detach().cpu().numpy()[0]

    valid = array(outputs["valid"]).astype(bool)
    scores, masks = {}, {}
    lag_valid = array(outputs["lag_valid"]).astype(bool) & valid & ("timing" in trained)
    lag = array(outputs["lag_steps"]) * float(step_s) * 1000
    confidence = array(outputs["lag_confidence"])
    for name in trained:
        if name == "timing":
            logits = outputs["lag_logits"][..., :-1]
            radius = (logits.shape[-1] - 1) // 2
            positions = torch.arange(-radius, radius + 1, device=logits.device)
            outside = positions.abs() > lag_tolerance_steps
            scores[name] = array(logits.softmax(-1)[..., outside].sum(-1))
            masks[name] = lag_valid
        else:
            scores[name] = array(outputs["logits"][name].sigmoid())
            masks[name] = valid
        if not np.isfinite(scores[name][masks[name]]).all():
            raise ValueError(f"Nonfinite scores: {name}")

    intervals = []
    for name in sorted(trained):
        active = masks[name] & (scores[name] >= thresholds[name])
        edges = np.diff(np.r_[False, active, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            intervals.append(
                {
                    "start": float(times[start]),
                    "end": float(times[end - 1] + step_s),
                    "relation": name,
                    "inconsistency_score": float(scores[name][start:end].max()),
                }
            )
    intervals.sort(key=lambda item: (item["start"], item["end"], item["relation"]))
    windows = []
    for i, start in enumerate(times):
        relations = {
            name: float(scores[name][i]) if masks[name][i] else None for name in sorted(trained)
        }
        observed = [value for value in relations.values() if value is not None]
        windows.append(
            {
                "start": float(start),
                "end": float(start + step_s),
                "inconsistency_score": max(observed) if observed else None,
                "relation_scores": relations,
                "estimated_lag_ms": float(lag[i]) if lag_valid[i] else None,
                "lag_confidence": float(confidence[i])
                if valid[i] and "timing" in trained
                else None,
            }
        )
    global_lag, consensus = None, None
    if lag_valid.any():
        values, weights = lag[lag_valid], confidence[lag_valid]
        order = np.argsort(values)
        median = values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]
        consensus = float(
            weights[np.abs(values - median) <= step_s * 1000 + 1e-6].sum() / weights.sum()
        )
        # A local estimate is not a global lag when much of the clip is unassessable.
        if consensus >= global_consensus and lag_valid.mean() >= global_consensus:
            global_lag = float(median)
    clip_scores = [
        window["inconsistency_score"]
        for window in windows
        if window["inconsistency_score"] is not None
    ]
    complete = trained == allowed and all(mask.all() for mask in masks.values())
    return {
        "schema_version": "av-relations-v1",
        "status": "not_assessable" if not clip_scores else "assessed" if complete else "partial",
        "score_kind": "anomaly_score_not_calibrated_probability",
        "clip_inconsistency_score": max(clip_scores) if clip_scores else None,
        "window_scores": windows,
        "suspicious_intervals": intervals,
        "global_lag_ms": global_lag,
        "lag_consensus": consensus,
        "lag_convention": "A(t) matches V(t+k); positive lag means audio leads",
        "head_availability": {
            name: "trained" if name in trained else "untrained" for name in sorted(allowed)
        },
        "coverage": {name: float(mask.mean()) for name, mask in masks.items()},
        "identity_status": "not_assessed",
        "identity_reason": "Active-speaker correspondence does not verify voice-face identity",
    }


class RelationPipeline:
    """Compose backbone -> shared heads -> temporal aggregation.

    This integration accepts preprocessed tensors plus verified token metadata.
    Raw-video face tracking and token-to-PTS mapping are separate preparation work.
    """

    def __init__(self, backbone, heads, trained_heads=(), thresholds=None):
        self.backbone = backbone
        self.heads = heads
        self.trained_heads = tuple(trained_heads)
        self.thresholds = dict(thresholds or {})

    @torch.no_grad()
    def analyze(self, processed_inputs, token_metadata, step_s):
        if not self.trained_heads:
            raise ValueError("Train relation heads before requesting detector scores")
        if set(token_metadata) != {"times_s", "audio_valid", "visual_valid", "padding_valid"}:
            raise ValueError("Supply verified token timestamps and modality/padding masks")
        self.backbone.eval()
        self.heads.eval()
        features = self.backbone(processed_inputs)
        device = next(self.heads.parameters()).device
        batch = {key: value.to(device) for key, value in features.items()}
        for key in ("audio_valid", "visual_valid", "padding_valid"):
            batch[key] = torch.as_tensor(token_metadata[key], device=device, dtype=torch.bool)
        outputs = self.heads(batch, use_lag_alignment="timing" in self.trained_heads)
        return aggregate_relations(
            outputs, token_metadata["times_s"], step_s, self.thresholds, self.trained_heads
        )


class RelationAnalyzer:
    """Raw video -> checkpoint-compatible FATE windows -> trained relation scores."""

    def __init__(self, cfg, encoder=None):
        from vn_av_training.features.fate import FATEVideoEncoder
        from vn_av_training.training.relations import load_relations

        self.cfg = cfg
        self.device = cfg.get("device", "cpu")
        self.model, self.state = load_relations(cfg["checkpoint"], self.device)
        if not self.state["trained_heads"]:
            raise ValueError("Checkpoint has no relation head with usable validation scores")
        self.encoder = encoder or FATEVideoEncoder(cfg["encoder"])
        if self.encoder.signature != self.state["feature_signature"]:
            raise ValueError("Encoder assets/window settings differ from the training cache")

    @torch.no_grad()
    def analyze(self, video, output, progress=None):
        from vn_av_training.training.relations import load_sample

        progress = progress or (lambda *_: None)
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=True)
        row = {"sample_id": "input", "video": str(video), "variant": {"kind": "clean"}}
        progress("features", "Đang trích đặc trưng âm thanh và vùng mặt theo thời gian")
        meta = self.encoder.extract(row, folder / "input.npz")
        row["duration_s"] = meta["duration_s"]
        batch, _, _, times = load_sample(row, folder, self.model.config["radius"], self.device)
        progress("relations", "Đang đánh giá quan hệ tiếng nói–chuyển động miệng")
        prediction = self.model(batch, use_lag_alignment="timing" in self.state["active_heads"])
        report = aggregate_relations(
            prediction,
            times,
            meta["step_s"],
            self.state["thresholds"],
            self.state["trained_heads"],
            lag_tolerance_steps=self.state.get("lag_tolerance_steps", 0),
        )
        for window in report["window_scores"]:
            window["end"] = min(window["end"], meta["duration_s"])
        for span in report["suspicious_intervals"]:
            span["end"] = min(span["end"], meta["duration_s"])
        report.update(
            duration_s=meta["duration_s"],
            context_window_s=meta["window_s"],
            step_s=meta["step_s"],
            thresholds=self.state["thresholds"],
            model_id=sha(self.cfg["checkpoint"])[:16],
            notice="Điểm bất nhất môi–tiếng; không phải kết luận thật/giả hoặc xác minh danh tính.",
        )
        receipt = Path(video).with_suffix(".json")
        provenance = None
        if receipt.is_file():
            from vn_av_training.common.runtime import read_json

            candidate = read_json(receipt)
            if candidate.get("format") == "curated-clip-v1" and candidate.get("sha256") == sha(
                video
            ):
                provenance = candidate
        if provenance is None:
            from vn_av_training.data.bundle import video_provenance

            provenance = video_provenance(video)
        if provenance is not None:
            offset = provenance["source_start_s"]
            report["source_id"] = provenance["source_id"]
            report["source_start_s"] = offset
            report["source_intervals"] = [
                {**span, "start": span["start"] + offset, "end": span["end"] + offset}
                for span in report["suspicious_intervals"]
            ]
        write_json(folder / "result.json", report)
        import csv

        with (folder / "timeline.csv").open("w", encoding="utf-8", newline="") as stream:
            keys = [
                "start",
                "end",
                "inconsistency_score",
                "estimated_lag_ms",
                *self.state["trained_heads"],
            ]
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for window in report["window_scores"]:
                writer.writerow(
                    {**{key: window[key] for key in keys[:4]}, **window["relation_scores"]}
                )
        progress("complete", "Đã lưu khoảng khả nghi và điểm theo thời gian")
        return report


def relation_health(cfg):
    import importlib.util

    ecfg = cfg.get("encoder", {})
    missing = [
        name
        for name in ("torch", "av", "cv2", "transformers", "peft", "accelerate")
        if importlib.util.find_spec(name) is None
    ]
    files = {
        "FATE source": Path(ecfg.get("repo", "")) / "models/pe_av/modeling_pe_audio_video.py",
        "base config": Path(ecfg.get("base", "")) / "config.json",
        "adapter config": Path(ecfg.get("adapter", "")) / "adapter_config.json",
        "adapter weights": Path(ecfg.get("adapter", "")) / "adapter_model.safetensors",
    }
    missing.extend(name for name, path in files.items() if not path.is_file())
    if ecfg.get("face_model") and not Path(ecfg["face_model"]).is_file():
        missing.append("YuNet face model")
    if not list(Path(ecfg.get("base", "")).glob("*.safetensors")):
        missing.append("base weights")
    preprocessing_ready = not missing
    if not Path(cfg.get("checkpoint", "")).is_file():
        missing.append("trained relation checkpoint")
    return {
        "ready": not missing,
        "preprocessing_ready": preprocessing_ready,
        "missing": missing,
        "validation": "File/module preflight; model loading checked on first use",
    }
