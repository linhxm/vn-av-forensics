"""Local FATE loader and two-stream feature contract.

Uses the author's PE-AV implementation, not a generic AutoModel substitute.
Inputs must already be prepared by the upstream processor. This adapter does
not assign timestamps to interpolated tokens or infer mouth visibility.
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vn_av_training.common.runtime import (
    fingerprint,
    read_json,
    require_file,
    save_npz,
    sha,
    write_json,
)


def asset_signature(repo, base, adapter):
    repo, base, adapter = (Path(p).resolve() for p in (repo, base, adapter))
    require_file(repo / "models/pe_av/modeling_pe_audio_video.py", "FATE source")
    require_file(base / "config.json", "PE-AV config")
    require_file(adapter / "adapter_config.json", "FATE adapter config")
    require_file(adapter / "adapter_model.safetensors", "FATE adapter weights")
    if not list(base.glob("*.safetensors")):
        raise FileNotFoundError("PE-AV base directory needs safetensors weights")
    files = {}
    for label, root in (("source", repo / "models/pe_av"), ("base", base), ("adapter", adapter)):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".safetensors"):
                files[f"{label}/{path.relative_to(root).as_posix()}"] = sha(path)
    return fingerprint({"implementation": sha(__file__), "assets": files})


def _upstream_classes(repo):
    """Use a private package name; do not replace this project's src.models."""
    directory = Path(repo).resolve() / "models/pe_av"
    name = "_vn_av_fate_" + fingerprint(str(directory))[:16]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        sys.modules[name] = package
    model = importlib.import_module(name + ".modeling_pe_audio_video")
    processor = importlib.import_module(name + ".processing_pe_audio_video")
    return model.PeAudioVideoModel, processor.PeAudioVideoProcessor


class FATEBackbone(nn.Module):
    """One frozen AV backbone; its two streams stay independent in content."""

    def __init__(self, model, processor, signature):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.processor = processor
        self.signature = signature
        self.vision_cache = None

    def enable_vision_cache(self):
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        if not hasattr(base, "video_encoder"):
            return
        vision = base.video_encoder.embedder.vision_model
        self.vision_cache = CachedFrameVision(vision)
        base.video_encoder.embedder.vision_model = self.vision_cache

    @classmethod
    def from_local(cls, repo, base, adapter, device="cpu", precision="float32"):
        signature = asset_signature(repo, base, adapter)
        from peft import PeftModel

        model_cls, processor_cls = _upstream_classes(repo)
        dtype = (
            {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
                precision
            ]
            if str(device).startswith("cuda")
            else torch.float32
        )
        model = load_base_weights(model_cls, base, dtype)
        processor = processor_cls.from_pretrained(str(base), local_files_only=True)
        model = PeftModel.from_pretrained(
            model, str(adapter), is_trainable=False, local_files_only=True
        )
        return cls(model.to(device), processor, signature)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, processed_inputs):
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in processed_inputs.items()}
        outputs = self.model(**inputs)
        audio = outputs.audio_frame_embeds.float()
        visual = outputs.video_frame_embeds.float()
        if audio.ndim != 3 or visual.ndim != 3 or audio.shape[:2] != visual.shape[:2]:
            raise ValueError("Expected aligned FATE [batch, time, feature] outputs")
        if not torch.isfinite(audio).all() or not torch.isfinite(visual).all():
            raise ValueError("FATE returned nonfinite features")
        return {"audio": audio, "visual": visual}


class CachedFrameVision(nn.Module):
    """Reuse independent, frozen per-image logits across overlapping windows.

    Temporal video layers still run on each complete window. Cache only one source
    at a time; audio variants of that source can share unchanged visual features.
    """

    def __init__(self, vision):
        super().__init__()
        self.vision = vision
        self.source = None
        self.indices = None
        self.cache = {}

    def select(self, source, indices):
        if source != self.source:
            self.cache.clear()
            self.source = source
        self.indices = tuple(int(i) for i in indices)

    def forward(self, pixels):
        if self.indices is None or len(self.indices) != len(pixels):
            return self.vision(pixels)
        positions = {}
        for position, index in enumerate(self.indices):
            if index not in self.cache:
                positions.setdefault(index, position)
        if positions:
            missing = list(positions)
            logits = self.vision(pixels[list(positions.values())]).logits
            for index, value in zip(missing, logits):
                self.cache[index] = value.detach().cpu()
        return types.SimpleNamespace(
            logits=torch.stack([self.cache[index] for index in self.indices]).to(pixels.device)
        )


def base_weight_name(name, available):
    """Official base contains audio_model/video_model wrappers absent in FATE's class."""
    if name in available:
        return name
    for stream in ("audio", "video"):
        if name.startswith((stream + "_encoder.", stream + "_head.")):
            candidate = stream + "_model." + name
            if candidate in available:
                return candidate
    raise ValueError(f"Pretrained base is missing required FATE tensor: {name}")


def load_base_weights(model_cls, base, dtype):
    """Load every backbone tensor strictly; never accept silently random initialization."""
    from accelerate import init_empty_weights
    from safetensors import safe_open

    config = model_cls.config_class.from_pretrained(str(base), local_files_only=True)
    with init_empty_weights():
        model = model_cls(config)
    files = sorted(Path(base).glob("*.safetensors"))
    locations = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 -- safetensors handle is not a dict
                if key in locations:
                    raise ValueError(f"Duplicate pretrained tensor: {key}")
                locations[key] = path
    weights = {}
    # The official base stores F32 tensors. NumPy reads avoid a large PyTorch
    # storage-slicing overhead on Windows; keep one mapping per shard open.
    with ExitStack() as stack:
        handles = {path: stack.enter_context(safe_open(path, framework="numpy")) for path in files}
        for name, expected in model.state_dict().items():
            key = base_weight_name(name, locations)
            tensor = torch.from_numpy(handles[locations[key]].get_tensor(key))
            if tensor.shape != expected.shape:
                raise ValueError(
                    f"Pretrained shape mismatch for {name}: {tensor.shape} vs {expected.shape}"
                )
            weights[name] = tensor.to(dtype) if tensor.is_floating_point() else tensor
    model.load_state_dict(weights, strict=True, assign=True)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("Pretrained model contains unmaterialized parameters")
    return model


def face_crops(frames):
    """Single-face crops include the mouth; missing/multiple detections stay invalid."""
    import cv2

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV face detector asset is missing")
    crops, valid = [], []
    for frame in frames:
        faces = detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        seen = len(faces) == 1
        if seen:
            x, y, w, h = faces[0]
            crop = frame[y : y + h, x : x + w]
            crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (336, 336))
        else:
            crop = np.zeros((336, 336, 3), np.uint8)
        crops.append(crop)
        valid.append(seen)
    return np.stack(crops), np.asarray(valid, bool)


def window_plan(n, window_s, step_s):
    window, stride = round(window_s * 25), round(step_s * 25)
    if (
        window < 10
        or stride < 1
        or not np.isclose(window / 25, window_s)
        or not np.isclose(stride / 25, step_s)
    ):
        raise ValueError("window_s and step_s must lie on the 25 Hz media grid")
    if stride > window:
        raise ValueError("step_s cannot exceed window_s")
    for start in range(0, n, stride):
        center = start + stride / 2
        left = int(np.floor(center - window / 2))
        index = np.arange(left, left + window)
        yield start / 25, index, (index >= 0) & (index < n)


class FATEVideoEncoder:
    """Frozen window features with explicit support, not guessed frame-token PTS.

    The averaging is an adaptation for trainable window heads. Reported cell
    spacing is not a claim that localization accuracy equals that spacing.
    """

    def __init__(self, cfg, backbone=None, cropper=None):
        self.cfg = dict(cfg)
        self.backbone = backbone or FATEBackbone.from_local(
            cfg["repo"],
            cfg["base"],
            cfg["adapter"],
            cfg.get("device", "cpu"),
            cfg.get("precision", "float32"),
        )
        if cfg.get("reuse_vision", True) and hasattr(self.backbone, "enable_vision_cache"):
            self.backbone.enable_vision_cache()
        if cropper is not None:
            self.cropper = cropper
        elif cfg.get("face_model"):
            from vn_av_training.features.face import FaceTracker

            self.cropper = FaceTracker(cfg["face_model"])
        else:
            self.cropper = face_crops
        options = {
            key: cfg.get(key, default)
            for key, default in (
                ("window_s", 2.0),
                ("step_s", 0.2),
                ("min_coverage", 0.75),
                ("max_duration", 60),
                ("max_side", 640),
                ("precision", "float32"),
                ("temporal_bins", 1),
            )
        }
        self.options = options
        self.signature = fingerprint(
            {
                "backbone": self.backbone.signature,
                "options": options,
                "face_model": sha(cfg["face_model"]) if cfg.get("face_model") else "opencv-haar",
                "implementation": {
                    "encoder": sha(__file__),
                    "media": sha(Path(__file__).parents[1] / "data/media.py"),
                    "face": sha(Path(__file__).with_name("face.py")),
                },
                "format": "fate-window-v1",
            }
        )
        if not 0 < options["min_coverage"] <= 1:
            raise ValueError("min_coverage must be in (0,1]")
        if options["precision"] not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported precision")
        if not isinstance(options["temporal_bins"], int) or options["temporal_bins"] < 1:
            raise ValueError("temporal_bins must be a positive integer")

    def source_fingerprint(self, row):
        assets = {"video": sha(row["video"])}
        if row.get("variant", {}).get("donor"):
            assets["donor"] = sha(row["variant"]["donor"])
        return fingerprint({"assets": assets, "variant": row.get("variant", {"kind": "clean"})})

    def decode(self, path):
        from vn_av_training.data.media import decode

        return decode(
            path, self.options["max_duration"], self.options["max_side"], sample_rate=48000
        )

    @torch.no_grad()
    def extract(self, row, output):
        if row.get("variant", {"kind": "clean"}) != {"kind": "clean"}:
            raise ValueError("Render media in the generation project before extraction")
        output = Path(output)
        source = self.source_fingerprint(row)
        if output.is_file() and output.with_suffix(".json").is_file():
            meta = read_json(output.with_suffix(".json"))
            if (
                meta.get("source_fingerprint") == source
                and meta.get("feature_signature") == self.signature
                and meta.get("feature_sha256") == sha(output)
            ):
                return meta
        decoded = self.decode(row["video"])
        crops, seen = self.cropper(decoded["frames"])
        visual_source = fingerprint({"video": sha(row["video"]), "encoder": self.signature})
        n = len(crops)
        af, vf, times, am, vm, support = [], [], [], [], [], []
        device = next(self.backbone.parameters()).device
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            self.options["precision"]
        ]
        if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support bfloat16; use float16 or float32")
        for time, index, inside in window_plan(n, self.options["window_s"], self.options["step_s"]):
            vi = index.clip(0, n - 1)
            video = crops[vi].copy()
            video[~inside] = 0
            # Exact PCM slices on the same physical grid, padded without wrapping.
            sample_index = index[0] * 1920 + np.arange(len(index) * 1920)
            audio_inside = (sample_index >= 0) & (sample_index < len(decoded["pcm"]))
            audio = np.zeros(len(sample_index), np.float32)
            audio[audio_inside] = decoded["pcm"][sample_index[audio_inside]]
            inputs = self.backbone.processor(
                videos=[torch.from_numpy(video).permute(0, 3, 1, 2)],
                audio=[audio],
                return_tensors="pt",
                padding=True,
                sampling_rate=48000,
            )
            context = (
                torch.autocast("cuda", dtype=dtype)
                if (device.type == "cuda" and dtype != torch.float32)
                else nullcontext()
            )
            with context:
                if self.backbone.vision_cache is not None:
                    self.backbone.vision_cache.select(visual_source, np.where(inside, vi, -1))
                features = self.backbone(inputs)
            af.append(temporal_descriptor(features["audio"][0], self.options["temporal_bins"]))
            vf.append(temporal_descriptor(features["visual"][0], self.options["temporal_bins"]))
            times.append(time)
            am.append(np.mean(inside & decoded["audio_valid"][vi]) >= self.options["min_coverage"])
            vm.append(
                np.mean(inside & decoded["visual_valid"][vi] & seen[vi])
                >= self.options["min_coverage"]
            )
            support.append([max(0, index[0] / 25), min(n / 25, (index[-1] + 1) / 25)])
        save_npz(
            output,
            audio=np.stack(af),
            visual=np.stack(vf),
            times_s=np.asarray(times),
            audio_valid=np.asarray(am),
            visual_valid=np.asarray(vm),
            support_s=np.asarray(support),
        )
        meta = {
            "format": "fate-window-v1",
            "feature_signature": self.signature,
            "feature_sha256": sha(output),
            "source_fingerprint": source,
            "step_s": self.options["step_s"],
            "window_s": self.options["window_s"],
            "duration_s": n / 25,
            "origin_s": decoded["origin_s"],
            "localization_note": "Window stride is not measured localization accuracy",
            "temporal_bins": self.options["temporal_bins"],
        }
        write_json(output.with_suffix(".json"), meta)
        return meta


def temporal_descriptor(sequence, bins):
    """Keep ordered temporal bins, never invent extra temporal samples by upsampling.

    Bin positions are relative to model tokens, not certified phoneme timestamps.
    """
    from torch.nn import functional as F

    if sequence.shape[0] < bins:
        raise ValueError("FATE returned fewer temporal tokens than temporal_bins")
    sequence = F.normalize(sequence.float(), dim=-1)
    pooled = F.adaptive_avg_pool1d(sequence.T[None], bins)[0].T
    return pooled.flatten().cpu().numpy()


def prepare_relations(cfg):
    from vn_av_training.data.manifest import read_manifest, validate

    rows = read_manifest(cfg["manifest"])
    validate(rows)
    encoder = FATEVideoEncoder(cfg["encoder"])
    for i, row in enumerate(rows):
        print(f"FATE {i + 1}/{len(rows)} {row['sample_id']}", flush=True)
        encoder.extract(row, Path(cfg["cache"]) / (row["sample_id"] + ".npz"))
    return {"samples": len(rows), "feature_signature": encoder.signature}
