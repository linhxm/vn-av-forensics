"""Portable compatibility fingerprints for checkpoints produced on another OS."""

from __future__ import annotations

import hashlib
from pathlib import Path

from vn_av_training.common.runtime import fingerprint


def portable_sha(path):
    path = Path(path)
    content = path.read_bytes()
    if path.suffix in {".json", ".py"}:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def portable_encoder_signature(cfg):
    """Match semantic assets while ignoring line endings and download metadata."""
    package = Path(__file__).resolve().parent
    files = {}
    roots = (
        ("source", Path(cfg["repo"]) / "models/pe_av"),
        ("base", Path(cfg["base"])),
        ("adapter", Path(cfg["adapter"])),
    )
    for label, root in roots:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and path.suffix in (".py", ".json", ".safetensors")
                and not any(part.startswith(".") for part in relative.parts)
            ):
                files[f"{label}/{relative.as_posix()}"] = portable_sha(path)
    backbone = fingerprint({"implementation": portable_sha(package / "fate.py"), "assets": files})
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
    return fingerprint(
        {
            "backbone": backbone,
            "options": options,
            "face_model": portable_sha(cfg["face_model"])
            if cfg.get("face_model")
            else "opencv-haar",
            "implementation": {
                "encoder": portable_sha(package / "fate.py"),
                "media": portable_sha(package.parent / "data/media.py"),
                "face": portable_sha(package / "face.py"),
            },
            "format": "fate-window-v1",
        }
    )
