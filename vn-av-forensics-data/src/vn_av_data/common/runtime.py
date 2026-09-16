"""Versioned, atomic artifacts shared by all pipeline stages. No implicit downloads."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

VERSION = "1.0"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    atomic_bytes(
        path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    )


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_npz(path, **arrays):
    import io

    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run(args, *, cwd=None, timeout=1800, log=None):
    result = subprocess.run(
        [str(x) for x in args],
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if log:
        atomic_bytes(log, (result.stdout + result.stderr).encode())
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {args[0]}\n{result.stderr[-4000:]}"
        )
    return result.stdout


def _merge_config(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_config(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _read_config(path, seen):
    import yaml

    path = Path(path).resolve()
    seen = set(seen)
    if path in seen:
        raise ValueError("Cyclic config inheritance")
    seen.add(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    parent = cfg.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str) or not parent:
            raise ValueError("extends must name one config file")
        cfg = _merge_config(_read_config(path.parent / parent, seen), cfg)
    return cfg


def load_config(path, profile=None):
    """Resolve file inheritance, then apply exactly one named profile to shared settings."""
    cfg = _read_config(path, set())
    profiles = cfg.pop("profiles", {})
    default = cfg.pop("default_profile", None)
    if not isinstance(profiles, dict) or any(
        not isinstance(name, str) or not name or not isinstance(value, dict)
        for name, value in profiles.items()
    ):
        raise ValueError("profiles must map nonempty names to configuration mappings")
    for value in profiles.values():
        if any(key in value for key in ("extends", "profiles", "default_profile")):
            raise ValueError("Profiles contain overrides only; inheritance belongs at file level")
    if default is not None and (not isinstance(default, str) or default not in profiles):
        raise ValueError("default_profile must name a defined profile")
    selected = profile if profile is not None else default
    if selected is None and profiles:
        raise ValueError("Choose --profile; available: " + ", ".join(profiles))
    if selected is not None:
        if not isinstance(selected, str) or selected not in profiles:
            raise ValueError(
                f"Unknown profile {selected!r}; available: {', '.join(profiles) or '(none)'}"
            )
        cfg = _merge_config(cfg, profiles[selected])
    # Asset paths are workspace-relative; only extends is config-relative.
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "cpu")
    return cfg


def require_file(path, name):
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Missing {name}: {path}. See README.md or run the setup command.")
    return Path(path).resolve()
