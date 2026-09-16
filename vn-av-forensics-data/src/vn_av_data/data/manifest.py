import json
from pathlib import Path

from vn_av_data.common.runtime import atomic_bytes


def read_manifest(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_manifest(path, rows):
    atomic_bytes(
        path,
        (
            "\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows) + "\n"
        ).encode(),
    )
