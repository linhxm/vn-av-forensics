import csv
import io
from pathlib import Path

from vn_av_data.common.runtime import atomic_bytes


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def unique_rows(rows, key="clip_id"):
    result = {}
    for row in rows:
        if not row.get(key) or row[key] in result:
            raise ValueError(f"Missing/duplicate {key}")
        result[row[key]] = row
    return result


def write_rows(path, rows, mutable=False):
    path = Path(path)
    if path.exists() and not mutable:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        if read_rows(path) == [{k: str(r.get(k, "") or "") for k in fields} for r in rows]:
            return
        raise FileExistsError(f"Existing source/manifest differs: {path}; choose a new run")
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, out.getvalue().encode("utf-8-sig"))
