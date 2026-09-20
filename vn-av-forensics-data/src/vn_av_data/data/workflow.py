"""Batch manifest/reviewer steps corresponding to the Capstone collection workflow."""

from pathlib import Path

from vn_av_data.data.source_io import read_rows, unique_rows, write_rows


def merge_candidates(root, output):
    root = Path(root).resolve()
    rows = []
    for manifest in sorted(root.rglob("candidates.csv")):
        for row in read_rows(manifest):
            video = (manifest.parent / row["file_path"]).resolve()
            if not video.is_relative_to(root) or not video.is_file():
                raise ValueError("Candidate media missing/outside root")
            rows.append({**row, "file_path": video.relative_to(root).as_posix()})
    if not rows:
        raise ValueError("No candidate batches")
    unique_rows(rows)
    write_rows(output, rows)
    return {"clips": len(rows), "output": str(output)}


def assign_reviewers(manifest, output, reviewers):
    reviewers = [x.strip() for x in reviewers.split(",") if x.strip()]
    if not reviewers or len(reviewers) != len(set(reviewers)):
        raise ValueError("Unique reviewer names required")
    if any(not x.replace("_", "").replace("-", "").isalnum() for x in reviewers):
        raise ValueError("Invalid reviewer name")
    rows = read_rows(manifest)
    unique_rows(rows)
    if len(rows) < len(reviewers):
        raise ValueError("Need at least one clip per reviewer")
    for i, name in enumerate(reviewers):
        write_rows(Path(output) / f"review_{name}.csv", rows[i :: len(reviewers)])
    return {"clips": len(rows), "reviewers": reviewers}


def merge_reviews(manifest, review_dir, output):
    expected = unique_rows(read_rows(manifest))
    combined = []
    for path in sorted(Path(review_dir).glob("review_*.csv")):
        combined.extend(read_rows(path))
    actual = unique_rows(combined)
    if set(actual) != set(expected):
        raise ValueError("Missing/extra reviews; every assigned clip needs one decision")
    for sid, row in actual.items():
        if row.get("decision") not in {"keep", "reject", "uncertain"}:
            raise ValueError(f"Invalid decision: {sid}")
        for key in ("sha256", "source_id", "source_start_s", "source_end_s", "file_path"):
            if row.get(key) != expected[sid].get(key):
                raise ValueError(f"Review provenance changed: {sid}/{key}")
    write_rows(output, list(actual.values()))
    return {"clips": len(actual), "output": str(output)}
