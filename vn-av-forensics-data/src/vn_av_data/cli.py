"""Independent data preparation CLI. No Torch or FATE imports."""

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vn-av-data")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "setup",
        "collect",
        "merge",
        "assign",
        "merge-reviews",
        "download",
        "index",
        "cut",
        "review",
        "export",
        "import-legacy-review",
        "consolidate-version",
        "assign-speaker",
        "validate",
    ):
        p = sub.add_parser(name)
        if name in ("setup", "cut"):
            p.add_argument("--config", default="configs/data.yaml")
        if name == "collect":
            p.add_argument("--input", required=True)
            p.add_argument("--output", required=True)
        elif name == "merge":
            p.add_argument("--root", required=True)
            p.add_argument("--output", required=True)
        elif name == "assign":
            p.add_argument("--manifest", required=True)
            p.add_argument("--output", required=True)
            p.add_argument("--reviewers", required=True)
        elif name == "merge-reviews":
            p.add_argument("--manifest", required=True)
            p.add_argument("--reviews", required=True)
            p.add_argument("--output", required=True)
        elif name == "download":
            p.add_argument("--sources", required=True)
            p.add_argument("--output", default="data/raw")
            p.add_argument(
                "--cookies-from-browser",
                choices=("firefox", "chrome", "edge", "brave"),
                help="Use a browser session when YouTube rejects anonymous media URLs",
            )
            p.add_argument("--force-ipv4", action="store_true")
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--dry-run", action="store_true")
        elif name == "index":
            p.add_argument("--root", required=True)
            p.add_argument("--output", default="data/raw/sources.jsonl")
        elif name == "cut":
            p.add_argument("--manifest", default="data/raw/sources.jsonl")
            p.add_argument("--output", default="data/candidates/v001")
        elif name in ("review", "export"):
            p.add_argument("--review", required=True)
            p.add_argument("--root", required=True)
            if name == "review":
                p.add_argument("--port", type=int, default=8001)
            else:
                p.add_argument("--output", required=True)
                p.add_argument("--dataset-id", required=True)
                p.add_argument("--annotations", help="JSONL: clip_id and relation_annotations")
        elif name == "import-legacy-review":
            p.add_argument("--review", required=True, help="Legacy review CSV")
            p.add_argument("--manifest", required=True, help="Legacy clip/assignment CSV")
            p.add_argument("--root", required=True, help="Folder containing legacy clip files")
            p.add_argument("--output", required=True)
            p.add_argument("--dataset-id", required=True)
        elif name == "consolidate-version":
            p.add_argument("--candidate", required=True, help="Native reviewed cut directory")
            p.add_argument("--raw", required=True, help="Native raw download directory")
            p.add_argument("--imported", required=True, help="Imported reviewed bundle")
            p.add_argument("--data-root", default="data")
            p.add_argument("--output", required=True, help="New merged export path")
            p.add_argument("--dataset-id", required=True)
        elif name == "assign-speaker":
            p.add_argument("--data-root", default="data")
            p.add_argument("--export", required=True)
            p.add_argument("--dataset-id", required=True)
            p.add_argument("--speaker-id", required=True)
            p.add_argument("--availability", default="imported_clips_only")
        elif name == "validate":
            p.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "config"):
        from vn_av_data.common.runtime import load_config

        cfg = load_config(args.config)
    if args.command == "collect":
        from vn_av_data.data.collect import collect_sources
        from vn_av_data.data.source_io import read_rows, write_rows

        rows = collect_sources(read_rows(args.input))
        write_rows(args.output, rows)
        result = {"sources": len(rows), "output": args.output}
    elif args.command == "merge":
        from vn_av_data.data.workflow import merge_candidates

        result = merge_candidates(args.root, args.output)
    elif args.command == "assign":
        from vn_av_data.data.workflow import assign_reviewers

        result = assign_reviewers(args.manifest, args.output, args.reviewers)
    elif args.command == "merge-reviews":
        from vn_av_data.data.workflow import merge_reviews

        result = merge_reviews(args.manifest, args.reviews, args.output)
    elif args.command == "setup":
        from vn_av_data.assets import setup_assets

        result = setup_assets(cfg)
    elif args.command == "download":
        from vn_av_data.data.acquisition import download_sources

        result = download_sources(
            args.sources,
            args.output,
            args.cookies_from_browser,
            args.force_ipv4,
            args.limit,
            args.dry_run,
        )
    elif args.command == "index":
        from vn_av_data.data.acquisition import index_sources

        result = index_sources(args.root, args.output)
    elif args.command == "cut":
        from vn_av_data.data.curation import curate_sources

        result = curate_sources(args.manifest, args.output, cfg)
    elif args.command == "review":
        import uvicorn

        from vn_av_data.serving.review import create_review_app

        uvicorn.run(create_review_app(args.review, args.root), host="127.0.0.1", port=args.port)
        return 0
    elif args.command == "export":
        from vn_av_data.data.export import export_dataset

        result = export_dataset(
            args.review, args.root, args.output, args.dataset_id, args.annotations
        )
    elif args.command == "import-legacy-review":
        from vn_av_data.data.legacy import import_legacy_review

        result = import_legacy_review(
            args.review, args.manifest, args.root, args.output, args.dataset_id
        )
    elif args.command == "consolidate-version":
        from vn_av_data.data.consolidate import consolidate_reviewed_version

        result = consolidate_reviewed_version(
            args.candidate,
            args.raw,
            args.imported,
            args.data_root,
            args.output,
            args.dataset_id,
        )
    elif args.command == "assign-speaker":
        from vn_av_data.data.speakers import assign_speaker

        result = assign_speaker(
            args.data_root,
            args.export,
            args.dataset_id,
            args.speaker_id,
            args.availability,
        )
    else:
        from vn_av_data.contract import validate_bundle

        _, result = validate_bundle(args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def entrypoint():
    try:
        return main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
