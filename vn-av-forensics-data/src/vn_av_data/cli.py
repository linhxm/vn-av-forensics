"""Independent data preparation CLI. No Torch or FATE imports."""

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vn-av-data")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "download", "index", "cut", "review", "export", "validate"):
        p = sub.add_parser(name)
        if name in ("setup", "cut"):
            p.add_argument("--config", default="configs/data.yaml")
        if name == "download":
            p.add_argument("--sources", required=True)
            p.add_argument("--output", default="data/raw")
            p.add_argument(
                "--cookies-from-browser",
                choices=("firefox", "chrome", "edge", "brave"),
                help="Use a browser session when YouTube rejects anonymous media URLs",
            )
            p.add_argument("--force-ipv4", action="store_true")
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
        elif name == "validate":
            p.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "config"):
        from vn_av_data.common.runtime import load_config

        cfg = load_config(args.config)
    if args.command == "setup":
        from vn_av_data.assets import setup_assets

        result = setup_assets(cfg)
    elif args.command == "download":
        from vn_av_data.data.acquisition import download_sources

        result = download_sources(
            args.sources, args.output, args.cookies_from_browser, args.force_ipv4
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
