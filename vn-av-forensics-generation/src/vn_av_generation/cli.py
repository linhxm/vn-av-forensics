import argparse
import json
import sys
from pathlib import Path

from vn_av_generation.common.runtime import load_config
from vn_av_generation.generate import finalize_labels, migrate_labels, plan_dataset, render_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vn-av-generate")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "render", "review", "finalize", "inspect", "migrate"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        if name == "plan":
            p.add_argument("--config", default="configs/generation.yaml")
            p.add_argument("--output", required=True)
        elif name == "render":
            p.add_argument("--plan", required=True)
            p.add_argument("--output", required=True)
        elif name == "review":
            p.add_argument("--port", type=int, default=8002)
        elif name == "migrate":
            p.add_argument("--manifest", default="manifest-reviewed.jsonl")
            p.add_argument("--output", required=True)
        elif name == "finalize":
            p.add_argument("--review", required=True)
            p.add_argument("--output", required=True)
        else:
            p.add_argument("--manifest", default="manifest.jsonl")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan_dataset(args.dataset, args.output, load_config(args.config))
    elif args.command == "render":
        result = render_dataset(args.dataset, args.plan, args.output)
    elif args.command == "migrate":
        result = migrate_labels(args.dataset, args.manifest, args.output)
    elif args.command == "finalize":
        result = finalize_labels(args.dataset, args.review, args.output)
    elif args.command == "review":
        import uvicorn

        from vn_av_generation.review import create_app

        uvicorn.run(create_app(args.dataset), host="127.0.0.1", port=args.port)
        return 0
    else:
        from collections import Counter

        from vn_av_generation.data.manifest import read_manifest

        rows = read_manifest(Path(args.dataset) / args.manifest)
        result = {
            split: {
                head: dict(
                    Counter(
                        "positive"
                        if r.get("relation_annotations", {}).get(head, {}).get("positive")
                        else "negative"
                        for r in rows
                        if r["split"] == split
                        and r.get("relation_annotations", {}).get(head, {}).get("known")
                    )
                )
                for head in ("lip_audio_mismatch",)
            }
            for split in ("train", "validation", "test")
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def entrypoint():
    try:
        return main()
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
