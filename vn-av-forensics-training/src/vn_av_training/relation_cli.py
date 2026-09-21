"""Commands for the FATE relation pipeline; no implicit model downloads."""

import argparse
import json

from vn_av_training.common.runtime import load_config

COMMANDS = (
    "validate",
    "import",
    "audit",
    "checkpoint-info",
    "doctor",
    "prepare",
    "train",
    "evaluate",
    "analyze",
    "serve",
    "setup",
    "run",
)


def main(argv):
    parser = argparse.ArgumentParser(prog="vn-av-train")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = sub.add_parser(name, aliases=["relations-" + name])
        name = "relations-" + name
        command.add_argument("--config", default="configs/relations.yaml")
        command.add_argument("--profile")
        if name == "relations-setup":
            command.add_argument(
                "--only", choices=("all", "media", "source", "fate"), default="all"
            )
        elif name == "relations-run":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--skip-evaluation", action="store_true")
        if name in ("relations-import", "relations-validate"):
            command.add_argument("--dataset")
            command.add_argument("--dataset-manifest")
            if name == "relations-import":
                command.add_argument("--output")
        elif name == "relations-train":
            command.add_argument("--resume", action="store_true")
        elif name == "relations-doctor":
            command.add_argument("--stage", choices=("features", "inference"), default="features")
            command.add_argument("--load", action="store_true")
        elif name == "relations-analyze":
            command.add_argument("--video", required=True)
            command.add_argument("--output", required=True)
        elif name == "relations-evaluate":
            command.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args(argv)
    args.command = "relations-" + args.command.removeprefix("relations-")
    cfg = load_config(args.config, args.profile)
    if args.command in ("relations-prepare", "relations-run"):
        from vn_av_training.data.generated import supervision_report

        supervision_report(cfg, enforce=True)
    result, status = None, 0
    if args.command == "relations-import":
        from vn_av_training.data.generated import import_generated

        result = import_generated(
            args.dataset or cfg["dataset"],
            args.output or cfg["manifest"],
            args.dataset_manifest or cfg.get("dataset_manifest", "manifest.jsonl"),
            cfg,
        )
    elif args.command == "relations-validate":
        from vn_av_training.data.generated import validate_generated

        _, result = validate_generated(
            args.dataset or cfg["dataset"],
            args.dataset_manifest or cfg.get("dataset_manifest", "manifest.jsonl"),
            cfg,
        )
    elif args.command == "relations-audit":
        from vn_av_training.data.generated import supervision_report

        result = supervision_report(cfg)
        status = 1 if result["missing_required_heads"] else 0
    elif args.command == "relations-checkpoint-info":
        from vn_av_training.training.quality import head_status
        from vn_av_training.training.relations import load_relations

        _, state = load_relations(cfg["checkpoint"])
        result = {
            key: state.get(key)
            for key in (
                "format",
                "phase",
                "epoch",
                "timing_warmup_epochs",
                "active_heads",
                "trained_heads",
                "deployable_heads",
                "thresholds",
                "validation_report",
                "label_coverage",
            )
        }
        result["head_status"] = head_status(state)
    elif args.command == "relations-setup":
        from vn_av_training.assets import setup_assets

        result = setup_assets(cfg, args.only)
    elif args.command == "relations-run":
        from vn_av_training.pipeline import train_pipeline

        result = train_pipeline(cfg, args.resume, not args.skip_evaluation)
    elif args.command == "relations-prepare":
        from vn_av_training.features.fate import prepare_relations

        result = prepare_relations(cfg)
    elif args.command == "relations-train":
        from vn_av_training.training.relations import fit_relations

        result = fit_relations(cfg, args.resume)
    elif args.command == "relations-evaluate":
        from vn_av_training.evaluation.relations import evaluate_relations

        result = evaluate_relations(cfg, args.split)
    elif args.command == "relations-analyze":
        from vn_av_training.serving.relations import RelationAnalyzer

        result = RelationAnalyzer(cfg).analyze(
            args.video, args.output, lambda *_args: print(*_args, flush=True)
        )
    elif args.command == "relations-doctor":
        from vn_av_training.serving.relations import relation_health

        result = relation_health(cfg)
        ready = result["preprocessing_ready"] if args.stage == "features" else result["ready"]
        if args.load and ready:
            from vn_av_training.features.fate import FATEVideoEncoder

            encoder = FATEVideoEncoder(cfg["encoder"])
            result["backbone_loaded"] = True
            result["feature_signature"] = encoder.signature
            if args.stage == "inference":
                from vn_av_training.serving.relations import RelationAnalyzer

                RelationAnalyzer(cfg, encoder=encoder)
                result["checkpoint_loaded"] = True
        status = 0 if ready else 1
    elif args.command == "relations-serve":
        import uvicorn

        from vn_av_training.serving.api import create_app

        cfg["pipeline"] = "relations"
        uvicorn.run(create_app(cfg), host=cfg.get("host", "127.0.0.1"), port=cfg.get("port", 8000))
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return status
