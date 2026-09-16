"""One-command cached extraction, training/resume and held-out evaluation."""

from pathlib import Path

from vn_av_training.common.runtime import write_json


def train_pipeline(cfg, resume=False, evaluate=True):
    from vn_av_training.evaluation.relations import evaluate_relations
    from vn_av_training.features.fate import prepare_relations
    from vn_av_training.training.relations import fit_relations

    result = {"features": prepare_relations(cfg)}
    result["training"] = fit_relations(cfg, resume)
    evaluation_cfg = {**cfg, "checkpoint": result["training"]["checkpoint"]}
    if evaluate:
        result["evaluation"] = evaluate_relations(evaluation_cfg, "test")
    write_json(Path(cfg["output"]) / "pipeline-result.json", result)
    return result
