import numpy as np

from vn_av_training.data.labels import labels_for
from vn_av_training.training.quality import calibrate, head_status


def test_legacy_union_keeps_partial_negatives_unknown():
    from vn_av_training.contract import mismatch_annotation

    assert mismatch_annotation({"sequence": {"known": [[0, 6]], "positive": []}}, 6) == {"known": [], "positive": []}
    labels = {name: {"known": [[0, 6]], "positive": []} for name in ("sequence", "phoneme_viseme", "motion_speech")}
    labels["sequence"]["positive"] = [[2, 3]]
    assert mismatch_annotation(labels, 6) == {"known": [[0, 6]], "positive": [[2, 3]]}
    assert mismatch_annotation({"source": {"known": [[0, 6]], "positive": [[0, 6]]}}, 6)["positive"] == []


def test_uncertain_lag_requires_validated_unaligned_mismatch():
    import torch
    from vn_av_training.training.quality import assessment_masks

    output = {"valid": torch.tensor([[True, True, False]]), "lag_valid": torch.tensor([[True, False, False]])}
    assert assessment_masks(output, {})["mismatch_valid"].tolist() == [[True, False, False]]
    state = {"validation_report": {"lip_audio_mismatch": {"unaligned_ready": True}}}
    assert assessment_masks(output, state)["mismatch_valid"].tolist() == [[True, True, False]]


def test_shift_only_is_negative_for_residual_mismatch_and_boundaries_unknown():
    row = {"duration_s": 8, "variant": {"kind": "clean"}, "materialized": True,
           "generation": {"edit": {"kind": "local_lag"}},
           "supervision": {"timing": [{"start": 0, "end": 2, "lag_s": 0}, {"start": 2, "end": 6, "lag_s": .4}, {"start": 6, "end": 8, "lag_s": 0}]}}
    targets = labels_for(row, np.arange(40)*.2, 2, .2, 4)
    assert targets["lag_class"][16] == 6
    assert targets["lip_audio_mismatch"][16] == 0
    assert targets["lag_class"][9] == -1
    assert targets["lip_audio_mismatch"][9] == -1


def test_constant_tiny_scores_are_not_a_usable_detector():
    result = calibrate([0] * 100 + [1] * 20, [0.009787] * 120)
    assert result["status"] == "quality_failed"
    assert result["false_alarm_rate"] == 0
    assert result["recall"] == 0


def test_good_separation_passes_only_validation_policy():
    result = calibrate([0] * 25 + [1] * 25, [0.1] * 25 + [0.9] * 25)
    assert result["status"] == "ready"
    assert result["false_alarm_rate"] == 0 and result["recall"] == 1
    assert (
        head_status(
            {"active_heads": ["lip_audio_mismatch"], "trained_heads": ["lip_audio_mismatch"]}
        )["lip_audio_mismatch"]
        == "uncalibrated"
    )


def test_materialized_media_does_not_become_clean_negative_or_receive_double_shift():
    row = {
        "duration_s": 6,
        "variant": {"kind": "clean"},
        "materialized": True,
        "supervision": {"timing": [{"start": 1, "end": 5, "lag_s": 0.4}]},
        "relation_annotations": {},
    }
    y = labels_for(row, np.arange(30) * 0.2, 2, 0.2, 4)
    assert set(y["lag_class"]) == {-1, 6}
    assert (y["lip_audio_mismatch"] == -1).all()
