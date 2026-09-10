"""Evaluate held-out identities and export measured tables, plots and a report."""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import rankdata

from .model import load_model
from .syncnet import intervals, write_json
from .train import collect


def evaluate(cache, checkpoint, output, device="cpu"):
    root, out = Path(cache), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((root / "index.json").read_text())
    model, info = load_model(checkpoint, device)
    from .syncnet import sha

    if info["signature"]["cache_sha256"] != sha(root / "index.json"):
        raise ValueError("Evaluation cache differs from the training split definition")
    rows = [r for r in index["rows"] if r["split"] == "test"]
    data = collect(model, root, rows, device)
    data["predicted"] = (data.score > info["threshold"]) & data.valid
    data.to_csv(out / "timeline.csv", index=False)
    metrics, segments = [], []
    for (source, variant), g in data.groupby(["source_id", "variant"], sort=False):
        x = g[g.valid]
        y = x.label.to_numpy()
        p = x.predicted.to_numpy()
        tp = int((p & y).sum())
        fp = int((p & ~y).sum())
        fn = int((~p & y).sum())
        tn = int((~p & ~y).sum())
        metrics.append(
            {
                "source_id": source,
                "speaker": g.speaker.iloc[0],
                "variant": variant,
                "coverage": len(x) / len(g),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "f1": 2 * tp / (2 * tp + fp + fn) if y.any() else None,
                "iou": tp / (tp + fp + fn) if y.any() else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            }
        )
        segments.append(
            {
                "source_id": source,
                "variant": variant,
                "predicted": intervals(g.time_s.values, g.predicted.values),
                "gold": intervals(g.time_s.values, g.label.values),
            }
        )
    per = pd.DataFrame(metrics)
    per.to_csv(out / "per_source.csv", index=False)
    summary = per.groupby("variant")[
        ["coverage", "f1", "iou", "recall", "false_positive_rate"]
    ].mean()
    summary.to_csv(out / "metrics.csv")
    g = data[data.valid]
    y = g.label.to_numpy()
    n1 = int(y.sum())
    n0 = int((~y).sum())
    auc = float((rankdata(g.score)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 * n0 else None
    write_json(
        out / "metrics.json",
        {
            "window_auc": auc,
            "test_sources": int(per.source_id.nunique()),
            "test_speakers": int(per.speaker.nunique()),
            "threshold": info["threshold"],
            "note": "Correlated windows; pilot metrics, not deepfake accuracy.",
        },
    )
    write_json(out / "segments.json", segments)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for source in data.source_id.unique()[:4]:
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for ax, variant in zip(axes, ["original", "local_+1", "synchronous_+1"]):
            g = data[(data.source_id == source) & (data.variant == variant)]
            ax.plot(g.time_s, g.score.where(g.valid), label=variant)
            ax.axhline(info["threshold"], color="red", linestyle="--")
            for left, right in intervals(g.time_s.values, g.label.values):
                ax.axvspan(left, right, color="orange", alpha=0.25)
            ax.legend()
        axes[-1].set_xlabel("Time in source video (seconds)")
        fig.tight_layout()
        fig.savefig(out / (source + ".png"))
        plt.close(fig)
    lines = [
        "# Kết quả huấn luyện và đánh giá bất nhất cục bộ",
        "",
        "Vietnamese Audio-Visual Deepfake Detection Using Cross-Modal Temporal Consistency",
        "",
        "## Huấn luyện",
        "",
        info["training"],
        "",
        f"- Số epoch đã chạy: {info['epochs_completed']}.",
        f"- Ngưỡng chọn trên validation: {info['threshold']:.5f}.",
        f"- Test: {per.source_id.nunique()} nguồn, {per.speaker.nunique()} người nói.",
        "- Tập test không dùng để chọn model hoặc ngưỡng.",
        "",
        "## Kết quả test",
        "",
        "| Biến thể | Coverage | F1 | IoU | Recall | Tỷ lệ báo nhầm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary.iterrows():
        cells = ["N/A" if pd.isna(value) else f"{value:.4f}" for value in values]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            f"AUC theo cửa sổ: {auc:.4f}." if auc is not None else "AUC: không đủ hai lớp.",
            "",
            ("Các số liệu trong bảng là trung bình theo nguồn cho từng biến thể. "
            "N/A ở F1/IoU/recall của mẫu âm vì không có vùng lỗi dương để đo."),
            "",
            "## Giới hạn",
            "",
            ("Nội suy thời gian có thể đổi pitch ở vùng chuyển tiếp. Đối chiếu tỷ lệ báo nhầm "
            "của original và synchronous-control trước khi kết luận model học quan hệ hai luồng. "
            "Đồng bộ nguồn được chuẩn hóa bằng model, chưa phải nhãn đồng bộ do người xác nhận. "
            "GRID tiếng Anh chưa chứng minh hiệu quả tiếng Việt hay deepfake. "
            "Các cửa sổ tương quan; không dùng chúng như mẫu độc lập để tính khoảng tin cậy."),
            "",
            "## Biểu đồ",
            "",
        ]
    )
    for path in sorted(out.glob("*.png")):
        lines.extend([f"![Timeline {path.stem}]({path.name})", ""])
    lines.extend(
        [
            "## Tài liệu",
            "",
            "- [SyncNet](https://github.com/joonson/syncnet_python)",
            "- [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/)",
            "",
            "## Tác dụng từng file kết quả",
            "",
            "| File | Tác dụng |",
            "|---|---|",
            "| KET_QUA.md | Báo cáo từ số liệu thực của lần chạy. |",
            "| metrics.csv | Chỉ số trung bình theo nguồn cho từng loại biến thể. |",
            "| metrics.json | AUC, số nguồn/người test và ngưỡng. |",
            "| per_source.csv | Chỉ số và TP/FP/FN/TN của từng nguồn, từng biến thể. |",
            "| timeline.csv | Điểm, nhãn, mask hợp lệ và dự đoán tại từng thời điểm. |",
            "| segments.json | Khoảng dự đoán và khoảng nhãn thật trên lưới cửa sổ. |",
            "| <source_id>.png | Biểu đồ original, local và synchronous-control của một nguồn. |",
            "| data/index.json | Danh sách mẫu, split và loại đặc trưng đã dùng. |",
            "| data/prepare_config.json | Seed, phiên bản phương pháp và dấu vân tay manifest. |",
            "| data/excluded.json | Những nguồn bị loại và lý do. |",
            "| data/sources.json | Nhãn đoạn, phép tạo lỗi và chuẩn hóa của từng nguồn. |",
            "",
            "Các file data/ được ô cuối notebook lưu lại sau khi đánh giá.",
        ]
    )
    (out / "KET_QUA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
