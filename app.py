"""Local audio-mouth inconsistency demo."""

import json
import subprocess
import uuid
from pathlib import Path

import streamlit as st

from avtc.inference import analyze

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="VN-AV Forensics", layout="wide")
st.title("Phát hiện bất nhất cục bộ giữa lời nói và chuyển động môi")
st.caption("Đánh dấu đoạn nghi vấn theo thời gian; chưa kết luận video thật hay deepfake.")
checkpoint = ROOT / "checkpoints/local.pt"
if not checkpoint.exists():
    st.info("Chạy notebook train.ipynb trên Kaggle, tải training/local.pt về checkpoints/local.pt trước.")
video = st.file_uploader(
    "Video một người nói, mặt và miệng rõ", type=["mp4", "avi", "mov", "mkv", "mpg", "webm"]
)
if video:
    st.video(video)
if st.button("Phân tích", disabled=video is None or not checkpoint.exists(), type="primary"):
    st.session_state.pop("analysis", None)
    out = ROOT / "outputs" / uuid.uuid4().hex
    out.mkdir(parents=True)
    path = out / ("input" + Path(video.name).suffix.lower())
    path.write_bytes(video.getvalue())
    try:
        with st.spinner("Đang phân tích; chạy CPU có thể mất vài phút…"):
            result, table = analyze(path, checkpoint, out, ROOT / "checkpoints/syncnet")
        st.session_state["analysis"] = (result, table, video.getvalue())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        st.error(f"Không phân tích được: {exc}")
if "analysis" in st.session_state:
    result, table, media = st.session_state["analysis"]
    labels = {
        "insufficient_evidence": "Không đủ dữ liệu để đánh giá",
        "inconsistency_detected": "Có đoạn bất nhất theo ngưỡng của model",
        "no_clear_inconsistency": "Chưa thấy bất nhất rõ theo ngưỡng của model",
    }
    st.subheader(labels[result["status"]])
    st.caption(
        "Điểm model chưa phải xác suất đã hiệu chuẩn. Cần đánh giá riêng khi dùng tiếng Việt."
    )
    chart = table.set_index("time_s")[["score"]].copy()
    chart["score"] = chart["score"].where(table.set_index("time_s").valid)
    chart["threshold"] = result["threshold"]
    st.line_chart(chart, x_label="Thời gian (giây)")
    st.write(f"Số cửa sổ đủ điều kiện: {result['valid_windows']}/{result['total_windows']}")
    for start, end in result["intervals"]:
        st.write(f"Đoạn nghi vấn: {start:.2f}–{end:.2f} giây")
    if result["intervals"]:
        i = st.selectbox(
            "Phát lại", range(len(result["intervals"])), format_func=lambda n: f"Đoạn {n + 1}"
        )
        st.video(media, start_time=max(0, int(result["intervals"][i][0]) - 1))
    st.download_button("Tải timeline", table.to_csv(index=False), "timeline.csv", "text/csv")
    st.download_button(
        "Tải kết quả",
        json.dumps(result, ensure_ascii=False, indent=2),
        "result.json",
        "application/json",
    )
