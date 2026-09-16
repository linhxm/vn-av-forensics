import { useRef, useState } from "react";
import type { RelationResult } from "./api";
import { Timeline } from "./Timeline";
import { formatTime } from "./research";

const labels: Record<string, string> = {
  timing: "Lệch thời gian", sequence: "Bất nhất chuỗi",
  phoneme_viseme: "Phát âm–khẩu hình", motion_speech: "Hoạt động miệng–lời nói",
  source: "Tương ứng người nói",
};

export function RelationResultView({ result, jobId, filename }: {
  result: RelationResult; jobId: string; filename: string;
}) {
  const available = Object.keys(result.thresholds);
  const [selected, setSelected] = useState(available[0] ?? "timing");
  const [current, setCurrent] = useState(0);
  const video = useRef<HTMLVideoElement>(null);
  const intervals = result.suspicious_intervals.filter(span => span.relation === selected);
  const segments = intervals.map(span => ({ start_s: span.start, end_s: span.end, score: span.inconsistency_score }));
  const points = result.window_scores.map(window => ({ time_s: window.start,
    valid: window.relation_scores[selected] != null, score: window.relation_scores[selected] ?? 0 }));
  const missing = Object.entries(result.head_availability).filter(([, value]) => value === "untrained");
  function seek(time: number) { if (video.current) video.current.currentTime = time; setCurrent(time); }
  return <>
    <section className="result-grid">
      <div className="player"><video ref={video} src={`/api/jobs/${jobId}/media`} controls
        onTimeUpdate={event => setCurrent(event.currentTarget.currentTime)} /><p>{filename}</p></div>
      <aside className="summary"><p className="eyebrow">QUAN HỆ ÂM THANH–CHUYỂN ĐỘNG MIỆNG</p>
        <h2>{result.clip_inconsistency_score === null ? "Chưa đủ bằng chứng"
          : result.suspicious_intervals.length ? "Có khoảng cần kiểm tra" : "Chưa thấy khoảng vượt ngưỡng"}</h2>
        <div className="score">{result.clip_inconsistency_score?.toFixed(3) ?? "—"}<span>điểm bất nhất</span></div>
        <dl><div><dt>Độ lệch toàn clip</dt><dd>{result.global_lag_ms === null ? "Chưa xác định" : `${result.global_lag_ms.toFixed(0)} ms`}</dd></div>
          <div><dt>Thời lượng</dt><dd>{formatTime(result.duration_s)}</dd></div></dl>
        <p className="muted">{result.notice}</p>
        {missing.length > 0 && <p>Chưa đánh giá: {missing.map(([name]) => labels[name] ?? name).join(", ")}.</p>}
        <p>Danh tính giọng nói: chưa xác minh.</p>
        <a className="download" href={`/api/jobs/${jobId}/report`} download>Tải báo cáo JSON</a>
      </aside>
    </section>
    {available.length > 0 && <section className="timeline-section">
      <div className="section-title"><h2>Kiểm tra theo thời gian</h2><div className="tabs">
        {available.map(name => <button key={name} className={selected === name ? "active" : ""}
          onClick={() => setSelected(name)}>{labels[name] ?? name}</button>)}
      </div></div>
      <Timeline points={points} segments={segments} threshold={result.thresholds[selected] ?? 0.5}
        current={current} duration={result.duration_s} onSeek={seek} />
      <p className="muted">Phần có đủ bằng chứng: {((result.coverage[selected] ?? 0) * 100).toFixed(1)}%.
        Khoảng trống không được xem là khớp. Lag dương nghĩa âm thanh đi trước.</p>
      <div className="segments">{intervals.length ? intervals.map((span, index) => <div className="segment" key={index}>
        <button onClick={() => seek(span.start)}>{formatTime(span.start)} → {formatTime(span.end)}</button>
        <span>Điểm {span.inconsistency_score.toFixed(3)}</span>
        <a href={`/api/jobs/${jobId}/clips/${selected}/${index}`} download>Tải đoạn</a>
      </div>) : <p>Không có đoạn vượt ngưỡng của nhánh này.</p>}</div>
      <a href={`/api/jobs/${jobId}/timeline.csv`} download>Tải điểm theo thời gian (CSV)</a>
    </section>}
  </>;
}
