import { useRef, useState } from "react";
import type { Head, RelationResult } from "./api";
import { Timeline } from "./Timeline";
import { formatTime } from "./research";

const labels: Record<Head, string> = {
  timing: "Lệch thời gian",
  lip_audio_mismatch: "Bất nhất môi–âm thanh",
};
const states: Record<string, string> = {
  ready: "Đạt kiểm tra validation",
  quality_failed: "Chưa đạt chất lượng",
  uncalibrated: "Chưa đủ calibration",
  untrained: "Chưa học",
};
const number = (v: number | null | undefined, digits = 3) => v == null ? "—" : v.toFixed(digits);

export function RelationResultView({ result, jobId, filename }: {
  result: RelationResult; jobId: string; filename: string;
}) {
  const [selected, setSelected] = useState<Head>("timing");
  const [diagnostic, setDiagnostic] = useState(false);
  const [current, setCurrent] = useState(0);
  const video = useRef<HTMLVideoElement>(null);
  const intervals = (diagnostic ? result.diagnostic_intervals ?? [] : result.suspicious_intervals)
    .filter(s => s.relation === selected);
  const scores = (w: RelationResult["window_scores"][number]) =>
    diagnostic ? w.relation_scores : w.accepted_relation_scores;
  const points = result.window_scores.map(w => ({
    time_s: w.start, valid: scores(w)[selected] != null, score: scores(w)[selected] ?? 0,
  }));
  const window = result.window_scores.find(w => w.start <= current && current < w.end);
  const prep = result.preprocessing;
  const features = result.feature_evidence;
  function seek(time: number) {
    if (video.current) video.current.currentTime = time;
    setCurrent(time);
  }

  return <>
    <section>
      <h2>3. Đặc trưng dùng chung</h2>
      <p>{filename} · {formatTime(result.duration_s)} · Cửa sổ {result.context_window_s}s · Bước {result.step_s}s</p>
      <p>FATE trích đặc trưng một lần. Hai head dùng chung đặc trưng; căn chỉnh chỉ thay cách ghép audio–hình, không sửa video.</p>
      {features && <p>Audio [{features.audio_shape.join(", ")}], hình [{features.visual_shape.join(", ")}].
        Hợp lệ: audio {features.audio_valid_windows}/{features.window_count}, hình {features.visual_valid_windows}/{features.window_count} cửa sổ.</p>}
      {prep && <>
        <p>Audio {prep.sample_rate_hz} Hz · Lưới hình {prep.video_grid_fps} fps · {prep.decoded_frames} frame.</p>
        <div className="frames">{prep.frames.map((f, i) => <figure key={i}>
          <img src={`/api/jobs/${jobId}/${f.image}`} alt={`Khung hình ${number(f.time_s)} giây`} />
          <figcaption><button onClick={() => seek(f.time_s)}>{number(f.time_s)}s</button>
            {" · "}{f.face_check?.valid ? "Mặt quan sát được" : f.face_check?.reason ?? "Không đo hình học"}</figcaption>
        </figure>)}</div>
        <details><summary>Năng lượng audio RMS — không phải nhãn đang nói</summary>
          <div className="scroll"><table><thead><tr><th>Thời gian</th><th>RMS</th></tr></thead>
            <tbody>{prep.audio_rms.map((r, i) => <tr key={i}><td>{number(r.start)}</td><td>{number(r.rms, 6)}</td></tr>)}</tbody>
          </table></div>
        </details>
      </>}
    </section>
    <section>
      <h2>4. Chất lượng hai nhánh</h2>
      <p className="warning">{result.status === "diagnostic_only"
        ? "Checkpoint chưa có nhánh đủ điều kiện kết luận. Có thể bật điểm chẩn đoán để kiểm tra."
        : result.notice}</p>
      <div className="scroll"><table>
        <thead><tr><th>Nhánh</th><th>Trạng thái</th><th>Ngưỡng</th><th>Vùng đủ điều kiện</th>
          <th>Validation: precision / recall / FAR</th><th>Chi tiết</th></tr></thead>
        <tbody>{(Object.keys(labels) as Head[]).map(name => {
          const q = result.checkpoint?.validation_report[name];
          const test = result.checkpoint?.test_report?.relations?.[name];
          return <tr key={name}>
            <td>{labels[name]}</td><td>{states[result.head_availability[name]] ?? result.head_availability[name]}</td>
            <td>{number(result.thresholds[name], 6)}</td><td>{number((result.accepted_coverage[name] ?? 0) * 100, 1)}%</td>
            <td>{number(q?.precision)} / {number(q?.recall)} / {number(q?.false_alarm_rate)}</td>
            <td>{q?.reasons?.join("; ")}
              {name === "timing" && <p>MAE lag: {number(q?.lag_mae_ms, 0)} ms</p>}
              {name === "lip_audio_mismatch" && <p>Khi lag chưa chắc: {q?.unaligned_ready ? "đã đạt kiểm tra riêng" : "chưa đủ bằng chứng để kết luận"}</p>}
              {test && <p>Test FAR {number(test.false_alarm_rate)}; precision {number(test.precision)}</p>}
            </td>
          </tr>;
        })}</tbody>
      </table></div>
      {result.checkpoint?.test_report_provenance && <small>{result.checkpoint.test_report_provenance}</small>}
    </section>
    <section>
      <h2>5. Kết quả và bằng chứng</h2>
      <video ref={video} src={`/api/jobs/${jobId}/media`} controls onTimeUpdate={e => setCurrent(e.currentTarget.currentTime)} />
      <p>Điểm đủ điều kiện: <strong>{number(result.clip_inconsistency_score)}</strong> · Lag toàn clip {number(result.global_lag_ms, 0)} ms.
        Lag dương: audio đi trước hình; dấu — nghĩa chưa xác định.</p>
      <label>Nhánh <select value={selected} onChange={e => setSelected(e.target.value as Head)}>
        {(Object.keys(labels) as Head[]).map(name => <option key={name} value={name}>{labels[name]}</option>)}
      </select></label>
      <label><input type="checkbox" checked={diagnostic} onChange={e => setDiagnostic(e.target.checked)} />
        Xem điểm chẩn đoán chưa đủ điều kiện kết luận</label>
      {diagnostic && <p className="warning">Đang xem điểm thô và khoảng vượt ngưỡng chẩn đoán, không phải kết luận.</p>}
      <Timeline points={points} segments={intervals.map(s => ({ start_s: s.start, end_s: s.end, score: s.inconsistency_score }))}
        threshold={result.thresholds[selected] ?? .5} current={current} duration={result.duration_s} onSeek={seek} />
      <p>Tại {formatTime(current)}: điểm {number(window ? scores(window)[selected] : null, 6)} · lag {number(diagnostic ? window?.diagnostic_estimated_lag_ms : window?.estimated_lag_ms, 0)} ms.</p>
      <p>{window?.alignment_used
        ? `Đã ghép đặc trưng audio tại ${number(window.start)}s với hình tại ${number(window.matched_visual_time_s)}s; confidence ${number(window.lag_confidence)}.`
        : "Chưa căn chỉnh đáng tin ở cửa sổ này. Không tìm được lag không tự chứng minh bất nhất môi–âm thanh."}</p>
      <table><thead><tr><th>Khoảng</th><th>Điểm</th><th>Bằng chứng</th></tr></thead>
        <tbody>{intervals.map((s, i) => <tr key={i}>
          <td><button onClick={() => seek(s.start)}>{formatTime(s.start)} → {formatTime(s.end)}</button></td>
          <td>{number(s.inconsistency_score, 6)}</td>
          <td><a href={`/api/jobs/${jobId}/clips/${selected}/${i}?diagnostic=${diagnostic}`} download>Tải đoạn video gốc</a></td>
        </tr>)}</tbody>
      </table>
      {!intervals.length && <p>Không có khoảng vượt ngưỡng trong chế độ đang xem; không có nghĩa mọi vùng đã được xác nhận khớp.</p>}
      <p><a href={`/api/jobs/${jobId}/report`} download>JSON đầy đủ</a> · <a href={`/api/jobs/${jobId}/timeline.csv`} download>CSV theo cửa sổ</a></p>
      <details><summary>Xem toàn bộ số liệu và bằng chứng</summary><pre>{JSON.stringify(result, null, 2)}</pre></details>
    </section>
  </>;
}
