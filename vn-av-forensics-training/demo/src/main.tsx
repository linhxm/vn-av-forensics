import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { createJob, getHealth, getJob, type Head, type Job } from "./api";
import { Timeline } from "./Timeline";
import { RelationResultView } from "./RelationResultView";
import { formatTime } from "./research";
import "./pipeline.css";

function App() {
  const [job, setJob] = useState<Job | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [health, setHealth] = useState<Awaited<ReturnType<typeof getHealth>> | null>(null);
  const [head, setHead] = useState<Head>("forgery");
  const [current, setCurrent] = useState(0);
  const video = useRef<HTMLVideoElement>(null);
  const busy = uploading || job?.status === "queued" || job?.status === "processing";
  useEffect(() => { getHealth().then(setHealth).catch(e => setError(String(e))); }, []);
  useEffect(() => {
    if (!job || !["queued", "processing"].includes(job.status)) return;
    let alive = true;
    const timer = window.setTimeout(async () => {
      try { const next = await getJob(job.id); if (alive) setJob(next); }
      catch (e) { if (alive) { setError(String(e)); setJob({ ...job, status: "interrupted" }); } }
    }, 1200);
    return () => { alive = false; window.clearTimeout(timer); };
  }, [job]);
  const relation = job?.result && "window_scores" in job.result ? job.result : null;
  const result = job?.result && !("window_scores" in job.result) ? job.result : null;
  useEffect(() => { if (result) setHead(result.head_availability.forgery ? "forgery" : "mismatch"); }, [result]);
  async function submit() {
    if (!file) return;
    setUploading(true); setError(""); setJob(null); setCurrent(0);
    try { setJob(await createJob(file)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setUploading(false); }
  }
  function seek(time: number) { if (video.current) video.current.currentTime = time; setCurrent(time); }
  const segments = result?.segments[head] ?? [];
  const temporal = Boolean(result?.head_availability[head]);
  const points = result?.timeline.map(p => ({ time_s:p.time_s, valid:p.valid && p[head] !== null, score:p[head] ?? 0 })) ?? [];
  return <main>
    <header><a className="brand" href="/">VN / AV <span>FORENSICS</span></a><span className="tag">Research workspace</span></header>
    <section className="intro"><p className="eyebrow">AUDIO–VISUAL SPEECH CONSISTENCY</p>
      <h1>Kiểm tra tiếng nói.<br />Đối chiếu chuyển động miệng.</h1>
      <p>Tải video một người nói, có âm thanh và thấy rõ miệng. Xem các khoảng khả nghi và điểm bất nhất giữa tiếng nói với chuyển động miệng.</p>
    </section>
    {health && !health.ready && <aside className="setup">Chưa sẵn sàng phân tích: {health.missing.join(", ")}. Cần hoàn tất thiết lập và nạp model đã huấn luyện.</aside>}
    <section className="upload"><label htmlFor="video-input">Chọn video</label>
      <input id="video-input" type="file" accept=".mp4,.webm,.mov,.mkv,.avi,.mpg" disabled={busy} onChange={e => setFile(e.target.files?.[0] ?? null)} />
      <button disabled={!file || busy || health?.ready === false} onClick={submit}>{uploading ? "Đang tải…" : busy ? "Đang phân tích…" : "Phân tích video"}</button>
      <small>Một người · Thấy rõ miệng · Có âm thanh · MP4 được khuyến nghị để phát trực tiếp trên trình duyệt</small>
    </section>
    {error && <p role="alert" className="error">{error}</p>}
    {job && <section className="job-status" aria-live="polite"><span className={busy ? "pulse" : "status-dot"} />{job.message}
      {(job.status === "failed" || job.status === "interrupted") && <p role="alert">{job.error ?? "Có thể tải lại video để thử lại."}</p>}
    </section>}
    {relation && job && <RelationResultView key={job.id} result={relation} jobId={job.id} filename={job.filename} />}
    {result && job && <>
      <section className="result-grid"><div className="player"><video ref={video} src={`/api/jobs/${job.id}/media`} controls onTimeUpdate={e => setCurrent(e.currentTarget.currentTime)} /><p>{job.filename}</p></div>
        <aside className="summary"><p className="eyebrow">DỰ ĐOÁN CỦA MODEL</p><h2>{result.prediction === "fake" ? "Có dấu hiệu giả mạo" : result.prediction === "real" ? "Nghiêng về video thật" : "Chưa đủ dữ liệu"}</h2>
          <div className="score">{result.clip_forgery_score === null ? "—" : result.clip_forgery_score.toFixed(3)}<span>điểm giả mạo</span></div>
          <dl><div><dt>Ngưỡng phân loại</dt><dd>{result.clip_threshold.toFixed(3)}</dd></div><div><dt>Dữ liệu hợp lệ</dt><dd>{(result.coverage*100).toFixed(1)}%</dd></div><div><dt>Thời lượng</dt><dd>{formatTime(result.duration_s)}</dd></div></dl>
          <p className="muted">{result.notice}</p><a className="download" href={`/api/jobs/${job.id}/report`} download>Tải báo cáo JSON</a>
        </aside>
      </section>
      {(result.head_availability.forgery || result.head_availability.mismatch) ? <section className="timeline-section">
        <div className="section-title"><h2>Kiểm tra theo thời gian</h2><div className="tabs">{(["forgery","mismatch"] as Head[]).filter(h => result.head_availability[h]).map(h => <button key={h} className={head===h?"active":""} onClick={() => setHead(h)}>{h === "forgery" ? "Giả mạo" : "Bất nhất tiếng–miệng"}</button>)}</div></div>
        {temporal && <><Timeline points={points} segments={segments} threshold={result.thresholds[head] ?? .5} current={current} duration={result.duration_s} onSeek={seek} />
          <p className="muted">{head === "mismatch" ? "Lệch tiếng–miệng có thể do xử lý kỹ thuật; đây không phải kết luận giả mạo." : "Các đoạn vượt ngưỡng của nhánh định vị giả mạo."}</p>
          <div className="segments">{segments.length ? segments.map((s,i) => <div className="segment" key={i}><button onClick={() => seek(s.start_s)}>{formatTime(s.start_s)} → {formatTime(s.end_s)}</button><span>Điểm {s.score.toFixed(3)}</span><a href={`/api/jobs/${job.id}/clips/${head}/${i}?context=0.75`} download>Tải đoạn</a></div>) : <p>Không có đoạn vượt ngưỡng.</p>}</div>
          <a href={`/api/jobs/${job.id}/timeline.csv`} download>Tải điểm theo thời gian (CSV)</a></>}
      </section> : <section className="timeline-section"><h2>Kết quả toàn clip</h2><p>Model này phân loại toàn clip. Chưa có đầu ra định vị theo thời gian.</p></section>}
    </>}
    <footer>VN-AV Forensics · Audio / Video / Temporal consistency</footer>
  </main>;
}

createRoot(document.getElementById("root")!).render(<App />);
