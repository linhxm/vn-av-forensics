import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { createJob, getJob, type Job } from "./api";
import { Timeline } from "./Timeline";
import { clipBounds, downloadJson, formatTime, peakScore, pointAt, type Review, type Translate } from "./research";
import "./styles.css";

const ACCEPTED = ".mp4,.avi,.mov,.mkv,.mpg,.webm";
const STAGES = [
  ["Kiểm tra mô hình", "Check model"], ["Theo dõi khuôn mặt", "Track face"],
  ["Đọc hình và âm thanh", "Decode video & audio"], ["Bù độ lệch toàn cục", "Correct global lag"],
  ["Trích xuất đặc trưng", "Extract features"], ["Định vị bất nhất", "Locate inconsistencies"],
  ["Lưu bằng chứng", "Save evidence"],
];

function Icon({ name, size = 20 }: { name: "scan" | "upload" | "play" | "pause" | "check" | "arrow" | "download" | "video" | "zoom"; size?: number }) {
  const paths = {
    scan: "M8 3H3v5M16 3h5v5M3 16v5h5M21 16v5h-5M7 9v6M11 7v10M15 9v6M19 11v2",
    upload: "M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5",
    play: "m8 5 11 7-11 7Z", pause: "M8 5v14M16 5v14",
    check: "m5 12 4 4L19 6", arrow: "M4 12h16m-6-6 6 6-6 6",
    download: "M12 3v13m-5-5 5 5 5-5M4 17v4h16v-4",
    video: "M3 5h13v14H3Zm13 5 5-3v10l-5-3", zoom: "M10 7v6m-3-3h6m2 5 6 6M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0",
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}

function loadReviews(id: string): Record<number, Review> {
  try {
    const parsed = JSON.parse(localStorage.getItem(`av-review:${id}`) ?? "{}");
    if (!parsed || typeof parsed !== "object") return {};
    return Object.fromEntries(Object.entries(parsed).filter((entry): entry is [string, Review] => {
      const value = entry[1];
      return Boolean(value && typeof value === "object" && typeof (value as Review).note === "string" && typeof (value as Review).reviewed === "boolean");
    }));
  } catch { return {}; }
}

function Workspace({ job, filename, t }: { job: Job; filename: string; t: Translate }) {
  const result = job.result!;
  const points = useMemo(() => job.timeline ?? [], [job.timeline]);
  const peaks = useMemo(() => result.intervals.map(range => peakScore(points, range)), [result.intervals, points]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [selected, setSelected] = useState<number | null>(result.intervals.length ? 0 : null);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(.5);
  const [zoom, setZoom] = useState(1);
  const [origin, setOrigin] = useState({ x: 50, y: 50 });
  const [context, setContext] = useState(.75);
  const [loop, setLoop] = useState(true);
  const [onlyUnreviewed, setOnlyUnreviewed] = useState(false);
  const [reviews, setReviews] = useState<Record<number, Review>>(() => loadReviews(job.id));
  const [storageError, setStorageError] = useState(false);
  const [mediaError, setMediaError] = useState(false);
  const [clipError, setClipError] = useState("");
  const [exporting, setExporting] = useState(false);
  const interval = selected === null ? undefined : result.intervals[selected];
  const bounds = interval ? clipBounds(interval, context, duration) : undefined;
  const point = pointAt(points, current);
  const reviewedCount = result.intervals.filter((_, index) => reviews[index]?.reviewed).length;
  const coverage = result.total_windows ? result.valid_windows / result.total_windows : 0;
  const review = selected === null ? undefined : reviews[selected];

  useEffect(() => {
    try { localStorage.setItem(`av-review:${job.id}`, JSON.stringify(reviews)); setStorageError(false); }
    catch { setStorageError(true); }
  }, [job.id, reviews]);

  useEffect(() => {
    const video = videoRef.current;
    if (video && bounds && video.readyState >= 1) {
      video.currentTime = bounds[0]; setCurrent(bounds[0]);
    }
    setClipError("");
  }, [selected, context]);

  useEffect(() => { if (videoRef.current) videoRef.current.playbackRate = rate; }, [rate]);

  // Check during playback, including short intervals that timeupdate can skip.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      const video = videoRef.current;
      if (!video) return;
      if (bounds && (video.currentTime >= bounds[1] || video.currentTime < bounds[0] - .03)) {
        if (loop) video.currentTime = bounds[0];
        else { video.pause(); video.currentTime = bounds[1]; }
      }
      setCurrent(video.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, bounds?.[0], bounds?.[1], loop]);

  const seek = (time: number, preserveRange = false) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(video.duration)) return;
    const target = Math.max(0, Math.min(video.duration, time));
    if (!preserveRange && bounds && (target < bounds[0] || target > bounds[1])) setSelected(null);
    video.currentTime = target; setCurrent(target);
  };
  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (!video.paused) video.pause();
    else {
      if (bounds && (video.currentTime >= bounds[1] || video.currentTime < bounds[0])) seek(bounds[0], true);
      void video.play().catch(() => setMediaError(true));
    }
  };
  const selectRange = (index: number) => {
    setSelected(index);
    seek(clipBounds(result.intervals[index], context, duration)[0], true);
  };
  const updateReview = (patch: Partial<Review>) => {
    if (selected === null) return;
    setReviews(previous => ({ ...previous, [selected]: { ...(previous[selected] ?? { note: "", reviewed: false }), ...patch } }));
  };
  const exportClip = async () => {
    if (selected === null) return;
    setExporting(true); setClipError("");
    try {
      const response = await fetch(`/api/jobs/${job.id}/clips/${selected}?context=${context}`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? t("Không thể cắt clip.", "Could not export clip."));
      }
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a"); link.href = url; link.download = `evidence-${job.id.slice(0, 8)}-${selected + 1}.mp4`; link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { setClipError(error instanceof Error ? error.message : String(error)); }
    finally { setExporting(false); }
  };
  const exportReport = () => downloadJson({
    schema_version: 1, job_id: job.id, source_filename: filename, exported_at: new Date().toISOString(),
    result, timeline: points, reviews: result.intervals.map((range, index) => ({ index, range, ...(reviews[index] ?? { note: "", reviewed: false }) })),
    viewer: { selected_interval: selected, context_s: context, playback_rate: rate, zoom, zoom_origin_percent: origin, source: "original", loop },
  }, `review-${job.id.slice(0, 8)}.json`);

  const verdict = result.status === "inconsistency_detected" ? t("Có khoảng cần kiểm tra", "Ranges need review") : result.status === "insufficient_evidence" ? t("Chưa đủ bằng chứng", "Insufficient evidence") : t("Chưa thấy bất nhất rõ", "No clear inconsistency");
  return <>
    <div className="workspace-title"><div><div className="breadcrumb">{t("KHÔNG GIAN NGHIÊN CỨU", "RESEARCH WORKSPACE")} <span>/</span> {job.id.slice(0, 8)}</div><h1 title={filename}>{filename}</h1><p>{t("Từ dấu hiệu của mô hình đến quan sát của bạn.", "From model signals to your own observations.")}</p></div><button className="button" onClick={exportReport}><Icon name="download" />{t("Xuất báo cáo JSON", "Export JSON report")}</button></div>
    <div className="summary-grid">
      <div className={`summary-card verdict-card ${result.status}`}><span>{t("Kết quả phân tích", "Analysis result")}</span><strong><i className="status-dot" />{verdict}</strong><small>{t("Bất nhất âm thanh–môi; không kết luận thật/giả", "Audio–mouth inconsistency; no real/fake verdict")}</small></div>
      <div className="summary-card"><span>{t("Khoảng nghi vấn", "Flagged ranges")}</span><strong>{result.intervals.length.toString().padStart(2, "0")} <small>/ {reviewedCount} {t("đã xem", "reviewed")}</small></strong></div>
      <div className="summary-card"><span>{t("Độ phủ cửa sổ hợp lệ", "Valid window coverage")}</span><strong>{Math.round(coverage * 100)}% <small>{result.valid_windows}/{result.total_windows}</small></strong></div>
      <div className="summary-card"><span>{t("Độ lệch toàn cục đã bù", "Corrected global lag")}</span><strong>{result.source_lag_corrected_ms} <small>ms</small></strong></div>
    </div>
    <div className="research-grid">
      <aside className="range-panel panel">
        <div className="panel-heading"><h2>{t("Đoạn cần xem", "Review queue")}</h2><span className="count-badge">{result.intervals.length}</span></div>
        <div className="filter-tabs"><button className={!onlyUnreviewed ? "active" : ""} onClick={() => setOnlyUnreviewed(false)}>{t("Tất cả", "All")}</button><button className={onlyUnreviewed ? "active" : ""} onClick={() => setOnlyUnreviewed(true)}>{t("Chưa xem", "Unreviewed")}</button></div>
        <div className="range-list">{result.intervals.map((range, index) => (!onlyUnreviewed || !reviews[index]?.reviewed) && <button className={`range-item ${selected === index ? "active" : ""}`} key={index} onClick={() => selectRange(index)} aria-pressed={selected === index}>
          <div className="range-top"><span className="range-index">{String(index + 1).padStart(2, "0")}</span><span>{t("Khoảng nghi vấn", "Flagged range")}</span>{reviews[index]?.reviewed && <Icon name="check" size={16} />}</div>
          <strong className="mono">{formatTime(range[0])} – {formatTime(range[1])}</strong>
          <div className="range-bottom"><span>{(range[1] - range[0]).toFixed(2)} s</span><span>{t("Đỉnh", "Peak")} <b>{peaks[index]?.toFixed(3) ?? "—"}</b></span></div>
        </button>)}</div>
        {(!result.intervals.length || (onlyUnreviewed && reviewedCount === result.intervals.length)) && <div className="empty-queue"><Icon name="check" size={28} /><strong>{result.intervals.length ? t("Đã xem hết các đoạn", "All ranges reviewed") : t("Không có đoạn được đánh dấu", "No flagged ranges")}</strong><p>{result.status === "insufficient_evidence" ? t("Chưa đủ cửa sổ hợp lệ. Hãy thử video có mặt và giọng nói rõ hơn.", "Not enough valid windows. Try a video with clearer face and speech.") : t("Bạn vẫn có thể kiểm tra toàn bộ video và timeline.", "You can still inspect the full video and timeline.")}</p></div>}
        <div className="queue-footer"><span>{reviewedCount}/{result.intervals.length} {t("đã kiểm tra", "reviewed")}</span><progress value={reviewedCount} max={result.intervals.length || 1} /></div>
      </aside>
      <div className="viewer-column">
        <section className="viewer-panel panel">
          <div className="panel-heading"><div><span className="eyebrow">{t("TRÌNH KIỂM TRA VIDEO", "VIDEO INSPECTOR")}</span><h2>{interval ? `${t("Đoạn", "Range")} ${String(selected! + 1).padStart(2, "0")} · ${formatTime(interval[0])} – ${formatTime(interval[1])}` : t("Toàn bộ video", "Full video")}</h2></div><button className={`button compact ${selected === null ? "selected" : ""}`} onClick={() => setSelected(null)}>{t("Toàn video", "Full video")}</button></div>
          <div className={`video-viewport ${zoom > 1 ? "zoomed" : ""}`} onPointerDown={event => {
            if (zoom === 1) return;
            const box = event.currentTarget.getBoundingClientRect();
            setOrigin({ x: (event.clientX - box.left) / box.width * 100, y: (event.clientY - box.top) / box.height * 100 });
          }}>
            <video ref={videoRef} src={job.media_url} playsInline preload="metadata" style={{ transform: `scale(${zoom})`, transformOrigin: `${origin.x}% ${origin.y}%` }}
              onLoadedMetadata={event => { const video = event.currentTarget; setDuration(video.duration); video.playbackRate = rate; if (bounds) { video.currentTime = bounds[0]; setCurrent(bounds[0]); } }}
              onTimeUpdate={event => setCurrent(event.currentTarget.currentTime)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
              onEnded={() => { if (loop && bounds) { seek(bounds[0], true); void videoRef.current?.play().catch(() => setMediaError(true)); } }} onError={() => setMediaError(true)} />
            <div className="video-label"><span className="record-dot" />{t("NGUỒN GỐC", "ORIGINAL SOURCE")}</div><div className="video-time mono">{formatTime(current)} <span>/ {formatTime(duration)}</span></div>
            {zoom > 1 && <span className="zoom-label">{zoom}× · {t("Bấm để đổi tâm phóng", "Click to move zoom centre")}</span>}
          </div>
          {mediaError && <p className="inline-error" role="alert">{t("Trình duyệt không phát được video. Thử MP4 H.264/AAC; bạn vẫn có thể tải clip MP4 bên dưới.", "Browser playback failed. Try H.264/AAC MP4; you can still download the MP4 clip below.")}</p>}
          <div className="transport">
            <button className="play-button" onClick={togglePlay} disabled={!duration || mediaError} aria-label={playing ? t("Tạm dừng", "Pause") : t("Phát video", "Play video")}><Icon name={playing ? "pause" : "play"} /></button>
            <button className="button compact" disabled={!duration} onClick={() => { videoRef.current?.pause(); seek(current - .04); }} aria-label={t("Lùi 0,04 giây", "Back 0.04 seconds")}>−.04s</button>
            <button className="button compact" disabled={!duration} onClick={() => { videoRef.current?.pause(); seek(current + .04); }} aria-label={t("Tiến 0,04 giây", "Forward 0.04 seconds")}>+.04s</button>
            <div className="rate-group" aria-label={t("Tốc độ phát", "Playback speed")}>{[.25, .5, 1].map(value => <button key={value} aria-pressed={rate === value} className={rate === value ? "selected" : ""} onClick={() => setRate(value)}>{value}×</button>)}</div>
            <label className="loop-toggle"><input type="checkbox" checked={loop} disabled={!interval} onChange={event => setLoop(event.target.checked)} />{t("Lặp đoạn", "Loop range")}</label>
          </div>
          <div className="inspection-tools"><label><Icon name="zoom" size={17} />{t("Phóng to", "Zoom")}<select value={zoom} onChange={event => { setZoom(Number(event.target.value)); setOrigin({ x: 50, y: 50 }); }}>{[1, 1.5, 2, 3, 4].map(value => <option key={value} value={value}>{value}×</option>)}</select></label><label>{t("Ngữ cảnh mỗi phía", "Context each side")}<select value={context} disabled={!interval} onChange={event => setContext(Number(event.target.value))}>{[0, .75, 1.5].map(value => <option key={value} value={value}>{value} s</option>)}</select></label><button className="text-button" onClick={() => { setZoom(1); setOrigin({ x: 50, y: 50 }); setRate(1); }}>{t("Đặt lại góc xem", "Reset view")}</button></div>
          {zoom > 1 && <div className="zoom-position"><label>{t("Tâm ngang", "Horizontal centre")}<input type="range" min="0" max="100" value={origin.x} onChange={event => setOrigin(previous => ({ ...previous, x: Number(event.target.value) }))} /></label><label>{t("Tâm dọc", "Vertical centre")}<input type="range" min="0" max="100" value={origin.y} onChange={event => setOrigin(previous => ({ ...previous, y: Number(event.target.value) }))} /></label></div>}
          <p className="viewer-caption">{bounds ? `${t("Giới hạn phát", "Playback range")}: ${formatTime(bounds[0])} – ${formatTime(bounds[1])}. ` : ""}{t("Zoom không tăng chi tiết nguồn. Bước ±0,04 s không đảm bảo đúng từng frame.", "Zoom adds no source detail. ±0.04 s steps are not frame-accurate.")}</p>
        </section>
        <Timeline points={points} result={result} current={current} duration={duration} onSeek={seek} t={t} />
      </div>
      <aside className="details-panel panel"><div className="panel-heading"><h2>{t("Bằng chứng & ghi chú", "Evidence & notes")}</h2></div>
        <div className="live-score"><span>{t("Điểm tại vị trí đang xem", "Score at playhead")}</span><strong>{point?.valid ? point.score.toFixed(3) : "—"}</strong><span className={point?.valid && point.score > result.threshold ? "signal-warning" : "muted"}>{point?.valid ? point.score > result.threshold ? t("Vượt ngưỡng mô hình", "Above model threshold") : t("Trong ngưỡng mô hình", "Within model threshold") : t("Không có cửa sổ hợp lệ tại đây", "No valid window here")}</span><div className="score-meter"><span style={{ width: `${point?.valid ? point.score * 100 : 0}%` }} /><i style={{ left: `${result.threshold * 100}%` }} /></div><div className="meter-labels"><span>0</span><span>{t("Ngưỡng", "Threshold")} {result.threshold.toFixed(3)}</span><span>1</span></div></div>
        <dl className="evidence-facts"><div><dt>{t("Cửa sổ gần nhất", "Nearest window")}</dt><dd className="mono">{point ? formatTime(point.time_s) : "—"}</dd></div><div><dt>{t("Đỉnh trong đoạn", "Range peak")}</dt><dd>{interval ? peakScore(points, interval)?.toFixed(3) ?? "—" : "—"}</dd></div><div><dt>{t("Thời lượng nghi vấn", "Flagged duration")}</dt><dd>{interval ? `${(interval[1] - interval[0]).toFixed(2)} s` : "—"}</dd></div></dl>
        <div className="review-note"><label htmlFor="research-note">{t("Quan sát của bạn", "Your observations")}</label><p>{t("Ghi lại chuyển động môi, âm tiết và thời điểm cần đối chiếu.", "Note lip movement, speech sounds and timestamps to compare.")}</p><textarea id="research-note" rows={5} maxLength={10000} disabled={selected === null} value={review?.note ?? ""} onChange={event => updateReview({ note: event.target.value })} placeholder={selected === null ? t("Chọn một đoạn để ghi chú…", "Select a range to add notes…") : t("Ví dụ: môi khép nhưng âm thanh vẫn tiếp tục tại…", "Example: lips close while speech continues at…")} /><label className="review-checkbox"><input type="checkbox" disabled={selected === null} checked={review?.reviewed ?? false} onChange={event => updateReview({ reviewed: event.target.checked })} />{t("Đã kiểm tra đoạn này", "I reviewed this range")}</label><small className={storageError ? "inline-error" : "muted"}>{storageError ? t("Không lưu được trên trình duyệt. Hãy xuất JSON để giữ ghi chú.", "Browser storage unavailable. Export JSON to keep your notes.") : t("Tự lưu trên trình duyệt theo phiên phân tích.", "Auto-saved in this browser per analysis.")}</small></div>
        <button className="button primary full-width" disabled={selected === null || exporting} onClick={() => void exportClip()}><Icon name="download" size={17} />{exporting ? t("Đang cắt clip…", "Preparing clip…") : t("Tải clip đoạn này · MP4", "Download this clip · MP4")}</button><p className="caption">{t("Clip giữ tốc độ gốc và ngữ cảnh đã chọn; không áp dụng zoom.", "Clip keeps original speed and selected context; zoom is not applied.")}</p>
        {clipError && <p className="inline-error" role="alert">{clipError}</p>}
        {job.timeline_csv_url && <a className="button full-width" href={job.timeline_csv_url} download><Icon name="download" size={16} />{t("Tải dữ liệu timeline CSV", "Download timeline CSV")}</a>}
        <details className="method-note"><summary>{t("Cách diễn giải kết quả", "How to interpret this")}</summary><p>{t("Điểm bất nhất không phải xác suất video giả. Độ phủ chỉ tính trên cửa sổ được phân tích. Model đã bù độ lệch toàn cục; trình phát hiển thị nguồn gốc, vì vậy hãy cân nhắc thông số độ lệch khi đối chiếu âm thanh–môi.", "Inconsistency scores are not deepfake probabilities. Coverage measures analysed windows only. The model corrects global lag; this player shows the original source, so consider that lag when comparing audio and lips.")}</p></details>
      </aside>
    </div>
  </>;
}

function App() {
  const [language, setLanguage] = useState<"vi" | "en">("vi");
  const t: Translate = (vi, en) => language === "vi" ? vi : en;
  const [file, setFile] = useState<File>();
  const [preview, setPreview] = useState<string>();
  const [job, setJob] = useState<Job>();
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const submitLock = useRef(false);
  const busy = submitting || job?.status === "queued" || job?.status === "processing";
  const complete = job?.status === "complete" && job.result;
  useEffect(() => { document.documentElement.lang = language; }, [language]);
  useEffect(() => {
    if (!file) { setPreview(undefined); return; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    if (!job || !["queued", "processing"].includes(job.status)) return;
    let cancelled = false;
    let timer: number;
    const poll = async () => {
      try {
        const next = await getJob(job.id);
        if (cancelled) return;
        setJob(next); setError("");
        if (["queued", "processing"].includes(next.status)) timer = window.setTimeout(poll, 1000);
      } catch (cause) {
        if (!cancelled) { setError(cause instanceof Error ? cause.message : String(cause)); timer = window.setTimeout(poll, 3000); }
      }
    };
    timer = window.setTimeout(poll, 700);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [job?.id, job?.status]);
  const selectFile = (next?: File) => {
    if (!next || busy) return;
    if (!/\.(mp4|avi|mov|mkv|mpg|webm)$/i.test(next.name) || next.size === 0) {
      setError(t("Chọn video không rỗng thuộc định dạng được hỗ trợ.", "Choose a non-empty video in a supported format.")); return;
    }
    setFile(next); setJob(undefined); setError("");
  };
  const start = async () => {
    if (!file || busy || submitLock.current) return;
    submitLock.current = true; setSubmitting(true); setError("");
    try { setJob(await createJob(file)); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { submitLock.current = false; setSubmitting(false); }
  };
  return <div className="app-shell">
    <a className="skip-link" href="#main">{t("Đến nội dung", "Skip to content")}</a>
    <header className="topbar"><a className="brand" href="/" aria-label="VN-AV Forensics"><span className="brand-mark"><Icon name="scan" size={25} /></span><span>VN-AV <b>Forensics</b><small>RESEARCH STUDIO</small></span></a><div className="header-divider" /><span className="header-location">{t("Bàn phân tích", "Analysis workspace")}</span><div className="header-actions"><span className="research-badge">{t("Nghiên cứu âm thanh–hình ảnh", "Audio–visual research")}</span><button className="button compact" onClick={() => setLanguage(language === "vi" ? "en" : "vi")}>{language === "vi" ? "EN" : "VI"}</button>{complete && <button className="button primary compact" disabled={busy} onClick={() => inputRef.current?.click()}><Icon name="upload" size={16} />{t("Video mới", "New video")}</button>}</div></header>
    <input ref={inputRef} type="file" accept={ACCEPTED} hidden disabled={busy} onChange={event => { selectFile(event.target.files?.[0]); event.target.value = ""; }} />
    <main id="main" className="main-content">
      {error && <div className="error-banner" role="alert">{error}</div>}
      {complete ? <Workspace key={job.id} job={job} filename={file?.name ?? `Analysis ${job.id.slice(0, 8)}`} t={t} /> : <>
        <section className="intro"><div className="intro-copy"><span className="eyebrow"><span className="tiny-line" />{t("QUAN SÁT KỸ HƠN. HIỂU RÕ HƠN.", "LOOK CLOSER. UNDERSTAND MORE.")}</span><h1>{t("Mỗi dấu hiệu,", "Every signal,")}<br /><em>{t("một điểm để kiểm chứng.", "a place to investigate.")}</em></h1><p>{t("Định vị những đoạn âm thanh và chuyển động môi không khớp. Xem chậm, phóng to và đối chiếu bằng chứng trong cùng một không gian nghiên cứu.", "Locate mismatches between speech and lip movement. Slow down, zoom in and inspect the evidence in one research workspace.")}</p><div className="intro-tags"><span><Icon name="scan" size={16} />{t("Định vị theo thời gian", "Temporal localisation")}</span><span><Icon name="video" size={16} />{t("Kiểm chứng trực quan", "Visual inspection")}</span></div></div><div className="workflow-art" aria-hidden="true"><div className="art-top"><span>AV / SIGNAL INSPECTION</span><span>01 — 03</span></div><div className="art-frame"><div className="face-outline"><div className="eye-line" /><div className="mouth-region"><span /><span /><span /><span /><span /><span /><span /></div></div><div className="scan-corner tl" /><div className="scan-corner br" /><span className="art-caption">REGION OF INTEREST</span></div><div className="art-wave">{Array.from({ length: 38 }, (_, i) => <i key={i} style={{ height: `${8 + (Math.sin(i * 1.8) + 1) * 12 + (i % 7) * 3}px` }} />)}<div className="art-window" /></div><div className="art-bottom"><span>{t("Định vị", "Locate")}</span><span>{t("Quan sát", "Inspect")}</span><span>{t("Đối chiếu", "Compare")}</span></div></div></section>
        <section className="intake-grid"><div className="upload-panel panel"><div className="panel-heading"><div><span className="eyebrow">01 / {t("VIDEO ĐẦU VÀO", "INPUT VIDEO")}</span><h2>{t("Bắt đầu một phiên phân tích", "Start an analysis")}</h2></div><span className="small-label">MP4 · MOV · AVI · MKV · MPG · WEBM</span></div><button disabled={busy} className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`} onClick={() => inputRef.current?.click()} onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files[0]); }}><span className="upload-icon"><Icon name={file ? "video" : "upload"} size={28} /></span><strong>{file?.name ?? t("Kéo thả video vào đây", "Drop your video here")}</strong><span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · ${t("Bấm để thay video", "Click to replace")}` : t("hoặc bấm để chọn từ máy tính", "or click to browse your files")}</span></button><div className="upload-bottom"><p>{t("Một người nói chính, khuôn mặt và miệng rõ, có âm thanh.", "One primary speaker, a clear face and mouth, with audio.")}</p><button className="button primary" disabled={!file || busy} onClick={() => void start()}>{busy ? t("Đang phân tích…", "Analysing…") : t("Phân tích video", "Analyse video")}<Icon name="arrow" size={18} /></button></div></div>
        <aside className="guide-panel"><span className="eyebrow">{t("QUY TRÌNH KIỂM CHỨNG", "REVIEW WORKFLOW")}</span>{[[t("Tìm đoạn cần xem", "Find the right moment"), t("Mô hình đánh dấu khoảng vượt ngưỡng trên timeline.", "The model flags ranges above threshold on the timeline.")], [t("Nhìn kỹ, nghe chậm", "Look closer, listen slower"), t("Lặp đoạn, phóng to và mở rộng ngữ cảnh trước/sau.", "Loop, zoom and include context before and after.")], [t("Lưu lại quan sát", "Keep your observations"), t("Ghi chú theo đoạn, tải clip và xuất dữ liệu nghiên cứu.", "Annotate ranges, download clips and export research data.")]].map(([title, body], i) => <div className="guide-step" key={i}><span>{String(i + 1).padStart(2, "0")}</span><div><h3>{title}</h3><p>{body}</p></div></div>)}</aside></section>
        {(submitting || job) && <section className="progress-panel panel" aria-live="polite"><div className="panel-heading"><h2>{job?.status === "failed" ? t("Phân tích chưa hoàn tất", "Analysis did not finish") : t("Tiến trình phân tích", "Analysis progress")}</h2><span className="small-label">{submitting ? t("Đang tải video lên…", "Uploading video…") : job?.status === "failed" ? t("Có lỗi · có thể thử lại", "Failed · retry available") : t("Đang xử lý", "Processing")}</span></div><ol className="stages">{STAGES.map(([vi, en], index) => <li className={job?.stages[index]?.status ?? "pending"} key={vi}><span>{job?.stages[index]?.status === "complete" ? <Icon name="check" size={15} /> : index + 1}</span><strong>{t(vi, en)}</strong></li>)}</ol>{job?.error && <p className="inline-error">{job.error}</p>}</section>}
        {preview && <details className="preview-panel panel"><summary>{t("Xem trước video đã chọn", "Preview selected video")}<span>{file?.name}</span></summary><video key={preview} src={preview} controls playsInline preload="metadata" /><p className="caption">{t("Nếu trình duyệt không hỗ trợ định dạng nguồn, bạn vẫn có thể gửi để phân tích.", "If your browser cannot play this source format, you can still submit it for analysis.")}</p></details>}
        <p className="research-notice"><Icon name="scan" size={18} />{t("Công cụ hỗ trợ nghiên cứu bất nhất âm thanh–môi. Kết quả cần được đối chiếu; không phải kết luận video thật hay giả.", "A research tool for audio–mouth inconsistency. Results require review; they are not a real/fake verdict.")}</p>
      </>}
    </main><footer className="site-footer"><span>VN-AV Forensics <span className="footer-separator">/</span> Research Studio</span><span>{t("Bằng chứng trước. Diễn giải sau.", "Evidence first. Interpretation follows.")}</span></footer>
  </div>;
}

createRoot(document.getElementById("root")!).render(<App />);
