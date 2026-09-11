import { useEffect, useMemo, useRef, useState } from "react";
import type { Result, TimelinePoint } from "./api";
import { formatTime, type Translate } from "./research";

export function Timeline({ points, result, current, duration, onSeek, t }: {
  points: TimelinePoint[]; result: Result; current: number; duration: number;
  onSeek: (time: number) => void; t: Translate;
}) {
  const chartRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(840);
  useEffect(() => {
    const element = chartRef.current;
    if (!element) return;
    const observer = new ResizeObserver(entries => setWidth(Math.max(280, entries[0].contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const left = 36, right = 16, top = 14, bottom = 125;
  const end = Math.max(duration, points.at(-1)?.time_s ?? 0, .01);
  const x = (time: number) => left + Math.max(0, Math.min(1, time / end)) * (width - left - right);
  const y = (score: number) => bottom - Math.max(0, Math.min(1, score)) * (bottom - top);
  const signals = useMemo(() => {
    let previousValid = false;
    const scorePath = points.map(point => {
      if (!point.valid) { previousValid = false; return ""; }
      const command = `${previousValid ? "L" : "M"}${x(point.time_s)},${y(point.score)}`;
      previousValid = true;
      return command;
    }).join(" ");
    const maxEnergy = points.reduce((max, point) => Math.max(max, point.audio_energy), .000001);
    const energyPath = points.map((point, i) => `${i ? "L" : "M"}${x(point.time_s)},${174 - point.audio_energy / maxEnergy * 24}`).join(" ");
    return <>
      <rect x={left} y={top} width={width - left - right} height={bottom - top} fill="url(#invalid-hatch)" />
      {points.map((point, index) => point.valid && <rect key={index} x={x(point.time_s - .02)} y={top} width={Math.max(.5, x(point.time_s + .02) - x(point.time_s - .02))} height={bottom - top} fill="#fff" />)}
      {[0, .5, 1].map(value => <g key={value}><line x1={left} x2={width - right} y1={y(value)} y2={y(value)} className="chart-grid" /><text x={left - 10} y={y(value) + 4} textAnchor="end">{value.toFixed(1)}</text></g>)}
      {result.intervals.map(([start, stop], i) => <rect key={i} x={x(start)} y={top} width={Math.max(2, x(stop) - x(start))} height={bottom - top} fill="#f59e0b" opacity=".17" />)}
      <line x1={left} x2={width - right} y1={y(result.threshold)} y2={y(result.threshold)} className="chart-threshold" />
      <path d={scorePath} className="chart-score" />
      <text x={left} y={145}>{t("Năng lượng âm thanh · chuẩn hóa theo đỉnh", "Audio energy · peak-normalised")}</text>
      <path d={energyPath} className="chart-energy" />
    </>;
  }, [points, result, width, end, t]);
  return <section className="timeline-panel">
    <div className="panel-heading"><div><span className="eyebrow">{t("ĐỊNH VỊ BẰNG CHỨNG", "EVIDENCE TIMELINE")}</span><h2>{t("Điểm bất nhất theo thời gian", "Inconsistency over time")}</h2></div><span className="mono">{formatTime(current)}</span></div>
    <div className="chart-legend"><span><i className="dot indigo" />{t("Điểm model", "Model score")}</span><span><i className="dash" />{t("Ngưỡng", "Threshold")} {result.threshold.toFixed(3)}</span><span><i className="dot amber" />{t("Khoảng nghi vấn", "Flagged range")}</span><span><i className="dot gray" />{t("Không hợp lệ / chưa phân tích", "Invalid / not analysed")}</span></div>
    <svg ref={chartRef} className="timeline-chart" viewBox={`0 0 ${width} 202`} role="img" aria-label={t("Bấm biểu đồ hoặc dùng thanh tua bên dưới để chọn thời điểm", "Click the chart or use the seek slider below")}
      onClick={event => { const box = event.currentTarget.getBoundingClientRect(); onSeek(Math.max(0, Math.min(end, ((event.clientX - box.left) / box.width * width - left) / (width - left - right) * end))); }}>
      <defs><pattern id="invalid-hatch" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M0 6L6 0" stroke="#dbe1ea" strokeWidth="1" /></pattern></defs>
      {signals}
      <line x1={x(current)} x2={x(current)} y1={top} y2={178} className="chart-playhead" /><circle cx={x(current)} cy={top} r="4" fill="#4f46e5" />
      {[0, 1, 2, 3, 4].map(i => <text key={i} x={x(end * i / 4)} y={198} textAnchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}>{formatTime(end * i / 4)}</text>)}
    </svg>
    <input className="timeline-seek" type="range" min="0" max={end} step="0.01" value={Math.min(current, end)} aria-label={t("Tua video theo timeline", "Seek video on timeline")} onChange={event => onSeek(Number(event.target.value))} />
    <p className="caption">{t("Năng lượng âm thanh phản ánh dữ liệu đầu vào; không phải bằng chứng thật/giả.", "Audio energy describes input data; it is not evidence of authenticity.")}</p>
  </section>;
}
