import type { Segment, TimelinePoint } from "./api";
import { formatTime } from "./research";

export function Timeline({ points, segments, threshold, current, duration, onSeek }: {
  points: TimelinePoint[]; segments: Segment[]; threshold: number; current: number;
  duration: number; onSeek: (time: number) => void;
}) {
  const width = 900, height = 180, left = 35, right = 15, top = 15, bottom = 145;
  const x = (t: number) => left + t / Math.max(.01, duration) * (width - left - right);
  const y = (s: number) => bottom - Math.max(0, Math.min(1, s)) * (bottom - top);
  let previous: TimelinePoint | undefined;
  const spacings = points.slice(1).map((p, i) => p.time_s - points[i].time_s).filter(d => d > 0).sort((a, b) => a - b);
  const maxGap = (spacings[Math.floor(spacings.length / 2)] ?? .04) * 1.5;
  const path = points.map(p => {
    if (!p.valid) { previous = undefined; return ""; }
    const command = previous && p.time_s - previous.time_s <= maxGap ? "L" : "M";
    previous = p;
    return `${command}${x(p.time_s)},${y(p.score)}`;
  }).join(" ");
  return <div className="signal">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Điểm theo thời gian. Dùng thanh tua bên dưới để chọn thời điểm."
      onClick={event => { const b = event.currentTarget.getBoundingClientRect(); onSeek(Math.max(0, Math.min(duration, ((event.clientX - b.left) / b.width * width - left) / (width - left - right) * duration))); }}>
      <rect x={left} y={top} width={width-left-right} height={bottom-top} fill="#f5f7fb" />
      {segments.map((s, i) => <rect key={i} x={x(s.start_s)} y={top} width={x(s.end_s)-x(s.start_s)} height={bottom-top} fill="#fee9ba" />)}
      {[0,.5,1].map(v => <g key={v}><line x1={left} x2={width-right} y1={y(v)} y2={y(v)} stroke="#dbe2ed" /><text x="8" y={y(v)+4}>{v}</text></g>)}
      <line x1={left} x2={width-right} y1={y(threshold)} y2={y(threshold)} stroke="#bf6a12" strokeDasharray="5 5" />
      <path d={path} fill="none" stroke="#3156d3" strokeWidth="2" />
      <line x1={x(current)} x2={x(current)} y1={top} y2={bottom} stroke="#111c36" />
      {[0,.25,.5,.75,1].map(v => <text key={v} x={x(duration*v)} y="172" textAnchor={v===0?"start":v===1?"end":"middle"}>{formatTime(duration*v)}</text>)}
    </svg>
    <input aria-label="Tua video" type="range" min="0" max={duration} step="0.04" value={Math.min(current,duration)} onChange={e => onSeek(Number(e.target.value))} />
    <small>Ngưỡng {threshold.toFixed(3)} · Vùng vàng: đoạn được đánh dấu · Khoảng đứt: thiếu dữ liệu hợp lệ</small>
  </div>;
}
