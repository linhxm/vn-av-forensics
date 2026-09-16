import type { Interval, TimelinePoint } from "./api";

export type Translate = (vi: string, en: string) => string;
export type Review = { note: string; reviewed: boolean };

export function formatTime(time: number) {
  const safe = Math.max(0, Number.isFinite(time) ? time : 0);
  return `${Math.floor(safe / 60).toString().padStart(2, "0")}:${(safe % 60).toFixed(2).padStart(5, "0")}`;
}

// Sampling cadence determines how far a point can represent the playhead.
export function pointAt(timeline: TimelinePoint[], time: number) {
  if (!timeline.length) return undefined;
  let low = 0, high = timeline.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (timeline[mid].time_s < time) low = mid + 1;
    else high = mid;
  }
  const index = low > 0 && Math.abs(timeline[low - 1].time_s - time) < Math.abs(timeline[low].time_s - time) ? low - 1 : low;
  const point = timeline[index];
  const gap = Math.min(
    index > 0 ? point.time_s - timeline[index - 1].time_s : Infinity,
    index + 1 < timeline.length ? timeline[index + 1].time_s - point.time_s : Infinity,
  );
  const tolerance = Number.isFinite(gap) && gap > 0 ? Math.min(gap / 2 + 1e-6, .1) : .02;
  return Math.abs(point.time_s - time) <= tolerance ? point : undefined;
}

export function peakScore(timeline: TimelinePoint[], interval: Interval) {
  const values = timeline.filter(p => p.valid && p.time_s >= interval[0] && p.time_s <= interval[1]);
  return values.length ? values.reduce((peak, p) => Math.max(peak, p.score), 0) : undefined;
}

export function clipBounds(interval: Interval, context: number, duration: number): Interval {
  return [Math.max(0, interval[0] - context), Math.min(duration || Infinity, interval[1] + context)];
}

export function downloadJson(value: unknown, filename: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
