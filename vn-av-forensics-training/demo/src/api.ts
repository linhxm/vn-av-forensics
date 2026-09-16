export type Interval = [number, number];
export type TimelinePoint = { time_s: number; score: number; valid: boolean; audio_energy?: number };
export type Head = "forgery" | "mismatch";
export type Segment = { start_s: number; end_s: number; score: number };
export type ForgeryResult = {
  schema_version: number; model_id: string; status: string;
  clip_forgery_score: number | null; clip_threshold: number; prediction: "real" | "fake" | null;
  coverage: number; duration_s: number; notice: string;
  head_availability: { clip: boolean; forgery: boolean; mismatch: boolean };
  thresholds: { clip: number; forgery?: number; mismatch?: number };
  segments: Partial<Record<Head, Segment[]>>;
  timeline: { time_s: number; valid: boolean; forgery: number | null; mismatch: number | null }[];
};
export type RelationResult = {
  schema_version: "av-relations-v1"; model_id: string; status: string;
  clip_inconsistency_score: number | null; duration_s: number;
  context_window_s: number; step_s: number; notice: string;
  global_lag_ms: number | null; identity_status: string;
  thresholds: Record<string, number>; coverage: Record<string, number>;
  head_availability: Record<string, "trained" | "untrained">;
  suspicious_intervals: { start: number; end: number; relation: string; inconsistency_score: number }[];
  window_scores: { start: number; end: number; inconsistency_score: number | null;
    estimated_lag_ms: number | null; relation_scores: Record<string, number | null> }[];
};
export type Result = ForgeryResult | RelationResult;
export type Job = {
  id: string; status: "queued" | "processing" | "complete" | "failed" | "interrupted";
  stage: string; message: string; filename: string; error: string | null; result: Result | null;
};
async function responseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const data = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  }
  return response.json() as Promise<T>;
}
export async function createJob(file: File): Promise<Job> {
  const form = new FormData(); form.append("video", file);
  return responseJson<Job>(await fetch("/api/jobs", { method: "POST", body: form }));
}
export async function getJob(id: string): Promise<Job> {
  return responseJson<Job>(await fetch(`/api/jobs/${id}`));
}
export async function getHealth(): Promise<{ ready: boolean; missing: string[]; model_loaded: boolean }> {
  return responseJson(await fetch("/api/health"));
}
