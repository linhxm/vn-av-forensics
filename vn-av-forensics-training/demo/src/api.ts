export type Interval = [number, number];
export type TimelinePoint = { time_s: number; score: number; valid: boolean; audio_energy?: number };
export type Head = "timing" | "lip_audio_mismatch";
export type Segment = { start_s: number; end_s: number; score: number };
export type RelationResult = {
  schema_version: "av-relations-v2"; model_id: string; status: string;
  clip_inconsistency_score: number | null; duration_s: number;
  context_window_s: number; step_s: number; notice: string;
  global_lag_ms: number | null; scope: string[];
  thresholds: Record<string, number>; coverage: Record<string, number>; accepted_coverage: Record<string, number>;
  head_availability: Record<string, string>;
  diagnostic_peak_score?: number | null;
  diagnostic_intervals?: RelationResult["suspicious_intervals"];
  checkpoint?: { active_heads: string[]; calibrated_heads: string[]; deployable_heads: string[];
    validation_report: Record<string, {status:string; reasons?:string[]; precision?:number; recall?:number; false_alarm_rate?:number; auroc?:number; positive_windows?:number; negative_windows?:number; inference_coverage?:number; lag_mae_ms?:number|null; unaligned_ready?:boolean}>;
    test_report?: {relations?:Record<string,{false_alarm_rate?:number;precision?:number;recall?:number}>}|null;
    checkpoint_sha256: string; label_coverage?:unknown; test_report_provenance?:string|null };
  feature_evidence?: {audio_shape:number[]; visual_shape:number[]; window_count:number; audio_valid_windows:number;visual_valid_windows:number;encoder_signature:string};
  preprocessing?: {sample_rate_hz:number;video_grid_fps:number;decoded_frames:number;origin_s:number;
    audio_rms:{start:number;rms:number}[]; frames:{time_s:number;image:string;face_check:{valid:boolean;reason:string;box?:number[]}|null}[];note:string};
  suspicious_intervals: { start: number; end: number; relation: string; inconsistency_score: number }[];
  window_scores: { start: number; end: number; inconsistency_score: number | null;
    alignment_used: boolean; matched_visual_time_s: number|null; accepted_relation_scores: Record<string, number|null>; lag_confidence: number|null;
    estimated_lag_ms: number | null; diagnostic_estimated_lag_ms?: number | null; diagnostic_inconsistency_score?: number | null; relation_scores: Record<string, number | null> }[];
};
export type Result = RelationResult;
export type Job = {
  id: string; status: "queued" | "processing" | "complete" | "failed" | "interrupted";
  stage: string; message: string; filename: string; error: string | null; result: Result | null;
  steps?: {stage:string;message:string;at:string}[];
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
