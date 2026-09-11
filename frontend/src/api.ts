export type StageStatus = "pending" | "active" | "complete" | "failed";

export type Stage = {
  id: string;
  label: string;
  status: StageStatus;
  message: string;
};

export type Interval = [number, number];

export type Result = {
  status: "insufficient_evidence" | "inconsistency_detected" | "no_clear_inconsistency";
  intervals: Interval[];
  threshold: number;
  source_lag_corrected_ms: number;
  valid_windows: number;
  total_windows: number;
  notice: string;
};

export type TimelinePoint = {
  time_s: number;
  score: number;
  valid: boolean;
  audio_energy: number;
};

export type Job = {
  id: string;
  status: "queued" | "processing" | "complete" | "failed";
  message: string;
  stages: Stage[];
  error?: string | null;
  result?: Result;
  timeline?: TimelinePoint[];
  media_url?: string;
  timeline_csv_url?: string;
};

async function responseJson<T>(response: Response): Promise<T> {
  if (response.ok) return response.json() as Promise<T>;
  const payload = await response.json().catch(() => ({ detail: response.statusText }));
  throw new Error(payload.detail ?? "Không thể kết nối tới analysis service.");
}

export async function createJob(file: File): Promise<Job> {
  const form = new FormData();
  form.append("video", file);
  return responseJson<Job>(await fetch("/api/jobs", { method: "POST", body: form }));
}

export async function getJob(id: string): Promise<Job> {
  return responseJson<Job>(await fetch(`/api/jobs/${id}`));
}
