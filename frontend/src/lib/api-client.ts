const BASE = "/api";

export interface Track {
  id: string;
  name: string;
  description: string;
  status: string;
}

export interface KataInfo {
  id: string;
  title: string;
  phase: number;
  sequence: number;
  track_id: string;
}

export interface KatasResponse {
  katas: KataInfo[];
  phases: Record<string, string>;
}

export interface ExecutionResult {
  stdout: string;
  stderr: string;
  error: string | null;
  execution_time_ms: number;
}

export async function fetchTracks(): Promise<Track[]> {
  const res = await fetch(`${BASE}/tracks`);
  if (!res.ok) throw new Error("Failed to fetch tracks");
  return res.json();
}

export async function fetchKatas(trackId: string): Promise<KatasResponse> {
  const res = await fetch(`${BASE}/tracks/${trackId}/katas`);
  if (!res.ok) throw new Error("Failed to fetch katas");
  return res.json();
}

export async function fetchKataContent(
  trackId: string,
  phaseId: number,
  kataId: string,
): Promise<string> {
  const res = await fetch(
    `${BASE}/tracks/${trackId}/katas/${phaseId}/${kataId}/content`,
  );
  if (!res.ok) throw new Error("Failed to fetch kata content");
  return res.text();
}

export async function executeCode(
  code: string,
  kataId: string,
  trackId: string = "python-nlp",
): Promise<ExecutionResult> {
  const res = await fetch(`${BASE}/execute/${trackId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, kata_id: kataId }),
  });
  if (!res.ok) throw new Error("Failed to execute code");
  return res.json();
}
