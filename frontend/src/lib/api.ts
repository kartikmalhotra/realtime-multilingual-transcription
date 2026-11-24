import type {
  EngineInfo,
  Stats,
  Transcription,
  TranscriptionPage,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

export interface ListParams {
  search?: string;
  language?: string;
  limit?: number;
  offset?: number;
}

export const api = {
  listTranscriptions(params: ListParams = {}): Promise<TranscriptionPage> {
    const qs = new URLSearchParams();
    if (params.search) qs.set("search", params.search);
    if (params.language && params.language !== "all")
      qs.set("language", params.language);
    qs.set("limit", String(params.limit ?? 20));
    qs.set("offset", String(params.offset ?? 0));
    return request<TranscriptionPage>(`/api/transcriptions?${qs}`);
  },

  createTranscription(body: {
    title: string;
    text: string;
    language?: string;
    engine?: string;
  }): Promise<Transcription> {
    return request<Transcription>("/api/transcriptions", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  updateTranscription(
    id: string,
    body: { title?: string; text?: string; language?: string }
  ): Promise<Transcription> {
    return request<Transcription>(`/api/transcriptions/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    });
  },

  deleteTranscription(id: string): Promise<void> {
    return request<void>(`/api/transcriptions/${id}`, { method: "DELETE" });
  },

  stats(): Promise<Stats> {
    return request<Stats>("/api/stats");
  },

  engine(): Promise<EngineInfo> {
    return request<EngineInfo>("/api/engine");
  },

  exportUrl(): string {
    return `${BASE}/api/transcriptions/export`;
  },
};

/** Absolute ws:// or wss:// URL for the streaming endpoint. */
export function websocketUrl(params: Record<string, string>): string {
  const qs = new URLSearchParams(params).toString();
  if (BASE) {
    return `${BASE.replace(/^http/, "ws")}/ws/transcribe?${qs}`;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/transcribe?${qs}`;
}
