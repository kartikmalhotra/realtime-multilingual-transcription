export type EngineName = "whisper" | "browser" | "gemini";

export interface Segment {
  id: string;
  text: string;
  language: string;
  start_ms: number;
  end_ms: number;
  confidence: number | null;
}

export interface Transcription {
  id: string;
  title: string;
  text: string;
  language: string;
  engine: string;
  duration_ms: number;
  word_count: number;
  created_at: string;
  updated_at: string;
  segments: Segment[];
}

export interface TranscriptionPage {
  items: Transcription[];
  total: number;
  limit: number;
  offset: number;
}

export interface Stats {
  total_transcriptions: number;
  total_words: number;
  total_duration_ms: number;
  by_language: Record<string, number>;
}

export interface EngineInfo {
  active: EngineName;
  available: EngineName[];
  model: string | null;
  ready: boolean;
  detail: string | null;
  database: string;
}

/** A committed utterance, held in the recorder panel before the page reloads. */
export interface LiveSegment {
  text: string;
  language: string;
  start_ms: number;
  end_ms: number;
}
