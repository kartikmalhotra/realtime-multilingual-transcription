import { useState } from "react";
import { browserSpeechSupported, useTranscriber } from "../hooks/useTranscriber";
import type { EngineInfo, EngineName, Transcription } from "../lib/types";
import { languageLabel } from "../lib/format";

interface Props {
  engineInfo: EngineInfo | null;
  onSaved: (t: Transcription) => void;
}

const LANGUAGES = [
  { value: "auto", label: "Auto-detect (Hinglish)" },
  { value: "hi", label: "Hindi" },
  { value: "en", label: "English" },
];

export default function Recorder({ engineInfo, onSaved }: Props) {
  const [engine, setEngine] = useState<EngineName>("whisper");
  const [language, setLanguage] = useState("auto");
  const [title, setTitle] = useState("");

  const {
    status,
    interim,
    segments,
    error,
    level,
    engineLabel,
    start,
    stop,
    cancel,
    clearError,
  } = useTranscriber((saved) => {
    setTitle("");
    onSaved(saved);
  });

  const live = status === "listening" || status === "saving";
  const connecting = status === "connecting";
  const committed = segments.map((s) => s.text).join(" ");

  return (
    <section className="panel recorder">
      <header className="panel-head">
        <div>
          <h2>Live transcription</h2>
          <p className="muted">
            Speak naturally — mix Hindi and English in the same sentence.
          </p>
        </div>
        {engineLabel && <span className="pill">{engineLabel}</span>}
      </header>

      <div className="controls">
        <label>
          <span>Engine</span>
          <select
            value={engine}
            onChange={(e) => setEngine(e.target.value as EngineName)}
            disabled={live}
          >
            <option value="whisper">Whisper (self-hosted)</option>
            <option value="browser" disabled={!browserSpeechSupported()}>
              Web Speech API (browser)
            </option>
            <option value="gemini">Gemini Live</option>
          </select>
        </label>

        <label>
          <span>Language</span>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            disabled={live}
          >
            {LANGUAGES.map((l) => (
              <option key={l.value} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
        </label>

        <label className="grow">
          <span>Title (optional)</span>
          <input
            type="text"
            value={title}
            placeholder="Auto-generated from the first few words"
            onChange={(e) => setTitle(e.target.value)}
            disabled={live}
          />
        </label>
      </div>

      <div className="record-row">
        <button
          type="button"
          className={live || connecting ? "btn danger" : "btn primary"}
          onClick={() => {
            if (connecting) return cancel();
            return live ? stop() : start({ engine, language, title });
          }}
          // Only the save round-trip is uninterruptible; waiting on the
          // microphone prompt must always be escapable.
          disabled={status === "saving"}
        >
          {status === "idle" && "Start recording"}
          {status === "connecting" && "Connecting… — cancel"}
          {status === "listening" && "Stop & save"}
          {status === "saving" && "Saving…"}
        </button>

        <div className="meter" aria-hidden="true">
          <div className="meter-fill" style={{ width: `${level * 100}%` }} />
        </div>

        {live && <span className="rec-dot" aria-label="recording" />}
      </div>

      {engine === "whisper" && engineInfo && !engineInfo.ready && (
        <p className="note">
          The Whisper model loads on the first utterance — expect a few seconds
          of delay once, then real-time results.
        </p>
      )}

      {engine === "gemini" && engineInfo?.active === "gemini" && !engineInfo.ready && (
        <p className="note warn">
          GEMINI_API_KEY is not set on the server. Switch to Whisper or add the key.
        </p>
      )}

      {error && (
        <p className="note error" role="alert">
          {error}
          <button type="button" className="link" onClick={clearError}>
            dismiss
          </button>
        </p>
      )}

      <div className="transcript" aria-live="polite">
        {!committed && !interim && (
          <p className="placeholder">
            Nothing yet. Press <strong>Start recording</strong> and say something
            like “Aaj ka meeting postpone ho gaya, so let's reschedule for
            tomorrow.”
          </p>
        )}
        {committed && <span className="final-text">{committed}</span>}
        {interim && <span className="interim-text"> {interim}</span>}
      </div>

      {segments.length > 0 && (
        <ul className="segment-list">
          {segments.map((s, i) => (
            <li key={`${s.start_ms}-${i}`}>
              <span className="seg-time">
                {(s.start_ms / 1000).toFixed(1)}s
              </span>
              <span className="seg-lang">{languageLabel(s.language)}</span>
              <span className="seg-text">{s.text}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
