import { useState } from "react";
import type { Transcription } from "../lib/types";
import {
  formatDuration,
  formatTimestamp,
  languageLabel,
  speak,
  stopSpeaking,
} from "../lib/format";

interface Props {
  record: Transcription;
  onEdit: (record: Transcription) => void;
  onDelete: (record: Transcription) => void;
}

export default function TranscriptionCard({ record, onEdit, onDelete }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [confirming, setConfirming] = useState(false);

  function toggleSpeech() {
    if (speaking) {
      stopSpeaking();
      setSpeaking(false);
      return;
    }
    speak(record.text, record.language);
    setSpeaking(true);
    // speechSynthesis has no reliable "done" event across browsers; poll.
    const timer = window.setInterval(() => {
      if (!window.speechSynthesis.speaking) {
        setSpeaking(false);
        window.clearInterval(timer);
      }
    }, 400);
  }

  return (
    <article className="card">
      <header>
        <h3>{record.title}</h3>
        <div className="tags">
          <span className={`tag lang-${record.language}`}>
            {languageLabel(record.language)}
          </span>
          <span className="tag">{record.engine}</span>
        </div>
      </header>

      <p className={expanded ? "card-text" : "card-text clamp"}>{record.text}</p>

      {record.text.length > 180 && (
        <button
          type="button"
          className="link"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}

      {expanded && record.segments.length > 0 && (
        <ul className="segment-list compact">
          {record.segments.map((s) => (
            <li key={s.id}>
              <span className="seg-time">{(s.start_ms / 1000).toFixed(1)}s</span>
              <span className="seg-lang">{languageLabel(s.language)}</span>
              <span className="seg-text">{s.text}</span>
            </li>
          ))}
        </ul>
      )}

      <footer>
        <span className="meta">
          {formatTimestamp(record.created_at)} · {record.word_count} words ·{" "}
          {formatDuration(record.duration_ms)}
        </span>

        <div className="card-actions">
          <button type="button" className="btn tiny" onClick={toggleSpeech}>
            {speaking ? "Stop" : "Speak"}
          </button>
          <button
            type="button"
            className="btn tiny"
            onClick={() => onEdit(record)}
          >
            Edit
          </button>
          {confirming ? (
            <>
              <button
                type="button"
                className="btn tiny danger"
                onClick={() => onDelete(record)}
              >
                Confirm
              </button>
              <button
                type="button"
                className="btn tiny ghost"
                onClick={() => setConfirming(false)}
              >
                No
              </button>
            </>
          ) : (
            <button
              type="button"
              className="btn tiny ghost"
              onClick={() => setConfirming(true)}
            >
              Delete
            </button>
          )}
        </div>
      </footer>
    </article>
  );
}
