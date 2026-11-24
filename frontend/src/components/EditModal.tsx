import { useEffect, useState } from "react";
import type { Transcription } from "../lib/types";

interface Props {
  record: Transcription;
  onCancel: () => void;
  onSave: (patch: { title: string; text: string }) => Promise<void>;
}

export default function EditModal({ record, onCancel, onSave }: Props) {
  const [title, setTitle] = useState(record.title);
  const [text, setText] = useState(record.text);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await onSave({ title, text });
    } catch (err) {
      setError(err instanceof Error ? err.message : "could not save");
      setBusy(false);
    }
  }

  return (
    <div className="backdrop" onClick={onCancel}>
      <form
        className="modal"
        onClick={(e) => e.stopPropagation()}
        onSubmit={submit}
      >
        <h3>Edit transcription</h3>

        <label>
          <span>Title</span>
          <input
            value={title}
            maxLength={200}
            onChange={(e) => setTitle(e.target.value)}
            autoFocus
          />
        </label>

        <label>
          <span>Transcript</span>
          <textarea
            value={text}
            rows={8}
            onChange={(e) => setText(e.target.value)}
          />
        </label>

        {error && <p className="note error">{error}</p>}

        <div className="modal-actions">
          <button type="button" className="btn ghost" onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" className="btn primary" disabled={busy}>
            {busy ? "Saving…" : "Save changes"}
          </button>
        </div>
      </form>
    </div>
  );
}
