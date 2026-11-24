import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import type { Stats, Transcription } from "../lib/types";
import { formatDuration } from "../lib/format";
import EditModal from "./EditModal";
import TranscriptionCard from "./TranscriptionCard";

const PAGE_SIZE = 10;

const DATABASE_LABELS: Record<string, string> = {
  postgresql: "PostgreSQL",
  mysql: "MySQL",
  sqlite: "SQLite",
};

interface Props {
  /** Bumped by the recorder after a save so the list refetches. */
  refreshKey: number;
  /** Named so the panel states the database actually in use. */
  database?: string;
}

export default function Dashboard({ refreshKey, database }: Props) {
  const [items, setItems] = useState<Transcription[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [search, setSearch] = useState("");
  const [language, setLanguage] = useState("all");
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<Transcription | null>(null);

  // Debounce so typing in the search box doesn't fire a request per keystroke.
  const [debouncedSearch, setDebouncedSearch] = useState("");
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedSearch(search);
      setPage(0);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [search]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    Promise.all([
      api.listTranscriptions({
        search: debouncedSearch,
        language,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
      api.stats(),
    ])
      .then(([listed, counted]) => {
        if (cancelled) return;
        setItems(listed.items);
        setTotal(listed.total);
        setStats(counted);
        setError(null);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "load failed");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [debouncedSearch, language, page, refreshKey]);

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const languageOptions = useMemo(() => {
    const keys = Object.keys(stats?.by_language ?? {});
    return ["all", ...keys.filter((k) => k !== "unknown"), "unknown"].filter(
      (k, i, arr) => arr.indexOf(k) === i
    );
  }, [stats]);

  async function handleUpdate(patch: { title: string; text: string }) {
    if (!editing) return;
    const updated = await api.updateTranscription(editing.id, patch);
    setItems((prev) => prev.map((i) => (i.id === updated.id ? updated : i)));
    setEditing(null);
    api.stats().then(setStats).catch(() => {});
  }

  async function handleDelete(record: Transcription) {
    const previous = items;
    setItems((prev) => prev.filter((i) => i.id !== record.id));
    setTotal((t) => Math.max(0, t - 1));
    try {
      await api.deleteTranscription(record.id);
      api.stats().then(setStats).catch(() => {});
    } catch (err) {
      setItems(previous); // roll the optimistic removal back
      setTotal(previous.length);
      setError(err instanceof Error ? err.message : "delete failed");
    }
  }

  return (
    <section className="panel dashboard">
      <header className="panel-head">
        <div>
          <h2>Transcription history</h2>
          <p className="muted">
            {database
              ? `Stored in ${DATABASE_LABELS[database] ?? database} · full CRUD`
              : "Full CRUD"}
          </p>
        </div>
        <a className="btn tiny ghost" href={api.exportUrl()}>
          Export CSV
        </a>
      </header>

      {stats && (
        <div className="stats">
          <div className="stat">
            <strong>{stats.total_transcriptions}</strong>
            <span>recordings</span>
          </div>
          <div className="stat">
            <strong>{stats.total_words}</strong>
            <span>words</span>
          </div>
          <div className="stat">
            <strong>{formatDuration(stats.total_duration_ms)}</strong>
            <span>audio</span>
          </div>
          <div className="stat">
            <strong>{Object.keys(stats.by_language).length}</strong>
            <span>languages</span>
          </div>
        </div>
      )}

      <div className="filters">
        <input
          type="search"
          placeholder="Search transcripts…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          value={language}
          onChange={(e) => {
            setLanguage(e.target.value);
            setPage(0);
          }}
        >
          {languageOptions.map((opt) => (
            <option key={opt} value={opt}>
              {opt === "all" ? "All languages" : opt}
            </option>
          ))}
        </select>
      </div>

      {error && (
        <p className="note error" role="alert">
          {error}
        </p>
      )}

      {loading && items.length === 0 && <p className="placeholder">Loading…</p>}

      {!loading && items.length === 0 && (
        <p className="placeholder">
          No transcriptions yet — record one above and it will show up here.
        </p>
      )}

      <div className="card-list">
        {items.map((record) => (
          <TranscriptionCard
            key={record.id}
            record={record}
            onEdit={setEditing}
            onDelete={handleDelete}
          />
        ))}
      </div>

      {total > PAGE_SIZE && (
        <div className="pager">
          <button
            type="button"
            className="btn tiny ghost"
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
          >
            Previous
          </button>
          <span className="muted">
            Page {page + 1} of {pageCount}
          </span>
          <button
            type="button"
            className="btn tiny ghost"
            disabled={page + 1 >= pageCount}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
          </button>
        </div>
      )}

      {editing && (
        <EditModal
          record={editing}
          onCancel={() => setEditing(null)}
          onSave={handleUpdate}
        />
      )}
    </section>
  );
}
