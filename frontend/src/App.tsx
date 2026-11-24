import { useCallback, useEffect, useState } from "react";
import Dashboard from "./components/Dashboard";
import Recorder from "./components/Recorder";
import { api } from "./lib/api";
import type { EngineInfo } from "./lib/types";

export default function App() {
  const [engineInfo, setEngineInfo] = useState<EngineInfo | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    api.engine().then(setEngineInfo).catch(() => setEngineInfo(null));
  }, []);

  const handleSaved = useCallback(() => {
    setRefreshKey((k) => k + 1);
  }, []);

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1>Real-time Multilingual Transcription</h1>
          <p className="muted">
            Streaming Hindi · English speech-to-text with a searchable,
            editable transcript history.
          </p>
        </div>
        <a
          className="btn tiny ghost"
          href="/docs"
          target="_blank"
          rel="noreferrer"
        >
          API docs
        </a>
      </header>

      <main>
        <Recorder engineInfo={engineInfo} onSaved={handleSaved} />
        <Dashboard refreshKey={refreshKey} database={engineInfo?.database} />
      </main>

      <footer className="footer muted">
        Audio streams over a WebSocket in ~128 ms frames; utterances are
        segmented server-side on silence, transcribed, and persisted.
      </footer>
    </div>
  );
}
