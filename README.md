# Real-time Multilingual Transcription

A "Speech to Text" web app that transcribes speech **as you talk**, handles
**Hindi–English code-switching** in a single sentence, stores every
transcript in a relational database, and exposes the history in a dashboard
with full CRUD.

Built for the Procucev engineering hiring assignment.

> **Stack:** FastAPI (Python) · React + TypeScript · PostgreSQL ·
> faster-whisper (self-hosted, open source)

---

## What it does

| | |
|---|---|
| **Real-time** | Audio streams over a WebSocket in ~128 ms frames. Interim text refreshes roughly every 900 ms; an utterance is committed after ~800 ms of silence. |
| **Multilingual** | Language detection is left to the model per utterance, so "कल की मीटिंग postpone हो गई, so please reschedule it" stays in both languages instead of being flattened into one. |
| **Open source STT** | Default engine is [faster-whisper](https://github.com/SYSTRAN/faster-whisper) running in-process. No API key, no third-party service, nothing leaves the host. |
| **Persistent** | Transcripts and their timed segments go into PostgreSQL via SQLAlchemy. |
| **CRUD dashboard** | Search, filter by language, paginate, edit title/text inline, delete, export CSV. |
| **Text-to-speech** | Each stored transcript can be played back (`SpeechSynthesis`), covering the brief's optional TTS note. |

---

## Quick start

### Option A — Docker (Postgres included)

```bash
git clone <your-repo-url> && cd procucev-realtime-transcription
docker compose up --build
```

Open <http://localhost:8000>. The first build downloads the Whisper weights
(~500 MB for `small`) into a named volume, so later starts are fast.

### Option B — Local dev (two terminals)

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # defaults to SQLite; no database to install
uvicorn app.main:app --reload --port 8000
```

```bash
# frontend
cd frontend
npm install
npm run dev                   # http://localhost:5173, proxies /api + /ws
```

The Whisper model loads lazily on the first utterance — expect a few seconds
once, then real-time results.

> Chrome or Edge is recommended. The app needs `AudioWorklet` and
> `getUserMedia`, and microphone access requires `localhost` or HTTPS.

---

## Choosing an STT engine

Set `STT_ENGINE` in `backend/.env`, or pick it from the dropdown in the UI
(the query parameter overrides the server default per connection).

| Engine | What it is | When to use it |
|---|---|---|
| `whisper` **(default)** | faster-whisper / CTranslate2, running on your own server. Open source, offline. | The real answer to the brief. Needs ~1–2 GB RAM for `small`. |
| `browser` | The browser's own Web Speech API. Recognition happens client-side; the server only persists the text. | Free-tier hosting that cannot hold a model in memory. Keeps the deployed demo usable. |
| `gemini` | Gemini Live API (`gemini-3.1-flash-live-preview`), as named in the brief. Needs `GEMINI_API_KEY`. | Real-time (rtf ~0.6) and needs no RAM on the host, so it is what makes a free-tier deploy work. Not self-hosted and not open source. |

### Model size (`WHISPER_MODEL`, default `auto`)

Measured on an M-series Mac with a Hindi/English code-switch clip. Real-time
factor is `decode time / audio length` — anything above 1.0 cannot keep up
with live speech.

| Model | CPU (int8) | Metal (float16) | Code-switching quality |
|---|---|---|---|
| `small` | 0.46 | 0.16 | Poor — transliterates English into Devanagari |
| `large-v3-turbo` | 2.17 ✗ | **0.42** | Correct — keeps each language in its own script |

Same sentence, same clip:

```
small           अगर की मेटिंग पोस्पोन होगगे है, सो प्लीस रिषे जुल इप फोर तमरो नोनिग
large-v3-turbo  कल की meeting postpone हो गई है, so please reschedule it for tomorrow morning.
```

Because neither model wins on both axes, `auto` picks per device:
`large-v3-turbo` on CUDA/Metal, `small` on CPU-only. Override with
`WHISPER_MODEL=small` (or `medium`, `large-v3`) and `WHISPER_DEVICE`.

---

## API

Interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/engine` | Active engine, model, readiness |
| `GET` | `/api/stats` | Totals and per-language counts |
| `POST` | `/api/transcriptions` | **Create** |
| `GET` | `/api/transcriptions` | **Read** — `?search=&language=&limit=&offset=` |
| `GET` | `/api/transcriptions/{id}` | **Read** one |
| `PATCH` | `/api/transcriptions/{id}` | **Update** title / text / language |
| `DELETE` | `/api/transcriptions/{id}` | **Delete** (cascades to segments) |
| `GET` | `/api/transcriptions/export` | CSV download |
| `WS` | `/ws/transcribe` | Streaming transcription |

### WebSocket protocol

```jsonc
// client → server
{"type": "config", "language": "auto", "title": "..."}   // optional
<binary>                          // 16 kHz mono little-endian int16 PCM
{"type": "flush"}                 // end the current utterance now
{"type": "stop"}                  // end utterance, save, return the record

// server → client
{"type": "ready",   "engine": "whisper", "model": "small"}
{"type": "partial", "text": "kal ki meeting"}             // replaces previous
{"type": "final",   "segment": {...}, "text": "full transcript so far"}
{"type": "saved",   "transcription": {...}}
{"type": "error",   "message": "..."}
```

`curl` example:

```bash
curl -X POST localhost:8000/api/transcriptions \
  -H 'Content-Type: application/json' \
  -d '{"title":"Demo","text":"kal ki meeting postpone ho gayi","language":"mixed"}'
```

---

## Tests

```bash
cd backend && .venv/bin/python -m pytest tests -q
```

53 tests covering the CRUD surface, utterance segmentation (VAD thresholds,
leading-silence trimming, forced finalisation), device/model auto-selection,
and the full WebSocket handshake against stubbed recognisers — including a
deliberately slow engine, which pins the fix for the decode backlog
described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

The same suite runs against PostgreSQL:

```bash
DATABASE_URL=postgresql+psycopg://postgres@localhost:5432/transcriptions \
  .venv/bin/python -m pytest tests -q
```

---

## Project layout

```
backend/
  app/
    main.py            FastAPI app; serves the built UI in production
    config.py          all tuning constants, env-driven
    models.py          Transcription + Segment (SQLAlchemy 2.0 typed)
    crud.py            query layer
    audio.py           PCM decode, RMS gate, utterance segmentation
    routers/
      transcriptions.py  REST CRUD
      ws.py              streaming endpoint
    stt/
      base.py            ASREngine protocol
      faster_whisper_engine.py
      gemini_engine.py
      browser_engine.py
  tests/
frontend/
  src/
    hooks/useTranscriber.ts   mic capture, AudioWorklet, WebSocket
    components/               Recorder, Dashboard, cards, edit modal
docs/ARCHITECTURE.md          design notes and trade-offs
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for why real-time Whisper
is done with a rolling re-transcribe loop, and how code-switching is kept
intact.

---

## Deploying

The `Dockerfile` is a two-stage build: it compiles the React app, then
serves it as static files from the same FastAPI container — one service, one
port, no CORS.

```bash
docker build -f backend/Dockerfile -t transcription .
docker run -p 8000:8000 -e DATABASE_URL=<postgres-url> transcription
```

### Render

`render.yaml` is a blueprint: in the Render dashboard choose **New →
Blueprint**, point it at this repo, and it provisions the web service and a
PostgreSQL instance together.

It intentionally defaults to `STT_ENGINE=browser`. Recognition then happens
in the browser and the server only persists text, so the instance needs no
model in memory and the image skips the ~500 MB weight download
(`PREFETCH_MODEL=false`). This is the configuration that actually stays up
on a small instance.

To run the self-hosted Whisper engine instead:

1. move the web service to a plan with **>= 2 GB RAM** - `small` alone is
   ~1.2 GB and will OOM a 512 MB instance on the first utterance;
2. set `STT_ENGINE=whisper` and `WHISPER_MODEL=small`;
3. set the build arg `PREFETCH_MODEL=true`, so the weights are baked into
   the image rather than downloaded on the first request, which would
   otherwise time out against the health check.

Render injects `DATABASE_URL` as `postgres://...`, a scheme SQLAlchemy 2.0
no longer loads. The app rewrites it to `postgresql+psycopg://` on startup,
so the provider's variable works untouched.

---

## Known limits

- **Interim text churns before it settles.** Partials come from
  re-transcribing a growing buffer, so you can watch the model change its
  mind — including transliterating English for a pass or two before getting
  it right. The committed `final` is correct; the interim wobble is inherent
  to using Whisper (a window transcriber) for streaming instead of a
  natively-streaming transducer model.
- **CPU-only hosts get the weaker model.** `large-v3-turbo` is the only
  model that reliably preserves code-switching, and it runs at ~2.2x
  real-time on CPU — too slow to stream. On CPU the app falls back to
  `small`, which transliterates. A GPU/Metal host, or the `browser` engine,
  avoids this.
- **Whole-utterance language detection.** Whisper assigns one language per
  decode window, so a mid-sentence switch is handled by the model's own
  script choice rather than by true per-word language ID. Pausing between
  languages gives noticeably better results.
- **Text-to-speech uses the browser's `SpeechSynthesis`, not an open-source
  model.** The brief allows a third-party TTS but says an open-source model
  is preferred, so this is a deliberate deviation: it adds no deployment
  weight and works offline on the client. Swapping in Piper behind a
  `/api/tts` endpoint would satisfy the preference and is the obvious next
  step if it matters.
- The Gemini engine needs an API key whose project has **Live** access.
  Without it the handshake returns a bare `HTTP 404`; the same key's
  non-Live models are not a usable substitute (three identical runs of one
  clip returned three different scripts). See docs/ARCHITECTURE.md.
- No authentication — every visitor shares one transcript history. A real
  deployment would scope records to a user.
