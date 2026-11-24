# Architecture

```
browser                          server                        storage
───────                          ──────                        ───────
getUserMedia 16 kHz
      │
AudioWorklet  ── float32 → int16, batched to ~128 ms
      │
      │  WebSocket (binary PCM frames)
      ▼
                        /ws/transcribe
                              │
                        UtteranceBuffer
                        · RMS energy gate
                        · drops leading silence
                        · every 900 ms  → partial
                        · 800 ms silence → final
                              │
                        ASREngine.transcribe()   (thread pool)
                        · faster-whisper  (default, self-hosted)
                        · Gemini Live     (optional)
                        · browser         (pass-through)
                              │
                     ┌────────┴────────┐
              {"type":"partial"}   SQLAlchemy ──► PostgreSQL
              {"type":"final"}                   transcriptions
              {"type":"saved"}                   segments
                              │
                        /api/transcriptions  (CRUD) ──► dashboard
```

## Why this shape

**Chunked streaming, not one long model call.** Whisper is an
offline-style encoder/decoder: it transcribes a window of audio, it does not
emit tokens as you speak. To get real-time behaviour out of it, the server
keeps a rolling buffer of the current utterance and re-transcribes it every
~900 ms. Each pass replaces the interim text on screen, which is what makes
the transcript look like it is being typed live. When the energy gate sees
~800 ms of silence, the utterance is transcribed once more, committed as a
`final` segment, and the buffer is cleared.

**Energy-based segmentation on the server.** The client stays dumb: it only
captures and forwards audio. Putting the VAD server-side means the browser,
a mobile client or a CLI tool all get identical segmentation, and the tuning
constants live in one place (`backend/app/config.py`).

**Ingestion and decoding are separate tasks.** This is the one piece of
the design that had to be rebuilt after measuring it. The first version
decoded inline in the receive loop, guarded by a "busy" flag. That looks
sufficient and is not: the flag prevents two decodes *overlapping*, but not
a backlog forming. Because an interim pass re-decodes the whole utterance,
each pass is more expensive than the last, and while one runs, more frames
arrive — so the next pass is immediately due again. Audio frames queue in
the socket, and a `stop` sent by the client sits behind all of them.

Measured on a 5-second clip: six interim decodes, the last finishing
**39 seconds** after the audio started. The client timed out before its
recording was ever saved. "Real-time" was false under exactly the condition
that matters.

The fix is a single decoder task per connection:

- the **receive loop** only buffers frames and sets flags, so it always
  drains the socket and reacts to `stop` or `flush` immediately;
- the **worker** loops over the *current* buffer state. Because it re-reads
  that state after each decode rather than consuming a queue, interim passes
  that went stale while it was busy are skipped instead of piling up;
- the interim interval **adapts**: the next pass needs at least
  `max(partial_interval_ms, duration of the last decode)` of new audio, so a
  slow host degrades to fewer interim updates rather than to a backlog;
- `stop` sets a flag and waits on a drain event, so the final utterance is
  always committed before the record is saved.

After the change the same clip finishes in ~10 seconds end to end
(including the save) and saves reliably. `tests/test_websocket.py` pins this
with a deliberately slow stub engine: it asserts the save still completes
and that interim decodes are coalesced rather than queued.

**Three engines behind one interface.** `ASREngine` takes float32 PCM and
returns text plus a language. That is the entire contract, which is what
lets the same WebSocket loop serve self-hosted Whisper, Gemini Live, or
browser-side Web Speech recognition. It also makes the trade-off explicit:
Whisper is the open-source default asked for in the brief, and the browser
engine is what keeps a free-tier deployment usable when there is not enough
RAM to hold a model.

## Handling Hindi–English code-switching

This is the part of the brief that actually decides whether the app works,
so the settings below were measured rather than guessed. Test clips were
built with macOS `say` — one Hindi voice (Lekha) reading a Hinglish
sentence, and one clip splicing an English voice (Rishi) to a Hindi voice so
each language is spoken natively. Everything was run on an M-series CPU and
on Metal.

**1. Model size is the single biggest factor — much bigger than any flag.**

Input: *"कल की meeting postpone हो गई है, so please reschedule it for
tomorrow morning."*

| Model | Output |
|---|---|
| `small` | `अगर की मेटिंग पोस्पोन होगगे है, सो प्लीस रिषे जुल इप फोर तमरो नोनिग` |
| `large-v3-turbo` | `कल की meeting postpone हो गई है, so please reschedule it for tomorrow morning.` |

`small` transliterates the English half into Devanagari — it is technically
a transcript and practically useless. `large-v3-turbo` returns the sentence
exactly as spoken, each language in its own script.

The catch is speed. Real-time factor (lower is better; >1.0 cannot keep up):

| | `small` | `large-v3-turbo` |
|---|---|---|
| CPU (int8) | **0.46** | 2.17 |
| Metal / MPS (float16) | **0.16** | **0.42** |

So there is no single best default, and `WHISPER_MODEL=auto` resolves it per
device: `large-v3-turbo` on CUDA or Metal, `small` on CPU-only. Note also
that CTranslate2's Metal backend is ~2.8x its CPU path on the same model, so
`WHISPER_DEVICE=auto` probes for `cuda`, then `mps`, then falls back to CPU.

**2. No forced language.** `language=auto` passes `None` to Whisper so it
detects per utterance rather than being pinned. Forcing `hi` transliterates
English into Devanagari; forcing `en` mangles or drops the Hindi.

**3. `task="transcribe"`, never `translate`.** The translate task silently
turns Hindi into English, which looks like a working transcript and is not
one.

**4. No `initial_prompt` — this one was a trap.** A Hinglish primer
("Yeh ek Hindi aur English mixed conversation hai…") is the obvious thing to
reach for, and it actively destroys output. Whisper conditions on the
prompt, so a prompt written mostly in English biases decoding toward English
and the decoder emits an end-of-segment at the language boundary instead of
switching scripts. Measured on `small`:

| Clip | With Hinglish prompt | No prompt |
|---|---|---|
| English→Hindi splice | 1 segment, 6 words (Hindi half lost) | 2 segments, 15 words |
| Hindi + English | **0 segments, 0 words** | 1 segment, 11 words |

A prompt cost an entire utterance. The default is now empty, and the setting
is kept only for what it is genuinely good for: domain vocabulary such as
names and product jargon.

**5. Segmentation helps code-switching for free.** Because utterances are
cut on silence, a speaker who pauses between languages gets a fresh language
detection per utterance. Per-segment languages are stored individually and
rolled up to `mixed` on the parent record when more than one appears, so the
dashboard can show that a recording was genuinely bilingual rather than
picking a winner.

### Gemini Live, and three things that make it fail silently

The brief names "gemini live 3.1", so the `gemini` engine targets
`gemini-3.1-flash-live-preview`. Getting it working needed three
non-obvious fixes, each of which fails in a way that looks like something
else:

1. **The Live endpoint is not on the default API version.** Constructing the
   client without `http_options={"api_version": "v1beta"}` makes the
   WebSocket handshake fail with a bare `HTTP 404`, which reads like a wrong
   model name. The model is fine; the route is not there on the default
   version.
2. **The transcript is not the model's reply.** A Live model answers with
   *audio*. Asking it for `response_modalities: ["TEXT"]` is rejected
   outright (error 1007). The transcript comes from
   `input_audio_transcription` - the server-side transcription of the audio
   that was *sent in* - and the spoken reply is discarded.
3. **`turn_complete` never arrives while the model is still speaking.**
   Looping until it appears hangs until the timeout, after the transcript
   has already been delivered. The read loop therefore stops once the
   transcript stops growing.

There is also a fourth, which cost the most time: **sending the utterance as
one large blob returns no transcription at all.** The endpoint expects a
stream, so the engine feeds it 100 ms chunks exactly as the browser would.
With one blob the session simply goes quiet and times out - no error, no
partial result.

Measured once working, on the same clips as the Whisper table above:

| Clip | Latency | Real-time factor | Output |
|---|---|---|---|
| Hindi + English (one voice) | 3.19 s / 5.01 s | **0.64** | English transliterated into Devanagari |
| English -> Hindi (two voices) | 3.67 s / 6.75 s | **0.54** | Both halves kept, each in its own script |

So Gemini Live is comfortably real-time and, unlike `small`, does not drop
the second language at a switch - but it still transliterates English on
Hindi-dominant audio. It sits between `small` and `large-v3-turbo` on
quality, and it needs no RAM on the host, which is what makes a free-tier
deployment viable.

One caveat worth stating plainly: the same key's non-Live models were not a
usable substitute. `generateContent` with inline audio ran at roughly
0.7x real time on `gemini-2.5-flash`, but three identical runs of the same
clip returned three different scripts (Devanagari, romanised, and correctly
mixed). Larger models were consistent but took 33-41 s for five seconds of
audio. Live is the only Gemini path here that is both fast and stable.

### Known residual weakness

Interim results are produced by re-transcribing a growing buffer, so the
wording visibly churns before it settles. A real captured sequence:

```
कल की मीटिंग पोस्ट
कल की मीटिंग पोस्पोन हो गई है।
कल की मीटिंग पोस्पोन हो गई है, सो प्लीज रिशेट रहे हैं,
कल की मीटिंग पोस्पोन हो गई है, सो प्लीज रिशेजूल इत फॉर टमाइब।
कल की मीटिंग पोस्पोन हो गई है, so please reschedule it for tomorrow morning.
```

Note the third and fourth passes transliterating the English before the
final pass gets it right. The committed `final` is correct, but anyone
reading the interim text sees it wobble. A natively-streaming model
(RNN-T / transducer style) would not do this; Whisper fundamentally
transcribes windows, not token streams.

## Data model

```
transcriptions                      segments
──────────────                      ────────
id            uuid pk               id               uuid pk
title         varchar(200)          transcription_id fk → cascade
text          text                  text             text
language      varchar(20)           language         varchar(20)
engine        varchar(40)           start_ms         int
duration_ms   int                   end_ms           int
word_count    int                   confidence       float  (avg_logprob)
created_at    timestamptz
updated_at    timestamptz
```

`word_count` is denormalised so the dashboard's stat tiles do not need a
scan, and it is recomputed on every write that touches `text`.
