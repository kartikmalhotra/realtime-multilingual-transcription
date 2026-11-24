import { useCallback, useEffect, useRef, useState } from "react";
import { websocketUrl } from "../lib/api";
import type { EngineName, LiveSegment, Transcription } from "../lib/types";

export type Status = "idle" | "connecting" | "listening" | "saving";

export interface StartOptions {
  engine: EngineName;
  language: string;
  title: string;
}

/**
 * Converts the mic's float32 frames to 16-bit PCM and batches them into
 * ~128 ms messages. Done in a worklet so resampling and conversion never
 * block the main thread (the render quantum is only 128 samples, so sending
 * every quantum would mean ~125 WebSocket messages a second).
 */
const WORKLET_SRC = `
class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this._batch = new Int16Array(2048);
    this._n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      let s = ch[i];
      s = s < -1 ? -1 : s > 1 ? 1 : s;
      this._batch[this._n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this._n === this._batch.length) {
        const out = this._batch.slice(0);
        this.port.postMessage(out.buffer, [out.buffer]);
        this._n = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-worklet', PCMWorklet);
`;

const TARGET_SAMPLE_RATE = 16000;

// getUserMedia never rejects on its own if the permission prompt is ignored
// or dismissed, which would otherwise wedge the UI on "Connecting…".
const MIC_PERMISSION_TIMEOUT_MS = 20000;

function requestMicrophone(): Promise<MediaStream> {
  const constraints: MediaStreamConstraints = {
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  };
  return Promise.race([
    navigator.mediaDevices.getUserMedia(constraints),
    new Promise<MediaStream>((_, reject) =>
      window.setTimeout(
        () =>
          reject(
            new Error(
              "no answer to the microphone request — allow access from the " +
                "icon in the address bar, then try again"
            )
          ),
        MIC_PERMISSION_TIMEOUT_MS
      )
    ),
  ]);
}

function speechApiLocale(language: string): string {
  // Chrome's hi-IN model copes with Hinglish better than en-* does, so it is
  // the sensible choice when the user has not pinned a language.
  if (language === "en") return "en-IN";
  if (language === "hi") return "hi-IN";
  return "hi-IN";
}

function getSpeechRecognition(): any {
  const w = window as any;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function browserSpeechSupported(): boolean {
  return getSpeechRecognition() !== null;
}

export function useTranscriber(onSaved: (t: Transcription) => void) {
  const [status, setStatus] = useState<Status>("idle");
  const [interim, setInterim] = useState("");
  const [segments, setSegments] = useState<LiveSegment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [engineLabel, setEngineLabel] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const recognitionRef = useRef<any>(null);
  const startedAtRef = useRef<number>(0);
  // Bumped on every start and cancel; a start whose id is stale must not
  // flip the UI back to "listening" after the user backed out.
  const runIdRef = useRef(0);
  const savedRef = useRef(onSaved);
  savedRef.current = onSaved;

  const teardown = useCallback(() => {
    recognitionRef.current?.stop?.();
    recognitionRef.current = null;

    nodeRef.current?.disconnect();
    nodeRef.current = null;

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;

    const ws = wsRef.current;
    wsRef.current = null;
    if (ws && ws.readyState <= WebSocket.OPEN) ws.close();

    setLevel(0);
  }, []);

  useEffect(() => teardown, [teardown]);

  const openSocket = useCallback(
    (opts: StartOptions) =>
      new Promise<WebSocket>((resolve, reject) => {
        const ws = new WebSocket(
          websocketUrl({
            engine: opts.engine,
            language: opts.language,
            title: opts.title,
          })
        );
        ws.binaryType = "arraybuffer";

        const timer = window.setTimeout(
          () => reject(new Error("timed out connecting to the server")),
          8000
        );

        ws.onopen = () => {
          window.clearTimeout(timer);
          resolve(ws);
        };
        ws.onerror = () => {
          window.clearTimeout(timer);
          reject(new Error("could not reach the transcription server"));
        };

        ws.onmessage = (event) => {
          let msg: any;
          try {
            msg = JSON.parse(event.data as string);
          } catch {
            return;
          }

          switch (msg.type) {
            case "ready":
              setEngineLabel(msg.model ? `${msg.engine} · ${msg.model}` : msg.engine);
              break;
            case "partial":
              setInterim(msg.text ?? "");
              break;
            case "final":
              setInterim("");
              if (msg.segment?.text) {
                setSegments((prev) => [...prev, msg.segment as LiveSegment]);
              }
              break;
            case "saved":
              savedRef.current(msg.transcription as Transcription);
              setStatus("idle");
              teardown();
              break;
            case "error":
              setError(msg.message ?? "transcription error");
              break;
            case "info":
              break;
          }
        };

        ws.onclose = () => {
          if (wsRef.current === ws) {
            wsRef.current = null;
            setStatus("idle");
          }
        };
      }),
    [teardown]
  );

  /** Browser-engine mode: recognition runs client-side, server just stores. */
  const startBrowserRecognition = useCallback(
    (ws: WebSocket, language: string) => {
      const Recognition = getSpeechRecognition();
      if (!Recognition) {
        throw new Error("this browser has no Web Speech API; use the Whisper engine");
      }
      const recognition = new Recognition();
      recognition.lang = speechApiLocale(language);
      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.onresult = (event: any) => {
        let pending = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          const text = result[0]?.transcript ?? "";
          if (result.isFinal) {
            ws.send(
              JSON.stringify({
                type: "client_transcript",
                text,
                is_final: true,
                language: recognition.lang.startsWith("hi") ? "hi" : "en",
                duration_ms: Math.round(performance.now() - startedAtRef.current),
              })
            );
            startedAtRef.current = performance.now();
          } else {
            pending += text;
          }
        }
        if (pending) setInterim(pending);
      };

      recognition.onerror = (event: any) => {
        if (event.error === "no-speech" || event.error === "aborted") return;
        setError(`speech recognition: ${event.error}`);
      };

      // Chrome ends the session on its own after a pause; restart while live.
      recognition.onend = () => {
        if (recognitionRef.current === recognition && wsRef.current === ws) {
          try {
            recognition.start();
          } catch {
            /* already restarting */
          }
        }
      };

      recognitionRef.current = recognition;
      recognition.start();
    },
    []
  );

  /** Whisper/Gemini mode: stream raw PCM to the server. */
  const startAudioStreaming = useCallback(async (ws: WebSocket) => {
    const stream = await requestMicrophone();
    streamRef.current = stream;

    // Asking for 16 kHz lets the browser resample for us, which is both
    // faster and better quality than doing it in JS.
    const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    ctxRef.current = ctx;
    if (ctx.state === "suspended") await ctx.resume();

    const blob = new Blob([WORKLET_SRC], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }

    const node = new AudioWorkletNode(ctx, "pcm-worklet");
    nodeRef.current = node;

    node.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const pcm = new Int16Array(event.data);

      // Cheap RMS for the level meter.
      let sum = 0;
      for (let i = 0; i < pcm.length; i += 8) sum += (pcm[i] / 32768) ** 2;
      setLevel(Math.min(1, Math.sqrt(sum / (pcm.length / 8)) * 6));

      ws.send(event.data);
    };

    ctx.createMediaStreamSource(stream).connect(node);
    // Keeps the graph pulling without routing mic audio back to the speakers.
    node.connect(ctx.destination);
  }, []);

  const start = useCallback(
    async (opts: StartOptions) => {
      setError(null);
      setInterim("");
      setSegments([]);
      setStatus("connecting");
      startedAtRef.current = performance.now();
      const runId = ++runIdRef.current;

      try {
        const ws = await openSocket(opts);
        if (runId !== runIdRef.current) {
          ws.close();
          return; // cancelled while the socket was opening
        }
        wsRef.current = ws;

        if (opts.engine === "browser") {
          startBrowserRecognition(ws, opts.language);
        } else {
          await startAudioStreaming(ws);
        }
        if (runId !== runIdRef.current) return; // cancelled while awaiting the mic
        setStatus("listening");
      } catch (err) {
        if (runId !== runIdRef.current) return;
        teardown();
        setStatus("idle");
        const message =
          err instanceof DOMException && err.name === "NotAllowedError"
            ? "microphone permission denied"
            : err instanceof Error
              ? err.message
              : "could not start recording";
        setError(message);
      }
    },
    [openSocket, startAudioStreaming, startBrowserRecognition, teardown]
  );

  /** Back out of a start that has not finished connecting. */
  const cancel = useCallback(() => {
    runIdRef.current += 1;
    teardown();
    setStatus("idle");
    setInterim("");
  }, [teardown]);

  const stop = useCallback(() => {
    runIdRef.current += 1;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      teardown();
      setStatus("idle");
      return;
    }

    setStatus("saving");
    setInterim("");
    recognitionRef.current?.stop?.();
    recognitionRef.current = null;
    nodeRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());

    ws.send(JSON.stringify({ type: "stop" }));

    // The server replies with "saved"; don't hang if it never does.
    window.setTimeout(() => {
      if (wsRef.current === ws) {
        teardown();
        setStatus("idle");
      }
    }, 20000);
  }, [teardown]);

  return {
    status,
    interim,
    segments,
    error,
    level,
    engineLabel,
    start,
    stop,
    cancel,
    clearError: () => setError(null),
  };
}
