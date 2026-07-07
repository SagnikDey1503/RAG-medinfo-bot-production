"use client";

import { useEffect, useRef, useState } from "react";
import { checkHealth, streamChat } from "@/lib/api";
import type { Message } from "@/lib/types";
import AssistantMessage from "@/components/AssistantMessage";

const SAMPLES = [
  "What are the symptoms of anemia?",
  "How is type 2 diabetes managed?",
  "What causes migraines and how are they treated?",
];

interface Health {
  pipeline_ready: boolean;
  chunks: number;
  model: string;
  web_search_enabled: boolean;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    checkHealth().then((h) => setHealth(h as Health | null));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  function autoGrow() {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }

  async function send(text: string) {
    const q = text.trim();
    if (!q || busy) return;

    const history = messages;
    const userMsg: Message = { role: "user", content: q };
    const assistantMsg: Message = { role: "assistant", content: "", streaming: true };
    setMessages((m) => [...m, userMsg, assistantMsg]);
    setInput("");
    setBusy(true);
    if (taRef.current) taRef.current.style.height = "auto";

    const patchLast = (patch: Partial<Message>) =>
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { ...copy[copy.length - 1], ...patch };
        return copy;
      });

    await streamChat(q, history, {
      onMeta: (meta) => patchLast({ citations: meta.citations }),
      onToken: (t) =>
        setMessages((m) => {
          const copy = [...m];
          const last = copy[copy.length - 1];
          copy[copy.length - 1] = { ...last, content: last.content + t };
          return copy;
        }),
      onDone: (final) =>
        patchLast({
          content: final.answer,
          streaming: false,
          citations: final.citations,
          confidence: final.confidence,
          confidenceLabel: final.confidence_label,
          usedWeb: final.used_web_search,
          healAttempts: final.heal_attempts,
          trace: final.trace,
        }),
      onError: (detail) =>
        patchLast({ content: detail, streaming: false, error: true }),
    });

    setBusy(false);
  }

  const notReady = health && !health.pipeline_ready;

  return (
    <div className="app">
      <header className="header">
        <h1>MedBot</h1>
        <div className="status">
          <span className={`dot ${health ? (health.pipeline_ready ? "ok" : "bad") : ""}`} />
          {health
            ? health.pipeline_ready
              ? "online"
              : "index not built"
            : "connecting…"}
        </div>
      </header>

      <div className="messages" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="empty">
            <h2>Ask a question</h2>
            {notReady && (
              <p style={{ color: "var(--yellow)", marginTop: 14 }}>
                Backend index isn't built yet. Set <code>OPENAI_API_KEY</code> and run{" "}
                <code>python -m rag.ingest</code>, then restart the backend.
              </p>
            )}
            <div className="chips">
              {SAMPLES.map((s) => (
                <button key={s} className="chip" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) =>
            m.role === "user" ? (
              <div className="row user" key={i}>
                <div className="bubble">{m.content}</div>
              </div>
            ) : (
              <AssistantMessage m={m} key={i} />
            )
          )
        )}
      </div>

      <div className="composer">
        <textarea
          ref={taRef}
          value={input}
          placeholder="Ask a question…"
          rows={1}
          onChange={(e) => {
            setInput(e.target.value);
            autoGrow();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          disabled={busy}
        />
        <button onClick={() => send(input)} disabled={busy || !input.trim()}>
          {busy ? "…" : "Send"}
        </button>
      </div>

      <div className="disclaimer">
        For informational purposes only — not a substitute for professional medical
        advice, diagnosis, or treatment. Always consult a qualified healthcare
        provider with questions about a medical condition.
      </div>
    </div>
  );
}
