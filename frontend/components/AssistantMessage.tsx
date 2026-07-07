import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "@/lib/types";

function confidencePct(m: Message) {
  return m.confidence != null ? `${Math.round(m.confidence * 100)}%` : "";
}

export default function AssistantMessage({ m }: { m: Message }) {
  return (
    <div className="row assistant">
      <div className={`bubble${m.error ? " error" : ""}`}>
        <div className="md">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content || ""}</ReactMarkdown>
          {m.streaming && <span className="cursor" />}
        </div>
      </div>

      {!m.streaming && !m.error && (m.confidence != null || m.citations?.length) ? (
        <div className="meta">
          {m.confidence != null && (
            <span className={`badge ${m.confidenceLabel}`}>
              {m.confidenceLabel} confidence · {confidencePct(m)}
            </span>
          )}
          {m.usedWeb && <span className="badge web">web result</span>}
          {m.citations?.length ? (
            <span className="badge">{m.citations.length} sources</span>
          ) : null}
        </div>
      ) : null}

      {!m.streaming && m.citations && m.citations.length > 0 && (
        <details>
          <summary>Sources ({m.citations.length})</summary>
          <div className="disclosure-body">
            {m.citations.map((c) => (
              <div className="source" key={c.id}>
                <div className="src-head">
                  <span className="cid">[{c.id}]</span>
                  <span className="src-name">{c.label}</span>
                  {c.origin === "web" && <span className="tag-web">web</span>}
                </div>
                <div className="snippet">{c.snippet}</div>
              </div>
            ))}
          </div>
        </details>
      )}

      {!m.streaming && m.trace && m.trace.length > 0 && (
        <details>
          <summary>Details</summary>
          <div className="disclosure-body">
            {m.trace.map((s, i) => (
              <div className="trace-step" key={i}>
                <span className="tname">{s.name}</span>
                <span className="tdetail">{s.detail}</span>
                {s.duration_ms != null && (
                  <span className="tms">{Math.round(s.duration_ms)}ms</span>
                )}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
