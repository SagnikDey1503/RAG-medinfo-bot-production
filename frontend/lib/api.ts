import type { ChatResponse, Message } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface StreamCallbacks {
  onMeta?: (meta: { trace_id: string; citations: ChatResponse["citations"] }) => void;
  onToken?: (text: string) => void;
  onDone?: (final: ChatResponse) => void;
  onError?: (detail: string) => void;
}

function toHistory(messages: Message[]) {
  return messages
    .filter((m) => !m.error && m.content.trim())
    .map((m) => ({ role: m.role, content: m.content }));
}

/**
 * POST /chat/stream and parse the Server-Sent-Events stream.
 * Events: meta -> token* -> done (or error).
 */
export async function streamChat(
  message: string,
  history: Message[],
  cb: StreamCallbacks
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: toHistory(history) }),
    });
  } catch (e) {
    cb.onError?.(`Cannot reach backend at ${API_URL}. Is it running?`);
    return;
  }

  if (!res.ok || !res.body) {
    let detail = `Request failed (${res.status})`;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* ignore */
    }
    cb.onError?.(detail);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      let event = "message";
      let dataStr = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
      }
      if (!dataStr) continue;
      let data: any;
      try {
        data = JSON.parse(dataStr);
      } catch {
        continue;
      }
      if (event === "meta") cb.onMeta?.(data);
      else if (event === "token") cb.onToken?.(data.text ?? "");
      else if (event === "done") cb.onDone?.(data as ChatResponse);
      else if (event === "error") cb.onError?.(data.detail || "stream error");
    }
  }
}

export async function checkHealth(): Promise<{
  pipeline_ready: boolean;
  chunks: number;
  model: string;
  web_search_enabled: boolean;
} | null> {
  try {
    const res = await fetch(`${API_URL}/health`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}
