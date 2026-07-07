export interface Citation {
  id: string;
  label: string;
  source: string;
  page: number | null;
  snippet: string;
  origin: "corpus" | "web";
}

export interface TraceStep {
  name: string;
  detail: string;
  duration_ms: number | null;
  data: Record<string, unknown> | null;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  confidence: number;
  confidence_label: string;
  used_web_search: boolean;
  heal_attempts: number;
  trace_id: string;
  trace: TraceStep[];
}

export type Role = "user" | "assistant";

export interface Message {
  role: Role;
  content: string;
  streaming?: boolean;
  citations?: Citation[];
  confidence?: number;
  confidenceLabel?: string;
  usedWeb?: boolean;
  healAttempts?: number;
  trace?: TraceStep[];
  error?: boolean;
}
