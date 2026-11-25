import React, { useEffect, useRef, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
  /\/+$/,
  ""
);

function createSessionId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(36).slice(2, 10)}`;
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => createSessionId());
  const [systemPrompt, setSystemPrompt] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorBanner, setErrorBanner] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    setSessionId(createSessionId());
    setMessages([]);
    setErrorBanner("");
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleNewSession = () => {
    setSessionId(createSessionId());
    setMessages([]);
    setErrorBanner("");
  };

  const handleSend = async (e) => {
    if (e) e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    setInput("");
    setErrorBanner("");
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setLoading(true);

    const body = {
      session_id: sessionId,
      message: trimmed,
    };
    if (systemPrompt.trim()) {
      body.system_prompt = systemPrompt.trim();
    }

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        const detailText =
          typeof data?.detail === "string"
            ? data.detail
            : Array.isArray(data?.detail)
            ? data.detail.map((item) => item.msg || JSON.stringify(item)).join("; ")
            : `Request failed with status ${res.status}`;

        setErrorBanner(detailText);
        setMessages((prev) => [
          ...prev,
          {
            role: "error",
            content: detailText,
          },
        ]);
        return;
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.reply,
          metadata: data.metadata,
        },
      ]);
    } catch (err) {
      const msg = err?.message || "Network error while contacting backend";
      setErrorBanner(msg);
      setMessages((prev) => [
        ...prev,
        {
          role: "error",
          content: msg,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Lyftr AI Chatbot</h1>
          <p className="subtitle">
            Session: <code>{sessionId}</code>
          </p>
        </div>
        <button type="button" className="secondary-btn" onClick={handleNewSession}>
          New Session
        </button>
      </header>

      <section className="prompt-bar">
        <label htmlFor="system-prompt">System Prompt (optional)</label>
        <input
          id="system-prompt"
          data-testid="system-prompt-input"
          type="text"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="e.g., Answer like a concise senior engineer..."
        />
      </section>

      {errorBanner && (
        <div className="error-banner" role="alert">
          {errorBanner}
        </div>
      )}

      <main className="chat-window">
        {messages.length === 0 && (
          <div className="empty-state">
            Start a conversation. Each reply includes cache resolution, model, latency, and
            per-request token usage.
          </div>
        )}

        {messages.map((msg, idx) => {
          if (msg.role === "user") {
            return (
              <div key={idx} className="message user-message">
                <div className="bubble">{msg.content}</div>
              </div>
            );
          }

          if (msg.role === "error") {
            return (
              <div key={idx} className="message error-message" role="alert">
                <div className="bubble">{msg.content}</div>
              </div>
            );
          }

          const meta = msg.metadata || {};
          return (
            <div key={idx} className="message assistant-row">
              <div className="bubble assistant-card" data-testid="assistant-message">
                <div className="reply-text">{msg.content}</div>
                <div className="meta-strip">
                  <span className="meta-chip" data-testid="meta-source">
                    {meta.resolution_source}
                  </span>
                  <span className="meta-chip" data-testid="meta-model">
                    {meta.model}
                  </span>
                  <span className="meta-chip" data-testid="meta-latency">
                    {meta.latency_ms} ms
                  </span>
                  <span className="meta-chip">
                    In: <strong data-testid="meta-input-tokens">{meta.input_tokens}</strong>
                  </span>
                  <span className="meta-chip">
                    Out: <strong data-testid="meta-output-tokens">{meta.output_tokens}</strong>
                  </span>
                </div>
              </div>
            </div>
          );
        })}

        {loading && (
          <div className="message assistant-row">
            <div className="bubble assistant-card">Thinking...</div>
          </div>
        )}

        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={handleSend}>
        <textarea
          data-testid="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message and press Enter..."
          rows={2}
        />
        <button
          type="submit"
          data-testid="send-button"
          disabled={loading || !input.trim()}
        >
          Send
        </button>
      </form>
    </div>
  );
}
