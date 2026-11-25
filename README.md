# Lyftr AI Chatbot

Full-stack AI chatbot service with a FastAPI + LiteLLM backend, SQLite session persistence, response caching, key/model failover, and a web frontend served via Docker Compose.

## Architecture Overview

- **Backend (`backend/app/main.py`, `config.py`, `llm_service.py`, `db.py`, `schemas.py`):** Exposes `/health`, `/chat`, and `/sessions/{session_id}/messages`. Routes completions through `LLMGateway` using LiteLLM across configured API keys and fallback models.
- **Frontend (`frontend/`):** Builds a static bundle and serves it with Nginx on port `3000`.
- **Storage:** Persists sessions and messages in SQLite (`DATABASE_URL`, defaulting to `/data/app.db` inside the backend container) and caches identical prompt/model completions in memory.

## Local Setup & Running with Docker Compose

1. Create a `.env` file in the project root with `LLM_API_KEYS` (comma-separated), `LLM_MODEL`, `LLM_FALLBACK_MODELS`, `DATABASE_URL`, and `ALLOWED_ORIGINS`.
2. Start the backend and frontend containers:
   `docker compose up -d --build`
3. Verify health at `http://localhost:8000/health` and open the frontend at `http://localhost:3000`.

## API Endpoints

- `GET /health` — Returns service status, configured key count (`keys_configured`), and cache status (`cache_enabled`).
- `POST /chat` — Accepts `session_id` and `message` (optional `X-Simulate-Failure` header when failure simulation is enabled), runs or caches the completion, saves both messages, and returns `session_id`, `reply`, and `metadata`.
- `GET /sessions/{session_id}/messages` — Returns the persisted conversation history for `session_id`.

## Key Failover & Response Caching

- **Multi-key & fallback model routing:** `LLMGateway` iterates through candidate models (`LLM_MODEL` followed by `LLM_FALLBACK_MODELS`) and configured keys (`key_1`, `key_2`, ...), tracking `fallback_attempts` and reporting the winning `key_id` and `model` in response metadata.
- **Failure simulation:** When `ENABLE_FAILURE_SIMULATION` is enabled, `X-Simulate-Failure` (for example `key_1=429`) simulates provider errors on specific keys to verify failover.
- **Response caching:** Identical requests hit the cache (`resolution_source: "LLM_CACHE"`, `fallback_attempts: 0`) with low latency.

## Persistence & Session History

- Every `/chat` request appends the user message and assistant reply (along with response metadata) to SQLite so `GET /sessions/{session_id}/messages` reconstructs prior turns across requests and container restarts when `/data` is volume-mounted.

## Deployment

- Run the stack locally with `docker compose up -d --build`, or deploy the `backend` and `frontend` services on your preferred container/cloud host with `LLM_API_KEYS`, `LLM_MODEL`, `LLM_FALLBACK_MODELS`, `DATABASE_URL`, and `ALLOWED_ORIGINS` configured in the environment.
