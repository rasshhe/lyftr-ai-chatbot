import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class KeyEntry:
    key_id: str
    api_key: str


@dataclass(frozen=True)
class Settings:
    keys: List[KeyEntry]
    model: str
    fallback_model: str | None
    fallback_models: List[str]
    default_system_prompt: str
    database_url: str
    allowed_origins: List[str]
    enable_failure_simulation: bool


def parse_sqlite_path(database_url: str) -> str:
    if database_url == "sqlite:///:memory:" or database_url == ":memory:":
        return ":memory:"
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///"):]
    if database_url.startswith("sqlite://"):
        return database_url[len("sqlite://"):]
    return database_url


def load_settings() -> Settings:
    raw_keys = os.getenv("LLM_API_KEYS", "")
    parsed_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    key_entries = [
        KeyEntry(key_id=f"key_{idx + 1}", api_key=key_val)
        for idx, key_val in enumerate(parsed_keys)
    ]

    model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash").strip() or "gemini/gemini-2.0-flash"
    fallback_raw = os.getenv("LLM_FALLBACK_MODELS") or os.getenv("FALLBACK_MODEL") or ""
    fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()]
    fallback_model = fallback_models[0] if fallback_models else None

    default_system_prompt = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are a helpful, concise AI assistant.",
    )
    database_url = os.getenv("DATABASE_URL", "sqlite:////data/app.db").strip()

    origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    allowed_origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
    if not allowed_origins:
        allowed_origins = ["*"]

    sim_flag = os.getenv("ENABLE_FAILURE_SIMULATION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return Settings(
        keys=key_entries,
        model=model,
        fallback_model=fallback_model,
        fallback_models=fallback_models,
        default_system_prompt=default_system_prompt,
        database_url=database_url,
        allowed_origins=allowed_origins,
        enable_failure_simulation=sim_flag,
    )
