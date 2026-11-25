from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import load_settings, parse_sqlite_path
from app.db import SQLiteStore
from app.llm_service import LLMGateway
from app.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionMessagesResponse,
)


def create_app() -> FastAPI:
    settings = load_settings()
    db_path = parse_sqlite_path(settings.database_url)
    store = SQLiteStore(db_path=db_path)
    gateway = LLMGateway(
        keys=settings.keys,
        default_model=settings.model,
        fallback_model=settings.fallback_model,
            fallback_models=settings.fallback_models,
        enable_failure_simulation=settings.enable_failure_simulation,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        store.close()

    app = FastAPI(
        title="Lyftr AI LLM Chatbot Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.store = store
    app.state.gateway = gateway

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        safe_errors = []
        for err in exc.errors():
            cleaned = dict(err)
            if "ctx" in cleaned and isinstance(cleaned["ctx"], dict):
                cleaned["ctx"] = {
                    k: (str(v) if isinstance(v, Exception) else v)
                    for k, v in cleaned["ctx"].items()
                }
            safe_errors.append(cleaned)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": safe_errors},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, __: Exception):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "all providers unavailable"},
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        db_ok = app.state.store.ping()
        keys_count = len(app.state.gateway.keys)
        if not db_ok or keys_count < 1:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="all providers unavailable",
            )
        return HealthResponse(
            status="ok",
            keys_configured=keys_count,
            cache_enabled=app.state.gateway.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        if payload.system_prompt is not None and payload.system_prompt.strip():
            app.state.store.set_session_prompt(payload.session_id, payload.system_prompt.strip())

        system_prompt = app.state.store.get_session_prompt(
            payload.session_id,
            app.state.settings.default_system_prompt,
        )
        prior_history = app.state.store.get_conversation_messages(payload.session_id)

        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(prior_history)
        full_messages.append({"role": "user", "content": payload.message})

        try:
            reply_text, meta_dict = app.state.gateway.generate_reply(
                messages=full_messages,
                simulate_failure_header=x_simulate_failure,
            )
        except Exception:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "all providers unavailable"},
            )

        app.state.store.append_turn(
            session_id=payload.session_id,
            user_message=payload.message,
            assistant_reply=reply_text,
            assistant_metadata=meta_dict,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=reply_text,
            metadata=ChatMetadata(**meta_dict),
        )

    @app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
    def get_session_messages(session_id: str) -> SessionMessagesResponse:
        cleaned_id = session_id.strip()
        if not cleaned_id or len(cleaned_id) > 128:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="session not found",
            )

        messages = app.state.store.get_session_history(cleaned_id)
        if messages is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "session not found"},
            )

        return SessionMessagesResponse(
            session_id=cleaned_id,
            messages=messages,
        )

    return app


app = create_app()
