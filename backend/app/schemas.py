from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok"]
    keys_configured: int
    cache_enabled: bool


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed or len(trimmed) > 128:
            raise ValueError("session_id must be between 1 and 128 characters")
        return trimmed

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        trimmed = v.strip()
        if len(trimmed) < 1:
            raise ValueError("message cannot be empty after trimming whitespace")
        if len(trimmed) > 4000:
            raise ValueError("message must be 4000 characters or fewer after trimming whitespace")
        return trimmed


class ChatMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str] = None
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    metadata: Optional[dict] = None


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: List[HistoryMessage]


class ErrorResponse(BaseModel):
    detail: str
