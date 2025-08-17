# app/schemas/dto.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    usage: Dict[str, Any] = {}

class IngestRequest(BaseModel):
    tags: List[str] = []
    tenant_id: Optional[str] = None

class IngestStatus(BaseModel):
    state: Literal["queued", "running", "done", "failed"]
    counts: Dict[str, int] = {}

class SearchResponse(BaseModel):
    hits: List[Dict[str, Any]]

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: Literal["up","down"]
    comment: Optional[str] = None
