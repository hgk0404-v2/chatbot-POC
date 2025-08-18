# app/routers/__init__.py
from fastapi import APIRouter

# 각 라우터를 import
from .chat import router as chat_router
from .ingest import router as ingest_router
from .search import router as search_router
# 필요하면 추가

# 최종 api_router에 다 모음
api_router = APIRouter()
api_router.include_router(chat_router, prefix="/chat", tags=["📘 chat"])
api_router.include_router(ingest_router, prefix="/ingest", tags=["🔬 ingest"])
api_router.include_router(search_router, prefix="/search", tags=["🔍 search"])
