# app/routers/search.py
from fastapi import APIRouter, Query
from app.services.embeddings import get_embeddings
from app.services.vectorstore import FaissStore
from app.core.config import settings

router = APIRouter()

@router.get("/search")
async def search(q: str, top_k: int = 5):
    store = FaissStore(get_embeddings())
    store.load_or_create()
    retriever = store.as_retriever(k=top_k)
    docs = retriever.invoke(q)
    hits = [{"score": None, "text": d.page_content, "metadata": d.metadata} for d in docs]
    return {"hits": hits}
