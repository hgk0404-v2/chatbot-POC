# app/routers/ingest.py
from fastapi import APIRouter, UploadFile, File, Form
from app.services.loaders import load_any, split_docs
from app.services.embeddings import get_embeddings
from app.services.vectorstore import FaissStore

router = APIRouter()

@router.post("/ingest")
async def ingest(files: list[UploadFile] = File(...), tags: str = Form("[]"), tenant_id: str | None = Form(None)):
    all_docs = []
    for f in files:
        path = f"data/{f.filename}"
        with open(path, "wb") as w:
            w.write(await f.read())
        docs = load_any(path)
        # 메타데이터 확장
        for d in docs:
            d.metadata.update({"tenant_id": tenant_id, "tags": tags})
        all_docs.extend(split_docs(docs))
    embeddings = get_embeddings()
    store = FaissStore(embeddings)
    try:
        store.load_or_create(all_docs)
    except RuntimeError:
        store.load_or_create(all_docs)  # 최초 생성
    else:
        store.upsert(all_docs)
    return {"job_id": "immediate", "count": len(all_docs)}
