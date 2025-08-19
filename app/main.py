# app/main.py
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.routers import api_router
from app.services.vectorstore import load_faiss
from app.services.local_model import load_llama
from app.services.rag_pipeline import build_chain
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="RAG FastAPI")
app.logger = logger

@app.on_event("startup")
async def startup():
    # 1) 무거운 것 전부 1회 로딩
    vs = load_faiss(path="index/faiss")           # 디스크 → 메모리
    retriever = vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.2,  # 0.2~0.4 사이에서 조정
            "k": 5
        },
    )
    llm = load_llama(  # llama-cpp-python 예시
        model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx=4096, 
        n_batch=512, 
        n_threads=8, 
        n_gpu_layers=999,
    )
    chain = build_chain(retriever=retriever, llm=llm)  # ★ 여기서만 조립

    # 2) state에 “객체” 그대로 보관
    app.state.vectorstore = vs
    app.state.retriever = retriever
    app.state.llm = llm
    app.state.chain = chain

    # (선택) 디버그: 동일 객체 재사용 여부 확인
    app.logger.info(f"CHAIN_ID={id(chain)}, LLM_ID={id(llm)}, RETR_ID={id(retriever)}")
    logger = logging.getLogger("uvicorn.error")
    logger.info("✅ [STARTUP] FAISS, Retriever, LLM, Chain loaded")
    logger.info(f"   IDs -> LLM:{id(llm)}, Retriever:{id(retriever)}, Chain:{id(chain)}")
    print("🟢 Application initialize")
    print("🟢 build_chaining_start")


app.include_router(api_router, prefix="/api")

# 실제 위치: /app/static/web
ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "static" / "web"

# 👉 /web → /app/static/web 로 서빙 (index.html 자동)
app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# 루트 접근 시 /web/index.html로 이동
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/web/index.html")

# 파비콘(선택)
FAVICON = WEB_DIR / "favicon.ico"
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(str(FAVICON), media_type="image/x-icon") if FAVICON.exists() else ("", 204)

@app.on_event('shutdown')
async def shutdown_event():
    print("🔴 Application shutdown")

# main.py 맨 아래에 테스트용 추가
@app.get("/debug/retriever")
async def debug_retriever():
    retriever = getattr(app.state, "retriever", None)
    if retriever is None:
        return {"error": "retriever not loaded"}
    docs = retriever.get_relevant_documents("강형근")
    return {
        "docs": [
            {
                "metadata": d.metadata,
                "content": d.page_content
            }
            for d in docs
        ]
    }
