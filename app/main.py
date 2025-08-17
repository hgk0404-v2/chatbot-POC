# app/main.py
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="RAG FastAPI")

# 0) web 정적 디렉토리 자동 탐색 (둘 중 존재하는 쪽 사용)
ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 추정: fastapi-project-v0.1/
CANDIDATES = [ROOT / "static" / "web", ROOT / "web"]
WEB_DIR = next((p for p in CANDIDATES if p.exists()), None)

if WEB_DIR is None:
    # web 폴더가 정말 없으면 루트에서 안내만 띄움
    @app.get("/", include_in_schema=False)
    def _no_web():
        return PlainTextResponse("web/ 또는 static/web/ 디렉토리가 없습니다.", status_code=200)
else:
    # 1) 정적 웹 서빙
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

    # 2) 루트(/) -> index.html 로 리다이렉트
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/web/index.html")

    # 3) 파비콘
    FAVICON = WEB_DIR / "favicon.ico"
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        if FAVICON.exists():
            return FileResponse(str(FAVICON), media_type="image/x-icon")
        return PlainTextResponse("", status_code=204)

# (기존 API 라우터 mount는 아래에 유지)
# from app.routers import chat, ingest, search
# app.include_router(chat.router)
# app.include_router(ingest.router)
# app.include_router(search.router)
