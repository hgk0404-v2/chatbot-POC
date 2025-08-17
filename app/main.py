# app/main.py
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="RAG FastAPI")

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
