# app/routers/chat.py
from fastapi import APIRouter
from fastapi import Request
from pydantic import BaseModel
from typing import Dict, List
from sse_starlette.sse import EventSourceResponse
from app.services.local_model import LocalModel
import os

router = APIRouter()

# ★ 환경변수로 간단 설정 (개발 편의)
#  BACKEND=hf  또는 BACKEND=llama_cpp
#  MODEL_PATH=models/your-hf-model  또는 models/xxx.gguf
BACKEND = os.getenv("BACKEND", "hf")
MODEL_PATH = os.getenv("MODEL_PATH", "models/your-hf-model")
_model = LocalModel(BACKEND, MODEL_PATH, max_new_tokens=512, temperature=0.2)

# 세션별 메모리(개발용 인메모리)
_SESS: Dict[str, List[Dict[str, str]]] = {}

class ChatReq(BaseModel):
    session_id: str
    message: str

@router.post("/chat")
def chat(req: ChatReq):
    sess = _SESS.setdefault(req.session_id, [])
    sess.append({"role": "user", "content": req.message})
    # 단발 응답(스트리밍 없이)
    text = "".join(_model.stream(sess))
    sess.append({"role": "assistant", "content": text})
    return {"answer": text}

@router.api_route("/chat/stream", methods=["GET", "POST"])
async def chat_stream(request: Request):
    # 1) 입력 파싱: GET은 query, POST는 json
    if request.method == "GET":
        session_id = request.query_params.get("session_id", "dev")
        message = request.query_params.get("message", "")
    else:  # POST
        body = await request.json()
        session_id = body.get("session_id", "dev")
        message = body.get("message", "")

    sess = _SESS.setdefault(session_id, [])
    sess.append({"role": "user", "content": message})

    def token_gen():
        for tok in _model.stream(sess):
            yield {"event": "delta", "data": tok}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(token_gen())
