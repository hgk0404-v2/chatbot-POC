# app/routers/chat.py
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from app.services.rag_pipeline import build_chain
from app.schemas.dto import ChatRequest, ChatResponse

router = APIRouter()
run_chain = build_chain()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    stream, citations = run_chain(req.message)
    text = "".join(list(stream)) # generator → list 변환
    return ChatResponse(answer=text, citations=citations, usage={})

@router.get("/stream")
async def chat_stream(message: str):
    stream, citations = run_chain(message)
    async def gen():
        yield {"event": "citations", "data": citations}
        for chunk in stream:   # generator 그대로 사용
            yield {"event": "token", "data": chunk}
        yield {"event": "done", "data": ""}
    return EventSourceResponse(gen(), media_type="text/event-stream")

