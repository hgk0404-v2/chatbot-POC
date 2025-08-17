# app/routers/chat.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from app.services.rag_pipeline import build_chain
from app.schemas.dto import ChatRequest, ChatResponse

router = APIRouter()
run_chain = build_chain()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    stream, citations = run_chain(req.message)
    # 스트리밍이 아니면 한 번에 합치기
    text = "".join([chunk for chunk in stream])
    return ChatResponse(answer=text, citations=citations, usage={})

@router.get("/chat/stream")
async def chat_stream(message: str):
    stream, citations = run_chain(message)
    async def gen():
        yield {"event": "citations", "data": citations}
        for chunk in stream:
            yield {"event": "token", "data": chunk}
        yield {"event": "done", "data": ""}
    return EventSourceResponse(gen(), media_type="text/event-stream")

