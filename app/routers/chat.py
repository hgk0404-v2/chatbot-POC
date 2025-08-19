# app/routers/chat.py
import asyncio, time, logging
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from langchain.callbacks import AsyncIteratorCallbackHandler

router = APIRouter()

@router.get("/stream")  # 최종 경로는 /api/chat/stream (main.py + __init__.py prefix)
async def chat_stream(request: Request, session_id: str, message: str):
    chain = request.app.state.chain
    logger = logging.getLogger("uvicorn.error")

    cb = AsyncIteratorCallbackHandler()
    started = time.monotonic()
    first = True

    # ★ callbacks를 체인에 넘겨야 토큰이 흘러나옴
    task = asyncio.create_task(
        chain.ainvoke(
            {"question": message, "session_id": session_id},
            config={"callbacks": [cb]},
        )
    )

    async def gen():
        nonlocal first
        try:
            async for token in cb.aiter():
                if first:
                    first = False
                    logger.info(f"⏱ first token = {time.monotonic()-started:.2f}s")
                yield {"event": "token", "data": token}
            await task  # 에러 전파/완료 대기
            logger.info(f"⏱ total = {time.monotonic()-started:.2f}s")
            yield {"event": "done", "data": "[DONE]"}
        except Exception as e:
            logger.exception("stream error")
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(gen())
