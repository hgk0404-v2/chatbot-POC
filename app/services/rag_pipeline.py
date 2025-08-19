# app/services/rag_pipeline.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch

CTX_SYSTEM = """당신은 RAG 어시스턴트입니다.
- '컨텍스트' 근거로만 1~2문장 답변.
- 컨텍스트에 없으면 “모르겠습니다.”만 말함.
- 반복/자문자답 금지.
"""

NOCTX_SYSTEM = """당신은 일반 상식 Q&A 어시스턴트입니다.
- 1~2문장 간결 답변.
- 불확실하면 “모르겠습니다.”.
"""

def build_chain(*, retriever, llm):
    prompt_ctx = ChatPromptTemplate.from_messages([
        ("system", CTX_SYSTEM),
        ("human", "질문: {question}\n\n컨텍스트:\n{context}")
    ])
    prompt_noctx = ChatPromptTemplate.from_messages([
        ("system", NOCTX_SYSTEM),
        ("human", "질문: {question}")
    ])

    def fetch_context(inputs, callbacks=None):  # ← callbacks 받도록
        q = inputs["question"]
        docs = retriever.invoke(q)
        ctx = "\n\n".join(d.page_content for d in docs[:5])  # ← 반드시 'context' 문자열 생성
        # has_ctx = bool(ctx.strip())
        has_ctx = len(docs) > 0

        print("🔎 [RAG DEBUG] Question:", q)
        print(f"🔎 [RAG DEBUG] Retrieved docs: {len(docs)} has_ctx: {has_ctx}")
        if has_ctx:
            print("🔎 [RAG DEBUG] Sources:")
            for i, d in enumerate(docs, 1):
                source = d.metadata.get("source", "unknown")
                print(f"   {i}. source={source}")

        return {"question": q, "context": ctx, "has_ctx": has_ctx}  # ← 프롬프트 키와 일치

    # langchain-core 0.3.74: default= 키워드 없음 → 마지막 인자가 default Runnable
    branch = RunnableBranch(
        (lambda x: x["has_ctx"], prompt_ctx | llm),
        (prompt_noctx | llm),   # default 분기
    )

    return (
        RunnableLambda(fetch_context)
        | branch
        | RunnableLambda(lambda x: x.content if hasattr(x, "content") else str(x))
    )
