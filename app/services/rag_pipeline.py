# app/services/rag_pipeline.py
import os
from app.core.config import settings
from app.services.embeddings import get_embeddings
from app.services.vectorstore import FaissStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp

SYSTEM_PROMPT = """당신은 한국어로 자연스럽고 간결하게 답하는 유능한 AI 어시스턴트입니다.
출력 규칙:
- 먼저 한두 문장으로 핵심 답변을 제공하세요(요약형).
- 그 다음 필요하면 한두 문단으로 추가 설명을 덧붙이되, 장황하게 반복하지 마세요.
- 제공된 참고 문서(context)는 '참고용'입니다. 참고 문서가 없으면 일반 지식으로 자연스럽게 답하세요.
- 절대 'Answer:', 'Assistant:' 같은 라벨을 반복해서 출력하지 마세요.
- '[1]', '[2]' 같은 번호 표기는 본문에 삽입하지 말고, 필요하면 맨 끝에 '참고: 파일명1, 파일명2' 형태로만 표시하세요.
- 가능한 한 자연스러운 한국어 문장으로 답하십시오.
"""

def build_chain():
    embeddings = get_embeddings()
    store = FaissStore(embeddings)
    store.load_or_create()
    retriever = store.as_retriever(k=settings.TOP_K)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        # context가 비어있어도 템플릿이 깨지지 않게 구성
        ("human", "질문: {question}\n\n(참고자료는 아래에 별도 제공됩니다.)\n\nContext:\n{context}\n\n답변:")
    ])

    # LLM 파라미터: 자연스러움 향상 위해 온도 약간 증가, top_p 사용 추천
    llm = LlamaCpp(
        model_path=settings.MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=-1,
        n_batch=512,
        temperature=0.4,   # 자연스러움 향상
        top_p=0.95,        # 다양성 허용
        streaming=True,
    )

    def format_docs(docs):
        # docs -> 긴 텍스트 하나로 합치지 말고, 간단 요약(첫 N자)만 붙이는 방식
        if not docs:
            return "", []
        snippets = []
        citations = []
        for i, d in enumerate(docs, 1):
            # snippet: 첫 300자 정도만 전달 (모델에게 너무 많은 컨텍스트를 주면 반복 유발)
            txt = (d.page_content[:300] + "...") if len(d.page_content) > 300 else d.page_content
            snippets.append(f"{txt}")
            citations.append(d.metadata.get("source", d.metadata.get("path", f"doc{i}")))
        # context는 snippet들을 줄바꿈으로 연결, 실제 번호 표시는 하지 않음
        return "\n\n".join(snippets), citations

    chain = prompt | llm | StrOutputParser()

    def run(question: str):
        try:
            docs = retriever.get_relevant_documents(question)
        except Exception:
            try:
                docs = retriever.invoke(question)
            except Exception:
                docs = []
        context, citations = format_docs(docs)
        # 빈 context는 빈 문자열
        context = context or ""
        stream = chain.stream({"question": question, "context": context})
        return stream, citations

    return run
