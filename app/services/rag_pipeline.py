# app/services/rag_pipeline.py
import os
from app.core.config import settings
from app.services.embeddings import get_embeddings
from app.services.vectorstore import FaissStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp

SYSTEM_PROMPT = """아래 컨텍스트에서만 근거하여 답변합니다.
근거가 없으면 '모르겠습니다'라고 답하십시오.
한국어로 간명하고 항목화해서 답하십시오.
각 단락 끝에 근거 출처 번호를 표기하십시오.
"""

def build_chain():
    # 1) 임베딩/벡터검색
    embeddings = get_embeddings()
    store = FaissStore(embeddings)
    store.load_or_create()
    retriever = store.as_retriever(k=settings.TOP_K)

    # 2) 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "질문: {question}\n\n컨텍스트:\n{context}")
    ])

    # 3) LLM (llama.cpp, CUDA 빌드)
    llm = LlamaCpp(
        model_path=settings.MODEL_PATH,   # ✅ .env에서 읽음
        n_ctx=4096,
        n_gpu_layers=-1,
        n_batch=512,
        temperature=0.2,
        streaming=True,
    )

    def format_docs(docs):
        chunks, cites = [], []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            tag = f"[{i}] {src}" + (f":p{page}" if page is not None else "")
            chunks.append(f"{d.page_content}\n{tag}")
            cites.append({"index": i, "source": d.metadata})
        return "\n\n".join(chunks), cites

    def run(question: str):
        docs = retriever.invoke(question)
        context, citations = format_docs(docs)
        chain = prompt | llm | StrOutputParser()
        # 항상 generator 반환
        stream = chain.stream({"question": question, "context": context})
        return stream, citations

    return run