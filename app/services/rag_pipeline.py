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

MODEL_PATH = os.path.join("models", "qwen2.5-3b-instruct-q4_k_m.gguf")  # GGUF 파일 위치

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
        model_path=MODEL_PATH,
        # 성능/메모리 파라미터 (RTX 4060 8GB 기준 안전값)
        n_ctx=4096,
        n_gpu_layers=-1,   # 가능한 모든 레이어를 GPU에
        n_batch=512,       # VRAM 여유에 따라 256~1024 사잇값 조정
        temperature=0.2,
        streaming=True,
        # 필요시 추가 옵션
        # f16_kv=True,
        # verbose=True,
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
        return chain.stream({"question": question, "context": context}), citations

    return run

