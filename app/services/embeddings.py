# app/services/embeddings.py

import torch
from app.core.config import settings
from langchain_community.embeddings import SentenceTransformerEmbeddings


def _sentence_backend(model_name: str) -> SentenceTransformerEmbeddings:
    """Sentence-Transformers 백엔드 생성기.
    - CUDA가 있으면 자동으로 사용
    - trust_remote_code=True 로 커스텀 모델 허용
    - 출력 벡터를 정규화해 코사인 유사도 안정화
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def get_embeddings():
    """rag_pipeline 에서 호출되는 팩토리 함수."""
    backend = (settings.EMBEDDING_BACKEND or "sentence").lower()

    if backend == "sentence":
        return _sentence_backend(settings.EMBEDDING_MODEL)

    # 추후 다른 백엔드(e5, openai 등)를 추가할 때 여기 분기
    return _sentence_backend(settings.EMBEDDING_MODEL)
