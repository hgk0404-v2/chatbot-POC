# app/core/config.py (pydantic 사용)
try:
    # pydantic <= v2.6 (또는 BaseSettings가 pydantic에 있는 경우)
    from pydantic import BaseSettings  # type: ignore
except Exception:
    # pydantic v2.7+에서 BaseSettings는 pydantic-settings 패키지로 이동
    from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    EMBEDDING_BACKEND: str = "openai"
    EMBEDDING_MODEL: str = "Alibaba-NLP/gte-multilingual-base"
    VECTORSTORE_BACKEND: str = "faiss"
    DATA_DIR: str = "./data"
    INDEX_DIR: str = "./index/faiss"

    # Retriever 설정
    TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.2

    # LLM 실행 옵션
    BACKEND: str = "hf" # 허깅페이스 로컬 폴더 쓸때 필요
    MODEL_PATH: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf" # 고정이 아님 기본값.
    N_CTX: int = 4096
    N_BATCH: int = 512
    N_THREADS: int = 8
    N_GPU_LAYERS: int = 999

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()