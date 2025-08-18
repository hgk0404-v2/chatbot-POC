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
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    VECTORSTORE_BACKEND: str = "faiss"
    DATA_DIR: str = "./data"
    INDEX_DIR: str = "./index/faiss"
    TOP_K: int = 5
    BACKEND: str = "hf"
    MODEL_PATH: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf" # 고정이 아님 기본값.

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()