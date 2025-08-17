# app/core/config.py
import os

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openai")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    VECTORSTORE_BACKEND = os.getenv("VECTORSTORE_BACKEND", "faiss")
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    INDEX_DIR = os.getenv("INDEX_DIR", "./index/faiss")
    TOP_K = int(os.getenv("TOP_K", "5"))

settings = Settings()
