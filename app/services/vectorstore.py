# app/services/vectorstore.py
from app.core.config import settings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path
import os, pickle

class FaissStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.index_dir = Path(settings.INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vs_path = self.index_dir / "faiss.index"
        self.pkl_path = self.index_dir / "store.pkl"
        self.vs = None

    def load_or_create(self, docs=None):
        if self.vs_path.exists() and self.pkl_path.exists():
            self.vs = FAISS.load_local(str(self.index_dir), self.embeddings, allow_dangerous_deserialization=True)
        elif docs:
            self.vs = FAISS.from_documents(docs, self.embeddings)
            self.vs.save_local(str(self.index_dir))
        else:
            # 빈 인덱스 생성
            self.vs = FAISS.from_documents([Document(page_content="")], self.embeddings)
            self.vs.save_local(str(self.index_dir))

    def upsert(self, docs):
        if self.vs is None:
            self.load_or_create(docs)
        else:
            self.vs.add_documents(docs)
            self.vs.save_local(str(self.index_dir))

    def as_retriever(self, k=5):
        return self.vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5})
