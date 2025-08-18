# app/services/vectorstore.py
from app.core.config import settings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path
import os, json

class FaissStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.index_dir = Path(settings.INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vs_path = self.index_dir / "faiss.index"
        self.pkl_path = self.index_dir / "store.pkl"
        self.vs = None

    # 간단한 텍스트 청크 분할기 (의존성 최소화)
    def _chunk_text(self, text, chunk_size=800, overlap=100):
        if not text:
            return []
        text = text.replace("\r\n", "\n")
        chunks = []
        start = 0
        L = len(text)
        while start < L:
            end = min(start + chunk_size, L)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = max(0, end - overlap)
            if end == L:
                break
        return chunks

    # data 디렉터리를 스캔해서 Document 리스트 반환
    def _load_docs_from_data_dir(self):
        data_dir = Path(settings.DATA_DIR)
        docs = []
        if not data_dir.exists():
            return docs

        for p in sorted(data_dir.rglob("*")):
            if p.is_file():
                try:
                    suffix = p.suffix.lower()
                    text = ""
                    if suffix in [".txt", ".md", ".csv"]:
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                    elif suffix in [".pdf"]:
                        # 간단한 pdf 처리: PyPDF2 있을 경우 사용, 없으면 skip
                        try:
                            import PyPDF2
                            reader = PyPDF2.PdfReader(str(p))
                            pages = []
                            for page in reader.pages:
                                pages.append(page.extract_text() or "")
                            text = "\n".join(pages)
                        except Exception:
                            # PyPDF2가 없거나 실패하면 skip pdf silently
                            text = ""
                    else:
                        # 다른 확장자(예: docx)는 무시(원하면 추가)
                        text = ""

                    if not text:
                        continue

                    # 파일 내용을 청크로 쪼개서 Document로 만듦
                    chunks = self._chunk_text(text, chunk_size=1200, overlap=200)
                    for i, c in enumerate(chunks, 1):
                        meta = {"source": str(p.name), "path": str(p), "chunk": i}
                        docs.append(Document(page_content=c, metadata=meta))
                except Exception as e:
                    # 파일 처리 실패 로그 대신 무시(필요시 로깅 추가)
                    # print(f"skip {p}: {e}")
                    continue
        return docs

    # 기존/새 인덱스 로드 또는 생성
    def load_or_create(self, docs=None):
        # 1) 이미 존재하면 로드
        if self.vs_path.exists() and self.pkl_path.exists():
            self.vs = FAISS.load_local(str(self.index_dir), self.embeddings, allow_dangerous_deserialization=True)
            return

        # 2) 외부에서 docs를 주면 그것으로 생성
        if docs:
            self.vs = FAISS.from_documents(docs, self.embeddings)
            self.vs.save_local(str(self.index_dir))
            return

        # 3) data 디렉토리에서 자동으로 문서를 로드하여 인덱스 생성
        docs_from_data = self._load_docs_from_data_dir()
        if docs_from_data:
            self.vs = FAISS.from_documents(docs_from_data, self.embeddings)
            self.vs.save_local(str(self.index_dir))
            return

        # 4) 그 외: 빈 인덱스(기본)
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
