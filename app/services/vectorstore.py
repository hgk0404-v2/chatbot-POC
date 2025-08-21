# app/services/vectorstore.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# (선택) settings 사용: 없으면 기본값 사용
try:
    from app.core.config import settings
    DEFAULT_INDEX_DIR = getattr(settings, "INDEX_DIR", "index/faiss")
    DEFAULT_EMBED_MODEL = getattr(settings, "EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")
except Exception:
    DEFAULT_INDEX_DIR = "index/faiss"
    DEFAULT_EMBED_MODEL = "Alibaba-NLP/gte-multilingual-base"


class FaissStore:
    """
    - LangChain 포맷: index.faiss + index.pkl
    - 미존재 시 data 디렉터리 스캔 → 색인 생성, 없으면 빈 인덱스라도 생성
    """
    def __init__(self, embeddings, index_dir: str | Path = DEFAULT_INDEX_DIR):
        self.embeddings = embeddings
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # ✅ 표준 파일명
        self.faiss_file = self.index_dir / "index.faiss"
        self.pkl_file = self.index_dir / "index.pkl"
        self.vs: Optional[FAISS] = None

    def _chunk_text(self, text: str, chunk_size=1200, overlap=200) -> List[str]:
        text = (text or "").replace("\r\n", "\n")
        if not text:
            return []
        out, start, L = [], 0, len(text)
        while start < L:
            end = min(start + chunk_size, L)
            chunk = text[start:end].strip()
            if chunk:
                out.append(chunk)
            if end == L:
                break
            start = max(0, end - overlap)
        return out

    def _load_docs_from_data_dir(self) -> List[Document]:
        # settings.DATA_DIR이 있다면 사용, 없으면 ./data
        data_dir = Path(getattr(globals().get("settings", object()), "DATA_DIR", "data"))
        docs: List[Document] = []
        if not data_dir.exists():
            return docs

        for p in sorted(data_dir.rglob("*")):
            if not p.is_file():
                continue
            text, suf = "", p.suffix.lower()
            try:
                if suf in [".txt", ".md", ".csv"]:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                elif suf == ".pdf":
                    try:
                        import PyPDF2
                        reader = PyPDF2.PdfReader(str(p))
                        text = "\n".join((page.extract_text() or "") for page in reader.pages)
                    except Exception:
                        text = ""
            except Exception:
                text = ""

            if not text:
                continue
            for i, c in enumerate(self._chunk_text(text), 1):
                docs.append(Document(page_content=c, metadata={"source": p.name, "path": str(p), "chunk": i}))
        return docs

    def load_or_create(self, docs: Optional[List[Document]] = None) -> None:
        # 1) 기존 인덱스(index.faiss + index.pkl) 존재하면 로드
        if self.faiss_file.exists() and self.pkl_file.exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            return

        # 2) 없고 docs 인자를 받았다면 -> docs로 새 인덱스 생성
        if docs:
            self.vs = FAISS.from_documents(docs, self.embeddings)
            self.vs.save_local(str(self.index_dir))
            return

        # 3) 없고 docs도 없지만 data 디렉터리에 파일이 있으면 -> 스캔해서 새 인덱스 생성
        scanned = self._load_docs_from_data_dir()
        if scanned:
            self.vs = FAISS.from_documents(scanned, self.embeddings)
            self.vs.save_local(str(self.index_dir))
            return

        # 4) 아무것도 없으면 빈 인덱스라도
        self.vs = FAISS.from_documents([Document(page_content="")], self.embeddings)
        self.vs.save_local(str(self.index_dir))

    def upsert(self, docs: List[Document]) -> None:
        if self.vs is None:
            self.load_or_create(docs)
        else:
            self.vs.add_documents(docs)
            self.vs.save_local(str(self.index_dir))

    def as_retriever(self, k=5):
        return self.vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5}
        )


# 🔹 main.py가 import 하는 진입점
def load_faiss(
    path: Optional[str] = None,
    model_name: Optional[str] = None,
    normalize_embeddings: bool = True,
) -> FAISS:
    """
    - path가 가리키는 디렉터리에서 LangChain-FAISS 인덱스를 로드/생성하여 반환
    - 반환값: langchain_community.vectorstores.FAISS 인스턴스
    """
    model_name = model_name or DEFAULT_EMBED_MODEL
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "trust_remote_code": True, # hugging face에서 모델이 custom python code를 포함할 수 있도록 허용.
            "device": "cuda"  # GPU가 있으면 활성화 가능
        },
        encode_kwargs={"normalize_embeddings": normalize_embeddings}, # 벡터를 정규화
    )
    index_dir = path or DEFAULT_INDEX_DIR
    store = FaissStore(embeddings, index_dir=index_dir)
    store.load_or_create()
    return store.vs
