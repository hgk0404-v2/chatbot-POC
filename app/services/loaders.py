# app/services/loaders.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF 로더는 선택적 의존성으로 처리 (없으면 친절한 에러)
try:
    from langchain_community.document_loaders import PyPDFLoader  # pypdf 필요
    _HAS_PYPDF = True
except Exception:
    PyPDFLoader = None  # type: ignore
    _HAS_PYPDF = False


# 업로드 파일(FastAPI UploadFile) 혹은 경로를 받아서 전부 저장 후 경로 리스트 반환
def _save_uploaded(files: Iterable, data_dir: Union[str, Path]) -> List[Path]:
    saved: List[Path] = []
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        # FastAPI UploadFile 인지, 문자열/Path 인지 구분
        if hasattr(f, "filename") and hasattr(f, "file"):
            # UploadFile
            filename = Path(getattr(f, "filename"))
            dest = data_dir / filename.name
            with open(dest, "wb") as out:
                out.write(f.file.read())
            saved.append(dest)
        else:
            # str | Path
            p = Path(f)
            if not p.exists():
                raise FileNotFoundError(p)
            saved.append(p)
    return saved


def _load_one(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        if not _HAS_PYPDF:
            raise RuntimeError(
                "PDF 로드를 위해 'pypdf' 기반 PyPDFLoader가 필요합니다. "
                "requirements.txt에 'pypdf' 또는 'langchain-community[pypdf]'를 추가해 주세요."
            )
        loader = PyPDFLoader(str(path))
        return loader.load()

    # 간단 텍스트 계열
    if ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": str(path)})]

    # 그 밖의 확장자는 텍스트로 일단 시도
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": str(path)})]


def load_any(files: Iterable, data_dir: Union[str, Path] = "./data") -> List[Document]:
    """
    UploadFile | str | Path 의 이터러블을 받아 문서 리스트로 변환.
    모든 업로드는 data_dir에 저장됩니다.
    """
    paths = _save_uploaded(files, data_dir)
    docs: List[Document] = []
    for p in paths:
        docs.extend(_load_one(Path(p)))
    return docs


def split_docs(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    단락/문장 구분을 최대한 보존하면서 조각내기.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n", "\n", "。", "！", "？",
            ". ", "! ", "? ", " ", ""
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(docs)
