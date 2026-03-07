"""문서 로드 및 청킹 모듈"""
import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import SUPPORTED_EXTENSIONS, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def detect_language(text: str) -> str:
    """문서 언어 감지. 한국어 문자 비율 기반으로 'ko' 또는 'en' 반환."""
    if not text.strip():
        return "en"
    korean_chars = len(re.findall(r"[가-힣]", text))
    total_chars = len(re.findall(r"[a-zA-Z가-힣]", text))
    if total_chars == 0:
        return "en"
    return "ko" if korean_chars / total_chars > 0.5 else "en"


def load_document(file_path: str) -> Document:
    """파일을 읽어 LangChain Document 객체로 변환.

    Args:
        file_path: 파일 경로 (.md 또는 .txt)

    Returns:
        Document 객체

    Raises:
        ValueError: 빈 파일이거나 지원하지 않는 형식일 때
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext} (지원: {SUPPORTED_EXTENSIONS})")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        raise ValueError("파일이 비어있습니다.")

    language = detect_language(content)
    return Document(
        page_content=content,
        metadata={"source": os.path.basename(file_path), "language": language},
    )


def split_document(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Document를 RecursiveCharacterTextSplitter로 청크 분할."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents([doc])
    # 각 청크에 chunk_id 메타데이터 추가
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
