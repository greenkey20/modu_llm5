"""검색 파이프라인 모듈"""
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from kiwipiepy import Kiwi

from config import EMBEDDING_MODEL, RERANKER_MODEL, DEFAULT_TOP_K

# Kiwi 형태소 분석기 (한국어용)
_kiwi = Kiwi()


def _kiwi_tokenizer(text: str) -> list[str]:
    """Kiwi 형태소 분석기를 사용한 한국어 토큰화."""
    return [t.form for t in _kiwi.tokenize(text)]


def _whitespace_tokenizer(text: str) -> list[str]:
    """공백 기반 영어 토큰화."""
    return text.split()


def build_vectorstore(
    docs: list[Document], collection_name: str, embeddings=None
) -> Chroma:
    """문서 목록으로 Chroma 벡터 저장소 생성."""
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
    )


def build_bm25_retriever(
    docs: list[Document], language: str = "ko", k: int = DEFAULT_TOP_K
) -> BM25Retriever:
    """BM25 검색기 생성. 한국어면 Kiwi 토크나이저, 영어면 공백 토크나이저 사용."""
    preprocess_func = _kiwi_tokenizer if language == "ko" else _whitespace_tokenizer
    return BM25Retriever.from_documents(
        documents=docs, preprocess_func=preprocess_func, k=k
    )


def build_hybrid_retriever(
    vector_retriever, bm25_retriever, weights: list[float]
) -> EnsembleRetriever:
    """Embedding + BM25 하이브리드 검색기 생성."""
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=weights,
    )


def build_rerank_retriever(
    base_retriever, top_n: int = DEFAULT_TOP_K
) -> ContextualCompressionRetriever:
    """Cross-Encoder Reranker를 적용한 검색기 생성."""
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )


def search(retriever, query: str) -> list[Document]:
    """검색 실행 및 결과 반환."""
    return retriever.invoke(query)
