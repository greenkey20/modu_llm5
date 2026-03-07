"""LLM 기반 맥락 생성 모듈"""
import logging
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Anthropic 스타일 컨텍스트 생성 프롬프트
CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 문서의 청크에 맥락을 추가하는 전문가입니다.
주어진 청크가 전체 문서에서 어떤 위치에 있고 무엇에 대한 내용인지 간결하게 설명하세요.
설명은 50-100자 이내로 작성하세요.
오직 맥락 설명만 출력하세요."""),
    ("user", """<document>
{whole_document}
</document>

위 문서에서 아래 청크의 맥락을 설명해주세요:

<chunk>
{chunk_content}
</chunk>

맥락 설명:""")
])


def generate_context(whole_document: str, chunk_content: str, llm: ChatOpenAI) -> str:
    """단일 청크에 대한 맥락 설명 생성."""
    chain = CONTEXT_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "whole_document": whole_document,
        "chunk_content": chunk_content,
    })


def generate_contexts_batch(
    chunks: list[Document], whole_document: str, llm: ChatOpenAI
) -> list[str]:
    """청크 목록에 대해 배치로 맥락 생성. 오류 시 빈 문자열 반환."""
    chain = CONTEXT_PROMPT | llm | StrOutputParser()
    inputs = [
        {"whole_document": whole_document, "chunk_content": chunk.page_content}
        for chunk in chunks
    ]
    contexts = []
    for i, inp in enumerate(inputs):
        try:
            result = chain.invoke(inp)
            contexts.append(result)
        except Exception as e:
            logger.warning(f"청크 {i} 맥락 생성 실패: {e}")
            contexts.append("")
    return contexts


def create_contextual_chunks(
    chunks: list[Document], contexts: list[str]
) -> list[Document]:
    """원본 청크 + 맥락을 결합하여 Contextual 청크 생성.

    형식: "[맥락] {context}\n\n{원본 청크}"
    """
    contextual_chunks = []
    for i, (chunk, context) in enumerate(zip(chunks, contexts)):
        if context:
            contextual_content = f"[맥락] {context}\n\n{chunk.page_content}"
        else:
            contextual_content = chunk.page_content

        contextual_chunk = Document(
            page_content=contextual_content,
            metadata={
                **chunk.metadata,
                "chunk_id": i,
                "original_content": chunk.page_content,
                "context": context,
            },
        )
        contextual_chunks.append(contextual_chunk)
    return contextual_chunks
