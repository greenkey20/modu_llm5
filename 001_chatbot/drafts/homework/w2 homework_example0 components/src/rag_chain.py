"""
RAGChain 컴포넌트
RAG 파이프라인을 구성하고 실행
"""
from typing import Iterator, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


class RAGChain:
    """RAG 파이프라인 관리 클래스"""
    
    def __init__(self):
        self.retriever: Optional[BaseRetriever] = None
        self.llm: Optional[BaseLanguageModel] = None
        self.chain: Optional[Runnable] = None
        self.last_retrieved_docs: List[Document] = []
    
    def initialize_llm(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        streaming: bool = True
    ) -> BaseLanguageModel:
        """
        LLM 모델 초기화
        
        Args:
            model_name: 모델 이름 (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
            temperature: 온도 파라미터 (0.0-2.0)
            max_tokens: 최대 토큰 수
            streaming: 스트리밍 모드 활성화 여부
            
        Returns:
            초기화된 LLM 객체
            
        Raises:
            Exception: API 키 누락 또는 모델 초기화 실패
        """
        # OpenAI API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception(
                "OpenAI API 키가 설정되지 않았습니다. "
                ".env 파일에 OPENAI_API_KEY를 설정해주세요."
            )
        
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                openai_api_key=api_key
            )
            return self.llm
        
        except Exception as e:
            raise Exception(f"LLM 초기화 실패: {str(e)}")
        
    def build_chain(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel
    ) -> Runnable:
        """
        RAG 체인 구성
        
        Args:
            retriever: 검색기 객체
            llm: LLM 객체
            
        Returns:
            구성된 Runnable 체인
        """
        self.retriever = retriever
        self.llm = llm
        
        # 프롬프트 템플릿 정의
        template = """당신은 주어진 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.
다음 문서 내용을 참고하여 질문에 답변해주세요.
문서에 없는 내용은 추측하지 말고, 모른다고 답변해주세요.

문서 내용:
{context}

질문: {question}

답변:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 체인 구성
        def format_docs(docs):
            self.last_retrieved_docs = docs
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.chain = (
            {"context": retriever | format_docs, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return self.chain
    
    def stream_response(
        self,
        query: str
    ) -> Iterator[str]:
        """
        스트리밍 방식으로 응답 생성
        
        Args:
            query: 사용자 질문
            
        Yields:
            응답 텍스트 청크
            
        Raises:
            ValueError: 체인이 구성되지 않았을 때
        """
        if not self.chain:
            raise ValueError("RAG 체인이 구성되지 않았습니다.")
        
        try:
            for chunk in self.chain.stream(query):
                yield chunk
        
        except Exception as e:
            raise Exception(f"응답 생성 실패: {str(e)}")
    
    def get_source_documents(
        self,
        query: str
    ) -> List[Document]:
        """
        검색된 소스 문서 반환
        
        Args:
            query: 사용자 질문
            
        Returns:
            검색된 Document 객체 리스트
            
        Raises:
            ValueError: 검색기가 설정되지 않았을 때
        """
        if not self.retriever:
            raise ValueError("검색기가 설정되지 않았습니다.")
        
        try:
            docs = self.retriever.get_relevant_documents(query)
            self.last_retrieved_docs = docs
            return docs
        
        except Exception as e:
            raise Exception(f"문서 검색 실패: {str(e)}")
    
    def get_last_retrieved_docs(self) -> List[Document]:
        """
        마지막으로 검색된 문서 반환
        
        Returns:
            마지막으로 검색된 Document 객체 리스트
        """
        return self.last_retrieved_docs


