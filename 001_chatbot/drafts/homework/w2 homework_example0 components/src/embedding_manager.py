"""
EmbeddingManager 컴포넌트
임베딩 모델 관리 및 벡터 생성을 담당
"""
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
import os


class EmbeddingManager:
    """임베딩 모델 관리 클래스"""
    
    # 임베딩 모델별 차원 정보
    EMBEDDING_DIMENSIONS = {
        "openai": {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        },
        "huggingface": {
            "BAAI/bge-m3": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768
        },
        "ollama": {
            "bge-m3": 1024,
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024
        }
    }
    
    def __init__(self):
        self.embedding_model: Optional[Embeddings] = None
        self.model_type: Optional[str] = None
        self.model_name: Optional[str] = None
        self.custom_dimensions: Optional[int] = None  # 사용자 지정 차원 저장
    
    def initialize_embedding_model(
        self,
        model_type: str,
        model_name: str,
        dimensions: Optional[int] = None
    ) -> Embeddings:
        """
        임베딩 모델 초기화
        
        Args:
            model_type: 모델 타입 ("openai", "huggingface", "ollama")
            model_name: 모델 이름
            dimensions: 임베딩 차원 (OpenAI text-embedding-3 모델만 지원)
            
        Returns:
            초기화된 Embeddings 객체
            
        Raises:
            ValueError: 지원하지 않는 모델 타입 또는 이름
            Exception: API 키 누락 또는 모델 로드 실패
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.custom_dimensions = dimensions  # 사용자 지정 차원 저장
        
        try:
            if self.model_type == "openai":
                # OpenAI API 키 확인
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise Exception(
                        "OpenAI API 키가 설정되지 않았습니다. "
                        ".env 파일에 OPENAI_API_KEY를 설정해주세요."
                    )
                
                # text-embedding-3 모델은 차원 설정 가능
                if "text-embedding-3" in model_name and dimensions:
                    self.embedding_model = OpenAIEmbeddings(
                        model=model_name,
                        openai_api_key=api_key,
                        dimensions=dimensions
                    )
                else:
                    self.embedding_model = OpenAIEmbeddings(
                        model=model_name,
                        openai_api_key=api_key
                    )
            
            elif self.model_type == "huggingface":
                # HuggingFace 모델 로드
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            elif self.model_type == "ollama":
                # Ollama 모델 로드
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self.embedding_model = OllamaEmbeddings(
                    model=model_name,
                    base_url=base_url
                )
            
            else:
                raise ValueError(
                    f"지원하지 않는 모델 타입: {model_type}. "
                    "지원 타입: openai, huggingface, ollama"
                )
            
            return self.embedding_model
        
        except Exception as e:
            raise Exception(f"임베딩 모델 초기화 실패: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        현재 임베딩 모델의 차원 수 반환
        
        Returns:
            임베딩 차원 수
            
        Raises:
            ValueError: 모델이 초기화되지 않았을 때
        """
        if not self.model_type or not self.model_name:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다.")
        
        # 사용자 지정 차원이 있으면 우선 반환 (OpenAI text-embedding-3)
        if self.custom_dimensions is not None:
            return self.custom_dimensions
        
        # 모델 타입과 이름으로 차원 조회
        if self.model_type in self.EMBEDDING_DIMENSIONS:
            dimensions = self.EMBEDDING_DIMENSIONS[self.model_type]
            if self.model_name in dimensions:
                return dimensions[self.model_name]
        
        # 기본값 반환 (알 수 없는 모델의 경우)
        return 1536
    
    def embed_query(self, text: str) -> List[float]:
        """
        단일 텍스트를 임베딩 벡터로 변환
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터 (float 리스트)
            
        Raises:
            ValueError: 모델이 초기화되지 않았을 때
        """
        if not self.embedding_model:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다.")
        
        return self.embedding_model.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트를 임베딩 벡터로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
            
        Raises:
            ValueError: 모델이 초기화되지 않았을 때
        """
        if not self.embedding_model:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다.")
        
        return self.embedding_model.embed_documents(texts)
