"""
VectorStoreManager 컴포넌트
벡터 저장소 관리를 담당
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.retrievers import BaseRetriever
import shutil
import os


class VectorStoreManager:
    """벡터 저장소 관리 클래스"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.vector_store: Optional[VectorStore] = None
        self.store_type: Optional[str] = None
        self.embedding_function: Optional[Embeddings] = None
        self.persist_directory = persist_directory
        self.current_embedding_dim: Optional[int] = None
    
    def initialize_vector_store(
        self,
        store_type: str,
        embedding_function: Embeddings,
        embedding_dim: Optional[int] = None
    ) -> VectorStore:
        """
        벡터 저장소 초기화
        
        Args:
            store_type: 저장소 타입 ("chroma" 또는 "faiss")
            embedding_function: 임베딩 함수
            embedding_dim: 임베딩 차원 (차원 변경 감지용)
            
        Returns:
            초기화된 VectorStore 객체
            
        Raises:
            ValueError: 지원하지 않는 저장소 타입
        """
        # 임베딩 차원이 변경되었으면 기존 벡터 스토어 삭제
        if embedding_dim is not None and self.current_embedding_dim is not None:
            if embedding_dim != self.current_embedding_dim:
                print(f"\n{'='*60}")
                print(f"⚠️  임베딩 차원 변경 감지!")
                print(f"   이전 차원: {self.current_embedding_dim}")
                print(f"   새 차원: {embedding_dim}")
                print(f"   → 기존 벡터 스토어를 초기화합니다...")
                print(f"{'='*60}\n")
                self.clear_store()
        
        self.store_type = store_type.lower()
        self.embedding_function = embedding_function
        self.current_embedding_dim = embedding_dim
        
        try:
            if self.store_type == "chroma":
                # Chroma 벡터 저장소 초기화
                self.vector_store = None  # 문서 추가 시 생성
                return self.vector_store
            
            elif self.store_type == "faiss":
                # FAISS 벡터 저장소 초기화
                self.vector_store = None  # 문서 추가 시 생성
                return self.vector_store
            
            else:
                raise ValueError(
                    f"지원하지 않는 저장소 타입: {store_type}. "
                    "지원 타입: chroma, faiss"
                )
        
        except Exception as e:
            raise Exception(f"벡터 저장소 초기화 실패: {str(e)}")
    
    def add_documents(
        self,
        documents: List[Document]
    ) -> List[str]:
        """
        문서를 벡터 저장소에 추가
        
        Args:
            documents: 추가할 Document 객체 리스트
            
        Returns:
            추가된 문서의 ID 리스트
            
        Raises:
            ValueError: 저장소가 초기화되지 않았거나 문서가 비어있을 때
        """
        if not self.embedding_function:
            raise ValueError("임베딩 함수가 설정되지 않았습니다.")
        
        if not documents:
            raise ValueError("추가할 문서가 없습니다.")
        
        try:
            if self.store_type == "chroma":
                # Chroma에 문서 추가
                if self.vector_store is None:
                    self.vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embedding_function,
                        persist_directory=self.persist_directory
                    )
                else:
                    self.vector_store.add_documents(documents)
                
                return [str(i) for i in range(len(documents))]
            
            elif self.store_type == "faiss":
                # FAISS에 문서 추가
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(
                        documents=documents,
                        embedding=self.embedding_function
                    )
                else:
                    self.vector_store.add_documents(documents)
                
                return [str(i) for i in range(len(documents))]
            
            else:
                raise ValueError("벡터 저장소가 초기화되지 않았습니다.")
        
        except Exception as e:
            raise Exception(f"문서 추가 실패: {str(e)}")
    
    def clear_store(self):
        """벡터 저장소 초기화"""
        import gc
        import time
        
        # Chroma 벡터 저장소를 먼저 명시적으로 삭제
        if self.vector_store is not None:
            try:
                # Chroma의 경우 delete_collection 호출
                if self.store_type == "chroma" and hasattr(self.vector_store, '_client'):
                    self.vector_store.delete_collection()
                    
                    # 클라이언트 명시적 종료
                    if hasattr(self.vector_store._client, 'clear_system_cache'):
                        self.vector_store._client.clear_system_cache()
                    
                    # 클라이언트 연결 완전히 닫기
                    if hasattr(self.vector_store._client, '_producer'):
                        try:
                            self.vector_store._client._producer = None
                        except:
                            pass
                    
                    if hasattr(self.vector_store._client, '_consumer'):
                        try:
                            self.vector_store._client._consumer = None
                        except:
                            pass
                    
                    # 클라이언트 자체를 None으로
                    try:
                        self.vector_store._client = None
                    except:
                        pass
                        
            except Exception as e:
                print(f"컬렉션 삭제 중 오류: {str(e)}")
        
        # 벡터 저장소 참조 완전히 제거
        self.vector_store = None
        
        # 가비지 컬렉션 강제 실행 (여러 번)
        for _ in range(3):
            gc.collect()
            time.sleep(0.2)
        
        # Chroma 저장소 디렉토리 삭제 (강화된 재시도 로직)
        if self.store_type == "chroma" and os.path.exists(self.persist_directory):
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    time.sleep(0.3)
                    
                    # 디렉토리 내 모든 파일을 읽기 전용 해제
                    try:
                        for root, dirs, files in os.walk(self.persist_directory):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    os.chmod(file_path, 0o777)
                                except:
                                    pass
                    except:
                        pass
                    
                    # 삭제 시도
                    shutil.rmtree(self.persist_directory, ignore_errors=True)
                    
                    # 삭제 확인
                    if not os.path.exists(self.persist_directory):
                        print("✅ 저장소 디렉토리가 성공적으로 삭제되었습니다.")
                        break
                    
                    # 여전히 존재하면 개별 파일 삭제 시도
                    if attempt >= 3:
                        try:
                            for root, dirs, files in os.walk(self.persist_directory, topdown=False):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    try:
                                        os.remove(file_path)
                                    except:
                                        pass
                                for dir in dirs:
                                    dir_path = os.path.join(root, dir)
                                    try:
                                        os.rmdir(dir_path)
                                    except:
                                        pass
                            os.rmdir(self.persist_directory)
                        except:
                            pass
                    
                    # 다시 확인
                    if not os.path.exists(self.persist_directory):
                        print("✅ 저장소 디렉토리가 성공적으로 삭제되었습니다.")
                        break
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"🔄 저장소 디렉토리 삭제 재시도 중... ({attempt + 1}/{max_retries})")
                        time.sleep(0.5)
                    else:
                        # 최종 실패 시 디렉토리 이름 변경으로 우회
                        try:
                            backup_dir = f"{self.persist_directory}_old_{int(time.time())}"
                            os.rename(self.persist_directory, backup_dir)
                            print(f"⚠️ 디렉토리를 삭제할 수 없어 이름을 변경했습니다: {backup_dir}")
                            print("   다음 실행 시 새로운 디렉토리가 생성됩니다.")
                        except:
                            print(f"❌ 저장소 디렉토리 삭제 실패: {str(e)}")
                            print("   다음 질문 시 새로운 컬렉션이 생성됩니다.")
    
    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """
        검색기 생성 및 반환
        
        Args:
            search_type: 검색 타입 ("similarity", "mmr", "similarity_score_threshold")
            search_kwargs: 검색 파라미터 (k, score_threshold, lambda_mult 등)
            
        Returns:
            BaseRetriever 객체
            
        Raises:
            ValueError: 벡터 저장소가 초기화되지 않았을 때
        """
        if not self.vector_store:
            raise ValueError(
                "벡터 저장소가 비어있습니다. 먼저 문서를 업로드해주세요."
            )
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        try:
            retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            return retriever
        
        except Exception as e:
            raise Exception(f"검색기 생성 실패: {str(e)}")
    
    def get_store_stats(self) -> Dict[str, Any]:
        """
        저장소 통계 정보 반환
        
        Returns:
            저장소 통계 정보를 담은 딕셔너리
        """
        if not self.vector_store:
            return {
                "document_count": 0,
                "store_type": self.store_type or "none"
            }
        
        # 저장소 타입에 따라 문서 수 계산
        try:
            if self.store_type == "chroma":
                # Chroma의 경우 컬렉션에서 문서 수 조회
                collection = self.vector_store._collection
                doc_count = collection.count()
            elif self.store_type == "faiss":
                # FAISS의 경우 인덱스 크기 조회
                doc_count = self.vector_store.index.ntotal
            else:
                doc_count = 0
        except:
            doc_count = 0
        
        return {
            "document_count": doc_count,
            "store_type": self.store_type or "none"
        }
