"""
DocumentProcessor 컴포넌트
문서 로드 및 전처리를 담당
"""
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """PDF 문서 로드 및 텍스트 분할을 처리하는 클래스"""
    
    def __init__(self):
        self.loader: Optional[PyPDFLoader] = None
        self.documents: List[Document] = []
        self.split_documents_list: List[Document] = []
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        PDF 문서를 로드하고 Document 객체 리스트 반환
        
        Args:
            file_path: PDF 파일 경로
            
        Returns:
            Document 객체 리스트
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            ValueError: PDF 파일이 아닐 때
        """
        # 파일 형식 검증
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("지원하지 않는 파일 형식입니다. PDF 파일만 업로드 가능합니다.")
        
        try:
            self.loader = PyPDFLoader(file_path)
            self.documents = self.loader.load()
            return self.documents
        except FileNotFoundError:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        except Exception as e:
            raise Exception(f"문서 로드 중 오류 발생: {str(e)}")
    
    def split_documents(
        self,
        documents: List[Document],
        splitter_type: str = "RecursiveCharacterTextSplitter",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ) -> List[Document]:
        """
        문서를 청크로 분할
        
        Args:
            documents: 분할할 Document 객체 리스트
            splitter_type: 분할기 타입 ("RecursiveCharacterTextSplitter" 또는 "CharacterTextSplitter")
            chunk_size: 청크 크기
            chunk_overlap: 청크 중복 크기
            separators: 분할 구분자 리스트
            
        Returns:
            분할된 Document 객체 리스트
            
        Raises:
            ValueError: 잘못된 파라미터 값
        """
        # 파라미터 검증
        if chunk_size <= 0:
            raise ValueError("chunk_size는 0보다 커야 합니다.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap은 0 이상이어야 합니다.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap은 chunk_size보다 작아야 합니다.")
        
        # 기본 구분자 설정
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        # 분할기 선택 및 초기화
        if splitter_type == "RecursiveCharacterTextSplitter":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len
            )
        elif splitter_type == "CharacterTextSplitter":
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separators[0] if separators else "\n\n",
                length_function=len
            )
        else:
            raise ValueError(f"지원하지 않는 분할기 타입: {splitter_type}")
        
        # 문서 분할
        self.split_documents_list = text_splitter.split_documents(documents)
        return self.split_documents_list
    
    def get_split_stats(self) -> Dict[str, Any]:
        """
        분할 결과 통계 정보 반환
        
        Returns:
            분할 통계 정보를 담은 딕셔너리
        """
        if not self.split_documents_list:
            return {
                "chunk_count": 0,
                "average_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0
            }
        
        chunk_lengths = [len(doc.page_content) for doc in self.split_documents_list]
        
        return {
            "chunk_count": len(self.split_documents_list),
            "average_chunk_length": sum(chunk_lengths) // len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths)
        }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        문서 통계 정보 반환 (페이지 수, 텍스트 길이 등)
        
        Returns:
            문서 통계 정보를 담은 딕셔너리
        """
        if not self.documents:
            return {
                "page_count": 0,
                "total_text_length": 0,
                "average_page_length": 0
            }
        
        total_text_length = sum(len(doc.page_content) for doc in self.documents)
        page_count = len(self.documents)
        
        return {
            "page_count": page_count,
            "total_text_length": total_text_length,
            "average_page_length": total_text_length // page_count if page_count > 0 else 0
        }
