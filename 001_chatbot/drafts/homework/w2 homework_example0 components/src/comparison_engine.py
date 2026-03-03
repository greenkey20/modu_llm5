"""
ComparisonEngine 컴포넌트
여러 템플릿으로 동시에 응답을 생성하고 비교
"""
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.template_manager import RAGConfig
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store_manager import VectorStoreManager
from src.rag_chain import RAGChain


@dataclass
class ComparisonResult:
    """템플릿 비교 결과 데이터 모델"""
    template_name: str
    query: str
    response: str
    response_length: int
    generation_time: float
    source_document_count: int
    config_summary: str
    timestamp: str


class ComparisonEngine:
    """템플릿 비교 엔진 클래스"""
    
    def __init__(self):
        self.results: List[ComparisonResult] = []
    
    def run_comparison(
        self,
        query: str,
        templates: List[Tuple[str, RAGConfig]],
        current_file_path: str,
        doc_processor: DocumentProcessor,
        embedding_manager: EmbeddingManager,
        vector_store_manager: VectorStoreManager,
        rag_chain: RAGChain
    ) -> List[ComparisonResult]:
        """
        여러 템플릿으로 응답 생성 및 비교 (병렬 처리)
        
        Args:
            query: 사용자 질문
            templates: (템플릿 이름, RAGConfig) 튜플 리스트
            current_file_path: 현재 로드된 문서 경로
            doc_processor: DocumentProcessor 인스턴스
            embedding_manager: EmbeddingManager 인스턴스
            vector_store_manager: VectorStoreManager 인스턴스
            rag_chain: RAGChain 인스턴스
            
        Returns:
            ComparisonResult 리스트
        """
        self.results = []
        
        # 병렬 처리를 위한 ThreadPoolExecutor 사용
        with ThreadPoolExecutor(max_workers=min(len(templates), 3)) as executor:
            # 각 템플릿에 대해 별도의 인스턴스 생성하여 병렬 처리
            futures = []
            for template_name, config in templates:
                future = executor.submit(
                    self._process_single_template,
                    template_name,
                    config,
                    query,
                    current_file_path
                )
                futures.append(future)
            
            # 결과 수집
            for future in as_completed(futures):
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    print(f"템플릿 처리 중 오류: {str(e)}")
        
        return self.results
    
    def _process_single_template(
        self,
        template_name: str,
        config: RAGConfig,
        query: str,
        current_file_path: str
    ) -> ComparisonResult:
        """
        단일 템플릿 처리 (병렬 실행용)
        
        Args:
            template_name: 템플릿 이름
            config: RAG 설정
            query: 사용자 질문
            current_file_path: 문서 경로
            
        Returns:
            ComparisonResult
        """
        try:
            start_time = time.time()
            
            # 각 스레드에서 독립적인 인스턴스 생성
            # 고유한 벡터 스토어 디렉토리 사용 (스레드 ID 기반)
            import threading
            thread_id = threading.get_ident()
            unique_dir = f"./chroma_db_temp_{thread_id}"
            
            doc_processor = DocumentProcessor()
            embedding_manager = EmbeddingManager()
            vector_store_manager = VectorStoreManager(persist_directory=unique_dir)
            rag_chain = RAGChain()
            
            # 1. 문서 로드 및 분할
            documents = doc_processor.load_document(current_file_path)
            sep_list = [s.strip().replace('\\n', '\n') for s in config.separators.split(',')]
            split_docs = doc_processor.split_documents(
                documents,
                splitter_type=config.splitter_type,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=sep_list
            )
            
            # 2. 임베딩 모델 초기화
            if config.embedding_type == "openai" and "text-embedding-3" in config.embedding_model:
                embedding_manager.initialize_embedding_model(
                    model_type=config.embedding_type,
                    model_name=config.embedding_model,
                    dimensions=config.embedding_dimensions
                )
            else:
                embedding_manager.initialize_embedding_model(
                    model_type=config.embedding_type,
                    model_name=config.embedding_model
                )
            
            # 3. 벡터 저장소 초기화 및 문서 추가
            embedding_dim = embedding_manager.get_embedding_dimension()
            vector_store_manager.initialize_vector_store(
                store_type=config.vector_store_type,
                embedding_function=embedding_manager.embedding_model,
                embedding_dim=embedding_dim
            )
            vector_store_manager.add_documents(split_docs)
            
            # 4. 검색 파라미터 설정
            search_kwargs = {"k": config.k}
            if config.search_type == "similarity_score_threshold":
                search_kwargs["score_threshold"] = config.score_threshold
            elif config.search_type == "mmr":
                search_kwargs["lambda_mult"] = config.lambda_mult
            
            # 5. 검색기 생성
            retriever = vector_store_manager.get_retriever(
                search_type=config.search_type,
                search_kwargs=search_kwargs
            )
            
            # 6. LLM 초기화
            rag_chain.initialize_llm(
                model_name=config.llm_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                streaming=False  # 비교 시에는 스트리밍 비활성화
            )
            
            # 7. RAG 체인 구성
            rag_chain.build_chain(retriever, rag_chain.llm)
            
            # 8. 응답 생성
            response = ""
            for chunk in rag_chain.stream_response(query):
                response += chunk
            
            # 9. 소스 문서 가져오기
            source_docs = rag_chain.get_last_retrieved_docs()
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # 10. 결과 생성
            config_summary = self._create_config_summary(config)
            result = ComparisonResult(
                template_name=template_name,
                query=query,
                response=response,
                response_length=len(response),
                generation_time=generation_time,
                source_document_count=len(source_docs),
                config_summary=config_summary,
                timestamp=datetime.now().isoformat()
            )
            
            # 벡터 저장소 정리
            vector_store_manager.clear_store()
            
            return result
        
        except Exception as e:
            # 오류 발생 시 오류 메시지를 응답으로 저장
            return ComparisonResult(
                template_name=template_name,
                query=query,
                response=f"오류 발생: {str(e)}",
                response_length=0,
                generation_time=0.0,
                source_document_count=0,
                config_summary=self._create_config_summary(config),
                timestamp=datetime.now().isoformat()
            )
    
    def _create_config_summary(self, config: RAGConfig) -> str:
        """설정 요약 생성"""
        return (
            f"Chunk: {config.chunk_size}/{config.chunk_overlap} | "
            f"Emb: {config.embedding_model} | "
            f"Search: {config.search_type}(k={config.k}) | "
            f"LLM: {config.llm_model}(T={config.temperature})"
        )
    
    def generate_comparison_report(self, results: List[ComparisonResult]) -> str:
        """
        비교 결과 리포트 생성
        
        Args:
            results: ComparisonResult 리스트
            
        Returns:
            포맷팅된 리포트 문자열
        """
        if not results:
            return "비교 결과가 없습니다."
        
        report_lines = ["# 템플릿 비교 결과\n"]
        
        for i, result in enumerate(results, 1):
            report_lines.append(f"## {i}. {result.template_name}")
            report_lines.append(f"**설정:** {result.config_summary}")
            report_lines.append(f"**응답 길이:** {result.response_length} 자")
            report_lines.append(f"**생성 시간:** {result.generation_time:.2f} 초")
            report_lines.append(f"**소스 문서 수:** {result.source_document_count}")
            report_lines.append(f"**응답:**\n{result.response}\n")
            report_lines.append("-" * 80 + "\n")
        
        return "\n".join(report_lines)
    
    def export_to_csv(self, results: List[ComparisonResult], output_path: str) -> bool:
        """
        비교 결과를 CSV로 내보내기
        
        Args:
            results: ComparisonResult 리스트
            output_path: 출력 파일 경로
            
        Returns:
            내보내기 성공 여부
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                
                # 헤더 작성
                writer.writerow([
                    '템플릿 이름',
                    '질문',
                    '응답',
                    '응답 길이',
                    '생성 시간(초)',
                    '소스 문서 수',
                    '설정 요약',
                    '타임스탬프'
                ])
                
                # 데이터 작성
                for result in results:
                    writer.writerow([
                        result.template_name,
                        result.query,
                        result.response,
                        result.response_length,
                        f"{result.generation_time:.2f}",
                        result.source_document_count,
                        result.config_summary,
                        result.timestamp
                    ])
            
            return True
        
        except Exception as e:
            print(f"CSV 내보내기 실패: {str(e)}")
            return False
