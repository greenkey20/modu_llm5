# Design Document

## Overview

RAG 기반 챗봇 웹 애플리케이션은 Gradio를 사용한 2탭 구조의 인터페이스를 제공합니다. 프롬프트 탭에서는 실시간 스트리밍 질의응답을, 성능 비교 탭에서는 여러 템플릿을 병렬로 비교할 수 있습니다. 시스템은 LangChain을 기반으로 구축되며, 임베딩 차원 변경 자동 감지, 템플릿 관리, 병렬 처리 등의 고급 기능을 제공합니다.

## Architecture

시스템은 다음과 같은 계층 구조로 설계됩니다:

```
┌─────────────────────────────────────────────────────────────┐
│         Gradio UI Layer (2 Tabs)                            │
│  - 프롬프트 탭: 실시간 스트리밍 질의응답                       │
│  - 성능 비교 탭: 병렬 템플릿 비교 및 CSV 내보내기              │
└─────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│      Application Layer                                      │
│  - GradioInterface: UI 관리 및 이벤트 처리                   │
│  - ComparisonEngine: 병렬 템플릿 비교 (ThreadPoolExecutor)   │
│  - TemplateManager: JSON 기반 템플릿 저장/로드               │
└─────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│      RAG Pipeline Layer                                     │
│  - DocumentProcessor: 문서 로드 및 분할                      │
│  - EmbeddingManager: 임베딩 생성 (차원 추적)                 │
│  - VectorStoreManager: 벡터 저장소 관리 (차원 변경 감지)      │
│  - RAGChain: 검색 및 응답 생성                               │
└─────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│    LangChain Components Layer                               │
│  (Loaders, Splitters, Embeddings, VectorStores, LLMs)      │
└─────────────────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│      External Services Layer                                │
│  (OpenAI API, HuggingFace, Ollama)                          │
└─────────────────────────────────────────────────────────────┘
```

### 주요 흐름

1. **프롬프트 탭 - 자동 처리 흐름**: 
   - 사용자 질문 제출 → 문서 자동 로드 및 분할 → 임베딩 생성 → 벡터 저장소 저장 → 검색 → LLM 응답 생성 → 스트리밍 출력

2. **성능 비교 탭 - 병렬 처리 흐름**:
   - 템플릿 선택 → ThreadPoolExecutor로 병렬 실행 (최대 3개) → 각 스레드별 독립 벡터 스토어 생성 → 결과 수집 → 마크다운 표시 → CSV 내보내기

3. **차원 변경 감지 흐름**:
   - 임베딩 모델 변경 → EmbeddingManager가 custom_dimensions 저장 → VectorStoreManager가 차원 비교 → 차원 불일치 시 자동 clear_store() 호출 → 새 벡터 스토어 생성

## Components and Interfaces

### 1. DocumentProcessor

문서 로드 및 전처리를 담당하는 컴포넌트

```python
class DocumentProcessor:
    def __init__(self):
        self.loader = None
        self.documents = []
    
    def load_document(self, file_path: str) -> List[Document]:
        """PDF 문서를 로드하고 Document 객체 리스트 반환"""
        pass
    
    def split_documents(
        self, 
        documents: List[Document],
        splitter_type: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str]
    ) -> List[Document]:
        """문서를 청크로 분할"""
        pass
    
    def get_document_stats(self) -> Dict[str, Any]:
        """문서 통계 정보 반환 (페이지 수, 텍스트 길이 등)"""
        pass
```

### 2. EmbeddingManager

임베딩 모델 관리 및 벡터 생성을 담당하는 컴포넌트

```python
class EmbeddingManager:
    def __init__(self):
        self.embedding_model = None
        self.model_type = None
        self.model_name = None
        self.custom_dimensions = None  # 사용자 지정 차원 추적
    
    def initialize_embedding_model(
        self,
        model_type: str,  # "openai", "huggingface", "ollama"
        model_name: str,
        dimensions: Optional[int] = None  # OpenAI text-embedding-3 모델용
    ) -> Embeddings:
        """임베딩 모델 초기화 및 차원 저장"""
        pass
    
    def get_embedding_dimension(self) -> int:
        """
        현재 임베딩 모델의 차원 수 반환
        custom_dimensions가 설정되어 있으면 우선 반환
        """
        pass
    
    def embed_query(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩 벡터로 변환"""
        pass
```

### 3. VectorStoreManager

벡터 저장소 관리를 담당하는 컴포넌트

```python
class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.vector_store = None
        self.store_type = None
        self.persist_directory = persist_directory
        self.current_embedding_dim = None  # 차원 변경 감지용
    
    def initialize_vector_store(
        self,
        store_type: str,  # "chroma", "faiss"
        embedding_function: Embeddings,
        embedding_dim: Optional[int] = None  # 차원 변경 감지용
    ) -> VectorStore:
        """
        벡터 저장소 초기화
        embedding_dim이 변경되면 자동으로 clear_store() 호출
        """
        pass
    
    def add_documents(
        self,
        documents: List[Document]
    ) -> List[str]:
        """문서를 벡터 저장소에 추가"""
        pass
    
    def get_retriever(
        self,
        search_type: str,  # "similarity", "mmr", "similarity_score_threshold"
        search_kwargs: Dict[str, Any]
    ) -> BaseRetriever:
        """검색기 생성 및 반환"""
        pass
    
    def clear_store(self):
        """
        벡터 저장소 초기화
        최대 10회 재시도, 실패 시 디렉토리 이름 변경으로 우회
        """
        pass
    
    def get_store_stats(self) -> Dict[str, Any]:
        """저장소 통계 정보 반환"""
        pass
```

### 4. RAGChain

RAG 파이프라인을 구성하고 실행하는 컴포넌트

```python
class RAGChain:
    def __init__(self):
        self.retriever = None
        self.llm = None
        self.chain = None
    
    def initialize_llm(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        streaming: bool = True
    ) -> BaseLLM:
        """LLM 모델 초기화"""
        pass
    
    def build_chain(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM
    ) -> Runnable:
        """RAG 체인 구성"""
        pass
    
    def stream_response(
        self,
        query: str
    ) -> Iterator[str]:
        """스트리밍 방식으로 응답 생성"""
        pass
    
    def get_source_documents(
        self,
        query: str
    ) -> List[Document]:
        """검색된 소스 문서 반환"""
        pass
```

### 5. GradioInterface

Gradio UI 구성 및 이벤트 처리를 담당하는 컴포넌트 (2탭 구조)

```python
class GradioInterface:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store_manager = VectorStoreManager()
        self.rag_chain = RAGChain()
        self.template_manager = TemplateManager()
        self.comparison_engine = ComparisonEngine()
        self.chat_history = []
        self.current_file_path = None
    
    def create_interface(self) -> gr.Blocks:
        """
        Gradio 인터페이스 생성 (2탭 구조)
        - 탭 1: 프롬프트 (실시간 질의응답)
        - 탭 2: 성능 비교 (템플릿 비교)
        """
        pass
    
    def _create_prompt_tab(self):
        """
        프롬프트 탭 생성
        - 파일 업로드 (자동 처리)
        - RAG 파라미터 설정
        - 템플릿 저장/로드
        - 질의응답 (스트리밍)
        """
        pass
    
    def _create_comparison_tab(self):
        """
        성능 비교 탭 생성
        - 템플릿 선택 (최대 4개)
        - 비교 실행 (병렬 처리)
        - 결과 표시 (마크다운)
        - CSV 내보내기
        """
        pass
    
    def handle_query_v2(
        self,
        query: str,
        file: gr.File,
        # ... 모든 RAG 파라미터
    ) -> Iterator[Tuple[str, str]]:
        """
        질의응답 이벤트 핸들러 (자동 처리 + 스트리밍)
        1. 파일이 있으면 자동으로 문서 처리
        2. 임베딩 차원 확인 및 벡터 스토어 초기화
        3. 스트리밍 응답 생성
        """
        pass
    
    def handle_save_template_v2(
        self,
        template_name: str,
        # ... 모든 RAG 파라미터
    ) -> Tuple[str, gr.update, gr.update]:
        """
        템플릿 저장 핸들러
        - JSON 파일에 저장
        - 드롭다운 및 체크박스 자동 업데이트
        """
        pass
    
    def handle_comparison(
        self,
        selected_templates: List[str],
        comparison_file: gr.File,
        comparison_query: str
    ) -> str:
        """
        템플릿 비교 핸들러
        - ComparisonEngine을 사용한 병렬 처리
        - 마크다운 형식으로 결과 반환
        """
        pass
    
    def update_embedding_options(
        self,
        embedding_type: str
    ) -> Tuple[gr.update, gr.update, gr.update]:
        """
        임베딩 타입 변경 시 모델 목록 및 차원 자동 업데이트
        - OpenAI: text-embedding-3 모델, 1536 차원
        - HuggingFace: BAAI/bge-m3 등, 1024 차원
        - Ollama: bge-m3 등, 1024 차원
        """
        pass
```

### 6. TemplateManager

템플릿 저장 및 관리를 담당하는 컴포넌트

```python
class TemplateManager:
    def __init__(self, template_file: str = "templates.json"):
        self.template_file = template_file
        self.templates: Dict[str, RAGConfig] = {}
    
    def save_template(
        self,
        name: str,
        config: RAGConfig
    ) -> bool:
        """
        템플릿을 JSON 파일에 저장
        """
        pass
    
    def load_template(
        self,
        name: str
    ) -> Optional[RAGConfig]:
        """
        템플릿을 JSON 파일에서 로드
        """
        pass
    
    def list_templates(self) -> List[str]:
        """
        저장된 템플릿 이름 목록 반환
        """
        pass
    
    def delete_template(
        self,
        name: str
    ) -> bool:
        """
        템플릿 삭제
        """
        pass
```

### 7. ComparisonEngine

여러 템플릿을 병렬로 실행하고 결과를 비교하는 컴포넌트

```python
class ComparisonEngine:
    def __init__(self):
        self.results: List[ComparisonResult] = []
    
    def run_comparison(
        self,
        query: str,
        templates: List[Tuple[str, RAGConfig]],
        current_file_path: str,
        # ... 컴포넌트 인스턴스들
    ) -> List[ComparisonResult]:
        """
        여러 템플릿으로 응답 생성 및 비교 (병렬 처리)
        - ThreadPoolExecutor 사용 (max_workers=3)
        - 각 스레드별 독립 벡터 스토어 디렉토리
        """
        pass
    
    def _process_single_template(
        self,
        template_name: str,
        config: RAGConfig,
        query: str,
        current_file_path: str
    ) -> ComparisonResult:
        """
        단일 템플릿 처리 (병렬 실행용)
        - 독립적인 컴포넌트 인스턴스 생성
        - 고유 벡터 스토어 디렉토리 사용 (thread_id 기반)
        """
        pass
    
    def export_to_csv(
        self,
        results: List[ComparisonResult],
        filename: str
    ) -> str:
        """
        비교 결과를 CSV 파일로 내보내기
        - UTF-8-sig 인코딩 사용
        - 타임스탬프 포함
        """
        pass
```

## Data Models

### Document

LangChain의 Document 객체를 사용

```python
@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ChatMessage

대화 이력을 저장하는 데이터 모델

```python
@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: Optional[List[Document]] = None
```

### RAGConfig

RAG 시스템 설정을 저장하는 데이터 모델 (템플릿 저장용)

```python
@dataclass
class RAGConfig:
    # 문서 처리 설정
    splitter_type: str = "RecursiveCharacterTextSplitter"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", " ", ""])
    
    # 임베딩 설정
    embedding_type: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # 벡터 저장소 설정
    vector_store_type: str = "chroma"
    
    # 검색 설정
    search_type: str = "similarity"
    k: int = 4
    score_threshold: float = 0.5
    lambda_mult: float = 0.5
    
    # LLM 설정
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
```

### ComparisonResult

템플릿 비교 결과를 저장하는 데이터 모델

```python
@dataclass
class ComparisonResult:
    template_name: str
    query: str
    response: str
    response_length: int
    generation_time: float
    source_document_count: int
    config_summary: str
    timestamp: str
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 문서 로드 후 청크 생성

*For any* valid PDF document, when loaded and split with valid parameters (chunk_size > 0, chunk_overlap >= 0), the system should produce at least one document chunk.

**Validates: Requirements 1.1, 2.1, 2.2**

### Property 2: 청크 크기 제약

*For any* document split with chunk_size parameter, each generated chunk (except possibly the last one) should have a length that does not exceed chunk_size by more than the length of the longest separator.

**Validates: Requirements 2.1**

### Property 3: 청크 중복 보존

*For any* two consecutive chunks generated with chunk_overlap > 0, the end of the first chunk should overlap with the beginning of the second chunk by approximately chunk_overlap characters.

**Validates: Requirements 2.2**

### Property 4: 임베딩 차원 일관성

*For any* embedding model, all generated embedding vectors should have the same dimension, and this dimension should match the model's specified embedding dimension.

**Validates: Requirements 3.4**

### Property 5: 벡터 저장소 문서 보존

*For any* set of documents added to the vector store, the number of documents retrieved by querying the store should equal the number of documents added (assuming no deletions).

**Validates: Requirements 4.2, 4.3**

### Property 6: 검색 결과 개수

*For any* query with parameter k, the retriever should return at most k documents (or fewer if the total number of documents in the store is less than k).

**Validates: Requirements 5.1**

### Property 7: 유사도 임계값 필터링

*For any* query using similarity_score_threshold search type with threshold t, all returned documents should have a similarity score greater than or equal to t.

**Validates: Requirements 5.3**

### Property 8: MMR 다양성

*For any* query using MMR search type with lambda_mult parameter, the returned documents should be more diverse than those returned by pure similarity search (measured by average pairwise similarity).

**Validates: Requirements 5.4**

### Property 9: 스트리밍 응답 완전성

*For any* query, the concatenation of all streamed response chunks should equal the complete response that would be generated in non-streaming mode.

**Validates: Requirements 7.4**

### Property 10: 소스 문서 추적

*For any* generated response, the system should be able to provide the source documents that were used to generate the response, and these documents should be among those retrieved by the retriever.

**Validates: Requirements 7.5, 8.1, 8.2, 8.3**

### Property 11: 대화 이력 순서 보존

*For any* sequence of questions and answers, the chat history should maintain the chronological order of all interactions.

**Validates: Requirements 9.1, 9.2**

### Property 12: 파라미터 변경 반영

*For any* parameter change (embedding model, chunk size, search type, etc.), subsequent operations should use the new parameter values.

**Validates: Requirements 10.4**

### Property 13: 빈 벡터 저장소 처리

*For any* query when the vector store is empty, the system should return an appropriate error message rather than attempting to generate a response.

**Validates: Requirements 7.6**

### Property 14: 파일 형식 검증

*For any* uploaded file, if the file format is not supported (not PDF), the system should reject the file and display an error message without attempting to process it.

**Validates: Requirements 1.3**

## Error Handling

### 1. 파일 업로드 오류

- **지원하지 않는 파일 형식**: 사용자에게 명확한 오류 메시지 표시 및 지원 형식 안내
- **파일 읽기 실패**: 파일 손상 또는 권한 문제 시 구체적인 오류 메시지 제공
- **파일 크기 제한**: 너무 큰 파일의 경우 경고 메시지 및 권장 크기 안내

### 2. 임베딩 생성 오류

- **API 키 누락**: OpenAI API 키가 설정되지 않은 경우 명확한 안내 메시지
- **API 호출 실패**: 네트워크 오류 또는 API 제한 시 재시도 로직 및 사용자 알림
- **모델 로드 실패**: HuggingFace 모델 다운로드 실패 시 대체 모델 제안
- **차원 불일치**: 임베딩 차원 변경 감지 시 자동으로 벡터 스토어 초기화

### 3. 벡터 저장소 오류

- **저장소 초기화 실패**: 메모리 부족 또는 권한 문제 시 오류 메시지 및 해결 방법 제시
- **문서 추가 실패**: 임베딩 차원 불일치 등의 문제 시 자동 초기화 시도
- **삭제 실패**: 최대 10회 재시도, 실패 시 디렉토리 이름 변경으로 우회
- **병렬 처리 충돌**: 각 스레드별 독립 디렉토리 사용으로 충돌 방지

### 4. 검색 오류

- **빈 벡터 저장소**: 문서를 먼저 업로드하라는 안내 메시지
- **검색 실패**: 쿼리 처리 중 오류 발생 시 사용자에게 알림 및 재시도 옵션 제공

### 5. LLM 응답 생성 오류

- **API 호출 실패**: OpenAI API 오류 시 재시도 로직 및 사용자 알림
- **토큰 제한 초과**: max_tokens 조정 권장 메시지
- **스트리밍 중단**: 네트워크 문제로 스트리밍이 중단된 경우 부분 응답 표시 및 재시도 옵션

### 6. 파라미터 검증 오류

- **잘못된 파라미터 값**: chunk_size < chunk_overlap 등의 논리적 오류 시 경고 메시지
- **범위 초과**: 슬라이더 범위를 벗어난 값 입력 시 자동 보정 및 알림
- **타입 변환 오류**: 템플릿 저장 시 명시적 타입 변환 (int, float) 수행

### 7. 템플릿 관리 오류

- **템플릿 저장 실패**: JSON 파일 쓰기 실패 시 오류 메시지
- **템플릿 로드 실패**: 존재하지 않는 템플릿 선택 시 안내 메시지
- **템플릿 삭제 실패**: 파일 권한 문제 시 오류 메시지

### 8. 병렬 처리 오류

- **스레드 실행 실패**: 개별 템플릿 처리 실패 시 해당 템플릿만 오류 표시, 나머지는 계속 진행
- **리소스 부족**: max_workers=3으로 제한하여 과도한 리소스 사용 방지
- **벡터 스토어 충돌**: 각 스레드별 고유 디렉토리 사용으로 충돌 방지

## Testing Strategy

### Unit Tests

각 컴포넌트의 개별 기능을 테스트합니다:

1. **DocumentProcessor 테스트**
   - PDF 로드 기능 테스트
   - 다양한 splitter 타입 테스트
   - 통계 정보 계산 테스트

2. **EmbeddingManager 테스트**
   - 각 임베딩 모델 초기화 테스트
   - 임베딩 차원 확인 테스트
   - 쿼리 임베딩 생성 테스트

3. **VectorStoreManager 테스트**
   - Chroma 및 FAISS 초기화 테스트
   - 문서 추가 및 검색 테스트
   - 다양한 검색 타입 테스트

4. **RAGChain 테스트**
   - LLM 초기화 테스트
   - 체인 구성 테스트
   - 스트리밍 응답 생성 테스트

### Property-Based Tests

각 correctness property를 검증하는 property-based tests를 작성합니다. Python의 `hypothesis` 라이브러리를 사용하여 최소 100회 반복 테스트를 수행합니다.

각 property test는 다음 형식의 주석을 포함해야 합니다:
```python
# Feature: chatbot, Property 1: 문서 로드 후 청크 생성
# Validates: Requirements 1.1, 2.1, 2.2
```

### Integration Tests

전체 RAG 파이프라인의 통합 테스트:

1. **End-to-End 테스트**: 파일 업로드부터 응답 생성까지 전체 흐름 테스트
2. **파라미터 변경 테스트**: 다양한 파라미터 조합으로 시스템 동작 검증
3. **오류 시나리오 테스트**: 각 오류 상황에서 적절한 처리 확인

### Performance Tests

1. **대용량 문서 처리**: 100페이지 이상의 PDF 처리 성능 측정
2. **검색 속도**: 1000개 이상의 청크에서 검색 속도 측정
3. **스트리밍 지연**: 응답 스트리밍의 첫 토큰 생성 시간 측정

### Test Configuration

- Property tests: 최소 100회 반복
- 각 테스트는 독립적으로 실행 가능해야 함
- Mock 사용 최소화, 실제 API 호출은 환경 변수로 제어
- CI/CD 파이프라인에서 자동 실행
