# Requirements Document

## Introduction

RAG 기반 챗봇 웹 애플리케이션으로, 사용자가 문서를 업로드하고 다양한 RAG 파라미터를 동적으로 조정하면서 질의응답을 수행할 수 있는 시스템입니다. Gradio를 사용한 2탭 구조의 인터페이스를 제공하며, 프롬프트 탭에서는 실시간 스트리밍 응답을, 성능 비교 탭에서는 여러 템플릿을 병렬로 비교할 수 있습니다.

## Glossary

- **RAG_System**: Retrieval-Augmented Generation 기반 질의응답 시스템
- **Embedding_Model**: 텍스트를 벡터로 변환하는 임베딩 모델
- **Vector_Store**: 임베딩 벡터를 저장하고 검색하는 벡터 데이터베이스
- **Text_Splitter**: 문서를 작은 청크로 분할하는 컴포넌트
- **LLM**: Large Language Model, 응답 생성을 위한 언어 모델
- **Streaming**: 응답을 실시간으로 점진적으로 표시하는 방식
- **Chunk**: 분할된 텍스트 조각
- **Retriever**: 벡터 저장소에서 관련 문서를 검색하는 컴포넌트
- **Template**: RAG 파라미터 조합을 저장한 설정 프리셋
- **Comparison_Engine**: 여러 템플릿을 병렬로 실행하고 결과를 비교하는 컴포넌트

## Requirements

### Requirement 1: 문서 업로드 및 자동 처리

**User Story:** As a 사용자, I want to 문서를 업로드하고 질문 시 자동으로 처리되기를, so that 별도의 처리 버튼 없이 즉시 질의응답을 수행할 수 있다

#### Acceptance Criteria

1. WHEN 사용자가 PDF 파일을 업로드하면, THE RAG_System SHALL 파일 경로를 저장한다
2. WHEN 사용자가 질문을 제출하면, THE RAG_System SHALL 자동으로 문서를 로드하고 처리한다
3. IF 문서가 업로드되지 않았으면, THEN THE RAG_System SHALL 오류 메시지를 표시한다
4. THE RAG_System SHALL PDF 파일만 업로드할 수 있도록 제한한다

### Requirement 2: 텍스트 분할 설정

**User Story:** As a 사용자, I want to 텍스트 분할 파라미터를 조정할 수 있기를, so that 다양한 청크 크기로 문서를 분할하고 성능을 비교할 수 있다

#### Acceptance Criteria

1. THE RAG_System SHALL chunk_size 파라미터를 100에서 2000 사이의 값으로 설정할 수 있는 슬라이더를 제공한다
2. THE RAG_System SHALL chunk_overlap 파라미터를 0에서 500 사이의 값으로 설정할 수 있는 슬라이더를 제공한다
3. THE RAG_System SHALL 텍스트 분할 방식을 선택할 수 있는 드롭다운을 제공한다 (RecursiveCharacterTextSplitter, CharacterTextSplitter)
4. THE RAG_System SHALL 분할 구분자를 사용자가 입력할 수 있는 텍스트 필드를 제공한다

### Requirement 3: 임베딩 모델 선택 및 자동 업데이트

**User Story:** As a 사용자, I want to 다양한 임베딩 모델을 선택하고 모델별 옵션이 자동으로 업데이트되기를, so that 각 모델의 성능과 특성을 쉽게 비교할 수 있다

#### Acceptance Criteria

1. THE RAG_System SHALL OpenAI, HuggingFace, Ollama 임베딩 타입 중 선택할 수 있는 드롭다운을 제공한다
2. WHEN OpenAI가 선택되면, THE RAG_System SHALL text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 모델 목록을 표시하고 차원을 1536으로 설정한다
3. WHEN HuggingFace가 선택되면, THE RAG_System SHALL BAAI/bge-m3, sentence-transformers/all-MiniLM-L6-v2 등의 모델 목록을 표시하고 차원을 1024로 설정한다
4. WHEN Ollama가 선택되면, THE RAG_System SHALL bge-m3, nomic-embed-text, mxbai-embed-large 모델 목록을 표시하고 차원을 1024로 설정한다
5. WHERE OpenAI text-embedding-3 모델이 선택되면, THE RAG_System SHALL 사용자가 차원을 256에서 3072 사이로 조정할 수 있는 슬라이더를 제공한다

### Requirement 4: 임베딩 차원 변경 감지 및 자동 초기화

**User Story:** As a 사용자, I want to 임베딩 차원이 변경되면 벡터 스토어가 자동으로 초기화되기를, so that 차원 불일치 오류 없이 시스템을 사용할 수 있다

#### Acceptance Criteria

1. WHEN 임베딩 차원이 변경되면, THE RAG_System SHALL 이전 차원과 새 차원을 비교한다
2. IF 차원이 다르면, THEN THE RAG_System SHALL 기존 벡터 스토어를 삭제한다
3. THE RAG_System SHALL 차원 변경 감지 시 콘솔에 경고 메시지를 출력한다
4. THE RAG_System SHALL 새로운 차원으로 벡터 스토어를 생성한다

### Requirement 5: 벡터 저장소 설정

**User Story:** As a 사용자, I want to 벡터 저장소 유형을 선택할 수 있기를, so that 다양한 저장소의 특성을 비교할 수 있다

#### Acceptance Criteria

1. THE RAG_System SHALL Chroma, FAISS 벡터 저장소 중 선택할 수 있는 드롭다운을 제공한다
2. WHEN 문서가 처리되면, THE RAG_System SHALL 선택된 벡터 저장소에 임베딩을 저장한다
3. THE RAG_System SHALL 벡터 저장소 초기화 시 최대 10회 재시도 로직을 수행한다
4. THE RAG_System SHALL 벡터 저장소 삭제 실패 시 디렉토리 이름을 변경하여 우회한다

### Requirement 6: 검색 파라미터 설정

**User Story:** As a 사용자, I want to 검색 파라미터를 조정할 수 있기를, so that 검색 결과의 품질과 개수를 제어할 수 있다

#### Acceptance Criteria

1. THE RAG_System SHALL 검색할 문서 개수(k)를 1에서 10 사이의 값으로 설정할 수 있는 슬라이더를 제공한다
2. THE RAG_System SHALL 검색 방식을 선택할 수 있는 드롭다운을 제공한다 (similarity, mmr, similarity_score_threshold)
3. WHERE similarity_score_threshold가 선택되면, THE RAG_System SHALL 임계값을 0.0에서 1.0 사이로 설정할 수 있는 슬라이더를 제공한다
4. WHERE mmr이 선택되면, THE RAG_System SHALL lambda_mult 파라미터를 0.0에서 1.0 사이로 설정할 수 있는 슬라이더를 제공한다

### Requirement 7: LLM 모델 선택

**User Story:** As a 사용자, I want to 응답 생성에 사용할 LLM 모델을 선택할 수 있기를, so that 다양한 모델의 응답 품질을 비교할 수 있다

#### Acceptance Criteria

1. THE RAG_System SHALL OpenAI GPT 모델을 선택할 수 있는 드롭다운을 제공한다 (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
2. THE RAG_System SHALL temperature 파라미터를 0.0에서 2.0 사이의 값으로 설정할 수 있는 슬라이더를 제공한다
3. THE RAG_System SHALL max_tokens 파라미터를 100에서 4000 사이의 값으로 설정할 수 있는 슬라이더를 제공한다

### Requirement 8: 프롬프트 탭 - 실시간 질의응답

**User Story:** As a 사용자, I want to 프롬프트 탭에서 질문을 입력하고 스트리밍 방식으로 응답을 받을 수 있기를, so that 실시간으로 답변을 확인할 수 있다

#### Acceptance Criteria

1. THE RAG_System SHALL 사용자 질문을 입력할 수 있는 텍스트 입력 필드를 제공한다
2. WHEN 사용자가 질문을 제출하면, THE RAG_System SHALL 자동으로 문서를 처리하고 벡터 저장소에 저장한다
3. THE RAG_System SHALL 벡터 저장소에서 관련 문서를 검색한다
4. THE RAG_System SHALL LLM을 사용하여 응답을 스트리밍 방식으로 생성한다
5. THE RAG_System SHALL 검색된 소스 문서 정보를 표시한다
6. THE RAG_System SHALL 대화 이력을 챗봇 형식으로 표시한다
7. THE RAG_System SHALL 대화 이력 초기화 버튼을 제공한다

### Requirement 9: 템플릿 관리

**User Story:** As a 사용자, I want to RAG 파라미터 조합을 템플릿으로 저장하고 관리할 수 있기를, so that 자주 사용하는 설정을 재사용하고 비교할 수 있다

#### Acceptance Criteria

1. THE RAG_System SHALL 템플릿 이름과 설명을 입력할 수 있는 필드를 제공한다
2. THE RAG_System SHALL 모든 RAG 파라미터를 포함한 템플릿을 JSON 파일로 저장한다
3. THE RAG_System SHALL 저장된 템플릿 목록을 드롭다운으로 표시한다
4. THE RAG_System SHALL 템플릿을 불러와서 모든 파라미터를 자동으로 설정한다
5. THE RAG_System SHALL 템플릿을 삭제할 수 있는 기능을 제공한다
6. THE RAG_System SHALL 템플릿 저장/삭제 시 드롭다운과 체크박스를 자동으로 업데이트한다

### Requirement 10: 성능 비교 탭 - 병렬 템플릿 비교

**User Story:** As a 사용자, I want to 여러 템플릿을 동시에 실행하고 결과를 비교할 수 있기를, so that 최적의 RAG 설정을 찾을 수 있다

#### Acceptance Criteria

1. THE RAG_System SHALL 최대 4개의 템플릿을 선택할 수 있는 체크박스 그룹을 제공한다
2. THE RAG_System SHALL 비교할 문서를 업로드할 수 있는 파일 업로드 필드를 제공한다
3. THE RAG_System SHALL 비교할 질문을 입력할 수 있는 텍스트 필드를 제공한다
4. WHEN 비교 실행 버튼을 클릭하면, THE RAG_System SHALL 선택된 템플릿들을 병렬로 실행한다
5. THE RAG_System SHALL ThreadPoolExecutor를 사용하여 최대 3개의 템플릿을 동시에 처리한다
6. THE RAG_System SHALL 각 템플릿마다 독립적인 벡터 스토어 디렉토리를 사용한다
7. THE RAG_System SHALL 각 템플릿의 응답, 응답 길이, 생성 시간, 소스 문서 수를 표시한다
8. THE RAG_System SHALL 비교 결과를 마크다운 형식으로 표시한다

### Requirement 11: CSV 내보내기

**User Story:** As a 사용자, I want to 비교 결과를 CSV 파일로 다운로드할 수 있기를, so that 결과를 분석하고 보고서를 작성할 수 있다

#### Acceptance Criteria

1. THE RAG_System SHALL CSV 다운로드 버튼을 제공한다
2. WHEN CSV 다운로드 버튼을 클릭하면, THE RAG_System SHALL 비교 결과를 CSV 파일로 저장한다
3. THE RAG_System SHALL CSV 파일 이름에 타임스탬프를 포함한다
4. THE RAG_System SHALL CSV 파일에 템플릿 이름, 질문, 응답, 응답 길이, 생성 시간, 소스 문서 수, 설정 요약을 포함한다
5. THE RAG_System SHALL UTF-8-sig 인코딩을 사용하여 한글을 올바르게 저장한다

### Requirement 12: 페이지 로드 시 템플릿 자동 로드

**User Story:** As a 사용자, I want to 페이지를 새로고침해도 저장된 템플릿이 자동으로 표시되기를, so that 매번 수동으로 새로고침할 필요가 없다

#### Acceptance Criteria

1. WHEN 페이지가 로드되면, THE RAG_System SHALL templates.json 파일에서 템플릿 목록을 읽는다
2. THE RAG_System SHALL 템플릿 드롭다운의 초기 choices를 템플릿 목록으로 설정한다
3. THE RAG_System SHALL 템플릿 체크박스 그룹의 초기 choices를 템플릿 목록으로 설정한다
4. THE RAG_System SHALL 페이지 새로고침 시에도 템플릿 목록이 표시되도록 한다

### Requirement 13: 에러 처리 및 복구

**User Story:** As a 사용자, I want to 에러 발생 시 명확한 메시지를 받고 시스템이 복구되기를, so that 문제를 이해하고 해결할 수 있다

#### Acceptance Criteria

1. IF 임베딩 모델 초기화 실패 시, THEN THE RAG_System SHALL 구체적인 오류 메시지를 표시한다
2. IF 벡터 스토어 추가 실패 시, THEN THE RAG_System SHALL 차원 불일치 여부를 확인하고 자동으로 초기화한다
3. IF 템플릿 비교 중 오류 발생 시, THEN THE RAG_System SHALL 해당 템플릿의 오류 메시지를 결과에 포함한다
4. THE RAG_System SHALL 모든 예외를 catch하고 사용자 친화적인 메시지로 변환한다

