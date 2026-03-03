# Implementation Plan: Chatbot

## Overview

RAG 기반 챗봇 웹 애플리케이션을 Python, LangChain, Gradio를 사용하여 구현합니다. 구현은 컴포넌트별로 점진적으로 진행하며, 각 단계에서 핵심 기능을 검증합니다.

**현재 상태**: 핵심 기능 구현 완료. 2탭 구조 (프롬프트, 성능 비교), 자동 처리, 템플릿 관리, 병렬 비교, 차원 변경 자동 감지 등 모든 주요 기능이 동작합니다.

## Tasks

- [x] 1. 프로젝트 구조 및 환경 설정
  - Python 가상환경 생성 및 필수 패키지 설치 (langchain, langchain-openai, langchain-community, langchain-chroma, faiss-cpu, gradio, python-dotenv, pypdf)
  - 프로젝트 디렉토리 구조 생성 (src/, tests/, data/)
  - .env 파일 템플릿 생성 (OPENAI_API_KEY)
  - _Requirements: 모든 requirements_

- [ ] 2. DocumentProcessor 컴포넌트 구현
  - [x] 2.1 PDF 문서 로더 구현
    - PyPDFLoader를 사용한 문서 로드 기능
    - 문서 통계 정보 추출 (페이지 수, 텍스트 길이)
    - _Requirements: 1.1, 1.2, 1.4_

  - [x] 2.2 텍스트 분할기 구현
    - RecursiveCharacterTextSplitter 및 CharacterTextSplitter 지원
    - 동적 파라미터 설정 (chunk_size, chunk_overlap, separators)
    - 분할 결과 통계 제공
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 2.3 DocumentProcessor 단위 테스트 작성
    - PDF 로드 테스트
    - 다양한 분할 파라미터 조합 테스트
    - _Requirements: 1.1, 2.1, 2.2_

  - [ ]* 2.4 Property test: 문서 로드 후 청크 생성
    - **Property 1: 문서 로드 후 청크 생성**
    - **Validates: Requirements 1.1, 2.1, 2.2**

  - [ ]* 2.5 Property test: 청크 크기 제약
    - **Property 2: 청크 크기 제약**
    - **Validates: Requirements 2.1**

  - [ ]* 2.6 Property test: 청크 중복 보존
    - **Property 3: 청크 중복 보존**
    - **Validates: Requirements 2.2**

- [ ] 3. EmbeddingManager 컴포넌트 구현
  - [x] 3.1 임베딩 모델 초기화 구현
    - OpenAI Embeddings 지원 (text-embedding-3-small, text-embedding-3-large)
    - HuggingFace Embeddings 지원 (BAAI/bge-m3)
    - Ollama Embeddings 지원 (선택적)
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 3.2 임베딩 차원 정보 제공
    - 각 모델의 임베딩 차원 수 반환
    - _Requirements: 3.4_

  - [ ]* 3.3 EmbeddingManager 단위 테스트 작성
    - 각 임베딩 모델 초기화 테스트
    - 임베딩 생성 테스트
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 3.4 Property test: 임베딩 차원 일관성
    - **Property 4: 임베딩 차원 일관성**
    - **Validates: Requirements 3.4**

- [ ] 4. VectorStoreManager 컴포넌트 구현
  - [x] 4.1 벡터 저장소 초기화 구현
    - Chroma 벡터 저장소 지원
    - FAISS 벡터 저장소 지원
    - 저장소 초기화 및 클리어 기능
    - _Requirements: 4.1, 4.2, 4.4_

  - [x] 4.2 문서 추가 및 검색기 생성
    - 문서를 벡터 저장소에 추가
    - 다양한 검색 타입 지원 (similarity, mmr, similarity_score_threshold)
    - 검색 파라미터 설정 (k, score_threshold, lambda_mult)
    - _Requirements: 4.2, 5.1, 5.2, 5.3, 5.4_

  - [x] 4.3 저장소 통계 정보 제공
    - 저장된 문서 수 반환
    - _Requirements: 4.3_

  - [ ]* 4.4 VectorStoreManager 단위 테스트 작성
    - 벡터 저장소 초기화 테스트
    - 문서 추가 및 검색 테스트
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 4.5 Property test: 벡터 저장소 문서 보존
    - **Property 5: 벡터 저장소 문서 보존**
    - **Validates: Requirements 4.2, 4.3**

  - [ ]* 4.6 Property test: 검색 결과 개수
    - **Property 6: 검색 결과 개수**
    - **Validates: Requirements 5.1**

  - [ ]* 4.7 Property test: 유사도 임계값 필터링
    - **Property 7: 유사도 임계값 필터링**
    - **Validates: Requirements 5.3**

  - [ ]* 4.8 Property test: MMR 다양성
    - **Property 8: MMR 다양성**
    - **Validates: Requirements 5.4**

- [ ] 5. RAGChain 컴포넌트 구현
  - [x] 5.1 LLM 초기화 구현
    - OpenAI ChatGPT 모델 지원 (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
    - 동적 파라미터 설정 (temperature, max_tokens)
    - 스트리밍 모드 활성화
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 5.2 RAG 체인 구성
    - Retriever와 LLM을 연결하는 체인 구성
    - 프롬프트 템플릿 정의
    - _Requirements: 7.2, 7.3_

  - [x] 5.3 스트리밍 응답 생성 구현
    - 스트리밍 방식으로 응답 생성
    - 소스 문서 추적 및 반환
    - _Requirements: 7.4, 7.5_

  - [ ]* 5.4 RAGChain 단위 테스트 작성
    - LLM 초기화 테스트
    - 체인 구성 테스트
    - 스트리밍 응답 테스트
    - _Requirements: 6.1, 7.3, 7.4_

  - [ ]* 5.5 Property test: 스트리밍 응답 완전성
    - **Property 9: 스트리밍 응답 완전성**
    - **Validates: Requirements 7.4**

  - [ ]* 5.6 Property test: 소스 문서 추적
    - **Property 10: 소스 문서 추적**
    - **Validates: Requirements 7.5, 8.1, 8.2, 8.3**

- [ ] 6. Checkpoint - 핵심 컴포넌트 검증
  - 모든 핵심 컴포넌트가 독립적으로 동작하는지 확인
  - 간단한 통합 테스트 실행 (문서 로드 → 임베딩 → 벡터 저장 → 검색 → 응답 생성)
  - 질문이 있으면 사용자에게 문의

- [x] 6.5 TemplateManager 컴포넌트 구현
  - [x] 6.5.1 템플릿 저장 기능
    - RAGConfig를 JSON 파일로 저장
    - 템플릿 이름 중복 확인
    - _Requirements: 9.1, 9.2_
  
  - [x] 6.5.2 템플릿 로드 기능
    - JSON 파일에서 템플릿 로드
    - RAGConfig 객체로 변환
    - _Requirements: 9.3, 9.4_
  
  - [x] 6.5.3 템플릿 목록 및 삭제 기능
    - 저장된 템플릿 목록 반환
    - 템플릿 삭제
    - _Requirements: 9.5, 9.6_

- [x] 6.6 ComparisonEngine 컴포넌트 구현
  - [x] 6.6.1 병렬 처리 구현
    - ThreadPoolExecutor 사용 (max_workers=3)
    - 각 템플릿별 독립 실행
    - _Requirements: 10.5, 10.6_
  
  - [x] 6.6.2 독립 벡터 스토어 관리
    - 각 스레드별 고유 디렉토리 생성 (thread_id 기반)
    - 벡터 스토어 충돌 방지
    - _Requirements: 10.6_
  
  - [x] 6.6.3 결과 수집 및 포맷팅
    - ComparisonResult 데이터 모델 사용
    - 마크다운 형식으로 결과 포맷팅
    - _Requirements: 10.7_
  
  - [x] 6.6.4 CSV 내보내기
    - UTF-8-sig 인코딩 사용
    - 타임스탬프 포함 파일명
    - _Requirements: 11.1-11.5_

- [x] 6.7 차원 변경 감지 및 자동 초기화
  - [x] 6.7.1 EmbeddingManager 차원 추적
    - custom_dimensions 속성 추가
    - get_embedding_dimension() 우선순위 수정
    - _Requirements: 3.5, 4.1_
  
  - [x] 6.7.2 VectorStoreManager 차원 감지
    - current_embedding_dim 속성 추가
    - initialize_vector_store()에서 차원 비교
    - 차원 변경 시 자동 clear_store() 호출
    - _Requirements: 4.1, 4.2_

- [ ] 7. GradioInterface 구현 - 기본 UI
  - [x] 7.1 파일 업로드 UI 구성
    - 파일 업로드 컴포넌트
    - 문서 처리 상태 표시
    - _Requirements: 1.1, 1.2, 1.4_

  - [x] 7.2 텍스트 분할 설정 UI 구성
    - chunk_size 슬라이더 (100-2000)
    - chunk_overlap 슬라이더 (0-500)
    - splitter_type 드롭다운
    - separators 텍스트 입력
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

  - [x] 7.3 임베딩 모델 선택 UI 구성
    - 임베딩 타입 드롭다운 (OpenAI, HuggingFace, Ollama)
    - 모델 이름 드롭다운 (타입별 옵션)
    - 임베딩 차원 표시 및 조정 (OpenAI text-embedding-3)
    - 임베딩 타입 변경 시 자동 모델 목록 업데이트
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 7.4 벡터 저장소 설정 UI 구성
    - 저장소 타입 드롭다운 (Chroma, FAISS)
    - 저장소 초기화 버튼
    - 저장된 문서 수 표시
    - _Requirements: 4.1, 4.3, 4.4_
  
  - [x] 7.5 템플릿 관리 UI 구성
    - 템플릿 이름 및 설명 입력 필드
    - 템플릿 저장 버튼
    - 템플릿 선택 드롭다운 (자동 로드)
    - 템플릿 삭제 버튼
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [ ] 8. GradioInterface 구현 - 검색 및 LLM 설정 UI
  - [x] 8.1 검색 파라미터 설정 UI 구성
    - k 슬라이더 (1-10)
    - search_type 드롭다운
    - score_threshold 슬라이더 (조건부 표시)
    - lambda_mult 슬라이더 (조건부 표시)
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 8.2 LLM 설정 UI 구성
    - 모델 선택 드롭다운
    - temperature 슬라이더 (0.0-2.0)
    - max_tokens 슬라이더 (100-4000)
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 8.3 질의응답 UI 구성 (프롬프트 탭)
    - 질문 입력 텍스트박스
    - 응답 표시 영역 (스트리밍 지원)
    - 검색된 소스 문서 표시 (아코디언)
    - 자동 문서 처리 (질문 제출 시)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4_

  - [x] 8.4 대화 이력 UI 구성
    - 대화 이력 표시 (챗봇 형식)
    - 이력 초기화 버튼
    - 타임스탬프 표시
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 8.5 시스템 상태 표시 UI 구성
    - 현재 설정 요약 표시
    - 문서 정보 표시
    - 벡터 저장소 상태 표시
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  
  - [x] 8.6 성능 비교 탭 UI 구성
    - 템플릿 선택 체크박스 (최대 4개)
    - 비교용 문서 업로드
    - 비교용 질문 입력
    - 비교 실행 버튼
    - 결과 표시 (마크다운)
    - CSV 다운로드 버튼
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 11.1-11.5_

- [ ] 9. 이벤트 핸들러 구현
  - [x] 9.1 질의응답 핸들러 (자동 처리 + 스트리밍)
    - 파일 업로드 시 자동 문서 처리
    - 임베딩 차원 확인 및 벡터 스토어 초기화
    - 벡터 저장소 상태 확인
    - RAGChain을 사용한 스트리밍 응답 생성
    - 소스 문서 정보 추출 및 포맷팅
    - 대화 이력 업데이트
    - _Requirements: 1.1-1.4, 2.1-2.5, 3.1-3.4, 4.1-4.3, 7.1-7.6, 8.1-8.4, 9.1, 9.2_

  - [x] 9.2 템플릿 저장/로드 핸들러
    - 템플릿 저장 (JSON 파일)
    - 템플릿 로드 및 파라미터 자동 설정
    - 템플릿 삭제
    - 드롭다운 및 체크박스 자동 업데이트
    - 페이지 로드 시 템플릿 자동 표시
    - _Requirements: 9.1-9.6, 12.1-12.4_

  - [x] 9.3 템플릿 비교 핸들러
    - 선택된 템플릿 검증 (최대 4개)
    - ComparisonEngine을 사용한 병렬 처리
    - 마크다운 형식으로 결과 표시
    - CSV 내보내기
    - _Requirements: 10.1-10.8, 11.1-11.5_

  - [x] 9.4 파라미터 변경 핸들러
    - 임베딩 타입 변경 시 모델 목록 자동 업데이트
    - 임베딩 차원 자동 설정
    - 검색 타입 변경 시 조건부 파라미터 표시
    - _Requirements: 3.1-3.5, 10.4_

  - [x] 9.5 대화 이력 초기화 핸들러
    - 대화 이력 클리어
    - _Requirements: 9.3_

  - [ ]* 9.6 이벤트 핸들러 통합 테스트
    - 파일 업로드부터 응답 생성까지 전체 흐름 테스트
    - 다양한 파라미터 조합 테스트
    - _Requirements: 모든 requirements_

  - [ ]* 9.7 Property test: 빈 벡터 저장소 처리
    - **Property 13: 빈 벡터 저장소 처리**
    - **Validates: Requirements 7.6**

  - [ ]* 9.8 Property test: 파일 형식 검증
    - **Property 14: 파일 형식 검증**
    - **Validates: Requirements 1.3**

- [ ] 10. 오류 처리 구현
  - [x] 10.1 파일 업로드 오류 처리
    - 지원하지 않는 파일 형식 처리
    - 파일 읽기 실패 처리
    - 파일 크기 제한 처리
    - _Requirements: 1.3_

  - [x] 10.2 API 오류 처리
    - OpenAI API 키 누락 처리
    - API 호출 실패 및 재시도 로직
    - 네트워크 오류 처리
    - _Requirements: 3.1, 6.1_

  - [x] 10.3 벡터 저장소 오류 처리
    - 저장소 초기화 실패 처리
    - 문서 추가 실패 처리 (차원 불일치 자동 해결)
    - 검색 실패 처리
    - 삭제 실패 시 재시도 및 디렉토리 이름 변경
    - _Requirements: 4.1, 4.2, 7.6_

  - [x] 10.4 파라미터 검증 및 오류 처리
    - 잘못된 파라미터 값 검증
    - 논리적 오류 검증 (chunk_size < chunk_overlap)
    - 범위 초과 값 자동 보정
    - 템플릿 저장 시 타입 변환 (int, float)
    - _Requirements: 2.1, 2.2, 5.1, 6.2, 6.3_

  - [x] 10.5 병렬 처리 오류 처리
    - 개별 템플릿 실행 실패 처리
    - 리소스 부족 방지 (max_workers=3)
    - 벡터 스토어 충돌 방지 (독립 디렉토리)
    - _Requirements: 10.5, 10.6, 13.3_

  - [ ]* 10.6 오류 처리 테스트
    - 각 오류 시나리오 테스트
    - 오류 메시지 검증
    - _Requirements: 1.3, 7.6, 13.1-13.4_

- [ ] 11. Checkpoint - 전체 시스템 통합 테스트
  - 모든 기능이 통합되어 정상 동작하는지 확인
  - 다양한 사용 시나리오 테스트
  - 성능 테스트 (대용량 문서, 많은 청크)
  - 병렬 처리 성능 테스트 (3개 템플릿 동시 실행)
  - 질문이 있으면 사용자에게 문의

- [x] 12. 최종 마무리
  - [x] 12.1 코드 정리 및 리팩토링
    - 중복 코드 제거
    - 함수 및 클래스 문서화 (docstring)
    - 타입 힌트 추가
    - _Requirements: 모든 requirements_

  - [x] 12.2 README 작성
    - 프로젝트 설명
    - 설치 방법
    - 사용 방법
    - 환경 변수 설정 가이드
    - _Requirements: 모든 requirements_

  - [x] 12.3 requirements.txt 생성
    - 모든 의존성 패키지 목록 작성
    - 버전 명시
    - _Requirements: 모든 requirements_
  
  - [x] 12.4 SPECS 파일 업데이트
    - requirements.md 업데이트 (13개 요구사항)
    - design.md 업데이트 (2탭 구조, 병렬 처리, 차원 감지)
    - tasks.md 업데이트 (완료된 작업 표시)
    - _Requirements: 모든 requirements_

  - [ ]* 12.5 Property test: 대화 이력 순서 보존
    - **Property 11: 대화 이력 순서 보존**
    - **Validates: Requirements 9.1, 9.2**

  - [ ]* 12.6 Property test: 파라미터 변경 반영
    - **Property 12: 파라미터 변경 반영**
    - **Validates: Requirements 10.4**

- [x] 13. 최종 검증
  - 모든 핵심 기능 테스트 완료
  - 사용자 시나리오 기반 최종 테스트 완료
  - 문서 및 코드 최종 검토 완료
  - 2탭 구조 (프롬프트, 성능 비교) 정상 동작
  - 자동 처리, 템플릿 관리, 병렬 비교, 차원 변경 감지 모두 동작

## 완료된 주요 기능

1. ✅ 2탭 구조 UI (프롬프트, 성능 비교)
2. ✅ 자동 문서 처리 (질문 제출 시)
3. ✅ 임베딩 타입 변경 시 모델 목록 자동 업데이트
4. ✅ 임베딩 차원 변경 감지 및 자동 벡터 스토어 초기화
5. ✅ 템플릿 저장/로드/삭제 (JSON 기반)
6. ✅ 페이지 로드 시 템플릿 자동 표시
7. ✅ 병렬 템플릿 비교 (ThreadPoolExecutor, max_workers=3)
8. ✅ 각 스레드별 독립 벡터 스토어 디렉토리
9. ✅ CSV 내보내기 (UTF-8-sig 인코딩)
10. ✅ 스트리밍 응답 생성
11. ✅ 소스 문서 추적 및 표시
12. ✅ 대화 이력 관리
13. ✅ 포괄적인 에러 처리 및 복구

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (minimum 100 iterations each)
- Unit tests validate specific examples and edge cases
- Python을 사용하여 구현하며, LangChain, Gradio, hypothesis 라이브러리를 활용합니다
