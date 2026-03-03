# 📚 RAG 기반 문서 질의응답 챗봇

Gradio를 사용한 2탭 구조의 RAG(Retrieval-Augmented Generation) 챗봇 웹 애플리케이션입니다. PDF 문서를 업로드하고 다양한 RAG 파라미터를 조정하면서 질의응답을 수행하거나, 여러 템플릿을 병렬로 비교할 수 있습니다.

## ✨ 주요 기능

### 프롬프트 탭
- 📄 **자동 문서 처리**: 질문 제출 시 자동으로 문서 로드 및 처리
- ✂️ **동적 텍스트 분할**: 청크 크기, 중복, 분할 방식 조정 가능
- 🔢 **다양한 임베딩 모델**: OpenAI, HuggingFace, Ollama 지원
- 🎛️ **임베딩 차원 조정**: OpenAI text-embedding-3 모델의 차원 커스터마이징 (256-3072)
- 🔄 **자동 차원 감지**: 임베딩 차원 변경 시 벡터 스토어 자동 초기화
- 💾 **벡터 저장소 선택**: Chroma, FAISS 중 선택 가능
- 🔍 **검색 방식 커스터마이징**: Similarity, MMR, Score Threshold 지원
- 🤖 **LLM 설정**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo 지원
- 💬 **스트리밍 응답**: 실시간으로 응답 생성 과정 확인
- 📑 **소스 문서 추적**: 답변 생성에 사용된 문서 확인 가능

### 성능 비교 탭
- 💾 **템플릿 관리**: RAG 파라미터 조합을 템플릿으로 저장/로드/삭제
- 🔄 **자동 템플릿 로드**: 페이지 새로고침 시 저장된 템플릿 자동 표시
- ⚡ **병렬 처리**: ThreadPoolExecutor를 사용한 최대 3개 템플릿 동시 비교
- 🔒 **충돌 방지**: 각 스레드별 독립 벡터 스토어 디렉토리 사용
- 📊 **비교 결과**: 응답, 응답 길이, 생성 시간, 소스 문서 수 표시
- 📥 **CSV 내보내기**: 비교 결과를 CSV 파일로 다운로드 (UTF-8-sig 인코딩)

## 🚀 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. 가상환경 생성 및 활성화

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
# OpenAI API Key (필수)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: HuggingFace Token
# HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

# Optional: Ollama settings
# OLLAMA_BASE_URL=http://localhost:11434
```

## 📖 사용 방법

### 1. 애플리케이션 실행

```bash
python app.py
```

브라우저에서 `http://localhost:7860`으로 접속하세요.

### 2. 프롬프트 탭 - 실시간 질의응답

1. **PDF 파일 업로드**: 왼쪽 패널에서 PDF 파일을 선택
2. **설정 조정** (선택사항):
   - 텍스트 분할: 청크 크기, 중복, 분할 방식
   - 임베딩 모델: 타입 선택 시 모델 목록 자동 업데이트
   - 임베딩 차원: OpenAI text-embedding-3 모델의 경우 256-3072 조정 가능
   - 벡터 저장소: Chroma 또는 FAISS
   - 검색 설정: 검색 방식, 문서 수(k)
   - LLM 설정: 모델, temperature, max_tokens
3. **질문 입력**: 오른쪽 패널 하단 입력창에 질문 입력
4. **자동 처리**: 질문 제출 시 자동으로 문서 처리 및 벡터 저장소 생성
5. **응답 확인**: 스트리밍 방식으로 실시간 응답 확인
6. **소스 문서 확인**: "📑 소스 문서" 아코디언에서 참조 문서 확인

### 3. 성능 비교 탭 - 템플릿 비교

#### 템플릿 저장
1. **프롬프트 탭에서 설정 조정**: 원하는 RAG 파라미터 설정
2. **성능 비교 탭으로 이동**: 왼쪽 "템플릿 관리" 섹션
3. **템플릿 이름 입력**: 예) "빠른검색", "정밀검색"
4. **설정 확인**: 아코디언에서 모든 파라미터 확인
5. **템플릿 저장**: "💾 템플릿 저장" 버튼 클릭

#### 템플릿 비교
1. **문서 업로드**: 비교할 문서 업로드
2. **템플릿 선택**: 체크박스에서 최대 4개 템플릿 선택
3. **질문 입력**: 비교할 질문 입력
4. **비교 실행**: "🔍 비교 실행" 버튼 클릭
5. **결과 확인**: 각 템플릿의 응답, 생성 시간, 소스 문서 수 비교
6. **CSV 다운로드**: "📥 CSV 다운로드" 버튼으로 결과 저장

## 🏗️ 프로젝트 구조

```
.
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # 문서 로드 및 분할
│   ├── embedding_manager.py     # 임베딩 모델 관리 (차원 추적)
│   ├── vector_store_manager.py  # 벡터 저장소 관리 (차원 변경 감지)
│   ├── rag_chain.py             # RAG 체인 구성
│   ├── template_manager.py      # 템플릿 저장/로드 (JSON)
│   ├── comparison_engine.py     # 병렬 템플릿 비교
│   └── gradio_interface_v2.py   # Gradio UI (2탭 구조)
├── .kiro/specs/chatbot/         # 프로젝트 스펙 문서
│   ├── requirements.md          # 요구사항 (13개)
│   ├── design.md                # 설계 문서
│   └── tasks.md                 # 구현 작업 목록
├── tests/                       # 테스트 파일
├── data/                        # 데이터 디렉토리
├── app.py                       # 메인 애플리케이션
├── templates.json               # 저장된 템플릿
├── requirements.txt             # 패키지 의존성
├── .env                         # 환경 변수 (생성 필요)
├── .gitignore                   # Git 제외 파일
├── README.md                    # 프로젝트 문서
└── USAGE_GUIDE.md               # 사용 가이드
```

## ⚙️ 설정 옵션

### 텍스트 분할

- **Chunk Size**: 100-2000 (기본값: 1000)
- **Chunk Overlap**: 0-500 (기본값: 200)
- **Splitter Type**: RecursiveCharacterTextSplitter, CharacterTextSplitter

### 임베딩 모델

- **OpenAI**: 
  - text-embedding-3-small (1536차원, 256-3072 조정 가능)
  - text-embedding-3-large (3072차원, 256-3072 조정 가능)
  - text-embedding-ada-002 (1536차원)
- **HuggingFace**: 
  - BAAI/bge-m3 (1024차원)
  - sentence-transformers/all-MiniLM-L6-v2 (384차원)
  - sentence-transformers/all-mpnet-base-v2 (768차원)
- **Ollama**: 
  - bge-m3 (1024차원)
  - nomic-embed-text (768차원)
  - mxbai-embed-large (1024차원)

### 검색 방식

- **Similarity**: 코사인 유사도 기반 검색
- **MMR**: Maximum Marginal Relevance (다양성 고려)
- **Similarity Score Threshold**: 임계값 이상의 문서만 검색

### LLM 모델

- **gpt-4o-mini**: 빠르고 경제적
- **gpt-4o**: 고성능
- **gpt-3.5-turbo**: 균형잡힌 성능

## 🔧 개발 환경

- Python 3.8+
- LangChain 0.1.0+
- Gradio 4.0.0+
- OpenAI API
- Chroma / FAISS

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 🤝 기여

버그 리포트, 기능 제안, Pull Request를 환영합니다!

## 📧 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 등록해주세요.



## PoC 결과
