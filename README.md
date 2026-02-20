# PRJ01 — LLM 챗봇 만들기

> LLM 실전 부트캠프 5기 | 12주 과정 중 1~3주차 프로젝트

## 개요

이 저장소는 **LLM 실전 부트캠프 5기** 커리큘럼의 첫 번째 프로젝트(`PRJ01`)입니다.
12주 전체 과정 중 처음 3주간 **챗봇 개발**을 주제로 진행되며, LLM API 활용부터 RAG 파이프라인 구축까지의 내용을 다룹니다.

---

## 학습 내용

### Week 1 — LLM 기초 & 챗봇 프레임워크

| 노트북 | 주제 |
|--------|------|
| `PRJ01_W1_001_Python_Clean_Code.ipynb` | Python 클린 코드 기초 |
| `PRJ01_W1_002_OpenAI_Chat_Completion.ipynb` | OpenAI Chat Completion API |
| `PRJ01_W1_003_Langchain_Components.ipynb` | LangChain 핵심 컴포넌트 |
| `PRJ01_W1_004_LangSmith_LCEL.ipynb` | LangSmith 모니터링 & LCEL |
| `PRJ01_W1_005_Gradio_Chatbot.ipynb` | Gradio UI로 챗봇 구현 |

### Week 2 — RAG (검색 증강 생성)

| 노트북 | 주제 |
|--------|------|
| `PRJ01_W2_001_Tokenizing_Embedding.ipynb` | 토크나이징 & 임베딩 개념 |
| `PRJ01_W2_002_Simple_RAG_Pipeline.ipynb` | 간단한 RAG 파이프라인 구축 |
| `PRJ01_W2_003_Document_Loader.ipynb` | Document Loader (PDF, CSV, JSON 등) |
| `PRJ01_W2_004_Text_Splitter.ipynb` | Text Splitter 전략 |
| `PRJ01_W2_005_Embedding_Model.ipynb` | 임베딩 모델 비교 (OpenAI, HuggingFace, Ollama) |
| `PRJ01_W2_006_Vectorstore.ipynb` | 벡터 저장소 (FAISS, Chroma, Pinecone) |
| `PRJ01_W2_007_Retriever.ipynb` | Retriever 구성 & 검색 전략 |

### Week 3 — (예정)

3주차 노트북은 수업 진행에 따라 추가됩니다.

---

## 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| LLM API | OpenAI GPT, Google Gemini, Ollama (로컬) |
| LLM 프레임워크 | LangChain, LCEL |
| 임베딩 | OpenAI Embeddings, HuggingFace (sentence-transformers) |
| 벡터 DB | FAISS, ChromaDB, Pinecone |
| UI | Gradio 4.x |
| 모니터링 | LangSmith, Langfuse |
| 패키지 관리 | uv |
| 런타임 | Python 3.11, Jupyter Notebook |

---

## 환경 설정

### 1. 의존성 설치

```bash
uv sync
```

### 2. 환경 변수 설정

`drafts/env_sample.md`를 참고하여 프로젝트 루트에 `.env` 파일을 생성합니다.

```bash
# .env 파일 예시
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=...
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=faq_bot
PINECONE_API_KEY=...
```

> `.env` 파일은 `.gitignore`에 의해 추적되지 않습니다.

### 3. Jupyter 실행

```bash
uv run jupyter notebook
```

---

## 디렉토리 구조

```
001_chatbot/
├── PRJ01_W1_*.ipynb        # Week 1 실습 노트북
├── PRJ01_W2_*.ipynb        # Week 2 실습 노트북
│
├── data/                   # 실습용 샘플 데이터
│   ├── *.pdf, *.txt, *.csv, *.json
│   └── ...
│
├── articles/               # RAG 실습용 문서
│   └── notionai.pdf
│
├── homework/               # 과제 제출 노트북
│
├── docs/                   # 프로젝트 문서
│   ├── DEPENDENCIES.md     # 의존성 관리 가이드
│   ├── RAG_Experiments.ipynb
│   └── LLM과정_5기_커리큘럼.pdf
│
├── drafts/                 # 작업 중 파일 & 유틸리티
│   ├── env_sample.md       # 환경 변수 템플릿
│   └── *.py
│
├── chroma_db/              # ChromaDB 로컬 데이터 (gitignore)
├── faiss_*_index/          # FAISS 인덱스 (gitignore)
│
├── pyproject.toml          # 프로젝트 의존성 (uv)
└── .env                    # API 키 (gitignore)
```

> `chroma_db/`, `faiss_*_index/`는 노트북 실행 시 자동 생성되며 git에 추적되지 않습니다.

---

## Intel Mac 사용자 주의사항

이 프로젝트는 **Intel Mac (macOS 13, x86_64)** 환경을 기준으로 구성되어 있습니다.
일부 ML 라이브러리 버전이 호환성 문제로 고정되어 있습니다.

자세한 내용은 [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md)를 참조하세요.

| 패키지 | 고정 버전 | 사유 |
|--------|-----------|------|
| `torch` | `==2.2.2` | 2.3+ macOS 13 미지원 |
| `sentence-transformers` | `==2.2.2` | torch 2.2.2 의존 |
| `faiss-cpu` | `<1.8` | 1.8+ macOS 14+ 전용 |
| `onnxruntime` | `<1.17` | 1.17+ 메모리 이슈 |

---

## 참고 자료

- [LangChain Docs](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [LangSmith](https://smith.langchain.com/)
