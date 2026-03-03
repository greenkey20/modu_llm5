# 의존성 관리 가이드

## 프로젝트 환경

- **OS**: macOS 13 (Ventura)
- **아키텍처**: Intel x86_64
- **Python**: 3.11.x
- **패키지 매니저**: uv

## 핵심 제약사항

### Intel Mac 제약

Intel Mac (특히 macOS 13)에서는 최신 ML 라이브러리들이 제대로 동작하지 않는 경우가 많습니다. 이는 Apple Silicon(M1/M2) 전환 이후 Intel Mac 지원이 점차 줄어들고 있기 때문입니다.

**주요 영향:**
- PyTorch, Transformers 등 최신 버전 사용 불가
- GPU 가속(CUDA) 미지원
- 일부 패키지는 특정 버전에서만 동작

---

## 의존성 그룹별 설명

### 1. ML Stack (Intel Mac 제약)

```toml
torch==2.2.2
sentence-transformers==2.2.2
transformers>=4.30,<4.40
onnxruntime<1.17
```

#### 고정 이유

**torch==2.2.2**
- Intel Mac에서 안정적으로 동작하는 마지막 버전
- 2.3.0 이상은 macOS 13에서 설치/실행 문제 발생
- CPU 전용 (CUDA 미지원)

**sentence-transformers==2.2.2**
- torch 2.2.2와 호환되는 버전
- 2.6.0 이상은 torch 2.3+ 요구
- 임베딩 모델 로딩에 필수

**transformers>=4.30,<4.40**
- 4.30: Gemini, GPT-4 시대 모델 지원
- 4.40 미만: torch 2.2.2와의 호환성 유지
- 4.40 이상은 torch 2.3+ 요구

**onnxruntime<1.17**
- 1.17 이상은 macOS 13 Intel에서 메모리 이슈 발생
- 일부 transformers 모델 최적화에 사용

### 2. 버전 고정 (충돌 방지)

```toml
huggingface-hub==0.19.4
pillow>=8.0,<11.0
```

#### 고정 이유

**huggingface-hub==0.19.4**
- sentence-transformers 2.2.2가 요구하는 버전
- 0.30.2 이상은 sentence-transformers 2.6.0+ 요구
- 모델 다운로드 및 캐싱 담당

**pillow>=8.0,<11.0**
- 11.0은 일부 이미지 처리에서 API 변경
- Gradio PDF 뷰어와의 호환성 유지

### 3. LangChain 생태계

```toml
langchain>=1.2.8
langchain-chroma>=1.1.0
langchain-community>=0.4.1
langchain-openai>=1.1.7
langchain-google-genai>=2.0,<3.0
langchain-ollama>=1.0.1
```

#### 주요 패키지 역할

**langchain-community**
- HuggingFaceEmbeddings 제공
- Intel Mac에서 `langchain-huggingface` 대체
- 다양한 오픈소스 통합 포함

**langchain-chroma**
- 벡터 데이터베이스 (Chroma) 연동
- 로컬 벡터 저장소로 사용

**langchain-openai**
- OpenAI API (GPT, Embedding) 연동
- text-embedding-3-small/large 사용

**langchain-google-genai**
- Google Gemini API 연동
- 2.x 버전 사용 (4.x는 의존성 충돌)

**langchain-ollama**
- 로컬 LLM/임베딩 실행 (Ollama)
- bge-m3 등 로컬 모델 사용

### 4. 벡터 저장소

```toml
faiss-cpu<1.8
```

#### 버전 제약

**faiss-cpu<1.8**
- 1.8 이상: macOS 14+ 전용 (macOS 13 미지원)
- 1.7.4: macOS 13 Intel Mac에서 안정 동작
- 고성능 벡터 유사도 검색

---

## 설치 불가 패키지 및 대안

### langchain-huggingface

**문제:**
```
langchain-huggingface requires:
- sentence-transformers>=2.6.0
- huggingface-hub>=0.30.2

현재 환경:
- sentence-transformers==2.2.2
- huggingface-hub==0.19.4
```

**대안:**
```python
# ❌ 설치 불가
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ 대신 사용
from langchain_community.embeddings import HuggingFaceEmbeddings
```

기능상 차이 없음. `langchain-community`가 동일한 클래스를 제공합니다.

---

## 트러블슈팅

### 1. 의존성 충돌 발생 시

**증상:**
```
No solution found when resolving dependencies
```

**해결:**
1. pyproject.toml에서 고정된 버전 확인
2. 새 패키지가 요구하는 버전 확인
3. 대안 패키지 또는 구버전 사용

**예시:**
```bash
# ❌ 최신 버전 설치 시도
uv add faiss-cpu

# ✅ 호환 버전 지정
uv add "faiss-cpu<1.8"
```

### 2. sentence-transformers 관련 에러

**증상:**
```
ImportError: cannot import name 'SentenceTransformer'
```

**원인:**
- torch 버전 불일치
- ONNX runtime 충돌

**해결:**
```bash
# 의존성 재설치
uv sync --reinstall-package sentence-transformers
```

### 3. Hugging Face 모델 다운로드 실패

**증상:**
```
HTTPError: 401 Unauthorized
```

**해결:**
```bash
# Hugging Face 토큰 설정 (.env)
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx
```

---

## 의존성 업데이트 가이드

### 업데이트 가능 패키지

다음 패키지는 Intel Mac 제약과 무관하므로 업데이트 가능:

```toml
beautifulsoup4
jupyter
notebook
python-dotenv
scikit-learn
seaborn
matplotlib
langchain (주의: 메이저 버전 변경 시 API 변경 확인)
```

### 업데이트 금지 패키지

Intel Mac 안정성을 위해 고정 유지:

```toml
torch==2.2.2
sentence-transformers==2.2.2
huggingface-hub==0.19.4
transformers (4.30~4.39 범위 유지)
onnxruntime (<1.17)
faiss-cpu (<1.8)
```

### 업데이트 명령

```bash
# 안전한 업데이트
uv lock --upgrade-package beautifulsoup4

# 특정 버전으로 업데이트
uv add "langchain>=1.3.0,<2.0"
```

---

## 환경별 설치 가이드

### Intel Mac (현재 환경)

```bash
# 정상 설치
uv sync
```

### Apple Silicon Mac (M1/M2/M3)

```bash
# pyproject.toml 수정 필요
# torch, transformers 최신 버전 사용 가능
uv add torch --upgrade
uv add transformers --upgrade
uv add sentence-transformers --upgrade
```

### Linux/Windows

```bash
# CUDA 사용 시
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 전용
uv sync
```

---

## 참고 자료

### 공식 문서

- [PyTorch - Start Locally](https://pytorch.org/get-started/locally/)
- [Sentence Transformers - Installation](https://www.sbert.net/docs/installation.html)
- [LangChain - Installation](https://python.langchain.com/docs/get_started/installation)
- [Hugging Face Hub - Authentication](https://huggingface.co/docs/huggingface_hub/quick-start)

### 관련 이슈

- [sentence-transformers #1000](https://github.com/UKPLab/sentence-transformers/issues/1000): Intel Mac 호환성
- [langchain #10000](https://github.com/langchain-ai/langchain/discussions/10000): HuggingFace 통합

---

## 변경 이력

### 2026-02-21
- `faiss-cpu<1.8` 추가 (macOS 13 호환성)
- `langchain-huggingface` 설치 포기, `langchain-community` 사용
- 의존성 문서 작성

### 이전
- 초기 Intel Mac 환경 구성
- torch 2.2.2, sentence-transformers 2.2.2 고정
- Gradio 4.x 다운그레이드 (5.x 호환성 문제)
