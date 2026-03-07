# 🧹 LLM 관련 캐시 정리 가이드

> **환경**: Intel Mac (macOS 13), Python 3.11  
> **목적**: 디스크 용량 부족 시 LLM 관련 캐시를 빠르게 정리하기 위한 긴급 참조 문서

---

## 1단계: 현재 용량 확인

```bash
df -h /
```

---

## 2단계: 각 캐시 크기 확인

```bash
du -sh ~/Library/Caches/Homebrew 2>/dev/null
du -sh ~/.cache/huggingface 2>/dev/null
du -sh ~/.ollama/models 2>/dev/null
du -sh ~/.cache/pip 2>/dev/null
du -sh ~/.local/share/uv/cache 2>/dev/null
```

---

## 3단계: 캐시 삭제

### 🍺 Homebrew 다운로드 캐시 (최우선 — 실패한 brew install 찌꺼기 포함)

```bash
brew cleanup --prune=all
rm -rf ~/Library/Caches/Homebrew
```

### 🤗 HuggingFace 모델 캐시 (수 GB 이상일 수 있음)

```bash
# 전체 삭제
rm -rf ~/.cache/huggingface

# 또는 특정 모델만 확인 후 삭제
ls ~/.cache/huggingface/hub/
rm -rf ~/.cache/huggingface/hub/<모델이름>
```

### 📦 pip 캐시

```bash
# 캐시 크기 먼저 확인
pip cache info

# 삭제
pip cache purge
```

> 💡 `ERROR: No matching packages` → 정상! 이미 캐시가 비어있다는 의미 (실제 에러 아님)

### ⚡ uv 캐시 (Python 패키지 빌드 캐시)

```bash
uv cache clean
```

> 💡 `No cache found at: /Users/<username>/.cache/uv` → 정상! 이미 캐시가 없다는 의미  
> 💡 uv 프로젝트 내부에서 실행해도 동일 결과면 캐시 없는 것

### 🪐 Jupyter 임시 파일

```bash
rm -rf ~/.local/share/jupyter/nbconvert/
jupyter lab clean 2>/dev/null
```

### 🦙 Ollama 모델 (주의: 재다운로드 필요, 모델당 수 GB)

```bash
# Ollama 전체 크기 확인
du -sh ~/.ollama 2>/dev/null

# 모델 목록 확인
ollama list

# 필요없는 모델만 삭제
ollama rm <모델이름>
```

### 🗑️ 기타 즉시 효과 있는 것들

```bash
# 휴지통 비우기 (수 GB 잡혀있을 수 있음)
rm -rf ~/.Trash/*

# 홈 디렉토리에서 큰 폴더 상위 10개 파악
du -sh ~/* 2>/dev/null | sort -rh | head -10

# 다운로드 폴더
du -sh ~/Downloads

# macOS 로컬 Time Machine 스냅샷 삭제 (자동 생성, 수 GB)
sudo tmutil deletelocalsnapshots /

# 현재 로컬 스냅샷 목록 확인
tmutil listlocalsnapshots /
```

---

## ⚠️ 주의사항

| 캐시 종류 | 삭제 후 영향 | 복구 방법 |
|---|---|---|
| Homebrew 캐시 | brew install 시 재다운로드 필요 | 자동 재다운로드 |
| HuggingFace 캐시 | 모델 사용 시 재다운로드 필요 | 자동 재다운로드 |
| Ollama 모델 | 해당 모델 사용 불가 | `ollama pull <모델이름>` |
| pip / uv 캐시 | 패키지 설치 시 약간 느려짐 | 자동 재생성 |

---

## 🔧 관련 이슈: Homebrew 빌드 실패

### 증상
```
meson.build:1:0: ERROR: Compiler clang cannot compile programs.
Write to restore size failed
```

### 원인
1. **디스크 용량 부족** → `Write to restore size failed`
2. **Xcode Command Line Tools 구버전** → `clang cannot compile programs`

### 해결 방법

**디스크 정리 후** Command Line Tools 재설치:
```bash
sudo rm -rf /Library/Developer/CommandLineTools
sudo xcode-select --install
```

또는 **System Settings → General → Software Update** 에서 CLT 업데이트 확인

---

## 🚀 brew 설치 실패 시 대안: pymupdf 사용

`poppler` / `tesseract` brew 설치가 실패하는 경우, Python 레벨 대안 사용:

### pdf2image (poppler 필요) → pymupdf로 대체

```python
import fitz          # pymupdf — pip install pymupdf, 외부 의존성 없음
import pytesseract
from PIL import Image
import io

# tesseract 경로 명시 (PATH 미등록 대비)
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

doc = fitz.open('./data/파일명.pdf')
pages = []
for page in doc:
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("ppm")))
    pages.append(img)

texts = [pytesseract.image_to_string(page, lang='kor') for page in pages]
print(texts[0][:500])
```

### 한국어 tessdata 수동 설치 (tesseract-lang brew 설치 실패 시)

```bash
curl -L https://github.com/tesseract-ocr/tessdata/raw/main/kor.traineddata \
  -o /usr/local/share/tessdata/kor.traineddata
```

### 텍스트 레이어 있는 PDF는 OCR 없이 바로 추출

```python
import fitz
doc = fitz.open('./data/파일명.pdf')
texts = [page.get_text() for page in doc]
print(texts[0][:500])  # 빈 문자열이면 스캔 이미지 PDF → OCR 필요
```
