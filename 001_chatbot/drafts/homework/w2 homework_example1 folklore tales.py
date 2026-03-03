# ===================================================================================
# RAG 동화 챗봇 - 한국 전래 동화 기반 질문-응답 시스템
# ===================================================================================
# RAG(Retrieval-Augmented Generation)란?
#   사용자가 질문하면, 먼저 관련 문서를 "검색(Retrieval)"한 뒤,
#   그 문서 내용을 참고하여 AI가 "답변을 생성(Generation)"하는 방식입니다.
#
# 왜 RAG가 필요한가?
#   AI(LLM)는 학습 데이터에 없는 내용(우리가 작성한 동화 파일 등)은 알지 못합니다.
#   RAG를 사용하면 AI가 모르는 내용도 문서에서 찾아서 정확하게 답변할 수 있습니다.
#   즉, "AI의 지식 + 우리 문서의 내용"을 결합하는 기술입니다.
#
# 전체 동작 흐름:
#   [사전 준비 - 앱 시작 시 1회]
#   동화 텍스트 파일 → 문서 로딩 → 작은 조각으로 분할(청킹)
#   → 임베딩(숫자 벡터로 변환) → 벡터 스토어에 저장
#
#   [질문 응답 - 사용자가 질문할 때마다]
#   사용자 질문 → 질문을 벡터로 변환 → 벡터 스토어에서 유사한 조각 검색
#   → 검색된 내용 + 질문을 LLM에 전달 → 답변 생성 → 화면에 표시
# ===================================================================================

# ============================================================
# 1. 임포트 & 환경 설정
# ============================================================
# 파이썬에서 외부 라이브러리의 기능을 가져와 사용하겠다는 선언입니다.
# 직접 코드를 작성하지 않아도, 다른 개발자가 만들어 놓은 도구를 가져다 쓸 수 있습니다.

# --- 문서 로딩 관련 ---
# DirectoryLoader: 폴더 안의 여러 파일을 한꺼번에 읽어오는 도구
# TextLoader: 텍스트(.txt) 파일 한 개를 읽는 도구 (DirectoryLoader가 내부적으로 사용)
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# RecursiveCharacterTextSplitter: 긴 문서를 적절한 크기의 작은 조각(청크)으로 나누는 도구
# "재귀적(Recursive)"이란 여러 구분자(문단→줄→문장→단어)를 순서대로 시도한다는 뜻
# 첫 번째 구분자로 나눠봤는데 조각이 너무 크면, 다음 구분자로 다시 나누기를 반복함
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 임베딩 & LLM 관련 ---
# OpenAIEmbeddings: 텍스트를 숫자 벡터(숫자 배열)로 변환하는 OpenAI의 임베딩 모델
# ChatOpenAI: 대화형 AI 모델 (질문에 대한 답변을 생성하는 LLM)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# HuggingFaceEmbeddings: OpenAI 대신 무료로 쓸 수 있는 오픈소스 임베딩 모델
# HuggingFace는 AI 모델을 공유하는 오픈소스 플랫폼 (github 같은 AI 모델 저장소)
from langchain_huggingface import HuggingFaceEmbeddings

# --- 벡터 스토어 관련 ---
# 벡터 스토어(Vector Store)란?
#   임베딩된 벡터들을 저장하고, "질문과 가장 비슷한 문서 조각"을 빠르게 찾아주는 특수 데이터베이스
#   일반 데이터베이스: "흥부"라는 키워드가 포함된 문서를 검색 (키워드 일치)
#   벡터 스토어: "착한 동생 이야기"로 검색해도 흥부 관련 문서가 검색됨 (의미 유사도)
#
# Chroma: 로컬 PC의 폴더에 파일로 저장하는 벡터 스토어 (설치가 간편함)
from langchain_chroma import Chroma
# FAISS(Facebook AI Similarity Search): Facebook이 만든 고속 유사도 검색 라이브러리
# 메모리에서 동작하여 검색 속도가 매우 빠르고, 파일로도 저장 가능
from langchain_community.vectorstores import FAISS
# Pinecone: 클라우드(인터넷 서버)에 벡터를 저장하는 서비스
# 내 PC가 꺼져도 데이터가 유지되고, 대규모 서비스에 적합
# ServerlessSpec: Pinecone 서버의 설정(어느 클라우드, 어느 지역)을 지정하는 객체
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# --- LangChain 핵심 도구 ---
# ChatPromptTemplate: AI에게 보낼 프롬프트(지시문)의 템플릿을 만드는 도구
#   템플릿이란? "당신은 {역할}입니다. {질문}에 답하세요" 처럼 빈칸을 두고,
#   실행 시점에 실제 값으로 채워 넣는 틀
# MessagesPlaceholder: 이전 대화 내역을 프롬프트에 끼워 넣을 자리를 만드는 도구
#   이것이 있어야 AI가 이전 대화를 기억하고 맥락에 맞는 답변을 할 수 있음
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# HumanMessage: 사용자가 보낸 메시지를 표현하는 객체
# AIMessage: AI가 보낸 메시지를 표현하는 객체
# 왜 필요한가? LangChain은 대화 내역을 이 객체들의 리스트로 관리함
#   예: [HumanMessage("흥부는 누구?"), AIMessage("흥부는 착한 동생입니다"), ...]
from langchain_core.messages import HumanMessage, AIMessage

# StrOutputParser: AI 응답에서 텍스트 문자열만 깔끔하게 추출하는 도구
# AI 응답은 원래 메타데이터(토큰 수, 모델명 등)가 포함된 복잡한 객체인데,
# 이 파서를 거치면 순수한 텍스트만 뽑아냄
from langchain_core.output_parsers import StrOutputParser

# LCEL(LangChain Expression Language) 체인 구성 도구들
# LCEL이란? "|" (파이프) 연산자로 여러 단계를 연결하는 LangChain의 문법
#   예: 프롬프트 | LLM | 출력파서 = "프롬프트를 만들고 → AI에 보내고 → 텍스트로 추출"
#   마치 공장의 컨베이어 벨트처럼 데이터가 단계별로 흘러감
#
# RunnablePassthrough: 입력 데이터를 변경 없이 그대로 다음 단계로 전달하는 도구
# RunnableParallel: 여러 작업을 동시에 병렬로 실행하는 도구
#   예: 질문으로 문서 검색 + 질문 텍스트 추출을 동시에 수행
# RunnableLambda: 일반 파이썬 함수를 LCEL 체인 안에서 하나의 단계로 쓸 수 있게 감싸주는 도구
#   예: lambda x: x["question"] → 딕셔너리에서 "question" 값만 꺼내는 단계
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

# --- 기타 ---
# gradio: 웹 UI(채팅 인터페이스)를 코드 몇 줄로 쉽게 만들어 주는 라이브러리
# HTML, CSS, JavaScript를 몰라도 웹 페이지를 만들 수 있음
import gradio as gr

# dotenv: .env 파일에서 API 키 등 환경 변수를 읽어오는 라이브러리
# .env 파일이란? 비밀번호나 API 키처럼 코드에 직접 쓰면 안 되는 값들을
# 별도 파일에 저장해두고 불러오는 방식 (보안 목적)
# 예: .env 파일 내용 → OPENAI_API_KEY=sk-abc123...
from dotenv import load_dotenv

# os: 운영체제(파일 경로, 환경 변수 등)와 상호작용하는 파이썬 기본 라이브러리
import os
# time: 시간 관련 기능 (여기서는 Pinecone 인덱스 생성 대기 시 sleep에 사용)
import time

# .env 파일에서 API 키(OPENAI_API_KEY, PINECONE_API_KEY 등)를 불러와서
# 환경 변수로 등록함 → 이후 os.environ["키이름"]으로 접근 가능
load_dotenv()

# ============================================================
# 2. 문서 로딩 & 텍스트 분할 (청킹)
# ============================================================
# 동화 텍스트 파일들이 있는 폴더 경로를 조합
# os.path.dirname(__file__) = 이 파이썬 파일(rag_chatbot.py)이 있는 폴더
# os.path.join()으로 경로를 합치면: "C:\Study\001_chatbot\data\fairy_tales"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "fairy_tales")


def load_and_split_documents(chunk_size=500, chunk_overlap=100):
    """
    동화 텍스트 파일을 모두 읽어온 뒤, 작은 조각(청크)으로 나누는 함수

    왜 나누는가? (청킹의 필요성)
    - AI에게 한 번에 보낼 수 있는 텍스트 양에 한계가 있음 (토큰 제한)
    - 긴 문서 전체보다 질문과 관련된 "부분"만 찾아서 보내는 것이 더 정확함
    - 예: "놀부에게 무슨 일이 일어났나요?" → 흥부와 놀부 전체가 아닌,
      "놀부가 박을 탔더니 도깨비가 나왔다" 부분만 검색하여 AI에게 전달

    Args:
        chunk_size: 한 조각의 최대 글자 수 (기본 500, 한국어 약 250자 분량)
        chunk_overlap: 조각 사이에 겹치는 글자 수 (기본 100)

    반환값:
    - Document 객체들의 리스트
    - Document 객체란? page_content(텍스트 내용) + metadata(출처 파일명 등) 를 담는 그릇
      예: Document(page_content="흥부는 착한...", metadata={"source": "흥부와_놀부.txt"})
    """
    # data/fairy_tales/ 폴더 안의 모든 .txt 파일을 읽어옴
    loader = DirectoryLoader(
        DATA_DIR,                              # 읽어올 폴더 경로
        glob="*.txt",                          # *.txt = 확장자가 .txt인 모든 파일 (와일드카드)
        loader_cls=TextLoader,                 # 텍스트 파일 전용 로더 사용
        loader_kwargs={"encoding": "utf-8"},   # 한글이 깨지지 않도록 UTF-8 인코딩 지정
    )
    # .load() 호출 시 실제로 파일을 읽어서 Document 객체 리스트로 반환
    # 5개 파일 → 5개 Document (각각 동화 전체 내용을 담고 있음)
    documents = loader.load()
    print(f"로딩된 문서 수: {len(documents)}")

    # 문서를 작은 조각(청크)으로 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # 한 조각의 최대 글자 수
        chunk_overlap=chunk_overlap, # 조각 사이에 겹치는 글자 수
        # 왜 겹치게 하는가? 문장이 조각 경계에서 잘리면 의미가 끊길 수 있음
        # 겹침을 두면 앞 조각의 끝부분이 다음 조각의 시작에도 포함되어 문맥이 이어짐
        separators=["\n\n", "\n", ".", " ", ""],
        # 나누는 기준 (우선순위 순서):
        # "\n\n" 빈 줄(문단) → "\n" 줄바꿈 → "." 마침표(문장) → " " 공백(단어) → "" 글자
        # 먼저 문단 단위로 나눠보고, 조각이 chunk_size보다 크면 줄 단위로 다시 나누기를 반복
    )
    chunks = splitter.split_documents(documents)
    print(f"분할된 청크 수: {len(chunks)} (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
    return chunks


# 앱이 시작될 때 문서를 읽고 조각으로 나눠서 CHUNKS 변수에 저장 (1회만 실행)
# 대문자 변수명(CHUNKS)은 파이썬 관례상 "상수" (한번 정하면 바꾸지 않는 값)를 의미
# 이후 사용자가 질문할 때마다 이 조각들을 재사용함 (매번 파일을 다시 읽지 않음)
CHUNKS = load_and_split_documents()

# ============================================================
# 3. 임베딩 모델 팩토리
# ============================================================
# 임베딩(Embedding)이란?
#   텍스트를 숫자 배열(벡터)로 변환하는 과정
#   예: "흥부는 착한 사람" → [0.12, -0.45, 0.78, ...] (숫자 수백~수천 개)
#   왜 숫자로 바꾸는가? 컴퓨터는 텍스트의 "의미"를 직접 이해하지 못하지만,
#   숫자 배열로 바꾸면 "두 문장이 얼마나 비슷한지"를 수학적으로 계산할 수 있음
#   예: "흥부는 착한 사람" ↔ "마음씨가 착한 동생" → 벡터가 비슷 → 유사도 높음
#       "흥부는 착한 사람" ↔ "호랑이가 나타났다" → 벡터가 다름 → 유사도 낮음
#
# "차원"이란?
#   벡터(숫자 배열)의 길이를 말함
#   1536차원 = 숫자 1536개짜리 배열, 384차원 = 숫자 384개짜리 배열
#   차원이 높을수록 텍스트의 의미를 더 세밀하게 표현할 수 있지만, 저장 공간과 계산 시간이 늘어남
#
# 팩토리 패턴이란?
#   사용자의 선택에 따라 다른 종류의 객체를 만들어 주는 함수 설계 방식
#   여기서는 "OpenAI" / "bge-m3" / "MiniLM" 중 선택하면 해당 임베딩 모델을 생성

# 캐시(Cache)란?
#   한번 만든 결과를 저장해두고, 같은 요청이 오면 다시 만들지 않고 저장된 것을 반환하는 기법
#   왜 필요한가? 임베딩 모델 생성은 느림 (특히 HuggingFace는 모델 파일 로딩에 시간이 걸림)
#   캐시가 없으면 질문할 때마다 모델을 다시 로딩해야 함 → 매우 느려짐
_embedding_cache: dict = {}  # 빈 딕셔너리로 시작, 모델이 생성되면 여기에 저장됨


def get_embedding_model(name: str):
    """
    사용자가 선택한 임베딩 모델을 생성하거나, 이미 만들어둔 것을 반환하는 함수

    세 가지 모델 비교:
    - OpenAI (text-embedding-3-small):
      유료 (API 호출 시 비용 발생), 빠름, 1536차원, API 키 필요
    - HuggingFace bge-m3:
      무료, 고성능, 1024차원, 첫 사용 시 모델 다운로드 필요 (약 2.2GB, 시간 소요)
    - HuggingFace MiniLM-multilingual:
      무료, 경량, 384차원, 모델 크기 작음 (약 470MB, 빠른 다운로드)
      50개 이상 언어(한국어 포함) 지원

    Args:
        name: 모델 이름 문자열 (Gradio 드롭다운에서 선택한 값)

    Returns:
        임베딩 모델 객체 (텍스트를 벡터로 변환하는 .embed_query() 메서드를 가짐)
    """
    # 이미 만들어둔 모델이 캐시에 있으면 그대로 반환 (재생성 방지)
    # 예: 처음 "OpenAI" 선택 시 모델 생성 후 캐시에 저장
    #     두 번째 "OpenAI" 선택 시 캐시에서 바로 꺼내 반환 (빠름)
    if name in _embedding_cache:
        return _embedding_cache[name]

    if name == "OpenAI (text-embedding-3-small)":
        # OpenAI의 임베딩 모델 (텍스트 → 1536차원 벡터로 변환)
        # API 호출 방식: 내 PC에서 계산하지 않고 OpenAI 서버에 요청 → 결과 수신
        model = OpenAIEmbeddings(model="text-embedding-3-small")
    elif name == "HuggingFace (bge-m3)":
        # 오픈소스 임베딩 모델 (텍스트 → 1024차원 벡터로 변환)
        # bge-m3: Beijing Academy of AI에서 만든 다국어 모델, 한국어 성능이 뛰어남
        # 로컬 실행 방식: 모델 파일을 내 PC에 다운로드하여 직접 계산
        model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",           # HuggingFace에서 모델을 식별하는 이름
            model_kwargs={"device": "cpu"},      # CPU에서 실행 (GPU가 없어도 동작)
            encode_kwargs={"normalize_embeddings": True},
            # normalize_embeddings=True: 벡터를 정규화 (길이를 1로 맞춤)
            # 정규화하면 코사인 유사도 계산이 더 정확해짐
        )
    else:  # HuggingFace (MiniLM-multilingual)
        # 경량 다국어 임베딩 모델 (텍스트 → 384차원 벡터로 변환)
        # MiniLM: 모델 크기가 작아서 다운로드/로딩이 빠르고 가벼움
        # "paraphrase-multilingual": 다국어 의미 유사도에 특화된 모델
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    _embedding_cache[name] = model  # 생성한 모델을 캐시에 저장
    return model

# ============================================================
# 4. 벡터 스토어 팩토리
# ============================================================
# 세 가지 벡터 스토어 비교:
#
#   Chroma (로컬 파일 저장)
#   - 저장 위치: 내 PC의 폴더 (chroma_db_fairy_tales/)
#   - 장점: 설치 간편, 무료, 별도 서비스 불필요
#   - 단점: 내 PC에서만 접근 가능, 대규모 데이터에는 부적합
#   - 적합한 경우: 개인 학습, 프로토타입 개발
#
#   FAISS (메모리 + 파일 저장)
#   - 저장 위치: 메모리(RAM)에서 동작, 파일로도 저장 가능
#   - 장점: 검색 속도가 매우 빠름, 무료
#   - 단점: 메모리 사용량이 큼, 실시간 업데이트가 불편
#   - 적합한 경우: 빠른 검색이 중요한 경우
#
#   Pinecone (클라우드 저장)
#   - 저장 위치: AWS 클라우드 서버 (인터넷)
#   - 장점: 내 PC가 꺼져도 데이터 유지, 대규모 확장 가능, 자동 관리
#   - 단점: API 키 필요, 무료 티어 제한 있음 (유료 플랜 존재)
#   - 적합한 경우: 실제 서비스 배포, 여러 사용자가 접속하는 환경

# 한번 생성한 벡터 스토어를 저장해두는 캐시
# 같은 임베딩+스토어 조합이면 매 질문마다 다시 만들지 않고 재사용
_vectorstore_cache: dict = {}


def get_vectorstore(embedding_name: str, store_name: str,
                    chunk_size: int = 500, chunk_overlap: int = 100):
    """
    사용자가 선택한 임베딩+벡터스토어 조합으로 벡터 스토어를 생성하는 함수

    동작 과정:
    1. 캐시 확인: 같은 조합이 이미 있으면 즉시 반환 (빠름)
    2. 없으면 새로 생성:
       a. 선택된 임베딩 모델로 각 문서 조각(청크)을 숫자 벡터로 변환
       b. 변환된 벡터들을 선택된 벡터 스토어에 저장
    3. 캐시에 저장 후 반환

    Args:
        embedding_name: 임베딩 모델 이름 (Gradio 드롭다운 값)
        store_name: 벡터 스토어 이름 ("Chroma" / "FAISS" / "Pinecone")
        chunk_size: 텍스트 분할 시 한 조각의 최대 글자 수 (기본 500)
        chunk_overlap: 텍스트 분할 시 조각 간 겹침 글자 수 (기본 100)

    Returns:
        벡터 스토어 객체 (.as_retriever() 메서드로 검색기를 만들 수 있음)
    """
    # 캐시 키: "OpenAI|Chroma|500|100" 형태의 문자열
    # 같은 임베딩+스토어+청크 설정 조합이면 이미 만들어둔 벡터 스토어를 그대로 반환
    cache_key = f"{embedding_name}|{store_name}|{chunk_size}|{chunk_overlap}"
    if cache_key in _vectorstore_cache:
        return _vectorstore_cache[cache_key]

    # 청크 설정이 변경되었을 수 있으므로 해당 설정으로 문서를 재분할
    chunks = load_and_split_documents(chunk_size, chunk_overlap)

    # 선택된 임베딩 모델 가져오기
    embedding = get_embedding_model(embedding_name)

    if store_name == "Chroma":
        # --- Chroma: 로컬 폴더에 저장하는 벡터 스토어 ---

        # 다른 임베딩으로 만든 Chroma가 캐시에 있으면 먼저 해제
        # (Windows에서 파일을 잡고 있으면 다른 작업이 실패할 수 있으므로)
        # 캐시 키 형태: "임베딩|Chroma|chunk_size|chunk_overlap"
        for k, v in list(_vectorstore_cache.items()):
            if "|Chroma|" in k:
                del _vectorstore_cache[k]
                del v

        # 벡터 데이터를 저장할 폴더 경로
        persist_dir = os.path.join(
            os.path.dirname(__file__), "chroma_db_fairy_tales"
        )
        # 임베딩별로 컬렉션(데이터 묶음) 이름을 다르게 설정
        # "OpenAI (text-embedding-3-small)" → "openai"
        # "HuggingFace (bge-m3)" → "bge-m3"
        # "HuggingFace (MiniLM-multilingual)" → "minilm-multilingual"
        # 왜? 같은 텍스트라도 임베딩 모델이 다르면 벡터가 다르므로 섞이면 안 됨
        # 괄호 안의 모델명을 추출하여 고유한 컬렉션 이름을 만듦
        collection_name = embedding_name.split("(")[-1].rstrip(")").strip().lower()

        # 먼저 기존 컬렉션에 연결하여 이미 데이터가 있는지 확인
        vectorstore = Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=embedding,
        )
        # 기존에 저장된 문서 조각 수를 확인
        existing_count = vectorstore._collection.count()

        if existing_count == len(chunks):
            # 이미 올바른 수의 문서가 저장되어 있으면 그대로 재사용
            # 앱을 재시작해도 디스크에 저장된 데이터를 불러오므로 빠름
            print(f"기존 Chroma 컬렉션 재사용: {collection_name} ({existing_count}개)")
        else:
            # 문서 수가 다르면 (중복 저장되었거나 비어있는 상태)
            # 컬렉션을 삭제하고 처음부터 새로 생성
            vectorstore.delete_collection()
            # from_documents: 문서 조각들을 임베딩하여 벡터 스토어에 저장하는 메서드
            # 내부적으로: 각 청크 텍스트 → 임베딩 모델로 벡터 변환 → Chroma에 저장
            vectorstore = Chroma.from_documents(
                documents=chunks,                    # 저장할 문서 조각들
                embedding=embedding,                 # 벡터 변환에 사용할 임베딩 모델
                collection_name=collection_name,     # 컬렉션 이름
                persist_directory=persist_dir,        # 저장 폴더 경로
            )

    elif store_name == "FAISS":
        # --- FAISS: 메모리 + 파일 기반 벡터 스토어 ---
        # 임베딩별로 저장 폴더를 다르게 설정 (섞이지 않도록)
        # 예: "faiss_fairy_tales_text-embedding-3-small", "faiss_fairy_tales_bge-m3"
        folder_suffix = embedding_name.split("(")[-1].rstrip(")").strip().lower()
        save_dir = os.path.join(
            os.path.dirname(__file__), f"faiss_fairy_tales_{folder_suffix}"
        )

        # 이미 저장된 인덱스 파일이 있으면 불러와서 재사용 (임베딩 생략 → 빠름)
        if os.path.exists(os.path.join(save_dir, "index.faiss")):
            vectorstore = FAISS.load_local(
                save_dir,
                embeddings=embedding,
                allow_dangerous_deserialization=True,
                # allow_dangerous_deserialization: 파일에서 객체를 복원할 때 필요한 옵션
                # FAISS가 pickle 파일을 사용하므로, 신뢰할 수 있는 파일임을 확인하는 안전장치
            )
            print(f"기존 FAISS 인덱스 재사용: {save_dir}")
        else:
            # 저장된 파일이 없으면 새로 생성
            # from_documents: 문서들을 임베딩하여 FAISS 인덱스를 메모리에 생성
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embedding,
            )
            # save_local: 메모리에 생성된 인덱스를 파일로 저장
            # 다음 실행 시 load_local()로 빠르게 불러올 수 있음
            vectorstore.save_local(save_dir)

    else:
        # --- Pinecone: 클라우드 벡터 스토어 ---

        # 임베딩 모델마다 출력하는 벡터의 길이(차원)가 다름
        # Pinecone 인덱스를 만들 때 이 차원을 정확히 맞춰야 함
        # (차원이 안 맞으면 벡터를 저장하거나 검색할 수 없음)
        embedding_dimensions = {
            "OpenAI (text-embedding-3-small)": 1536,     # OpenAI는 1536개의 숫자
            "HuggingFace (bge-m3)": 1024,                # bge-m3는 1024개의 숫자
            "HuggingFace (MiniLM-multilingual)": 384,    # MiniLM은 384개의 숫자
        }
        dimension = embedding_dimensions[embedding_name]
        index_name = "fairy-tales"  # Pinecone에서 이 인덱스를 식별하는 이름

        # Pinecone 클라이언트 초기화 (.env 파일의 PINECONE_API_KEY 사용)
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        # 현재 Pinecone 계정에 존재하는 인덱스 목록 조회
        existing = [idx["name"] for idx in pc.list_indexes()]

        # 인덱스가 이미 있는데 차원이 다르면 삭제 후 재생성 필요
        # 예: 이전에 OpenAI(1536차원)로 만들었는데 MiniLM(384차원)으로 바꾸는 경우
        if index_name in existing:
            info = pc.describe_index(index_name)
            if info.dimension != dimension:
                pc.delete_index(index_name)  # 기존 인덱스 삭제
                existing.remove(index_name)

        # 인덱스가 없으면 새로 생성
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=dimension,       # 벡터 차원 (임베딩 모델에 맞춰야 함)
                metric="cosine",           # 유사도 측정 방식
                # cosine(코사인 유사도): 두 벡터의 방향이 비슷할수록 유사하다고 판단
                # 텍스트 검색에서 가장 널리 사용되는 방식
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                # Serverless: 서버를 직접 관리하지 않아도 되는 방식
                # AWS us-east-1 리전(미국 동부)에 데이터 저장
            )
            # 인덱스 생성은 시간이 걸리므로, 준비될 때까지 1초 간격으로 대기
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        # 생성된 인덱스에 연결
        index = pc.Index(index_name)

        # 이미 올바른 수의 벡터가 저장되어 있으면 재사용 (업로드 생략 → 빠름)
        stats = index.describe_index_stats()
        if stats.total_vector_count == len(chunks):
            vectorstore = PineconeVectorStore(index=index, embedding=embedding)
            print(f"기존 Pinecone 인덱스 재사용: {index_name} ({stats.total_vector_count}개)")
        else:
            # 벡터 수가 다르면 (중복이거나 비어있는 상태) 정리 후 새로 업로드
            if stats.total_vector_count > 0:
                index.delete(delete_all=True)
            vectorstore = PineconeVectorStore(index=index, embedding=embedding)
            vectorstore.add_documents(documents=chunks)

    # 완성된 벡터 스토어를 캐시에 저장하여 다음 질문 때 재사용
    _vectorstore_cache[cache_key] = vectorstore
    print(f"벡터 스토어 생성 완료: {cache_key}")
    return vectorstore

# ============================================================
# 5. RAG 체인 구성
# ============================================================
# RAG 체인이란?
#   "검색 → 프롬프트 구성 → AI 답변 생성"을 하나의 파이프라인(체인)으로 연결한 것
#
# 비유: 음식점 주문 시스템
#   손님 주문(질문) → 주방에서 재료 꺼냄(문서 검색) → 레시피대로 조리(프롬프트+LLM) → 음식 서빙(답변)
#   이 과정을 LCEL 체인으로 자동화하면: 프롬프트 | LLM | 출력파서

# AI에게 보내는 시스템 프롬프트 (AI의 역할과 행동 규칙을 정의하는 지시문)
# {context} 자리에는 실행 시 검색된 동화 내용이 채워짐
# {question} 자리에는 사용자의 질문이 채워짐
RAG_SYSTEM_PROMPT = """\
당신은 한국 전래 동화 전문가입니다.
아래 제공된 동화 내용(context)을 바탕으로 사용자의 질문에 친절하고 정확하게 답변하세요.

규칙:
- 반드시 context에 포함된 정보를 기반으로 답변하세요.
- context에 없는 내용은 "제공된 동화에서는 해당 내용을 찾을 수 없습니다."라고 안내하세요.
- 답변은 한국어로 작성하세요.
- 동화의 교훈이나 의미에 대한 질문에도 답변할 수 있습니다.

context:
{context}"""

# 프롬프트 템플릿 구성 - AI에게 보내는 메시지의 전체 구조를 정의
# 대화가 이루어질 때 실제로 AI에게 전달되는 메시지는 이런 순서:
#   1. 시스템 메시지: "당신은 동화 전문가입니다..." (AI의 역할 설정)
#   2. 대화 히스토리: 이전에 주고받은 질문/답변들 (AI가 맥락을 기억하도록)
#   3. 사용자 메시지: 현재 질문 (AI가 답변할 대상)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),          # AI 역할 + 검색된 동화 내용
    MessagesPlaceholder("chat_history"),     # 이전 대화 내역이 들어갈 자리
    ("human", "{question}"),                 # 사용자의 현재 질문
])

# LLM(Large Language Model, 대규모 언어 모델) 설정
# gpt-4.1-nano: OpenAI의 경량 모델 (빠르고 저렴하면서도 품질 좋음)
# temperature=0.3: AI 답변의 창의성 조절 (0.0~2.0)
#   - 0.0에 가까울수록: 항상 같은 답변, 사실적이고 일관됨
#   - 1.0 이상: 매번 다른 답변, 창의적이지만 사실과 다를 수 있음
#   - 0.3: 사실적이면서 약간의 자연스러움을 가진 균형 잡힌 설정
# streaming=True: 답변을 한꺼번에가 아니라 한 글자씩 실시간으로 받음
#   이렇게 하면 사용자가 답변이 생성되는 과정을 실시간으로 볼 수 있음 (타이핑 효과)
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3, streaming=True)


def format_docs(docs):
    """
    검색된 문서 조각(Document 객체 리스트)을 하나의 문자열로 합치는 함수

    입력 예시: [Document("흥부는 착한..."), Document("놀부는 욕심이...")]
    출력 예시: "흥부는 착한...\n\n---\n\n놀부는 욕심이..."

    각 조각 사이에 "---" 구분선을 넣어서 AI가 서로 다른 조각임을 인식할 수 있게 함
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever):
    """
    검색기(Retriever)를 받아 전체 RAG 체인을 하나로 구성하는 함수
    (현재 answer() 함수에서는 단계를 분리하여 직접 처리하므로, 참고용 함수)

    체인 흐름 (LCEL 파이프라인):
    입력: {"question": "흥부는?", "chat_history": [...]}
                           ↓
    RunnableParallel (세 가지 작업을 동시에 처리):
      ├─ context: 질문 추출 → 검색기로 문서 검색 → 텍스트로 합침
      ├─ question: 질문 문자열 추출
      └─ chat_history: 대화 히스토리 추출
                           ↓
    rag_prompt: 위 결과를 프롬프트 템플릿에 삽입
                           ↓
    llm: AI 모델이 답변 생성
                           ↓
    StrOutputParser: 텍스트만 추출하여 반환
    """
    chain = (
        RunnableParallel(
            # RunnableLambda: 파이썬 함수를 체인의 한 단계로 변환
            # lambda x: x["question"] → 입력 딕셔너리에서 "question" 값만 꺼냄
            # | retriever → 꺼낸 질문 문자열로 유사 문서 검색
            # | format_docs → 검색된 문서들을 하나의 문자열로 합침
            context=RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
            question=RunnableLambda(lambda x: x["question"]),
            chat_history=RunnableLambda(lambda x: x.get("chat_history", [])),
        )
        | rag_prompt         # 검색 결과 + 질문 + 히스토리를 프롬프트에 삽입
        | llm                # AI 모델이 답변 생성
        | StrOutputParser()  # AI 응답에서 텍스트만 추출
    )
    return chain

# ============================================================
# 6. Gradio 챗봇 함수 (스트리밍 응답)
# ============================================================
# 이 함수가 RAG 챗봇의 핵심 로직입니다.
# Gradio가 사용자의 메시지를 이 함수에 전달하고, 이 함수가 답변을 반환합니다.


def answer(message, chat_history, embedding_name, store_name, search_type, top_k,
           chunk_size, chunk_overlap):
    """
    사용자 질문을 받아서 RAG 파이프라인으로 답변을 생성하는 핵심 함수
    gr.Blocks 기반 수동 채팅 UI에서 호출되며, 채팅 히스토리 + 오른쪽 패널 정보를 함께 반환

    동작 순서:
    1. 대화 히스토리를 LangChain이 이해하는 형식으로 변환
    2. 선택된 설정으로 벡터 스토어 & 검색기 준비
    3. 질문과 유사한 동화 조각을 벡터 스토어에서 검색
    4. 오른쪽 패널에 RAG 파이프라인 과정 표시 (설정 → 검색 결과 → 프롬프트)
    5. 검색된 조각 + 질문을 LLM에 전달하여 스트리밍 답변 생성

    Args:
        message: 사용자가 입력한 질문 (예: "흥부는 어떤 사람이었나요?")
        chat_history: gr.State로 관리되는 대화 히스토리 리스트
                      형태: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        embedding_name: 선택한 임베딩 모델 이름 (Gradio 드롭다운에서 선택한 값)
        store_name: 선택한 벡터 스토어 이름 ("Chroma" / "FAISS" / "Pinecone")
        search_type: 검색 방식 ("유사도 검색 (Top-K)" 또는 "MMR (다양성 고려)")
        top_k: 검색할 문서 조각의 수 (기본 3개, Gradio 슬라이더에서 설정)
        chunk_size: 텍스트 분할 시 한 조각의 최대 글자 수 (Gradio 슬라이더에서 설정)
        chunk_overlap: 텍스트 분할 시 조각 간 겹침 글자 수 (Gradio 슬라이더에서 설정)

    Yields:
        chat_history, step1_md, step2_md, step3_md 튜플을 단계별로 yield
        - chat_history: 현재까지의 전체 대화 내역 (gr.Chatbot에 표시)
        - step1_md: 1단계 설정 정보 (오른쪽 패널 상단)
        - step2_md: 2단계 검색된 청크 (오른쪽 패널 중간)
        - step3_md: 3단계 LLM에 전달된 프롬프트 (오른쪽 패널 하단)
    """
    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)
    # --- 사용자 메시지를 히스토리에 추가 ---
    chat_history = chat_history + [{"role": "user", "content": message}]

    # --- 1단계: 오른쪽 패널 - 사용된 설정 표시 ---
    step1_md = (
        "### 1단계: 사용된 설정\n"
        f"- **임베딩 모델**: {embedding_name}\n"
        f"- **벡터 스토어**: {store_name}\n"
        f"- **검색 방식**: {search_type}\n"
        f"- **검색 결과 수 (k)**: {int(top_k)}\n"
        f"- **청크 크기**: {chunk_size}자\n"
        f"- **청크 겹침**: {chunk_overlap}자\n"
    )
    # 설정 정보를 먼저 표시 (아직 검색/프롬프트는 비어 있음)
    yield chat_history, step1_md, "*검색 중...*", ""

    # --- 대화 히스토리를 LangChain 형식으로 변환 ---
    # Gradio 히스토리(딕셔너리) → LangChain 메시지 객체
    lc_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))
    # 마지막 사용자 메시지는 question으로 따로 전달하므로 히스토리에서 제외
    lc_history = lc_history[:-1]

    # --- 2단계: 벡터 스토어 & 검색기 구성 ---
    vectorstore = get_vectorstore(embedding_name, store_name, chunk_size, chunk_overlap)

    if search_type == "MMR (다양성 고려)":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": int(top_k),
                "fetch_k": int(top_k) * 3,
            },
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": int(top_k)},
        )

    # --- 3단계: 질문으로 관련 동화 조각 검색 ---
    retrieved_docs = retriever.invoke(message)

    # --- 오른쪽 패널 - 검색된 청크 표시 ---
    step2_md = "### 2단계: 검색된 청크\n"
    for i, doc in enumerate(retrieved_docs, 1):
        filename = os.path.basename(doc.metadata.get("source", ""))
        name = filename.replace(".txt", "").replace("_", " ")
        # 전체 내용을 표시 (접기 블록 안에 넣음)
        content = doc.page_content.replace("\n", "  \n")  # Markdown 줄바꿈
        step2_md += (
            f"\n**청크 {i}** — 출처: `{name}`\n"
            f"<details><summary>내용 보기 ({len(doc.page_content)}자)</summary>\n\n"
            f"{content}\n\n</details>\n"
        )
    yield chat_history, step1_md, step2_md, "*프롬프트 구성 중...*"

    # --- 4단계: 검색 결과 + 질문을 LLM에 전달 ---
    context_str = format_docs(retrieved_docs)

    rag_input = {
        "context": context_str,
        "question": message,
        "chat_history": lc_history,
    }

    # --- 오른쪽 패널 - LLM에 전달된 프롬프트 표시 ---
    # 실제 프롬프트를 구성하여 사용자에게 보여줌
    filled_system = RAG_SYSTEM_PROMPT.replace("{context}", context_str).replace(
        "{question}", message
    )
    step3_md = (
        "### 3단계: LLM에 전달된 프롬프트\n"
        f"<details><summary>시스템 프롬프트 보기</summary>\n\n"
        f"```\n{filled_system}\n```\n\n</details>\n\n"
        f"<details><summary>대화 히스토리 ({len(lc_history)}개 메시지)</summary>\n\n"
    )
    if lc_history:
        for msg in lc_history:
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            preview = msg.content[:100] + ("..." if len(msg.content) > 100 else "")
            step3_md += f"- **{role}**: {preview}\n"
    else:
        step3_md += "- (첫 번째 질문 — 히스토리 없음)\n"
    step3_md += (
        f"\n</details>\n\n"
        f"**사용자 질문**: {message}\n"
    )
    yield chat_history, step1_md, step2_md, step3_md

    # --- 5단계: 스트리밍 답변 생성 ---
    chain = rag_prompt | llm | StrOutputParser()

    response = ""
    # AI의 빈 응답을 히스토리에 추가 (스트리밍으로 채워나감)
    chat_history = chat_history + [{"role": "assistant", "content": ""}]
    for chunk in chain.stream(rag_input):
        response += chunk
        # 마지막 assistant 메시지의 content를 업데이트
        chat_history[-1]["content"] = response
        yield chat_history, step1_md, step2_md, step3_md

    # 답변 완료 (참고 자료는 오른쪽 패널 "검색된 청크"에 표시되므로 생략)


# ============================================================
# 7. Gradio 인터페이스 (좌우 분할 레이아웃)
# ============================================================
# gr.ChatInterface 대신 gr.Blocks를 사용하여 직접 UI를 구성
# 왼쪽: 채팅 영역 (대화 히스토리 + 질문 입력)
# 오른쪽: RAG 파이프라인 실행 과정 (설정 → 검색 결과 → 프롬프트)

# 예시 질문 목록 (Examples 컴포넌트에서 사용)
EXAMPLE_QUESTIONS = [
    ["흥부는 어떤 사람이었나요?"],
    ["토끼는 어떻게 용궁에서 탈출했나요?"],
    ["콩쥐를 도와준 동물들은 누구인가요?"],
    ["나무꾼은 왜 선녀의 날개옷을 숨겼나요?"],
    ["해와 달이 된 오누이에서 호랑이는 어떻게 되었나요?"],
    ["이 동화들의 공통된 교훈은 무엇인가요?"],
]

FLOWCHART_HTML = """
<style>
  #flowchart-overlay,
  #flowchart-overlay * {
    color: #000 !important;
  }
  #flowchart-overlay .fc-arrow {
    color: #555 !important;
  }
  #flowchart-overlay .fc-badge {
    color: #fff !important;
  }
  #flowchart-overlay .fc-close {
    color: #fff !important;
  }
  #flowchart-overlay .fc-yes {
    color: #2e7d32 !important;
  }
  #flowchart-overlay .fc-no {
    color: #c62828 !important;
  }
  #flowchart-overlay .fc-pipe {
    color: #4527a0 !important;
  }
</style>
<div id="flowchart-overlay" onclick="if(event.target===this)this.style.display='none'"
     style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
            background:rgba(0,0,0,0.6);z-index:9999;overflow-y:auto;padding:30px 10px;">
<div style="font-family:'Segoe UI',sans-serif;max-width:750px;margin:0 auto;
            background:#fff;border-radius:16px;padding:30px 24px;position:relative;
            box-shadow:0 8px 32px rgba(0,0,0,0.3);">

  <!-- 닫기 버튼 -->
  <button class="fc-close"
          onclick="document.getElementById('flowchart-overlay').style.display='none'"
          style="position:sticky;top:0;float:right;background:#e53935;
                 border:none;border-radius:50%;width:36px;height:36px;font-size:20px;
                 cursor:pointer;z-index:10;box-shadow:0 2px 8px rgba(0,0,0,0.2);">
    ✕
  </button>

  <!-- ========== 앱 시작 플로우 ========== -->
  <h2 style="text-align:center;margin-bottom:18px;">
    A. 앱 시작 플로우 (1회 실행)
  </h2>
  <div style="display:flex;flex-direction:column;align-items:center;gap:0;">

    <div style="background:#e8f5e9;border:2px solid #43a047;border-radius:10px;
                padding:12px 28px;text-align:center;min-width:340px;">
      <b>python rag_chatbot.py</b>
    </div>
    <div class="fc-arrow" style="font-size:22px;">▼</div>

    <div style="background:#fff3e0;border:2px solid #fb8c00;border-radius:10px;
                padding:12px 28px;text-align:center;min-width:340px;">
      <b>load_dotenv()</b><br>
      <span style="font-size:13px;">.env에서 API 키 로딩</span>
    </div>
    <div class="fc-arrow" style="font-size:22px;">▼</div>

    <div style="background:#e3f2fd;border:2px solid #1e88e5;border-radius:10px;
                padding:14px 28px;text-align:center;min-width:340px;">
      <b>load_and_split_documents(500, 100)</b><br>
      <span style="font-size:13px;">
        DirectoryLoader → 동화 .txt 5개 로딩<br>
        RecursiveCharacterTextSplitter<br>
        → 약 24개 청크 생성 (CHUNKS)
      </span>
    </div>
    <div class="fc-arrow" style="font-size:22px;">▼</div>

    <div style="background:#f3e5f5;border:2px solid #8e24aa;border-radius:10px;
                padding:14px 28px;text-align:center;min-width:340px;">
      <b>Gradio UI 구성</b><br>
      <span style="font-size:13px;">
        드롭다운 3개 · 슬라이더 3개 · 채팅창 · 파이프라인 패널
      </span>
    </div>
    <div class="fc-arrow" style="font-size:22px;">▼</div>

    <div style="background:#e8f5e9;border:2px solid #43a047;border-radius:10px;
                padding:12px 28px;text-align:center;min-width:340px;">
      <b>demo.launch()</b><br>
      <span style="font-size:13px;">http://127.0.0.1:7860 서버 시작</span>
    </div>
  </div>

  <hr style="margin:32px 0;border:none;border-top:2px dashed #aaa;">

  <!-- ========== 질문-응답 플로우 ========== -->
  <h2 style="text-align:center;margin-bottom:18px;">
    B. 질문-응답 플로우 (매 질문마다)
  </h2>
  <div style="display:flex;flex-direction:column;align-items:center;gap:0;">

    <!-- 사용자 입력 -->
    <div style="background:#e8f5e9;border:2px solid #43a047;border-radius:24px;
                padding:12px 32px;text-align:center;">
      <b>사용자 질문 입력</b> (Enter / 전송 버튼)
    </div>
    <div class="fc-arrow" style="font-size:22px;">▼</div>

    <!-- on_submit -->
    <div style="background:#fff3e0;border:2px solid #fb8c00;border-radius:10px;
                padding:10px 24px;text-align:center;min-width:400px;">
      <b>on_submit()</b><br>
      <span style="font-size:13px;">
        Gradio inputs 8개 수집 → answer() 호출
      </span>
    </div>
    <div class="fc-arrow" style="font-size:22px;">▼</div>

    <!-- answer() 큰 박스 -->
    <div style="border:3px solid #1565c0;border-radius:14px;padding:20px 18px;
                min-width:500px;max-width:700px;background:#f5f7ff;">

      <div style="text-align:center;margin-bottom:14px;">
        <span class="fc-badge"
              style="background:#1565c0;padding:5px 18px;border-radius:8px;
                     font-weight:bold;font-size:15px;">
          answer() 제너레이터 — yield로 단계별 UI 업데이트
        </span>
      </div>

      <div style="display:flex;flex-direction:column;align-items:center;gap:0;">

        <!-- Step 1 -->
        <div style="background:#e3f2fd;border:2px solid #1e88e5;border-radius:8px;
                    padding:10px 20px;text-align:center;width:90%;">
          <b>Step 1: 설정 표시</b><br>
          <span style="font-size:13px;">
            임베딩 · 스토어 · 검색방식 · k · chunk_size · chunk_overlap<br>
            → <i>yield: step1 패널 업데이트</i>
          </span>
        </div>
        <div class="fc-arrow" style="font-size:20px;">▼</div>

        <!-- Step 2 -->
        <div style="background:#fff3e0;border:2px solid #fb8c00;border-radius:8px;
                    padding:10px 20px;text-align:center;width:90%;">
          <b>Step 2: 벡터 스토어 준비</b><br>
          <span style="font-size:13px;">get_vectorstore(embedding, store, chunk_size, overlap)</span>
        </div>

        <!-- 캐시 분기 -->
        <div class="fc-arrow" style="font-size:20px;">▼</div>
        <div style="display:flex;gap:16px;width:90%;justify-content:center;flex-wrap:wrap;">
          <div style="background:#fffde7;border:1.5px solid #f9a825;border-radius:50%;
                      width:110px;height:50px;display:flex;align-items:center;
                      justify-content:center;font-size:13px;font-weight:bold;">
            캐시 hit?
          </div>
        </div>
        <div style="display:flex;width:90%;justify-content:center;gap:40px;margin:4px 0;">
          <div style="text-align:center;">
            <span class="fc-yes" style="font-weight:bold;font-size:13px;">Yes → 즉시 반환</span>
          </div>
          <div style="text-align:center;">
            <span class="fc-no" style="font-weight:bold;font-size:13px;">No ▼</span>
          </div>
        </div>

        <!-- 캐시 miss 상세 -->
        <div style="background:#fce4ec;border:1.5px dashed #c62828;border-radius:8px;
                    padding:10px 14px;width:85%;margin-bottom:4px;">
          <div style="display:flex;flex-direction:column;align-items:center;gap:2px;">
            <div style="background:#fff;border:1px solid #bbb;border-radius:6px;
                        padding:6px 14px;font-size:13px;text-align:center;width:85%;">
              <b>load_and_split_documents</b>(chunk_size, overlap)<br>
              문서 재분할 → 새 chunks 생성
            </div>
            <div class="fc-arrow" style="font-size:16px;">▼</div>
            <div style="background:#fff;border:1px solid #bbb;border-radius:6px;
                        padding:6px 14px;font-size:13px;text-align:center;width:85%;">
              <b>get_embedding_model</b>(name)<br>
              OpenAI(1536d) / bge-m3(1024d) / MiniLM(384d)
            </div>
            <div class="fc-arrow" style="font-size:16px;">▼</div>
            <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
              <div style="background:#e8f5e9;border:1px solid #66bb6a;border-radius:6px;
                          padding:6px 10px;font-size:12px;text-align:center;">
                <b>Chroma</b><br>로컬 폴더
              </div>
              <div style="background:#e3f2fd;border:1px solid #42a5f5;border-radius:6px;
                          padding:6px 10px;font-size:12px;text-align:center;">
                <b>FAISS</b><br>메모리+파일
              </div>
              <div style="background:#f3e5f5;border:1px solid #ab47bc;border-radius:6px;
                          padding:6px 10px;font-size:12px;text-align:center;">
                <b>Pinecone</b><br>클라우드
              </div>
            </div>
          </div>
        </div>
        <div class="fc-arrow" style="font-size:20px;">▼</div>

        <!-- Step 3 -->
        <div style="background:#e8f5e9;border:2px solid #43a047;border-radius:8px;
                    padding:10px 20px;text-align:center;width:90%;">
          <b>Step 3: Retriever 생성</b><br>
          <span style="font-size:13px;">
            유사도 검색 (Top-K) 또는 MMR (다양성 고려)
          </span>
        </div>
        <div class="fc-arrow" style="font-size:20px;">▼</div>

        <!-- Step 4 -->
        <div style="background:#e3f2fd;border:2px solid #1e88e5;border-radius:8px;
                    padding:10px 20px;text-align:center;width:90%;">
          <b>Step 4: 문서 검색</b><br>
          <span style="font-size:13px;">
            retriever.invoke(질문) → 유사한 청크 k개 반환<br>
            → <i>yield: 검색 결과 패널 업데이트</i>
          </span>
        </div>
        <div class="fc-arrow" style="font-size:20px;">▼</div>

        <!-- Step 5 -->
        <div style="background:#f3e5f5;border:2px solid #8e24aa;border-radius:8px;
                    padding:10px 20px;text-align:center;width:90%;">
          <b>Step 5: 프롬프트 구성</b><br>
          <span style="font-size:13px;">
            context(검색결과) + question + chat_history<br>
            → <i>yield: 프롬프트 패널 업데이트</i>
          </span>
        </div>
        <div class="fc-arrow" style="font-size:20px;">▼</div>

        <!-- Step 6 -->
        <div style="background:linear-gradient(135deg,#e3f2fd,#f3e5f5);
                    border:2px solid #5e35b1;border-radius:8px;
                    padding:12px 20px;text-align:center;width:90%;">
          <b>Step 6: LLM 스트리밍 답변</b><br>
          <div style="display:flex;justify-content:center;align-items:center;
                      gap:8px;margin-top:6px;flex-wrap:wrap;">
            <span style="background:#fff;border:1px solid #999;border-radius:5px;
                         padding:4px 10px;font-size:12px;">rag_prompt</span>
            <span class="fc-pipe" style="font-weight:bold;">→</span>
            <span style="background:#fff;border:1px solid #999;border-radius:5px;
                         padding:4px 10px;font-size:12px;">gpt-4.1-nano</span>
            <span class="fc-pipe" style="font-weight:bold;">→</span>
            <span style="background:#fff;border:1px solid #999;border-radius:5px;
                         padding:4px 10px;font-size:12px;">StrOutputParser</span>
          </div>
          <span style="font-size:13px;">
            <br>chain.stream() → <i>yield: 답변 실시간 타이핑</i>
          </span>
        </div>

      </div>
    </div><!-- answer() 박스 끝 -->

    <div class="fc-arrow" style="font-size:22px;">▼</div>
    <div style="background:#e8f5e9;border:2px solid #43a047;border-radius:24px;
                padding:12px 32px;text-align:center;">
      <b>답변 표시 완료</b> → 입력창 비움 → 다음 질문 대기
    </div>
  </div>

  <hr style="margin:32px 0;border:none;border-top:2px dashed #aaa;">

  <!-- ========== 캐시 구조 ========== -->
  <h2 style="text-align:center;margin-bottom:14px;">
    C. 캐시 구조
  </h2>
  <div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;">
    <div style="background:#e3f2fd;border:2px solid #1e88e5;border-radius:10px;
                padding:14px 20px;min-width:280px;">
      <b>_embedding_cache</b><br>
      <span style="font-size:13px;">
        키: 모델 이름<br>
        예: "OpenAI (text-embedding-3-small)" → 모델 객체<br>
        → 모델 재로딩 방지
      </span>
    </div>
    <div style="background:#fff3e0;border:2px solid #fb8c00;border-radius:10px;
                padding:14px 20px;min-width:280px;">
      <b>_vectorstore_cache</b><br>
      <span style="font-size:13px;">
        키: "임베딩|스토어|chunk_size|overlap"<br>
        예: "OpenAI|Chroma|500|100" → 벡터스토어<br>
        → chunk 설정 변경 시 별도 캐시
      </span>
    </div>
  </div>

</div><!-- 흰색 박스 끝 -->
</div><!-- 오버레이 끝 -->
"""

with gr.Blocks(
    title="한국 전래 동화 RAG 챗봇",
    analytics_enabled=False,
) as demo:
    # --- 플로우차트 팝업 (HTML 오버레이) ---
    gr.HTML(FLOWCHART_HTML)

    # --- 타이틀 & 설명 ---
    with gr.Row():
        gr.Markdown(
            "# 한국 전래 동화 RAG 챗봇\n"
            "한국 전래 동화(흥부와 놀부, 토끼와 거북이, 콩쥐 팥쥐, "
            "선녀와 나무꾼, 해와 달이 된 오누이)에 대해 질문해 보세요.\n\n"
            "RAG 파이프라인이 동화 내용을 검색하여 답변합니다."
        )
        flowchart_btn = gr.Button("플로우차트 보기", scale=0, min_width=140)
        flowchart_btn.click(
            fn=None,
            js="() => { document.getElementById('flowchart-overlay').style.display = 'block'; }",
        )

    # --- RAG 설정 (가로 배치) ---
    with gr.Row(equal_height=True):
        embedding_dd = gr.Dropdown(
            choices=[
                "OpenAI (text-embedding-3-small)",
                "HuggingFace (bge-m3)",
                "HuggingFace (MiniLM-multilingual)",
            ],
            value="OpenAI (text-embedding-3-small)",
            label="임베딩 모델",
            info="텍스트를 벡터로 변환하는 모델",
        )
        store_dd = gr.Dropdown(
            choices=["Chroma", "FAISS", "Pinecone"],
            value="Chroma",
            label="벡터 스토어",
            info="벡터를 저장하고 검색하는 DB",
        )
        search_dd = gr.Dropdown(
            choices=["유사도 검색 (Top-K)", "MMR (다양성 고려)"],
            value="유사도 검색 (Top-K)",
            label="검색 방식",
            info="유사한 문서 조각을 찾는 방식",
        )
        top_k_slider = gr.Slider(
            minimum=1, maximum=10, value=3, step=1,
            label="검색 결과 수 (k)",
            info="AI에게 전달할 문서 조각 수",
        )
    with gr.Row(equal_height=True):
        chunk_size_slider = gr.Slider(
            minimum=100, maximum=1000, value=500, step=50,
            label="청크 크기 (chunk_size)",
            info="한 조각의 최대 글자 수 (변경 시 벡터스토어 재생성)",
        )
        chunk_overlap_slider = gr.Slider(
            minimum=0, maximum=200, value=100, step=10,
            label="청크 겹침 (chunk_overlap)",
            info="조각 사이에 겹치는 글자 수 (문맥 연결용)",
        )

    # --- 설정 변경 시 토스트 알림 (말풍선) ---
    # gr.Info()는 화면 우측 하단에 잠시 나타났다 사라지는 토스트 알림을 표시
    EMBEDDING_DESCRIPTIONS = {
        "OpenAI (text-embedding-3-small)": "OpenAI 임베딩 — 유료 API, 빠른 속도, 1536차원",
        "HuggingFace (bge-m3)": "BGE-M3 임베딩 — 무료 오픈소스, 고성능 다국어, 1024차원 (첫 로딩 느림)",
        "HuggingFace (MiniLM-multilingual)": "MiniLM 임베딩 — 무료 경량 모델, 50개+ 언어 지원, 384차원",
    }
    STORE_DESCRIPTIONS = {
        "Chroma": "Chroma — 로컬 폴더에 파일로 저장, 설치 간편, 개인 학습용",
        "FAISS": "FAISS — 메모리 기반 고속 검색, Facebook 개발, 빠른 유사도 탐색",
        "Pinecone": "Pinecone — 클라우드 저장, 대규모 서비스에 적합, API 키 필요",
    }
    SEARCH_DESCRIPTIONS = {
        "유사도 검색 (Top-K)": "유사도 검색 — 질문과 가장 비슷한 k개 조각을 단순 선택",
        "MMR (다양성 고려)": "MMR 검색 — 유사하면서도 서로 다양한 내용을 골고루 선택",
    }

    def on_embedding_change(name):
        gr.Info(EMBEDDING_DESCRIPTIONS.get(name, name))

    def on_store_change(name):
        gr.Info(STORE_DESCRIPTIONS.get(name, name))

    def on_search_change(name):
        gr.Info(SEARCH_DESCRIPTIONS.get(name, name))

    def on_chunk_size_change(val):
        gr.Info(f"청크 크기 → {int(val)}자 (다음 질문 시 벡터스토어가 재생성됩니다)")

    def on_chunk_overlap_change(val):
        gr.Info(f"청크 겹침 → {int(val)}자 (다음 질문 시 벡터스토어가 재생성됩니다)")

    embedding_dd.change(fn=on_embedding_change, inputs=embedding_dd)
    store_dd.change(fn=on_store_change, inputs=store_dd)
    search_dd.change(fn=on_search_change, inputs=search_dd)
    chunk_size_slider.change(fn=on_chunk_size_change, inputs=chunk_size_slider)
    chunk_overlap_slider.change(fn=on_chunk_overlap_change, inputs=chunk_overlap_slider)

    # --- 대화 히스토리를 수동으로 관리하는 State ---
    # gr.ChatInterface는 히스토리를 자동 관리하지만,
    # gr.Blocks에서는 gr.State를 사용하여 직접 관리해야 함
    chat_state = gr.State([])

    # --- 상단: 채팅 영역 ---
    chatbot = gr.Chatbot(
        label="대화",
        height=400,
    )
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="질문을 입력하세요...",
            label="질문",
            scale=4,
            show_label=False,
        )
        send_btn = gr.Button("전송", variant="primary", scale=1)
    gr.Examples(
        examples=EXAMPLE_QUESTIONS,
        inputs=user_input,
        label="예시 질문",
    )

    # --- 하단: RAG 파이프라인 실행 과정 ---
    gr.Markdown("## RAG 파이프라인 실행 과정")
    step1_settings = gr.Markdown(
        "*질문을 입력하면 RAG 파이프라인 과정이 여기에 표시됩니다.*"
    )
    step2_chunks = gr.Markdown("")
    step3_prompt = gr.Markdown("")

    # --- 이벤트 핸들링 ---
    # 전송 버튼 클릭 또는 Enter 키로 answer 함수 호출
    # inputs: answer 함수에 전달할 값들
    # outputs: answer 함수가 yield하는 값들이 업데이트할 컴포넌트들

    # answer 함수의 반환값: (chat_history, step1_md, step2_md, step3_md)
    # → chat_state와 chatbot 모두에 chat_history를 전달
    inputs = [user_input, chat_state, embedding_dd, store_dd, search_dd, top_k_slider,
              chunk_size_slider, chunk_overlap_slider]
    outputs = [chatbot, step1_settings, step2_chunks, step3_prompt]

    def on_submit(message, history, embedding_name, store_name, search_type, top_k,
                  chunk_size, chunk_overlap):
        """
        answer 제너레이터를 감싸서 chatbot + chat_state + 오른쪽 패널을 동시에 업데이트
        chatbot과 chat_state는 같은 chat_history 값을 받으므로 함께 업데이트
        """
        for chat_history, s1, s2, s3 in answer(
            message, history, embedding_name, store_name, search_type, top_k,
            chunk_size, chunk_overlap
        ):
            # chat_state도 업데이트하기 위해 5개 값을 yield
            yield chat_history, s1, s2, s3, chat_history

    all_outputs = outputs + [chat_state]

    # 전송 버튼 클릭 이벤트
    send_btn.click(
        fn=on_submit,
        inputs=inputs,
        outputs=all_outputs,
    ).then(
        # 전송 후 입력창 비우기
        fn=lambda: "",
        outputs=user_input,
    )

    # Enter 키 이벤트 (submit = Enter 키 누름)
    user_input.submit(
        fn=on_submit,
        inputs=inputs,
        outputs=all_outputs,
    ).then(
        fn=lambda: "",
        outputs=user_input,
    )

# ============================================================
# 8. 실행
# ============================================================
# if __name__ == "__main__": 이란?
#   이 파일을 직접 실행했을 때(python rag_chatbot.py)만 아래 코드를 실행하라는 뜻
#   다른 파일에서 import rag_chatbot 으로 불러올 때는 실행되지 않음
#
# demo.launch(): Gradio 웹 서버를 시작
#   실행 후 브라우저에서 http://127.0.0.1:7860 으로 접속하면 채팅 UI가 나타남
if __name__ == "__main__":
    demo.launch()
