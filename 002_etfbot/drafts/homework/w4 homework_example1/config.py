"""설정 및 상수 정의"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-nano"

# 청킹 기본값
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50

# 검색 기본값
DEFAULT_TOP_K = 3
SUPPORTED_EXTENSIONS = [".md", ".txt"]

# Reranker
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Hybrid 가중치 프리셋
HYBRID_WEIGHT_PRESETS = {
    "Embedding 중심 (0.7:0.3)": [0.7, 0.3],
    "균등 (0.5:0.5)": [0.5, 0.5],
    "BM25 중심 (0.3:0.7)": [0.3, 0.7],
}
