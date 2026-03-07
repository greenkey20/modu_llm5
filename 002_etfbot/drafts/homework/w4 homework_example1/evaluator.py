"""검색 성능 평가 모듈 (HitRate, MRR)"""
import pandas as pd
from langchain_core.documents import Document
from config import DEFAULT_TOP_K


def compute_hit_rate(
    retrieved_docs: list[list[Document]],
    ground_truths: list[list[str]],
    k: int = DEFAULT_TOP_K,
) -> float:
    """HitRate@k 계산.

    각 쿼리에 대해 상위 k개 검색 결과 중 정답이 하나라도 포함되면 hit.
    """
    if not retrieved_docs:
        return 0.0
    hits = 0
    for docs, truths in zip(retrieved_docs, ground_truths):
        top_k = docs[:k]
        contents = " ".join(
            d.metadata.get("original_content", d.page_content) for d in top_k
        )
        if any(truth in contents for truth in truths):
            hits += 1
    return hits / len(retrieved_docs)


def compute_mrr(
    retrieved_docs: list[list[Document]],
    ground_truths: list[list[str]],
) -> float:
    """MRR (Mean Reciprocal Rank) 계산."""
    if not retrieved_docs:
        return 0.0
    reciprocal_ranks = []
    for docs, truths in zip(retrieved_docs, ground_truths):
        rr = 0.0
        for rank, doc in enumerate(docs, start=1):
            content = doc.metadata.get("original_content", doc.page_content)
            if any(truth in content for truth in truths):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_pipeline(
    retriever,
    questions: list[str],
    ground_truths: list[list[str]],
    k: int = DEFAULT_TOP_K,
) -> dict:
    """단일 파이프라인 평가. {'hit_rate': float, 'mrr': float} 반환."""
    all_results = [retriever.invoke(q) for q in questions]
    return {
        "hit_rate": compute_hit_rate(all_results, ground_truths, k),
        "mrr": compute_mrr(all_results, ground_truths),
        "num_queries": len(questions),
        "k": k,
    }


def compare_pipelines(
    pipelines: dict,
    questions: list[str],
    ground_truths: list[list[str]],
    k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """여러 파이프라인 비교 평가. DataFrame 반환."""
    rows = []
    for name, retriever in pipelines.items():
        result = evaluate_pipeline(retriever, questions, ground_truths, k)
        rows.append({
            "파이프라인": name,
            "HitRate@k": round(result["hit_rate"], 4),
            "MRR": round(result["mrr"], 4),
            "쿼리 수": result["num_queries"],
            "k": result["k"],
        })
    return pd.DataFrame(rows)
