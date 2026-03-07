"""Contextual Retrieval Lab - Gradio 메인 앱"""
import os
import gradio as gr
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K,
    HYBRID_WEIGHT_PRESETS,
)
from document_loader import load_document, split_document, detect_language
from context_generator import generate_contexts_batch, create_contextual_chunks
from search_engine import (
    build_vectorstore, build_bm25_retriever,
    build_hybrid_retriever, build_rerank_retriever, search,
)
from evaluator import compute_hit_rate, compute_mrr

# ── 전역 상태 ──
state = {
    "doc": None,
    "chunks": [],
    "contextual_chunks": [],
    "language": "ko",
    "retrievers": {},
}


# ── CSS 테마 ──
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.tab-nav button {
    font-size: 15px !important;
    padding: 12px 24px !important;
    border-radius: 12px 12px 0 0 !important;
}
.panel {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}
.stat-box {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
footer { display: none !important; }
"""


# ═══════════════════════════════════════════
# Tab 1: 문서 업로드 / 청킹
# ═══════════════════════════════════════════
def process_upload(file, chunk_size, chunk_overlap, lang_choice):
    """파일 업로드 → 청킹 처리."""
    if file is None:
        return "⚠️ 파일을 업로드해주세요.", "", pd.DataFrame()

    try:
        doc = load_document(file.name)
        if lang_choice != "자동 감지":
            doc.metadata["language"] = "ko" if lang_choice == "한국어" else "en"
        state["language"] = doc.metadata["language"]
        state["doc"] = doc

        chunks = split_document(doc, chunk_size, chunk_overlap)
        state["chunks"] = chunks

        # 미리보기 테이블
        preview_data = []
        for i, c in enumerate(chunks):
            preview_data.append({
                "번호": i + 1,
                "내용 미리보기": c.page_content[:120] + ("..." if len(c.page_content) > 120 else ""),
                "글자수": len(c.page_content),
            })
        df = pd.DataFrame(preview_data)

        info = f"✅ 청크 생성 완료: {len(chunks)}개 | 언어: {state['language']} | 원본: {len(doc.page_content)}자"
        return info, doc.page_content[:500] + "...", df

    except Exception as e:
        return f"❌ 오류: {e}", "", pd.DataFrame()


# ═══════════════════════════════════════════
# Tab 2: 컨텍스트 생성
# ═══════════════════════════════════════════
def generate_contexts_ui(progress=gr.Progress()):
    """청크에 대해 LLM 맥락 생성."""
    if not state["chunks"]:
        return "⚠️ 먼저 Tab 1에서 문서를 업로드하고 청킹하세요.", pd.DataFrame()

    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.", pd.DataFrame()

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    whole_doc = state["doc"].page_content
    chunks = state["chunks"]

    progress(0, desc="맥락 생성 중...")
    contexts = generate_contexts_batch(chunks, whole_doc, llm)
    contextual_chunks = create_contextual_chunks(chunks, contexts)
    state["contextual_chunks"] = contextual_chunks
    progress(1, desc="완료")

    # 비교 테이블
    rows = []
    for i, (orig, ctx) in enumerate(zip(chunks, contextual_chunks)):
        rows.append({
            "번호": i + 1,
            "원본 청크": orig.page_content[:100] + "...",
            "생성된 맥락": ctx.metadata.get("context", "")[:100],
        })
    df = pd.DataFrame(rows)

    return f"✅ {len(contextual_chunks)}개 청크에 맥락 생성 완료", df


# ═══════════════════════════════════════════
# Tab 3: 검색 비교
# ═══════════════════════════════════════════
def build_pipelines_ui(weight_preset, progress=gr.Progress()):
    """검색 파이프라인 구축."""
    if not state["chunks"]:
        return "⚠️ 먼저 Tab 1에서 문서를 업로드하세요."
    if not state["contextual_chunks"]:
        return "⚠️ 먼저 Tab 2에서 컨텍스트를 생성하세요."

    progress(0.1, desc="벡터 저장소 생성 중...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    lang = state["language"]
    chunks = state["chunks"]
    ctx_chunks = state["contextual_chunks"]

    # 일반 Embedding
    normal_vs = build_vectorstore(chunks, "normal", embeddings)
    normal_retriever = normal_vs.as_retriever(search_kwargs={"k": DEFAULT_TOP_K})

    # Contextual Embedding
    ctx_vs = build_vectorstore(ctx_chunks, "contextual", embeddings)
    ctx_retriever = ctx_vs.as_retriever(search_kwargs={"k": DEFAULT_TOP_K})

    progress(0.4, desc="BM25 인덱스 생성 중...")
    # BM25
    normal_bm25 = build_bm25_retriever(chunks, lang)
    ctx_bm25 = build_bm25_retriever(ctx_chunks, lang)

    # Hybrid
    weights = HYBRID_WEIGHT_PRESETS.get(weight_preset, [0.5, 0.5])
    ctx_hybrid = build_hybrid_retriever(ctx_retriever, ctx_bm25, weights)

    progress(0.7, desc="Reranker 로드 중...")
    # Reranker
    try:
        ctx_rerank = build_rerank_retriever(ctx_hybrid)
    except Exception:
        ctx_rerank = ctx_hybrid  # fallback

    state["retrievers"] = {
        "일반 Embedding": normal_retriever,
        "Contextual Embedding": ctx_retriever,
        "BM25": normal_bm25,
        "Contextual Hybrid": ctx_hybrid,
        "Contextual Hybrid + Reranker": ctx_rerank,
    }
    progress(1, desc="완료")
    return f"✅ 5개 파이프라인 구축 완료 (Hybrid 가중치: {weights})"


def search_compare_ui(query):
    """쿼리로 각 파이프라인 검색 비교."""
    if not query.strip():
        return pd.DataFrame(), "", "", "", "", ""
    if not state["retrievers"]:
        empty = "⚠️ 먼저 파이프라인을 구축하세요."
        return pd.DataFrame(), empty, empty, empty, empty, empty

    results = {}
    for name, ret in state["retrievers"].items():
        try:
            docs = search(ret, query)
            results[name] = docs
        except Exception as e:
            results[name] = []

    # 요약 테이블
    summary_rows = []
    for name, docs in results.items():
        summary_rows.append({
            "파이프라인": name,
            "결과 수": len(docs),
            "Top-1 미리보기": (
                docs[0].metadata.get("original_content", docs[0].page_content)[:80] + "..."
                if docs else "결과 없음"
            ),
        })
    df = pd.DataFrame(summary_rows)

    # 각 파이프라인 상세 결과
    detail_texts = []
    for name in ["일반 Embedding", "Contextual Embedding", "BM25", "Contextual Hybrid", "Contextual Hybrid + Reranker"]:
        docs = results.get(name, [])
        lines = [f"### {name}\n"]
        for i, doc in enumerate(docs[:3]):
            content = doc.metadata.get("original_content", doc.page_content)
            context = doc.metadata.get("context", "")
            lines.append(f"**[{i+1}]** {content[:150]}")
            if context:
                lines.append(f"  _맥락: {context[:80]}_\n")
        detail_texts.append("\n".join(lines) if docs else f"### {name}\n결과 없음")

    return df, *detail_texts


# ═══════════════════════════════════════════
# Tab 4: 성능 평가
# ═══════════════════════════════════════════
def run_evaluation_ui(testset_file, progress=gr.Progress()):
    """테스트셋으로 파이프라인 성능 평가."""
    if not state["retrievers"]:
        return "⚠️ 먼저 Tab 3에서 파이프라인을 구축하세요.", pd.DataFrame(), None

    # 테스트셋 로드
    try:
        if testset_file is not None:
            df_test = pd.read_excel(testset_file.name)
        else:
            # 기본 testset.xlsx 사용
            default_path = os.path.join(os.path.dirname(__file__), "data", "testset.xlsx")
            if os.path.exists(default_path):
                df_test = pd.read_excel(default_path)
            else:
                return "⚠️ 테스트셋 파일을 업로드하거나 data/testset.xlsx를 준비하세요.", pd.DataFrame(), None
    except Exception as e:
        return f"❌ 테스트셋 로드 오류: {e}", pd.DataFrame(), None

    # 쿼리와 정답 추출 (다양한 컬럼명 지원)
    q_col = next((c for c in ["question", "query", "user_input"] if c in df_test.columns), None)
    a_col = next((c for c in ["ground_truth", "answer", "reference"] if c in df_test.columns), None)
    if q_col is None or a_col is None:
        cols = ", ".join(df_test.columns.tolist())
        return f"⚠️ 테스트셋에 질문/정답 컬럼이 필요합니다. 현재 컬럼: {cols}", pd.DataFrame(), None
    questions = df_test[q_col].tolist()
    ground_truths = [[str(gt)] for gt in df_test[a_col].tolist()]

    # 평가 실행
    progress(0, desc="평가 중...")
    eval_rows = []
    for idx, (name, ret) in enumerate(state["retrievers"].items()):
        progress((idx + 1) / len(state["retrievers"]), desc=f"{name} 평가 중...")
        try:
            all_results = [ret.invoke(q) for q in questions]
            hr = compute_hit_rate(all_results, ground_truths, DEFAULT_TOP_K)
            mrr = compute_mrr(all_results, ground_truths)
            eval_rows.append({
                "파이프라인": name,
                "HitRate@3": round(hr, 4),
                "MRR": round(mrr, 4),
            })
        except Exception as e:
            eval_rows.append({
                "파이프라인": name,
                "HitRate@3": 0.0,
                "MRR": 0.0,
            })

    df_result = pd.DataFrame(eval_rows)

    # 차트 생성
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    colors = ["#a8d8ea", "#aa96da", "#fcbad3", "#ffffd2", "#b5eaea"]

    # HitRate 차트
    axes[0].barh(df_result["파이프라인"], df_result["HitRate@3"], color=colors[:len(df_result)])
    axes[0].set_title("HitRate@3", fontsize=14, fontweight="bold")
    axes[0].set_xlim(0, 1)
    axes[0].spines[["top", "right"]].set_visible(False)

    # MRR 차트
    axes[1].barh(df_result["파이프라인"], df_result["MRR"], color=colors[:len(df_result)])
    axes[1].set_title("MRR", fontsize=14, fontweight="bold")
    axes[1].set_xlim(0, 1)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    progress(1, desc="완료")

    return f"✅ {len(questions)}개 쿼리로 평가 완료", df_result, fig


# ═══════════════════════════════════════════
# Gradio UI 구성
# ═══════════════════════════════════════════
def create_app():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
    ).set(
        body_background_fill="#f8f9fa",
        block_background_fill="#ffffff",
        block_shadow="0 2px 12px rgba(0,0,0,0.06)",
        block_border_width="0px",
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="Contextual Retrieval Lab") as app:
        gr.Markdown("# 🔍 Contextual Retrieval Lab\nRAG 검색 성능 비교 대시보드")

        # API 키 확인
        if not OPENAI_API_KEY:
            gr.Markdown("⚠️ **OPENAI_API_KEY**가 설정되지 않았습니다. `.env` 파일에 키를 추가하세요.")

        with gr.Tabs():
            # ── Tab 1: 문서 업로드 / 청킹 ──
            with gr.Tab("📄 문서 업로드 / 청킹"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(label="문서 파일 업로드 (.md, .txt)", file_types=[".md", ".txt"])
                        chunk_size = gr.Slider(100, 1000, value=DEFAULT_CHUNK_SIZE, step=50, label="Chunk Size")
                        chunk_overlap = gr.Slider(0, 200, value=DEFAULT_CHUNK_OVERLAP, step=10, label="Chunk Overlap")
                        lang_radio = gr.Radio(["자동 감지", "한국어", "English"], value="자동 감지", label="문서 언어")
                        upload_btn = gr.Button("📥 업로드 & 청킹", variant="primary")
                    with gr.Column(scale=2):
                        upload_status = gr.Textbox(label="상태", interactive=False)
                        doc_preview = gr.Textbox(label="문서 미리보기", lines=5, interactive=False)
                        chunk_table = gr.Dataframe(label="청크 목록", interactive=False)

                upload_btn.click(
                    process_upload,
                    inputs=[file_input, chunk_size, chunk_overlap, lang_radio],
                    outputs=[upload_status, doc_preview, chunk_table],
                )

            # ── Tab 2: 컨텍스트 생성 ──
            with gr.Tab("🧠 컨텍스트 생성"):
                gr.Markdown("LLM을 사용하여 각 청크에 맥락 설명을 추가합니다.")
                ctx_btn = gr.Button("🚀 컨텍스트 생성 시작", variant="primary")
                ctx_status = gr.Textbox(label="상태", interactive=False)
                ctx_table = gr.Dataframe(label="원본 vs 맥락 비교", interactive=False)

                ctx_btn.click(
                    generate_contexts_ui,
                    outputs=[ctx_status, ctx_table],
                )

            # ── Tab 3: 검색 비교 ──
            with gr.Tab("🔎 검색 비교"):
                with gr.Row():
                    weight_dropdown = gr.Dropdown(
                        choices=list(HYBRID_WEIGHT_PRESETS.keys()),
                        value="균등 (0.5:0.5)",
                        label="Hybrid 가중치",
                    )
                    build_btn = gr.Button("🔧 파이프라인 구축", variant="primary")
                build_status = gr.Textbox(label="상태", interactive=False)

                build_btn.click(build_pipelines_ui, inputs=[weight_dropdown], outputs=[build_status])

                gr.Markdown("---")
                query_input = gr.Textbox(label="검색 쿼리 입력", placeholder="예: 테슬라의 2023년 매출은?")
                search_btn = gr.Button("🔍 검색", variant="primary")

                search_summary = gr.Dataframe(label="검색 결과 요약", interactive=False)
                with gr.Row():
                    detail_1 = gr.Markdown(label="일반 Embedding")
                    detail_2 = gr.Markdown(label="Contextual Embedding")
                with gr.Row():
                    detail_3 = gr.Markdown(label="BM25")
                    detail_4 = gr.Markdown(label="Contextual Hybrid")
                detail_5 = gr.Markdown(label="Contextual Hybrid + Reranker")

                search_btn.click(
                    search_compare_ui,
                    inputs=[query_input],
                    outputs=[search_summary, detail_1, detail_2, detail_3, detail_4, detail_5],
                )

            # ── Tab 4: 성능 평가 ──
            with gr.Tab("📊 성능 평가"):
                gr.Markdown("테스트 쿼리 세트로 각 파이프라인의 HitRate@k, MRR을 측정합니다.")
                testset_file = gr.File(label="테스트셋 업로드 (.xlsx) — 비워두면 기본 testset.xlsx 사용", file_types=[".xlsx"])
                eval_btn = gr.Button("📈 평가 실행", variant="primary")
                eval_status = gr.Textbox(label="상태", interactive=False)
                eval_table = gr.Dataframe(label="평가 결과", interactive=False)
                eval_chart = gr.Plot(label="성능 비교 차트")

                eval_btn.click(
                    run_evaluation_ui,
                    inputs=[testset_file],
                    outputs=[eval_status, eval_table, eval_chart],
                )

    return app


# ── 실행 ──
if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
