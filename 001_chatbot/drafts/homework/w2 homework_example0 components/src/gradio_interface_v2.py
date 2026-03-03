"""
GradioInterface 컴포넌트 (2탭 구조)
프롬프트 탭 + 성능 비교 탭
"""
import gradio as gr
from typing import Iterator, Tuple, List, Optional, Dict
from datetime import datetime
import os

from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store_manager import VectorStoreManager
from src.rag_chain import RAGChain
from src.template_manager import TemplateManager, RAGConfig
from src.comparison_engine import ComparisonEngine


class GradioInterface:
    """Gradio UI 관리 클래스 (2탭 구조)"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store_manager = VectorStoreManager()
        self.rag_chain = RAGChain()
        self.template_manager = TemplateManager()
        self.comparison_engine = ComparisonEngine()
        self.chat_history = []
        self.current_file_path = None
    
    def create_interface(self) -> gr.Blocks:
        """Gradio 인터페이스 생성 (2개 탭 구조)"""
        
        with gr.Blocks(title="RAG 챗봇") as demo:
            gr.Markdown("# 📚 RAG 기반 문서 질의응답 챗봇")
            
            with gr.Tabs() as tabs:
                # ==================== 탭 1: 프롬프트 ====================
                with gr.Tab("💬 프롬프트"):
                    prompt_components = self._create_prompt_tab()
                
                # ==================== 탭 2: 성능 비교 ====================
                with gr.Tab("📊 성능 비교"):
                    comparison_components = self._create_comparison_tab()
        
        return demo

    def _create_prompt_tab(self):
        """프롬프트 탭 생성"""
        with gr.Row():
            # 왼쪽: 설정
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 설정")
                
                # 파일 업로드
                file_upload = gr.File(label="📄 PDF 파일 업로드", file_types=[".pdf"])
                
                # 텍스트 분할
                with gr.Accordion("✂️ 텍스트 분할", open=True):
                    splitter_type = gr.Dropdown(
                        choices=["RecursiveCharacterTextSplitter", "CharacterTextSplitter"],
                        value="RecursiveCharacterTextSplitter",
                        label="분할 방식"
                    )
                    chunk_size = gr.Slider(100, 2000, value=1000, step=100, label="청크 크기")
                    chunk_overlap = gr.Slider(0, 500, value=200, step=50, label="청크 중복")
                    separators = gr.Textbox(value="\\n\\n,\\n, ,", label="구분자")
                
                # 임베딩 모델
                with gr.Accordion("🔢 임베딩 모델", open=True):
                    embedding_type = gr.Dropdown(
                        choices=["openai", "huggingface", "ollama"],
                        value="openai",
                        label="임베딩 타입"
                    )
                    embedding_model = gr.Dropdown(
                        choices=["text-embedding-3-small", "text-embedding-3-large"],
                        value="text-embedding-3-small",
                        label="모델"
                    )
                    embedding_dimensions = gr.Slider(
                        256, 3072, value=1536, step=256,
                        label="차원 (OpenAI만)", visible=True
                    )
                
                # 벡터 저장소
                with gr.Accordion("💾 벡터 저장소", open=True):
                    vector_store_type = gr.Dropdown(
                        choices=["chroma", "faiss"],
                        value="chroma",
                        label="저장소 타입"
                    )
                
                # 검색 설정
                with gr.Accordion("🔍 검색 설정", open=True):
                    search_type = gr.Dropdown(
                        choices=["similarity", "mmr", "similarity_score_threshold"],
                        value="similarity",
                        label="검색 방식"
                    )
                    k = gr.Slider(1, 10, value=4, step=1, label="검색 문서 수 (k)")
                    score_threshold = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.1,
                        label="유사도 임계값", visible=False
                    )
                    lambda_mult = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.1,
                        label="Lambda", visible=False
                    )
                
                # LLM 설정
                with gr.Accordion("🤖 LLM 설정", open=True):
                    llm_model = gr.Dropdown(
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                        value="gpt-4o-mini",
                        label="모델"
                    )
                    temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
                    max_tokens = gr.Slider(100, 4000, value=1000, step=100, label="Max Tokens")
            
            # 오른쪽: 채팅
            with gr.Column(scale=2):
                gr.Markdown("### 💬 대화")
                chatbot = gr.Chatbot(label="대화", height=500)
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="질문",
                        placeholder="문서에 대해 질문하세요...",
                        scale=4
                    )
                    submit_btn = gr.Button("전송", variant="primary", scale=1)
                
                clear_btn = gr.Button("🗑️ 대화 초기화")
                
                with gr.Accordion("📑 소스 문서", open=False):
                    source_docs = gr.Textbox(label="검색된 문서", lines=10, interactive=False)
        
        # 이벤트 핸들러
        def update_embedding_models(emb_type):
            if emb_type == "openai":
                return gr.Dropdown(
                    choices=["text-embedding-3-small", "text-embedding-3-large"],
                    value="text-embedding-3-small"
                ), gr.Slider(visible=True)
            elif emb_type == "huggingface":
                return gr.Dropdown(
                    choices=["BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"],
                    value="BAAI/bge-m3"
                ), gr.Slider(visible=False)
            else:
                return gr.Dropdown(
                    choices=["bge-m3", "nomic-embed-text"],
                    value="bge-m3"
                ), gr.Slider(visible=False)
        
        def update_search_params(search_type_val):
            if search_type_val == "similarity_score_threshold":
                return gr.Slider(visible=True), gr.Slider(visible=False)
            elif search_type_val == "mmr":
                return gr.Slider(visible=False), gr.Slider(visible=True)
            else:
                return gr.Slider(visible=False), gr.Slider(visible=False)
        
        embedding_type.change(
            fn=update_embedding_models,
            inputs=[embedding_type],
            outputs=[embedding_model, embedding_dimensions]
        )
        
        search_type.change(
            fn=update_search_params,
            inputs=[search_type],
            outputs=[score_threshold, lambda_mult]
        )
        
        file_upload.change(
            fn=self.handle_file_upload_simple,
            inputs=[file_upload],
            outputs=[]
        )
        
        submit_btn.click(
            fn=self.handle_query_auto,
            inputs=[
                query_input, file_upload,
                splitter_type, chunk_size, chunk_overlap, separators,
                embedding_type, embedding_model, embedding_dimensions,
                vector_store_type, search_type, k, score_threshold, lambda_mult,
                llm_model, temperature, max_tokens
            ],
            outputs=[chatbot, source_docs, query_input]
        )
        
        query_input.submit(
            fn=self.handle_query_auto,
            inputs=[
                query_input, file_upload,
                splitter_type, chunk_size, chunk_overlap, separators,
                embedding_type, embedding_model, embedding_dimensions,
                vector_store_type, search_type, k, score_threshold, lambda_mult,
                llm_model, temperature, max_tokens
            ],
            outputs=[chatbot, source_docs, query_input]
        )
        
        clear_btn.click(
            fn=self.handle_clear_history,
            outputs=[chatbot]
        )
        
        return {}  # 프롬프트 탭은 초기 로드가 필요 없음

    def _create_comparison_tab(self):
        """성능 비교 탭 생성"""
        gr.Markdown("### 여러 템플릿으로 성능 비교")
        
        with gr.Row():
            # 왼쪽: 템플릿 관리
            with gr.Column(scale=1):
                gr.Markdown("#### 💾 템플릿 관리")
                
                # 템플릿 저장
                with gr.Group():
                    gr.Markdown("**새 템플릿 만들기**")
                    template_name = gr.Textbox(label="템플릿 이름", placeholder="예: 빠른검색")
                    template_desc = gr.Textbox(label="설명", placeholder="템플릿 설명")
                    
                    # 설정
                    with gr.Accordion("⚙️ 템플릿 설정", open=True):
                        t_splitter = gr.Dropdown(
                            ["RecursiveCharacterTextSplitter", "CharacterTextSplitter"],
                            value="RecursiveCharacterTextSplitter", label="분할 방식"
                        )
                        t_chunk_size = gr.Slider(100, 2000, 1000, step=100, label="청크 크기")
                        t_chunk_overlap = gr.Slider(0, 500, 200, step=50, label="청크 중복")
                        t_separators = gr.Textbox(value="\\n\\n,\\n, ,", label="구분자")
                        
                        t_emb_type = gr.Dropdown(
                            ["openai", "huggingface", "ollama"],
                            value="openai", label="임베딩 타입"
                        )
                        t_emb_model = gr.Dropdown(
                            ["text-embedding-3-small", "text-embedding-3-large"],
                            value="text-embedding-3-small", label="임베딩 모델"
                        )
                        t_emb_dim = gr.Slider(256, 3072, 1536, step=256, label="차원")
                        
                        t_vector_store = gr.Dropdown(["chroma", "faiss"], value="chroma", label="저장소")
                        
                        t_search_type = gr.Dropdown(
                            ["similarity", "mmr", "similarity_score_threshold"],
                            value="similarity", label="검색 방식"
                        )
                        t_k = gr.Slider(1, 10, 4, step=1, label="k")
                        t_score_threshold = gr.Slider(0.0, 1.0, 0.5, step=0.1, label="임계값")
                        t_lambda_mult = gr.Slider(0.0, 1.0, 0.5, step=0.1, label="Lambda")
                        
                        t_llm_model = gr.Dropdown(
                            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                            value="gpt-4o-mini", label="LLM"
                        )
                        t_temperature = gr.Slider(0.0, 2.0, 0.7, step=0.1, label="Temperature")
                        t_max_tokens = gr.Slider(100, 4000, 1000, step=100, label="Max Tokens")
                    
                    save_template_btn = gr.Button("💾 템플릿 저장", variant="primary")
                
                # 템플릿 관리
                with gr.Group():
                    gr.Markdown("**저장된 템플릿**")
                    # 초기 템플릿 목록 로드
                    initial_templates = self.template_manager.list_templates()
                    template_list = gr.Dropdown(
                        label="템플릿 선택", 
                        choices=initial_templates, 
                        interactive=True
                    )
                    with gr.Row():
                        load_template_btn = gr.Button("📥 불러오기", scale=1)
                        delete_template_btn = gr.Button("🗑️ 삭제", variant="stop", scale=1)
                
                template_status = gr.Textbox(label="상태", interactive=False, lines=2)
            
            # 오른쪽: 비교 실행
            with gr.Column(scale=2):
                gr.Markdown("#### 📊 템플릿 비교")
                
                comparison_file = gr.File(label="📄 비교할 문서 업로드", file_types=[".pdf"])
                
                # 초기 템플릿 목록 로드
                initial_templates = self.template_manager.list_templates()
                comparison_templates = gr.CheckboxGroup(
                    label="비교할 템플릿 선택 (최대 4개)",
                    choices=initial_templates
                )
                
                comparison_query = gr.Textbox(
                    label="비교할 질문",
                    placeholder="여러 템플릿으로 비교할 질문을 입력하세요..."
                )
                
                run_comparison_btn = gr.Button("🔍 비교 실행", variant="primary", size="lg")
                
                comparison_status = gr.Textbox(label="비교 상태", interactive=False, lines=2)
                
                comparison_results = gr.Markdown(value="비교 결과가 여기에 표시됩니다.")
                
                with gr.Row():
                    export_csv_btn = gr.Button("📥 CSV 다운로드")
                    csv_status = gr.Textbox(label="다운로드 상태", interactive=False, scale=2)
        
        # 이벤트 핸들러
        
        # 임베딩 타입 변경 시 모델 목록 업데이트
        def update_embedding_options(emb_type):
            if emb_type == "openai":
                return (
                    gr.update(choices=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"], value="text-embedding-3-small"),
                    gr.update(value=1536)
                )
            elif emb_type == "huggingface":
                return (
                    gr.update(choices=["BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"], value="BAAI/bge-m3"),
                    gr.update(value=1024)
                )
            elif emb_type == "ollama":
                return (
                    gr.update(choices=["bge-m3", "nomic-embed-text", "mxbai-embed-large"], value="bge-m3"),
                    gr.update(value=1024)
                )
            return gr.update(), gr.update()
        
        t_emb_type.change(
            fn=update_embedding_options,
            inputs=[t_emb_type],
            outputs=[t_emb_model, t_emb_dim]
        )
        
        save_template_btn.click(
            fn=self.handle_save_template_v2,
            inputs=[
                template_name, template_desc,
                t_splitter, t_chunk_size, t_chunk_overlap, t_separators,
                t_emb_type, t_emb_model, t_emb_dim,
                t_vector_store, t_search_type, t_k, t_score_threshold, t_lambda_mult,
                t_llm_model, t_temperature, t_max_tokens
            ],
            outputs=[template_status, template_list, comparison_templates]
        )
        
        load_template_btn.click(
            fn=self.handle_load_template_v2,
            inputs=[template_list],
            outputs=[
                template_status,
                t_splitter, t_chunk_size, t_chunk_overlap, t_separators,
                t_emb_type, t_emb_model, t_emb_dim,
                t_vector_store, t_search_type, t_k, t_score_threshold, t_lambda_mult,
                t_llm_model, t_temperature, t_max_tokens
            ]
        )
        
        delete_template_btn.click(
            fn=self.handle_delete_template,
            inputs=[template_list],
            outputs=[template_status, template_list, comparison_templates]
        )
        
        run_comparison_btn.click(
            fn=self.handle_comparison_v2,
            inputs=[comparison_query, comparison_templates, comparison_file],
            outputs=[comparison_status, comparison_results]
        )
        
        export_csv_btn.click(
            fn=self.handle_export_csv,
            outputs=[csv_status]
        )
        
        # 컴포넌트 반환 (초기 로드용)
        return {
            'template_list': template_list,
            'comparison_templates': comparison_templates
        }

    def handle_file_upload_simple(self, file):
        """파일 업로드 (간단)"""
        if file:
            self.current_file_path = file.name
        return None
    
    def handle_query_auto(
        self, query, file,
        splitter_type, chunk_size, chunk_overlap, separators,
        embedding_type, embedding_model, embedding_dimensions,
        vector_store_type, search_type, k, score_threshold, lambda_mult,
        llm_model, temperature, max_tokens
    ) -> Tuple[List, str, str]:
        """질문 시 자동으로 문서 처리 후 응답"""
        try:
            if not query.strip():
                return self.chat_history, "", ""
            
            # 파일 확인
            if not file:
                error_msg = "먼저 PDF 파일을 업로드해주세요."
                self.chat_history.append({"role": "user", "content": query})
                self.chat_history.append({"role": "assistant", "content": error_msg})
                return self.chat_history, "", ""
            
            # 파일 경로 저장
            self.current_file_path = file.name
            
            # 1. 문서 로드 및 분할
            documents = self.doc_processor.load_document(file.name)
            sep_list = [s.strip().replace('\\n', '\n') for s in separators.split(',')]
            split_docs = self.doc_processor.split_documents(
                documents,
                splitter_type=splitter_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=sep_list
            )
            
            # 2. 임베딩 모델 초기화
            if embedding_type == "openai" and "text-embedding-3" in embedding_model:
                self.embedding_manager.initialize_embedding_model(
                    model_type=embedding_type,
                    model_name=embedding_model,
                    dimensions=int(embedding_dimensions)
                )
            else:
                self.embedding_manager.initialize_embedding_model(
                    model_type=embedding_type,
                    model_name=embedding_model
                )
            
            # 3. 벡터 저장소 초기화 및 문서 추가
            embedding_dim = self.embedding_manager.get_embedding_dimension()
            self.vector_store_manager.initialize_vector_store(
                store_type=vector_store_type,
                embedding_function=self.embedding_manager.embedding_model,
                embedding_dim=embedding_dim
            )
            self.vector_store_manager.add_documents(split_docs)
            
            # 4. 검색 파라미터 설정
            search_kwargs = {"k": k}
            if search_type == "similarity_score_threshold":
                search_kwargs["score_threshold"] = score_threshold
            elif search_type == "mmr":
                search_kwargs["lambda_mult"] = lambda_mult
            
            # 5. 검색기 생성
            retriever = self.vector_store_manager.get_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            
            # 6. LLM 초기화
            self.rag_chain.initialize_llm(
                model_name=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=True
            )
            
            # 7. RAG 체인 구성
            self.rag_chain.build_chain(retriever, self.rag_chain.llm)
            
            # 8. 스트리밍 응답 생성
            response = ""
            for chunk in self.rag_chain.stream_response(query):
                response += chunk
            
            # 9. 소스 문서 가져오기
            source_docs = self.rag_chain.get_last_retrieved_docs()
            source_text = self._format_source_docs(source_docs)
            
            # 10. 대화 이력 업데이트
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": response})
            
            return self.chat_history, source_text, ""
        
        except Exception as e:
            error_msg = f"❌ 오류 발생: {str(e)}"
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": error_msg})
            return self.chat_history, "", ""
    
    def handle_clear_history(self) -> List:
        """대화 이력 초기화"""
        self.chat_history = []
        return []

    def handle_save_template_v2(
        self, name, desc,
        splitter_type, chunk_size, chunk_overlap, separators,
        embedding_type, embedding_model, embedding_dimensions,
        vector_store_type, search_type, k, score_threshold, lambda_mult,
        llm_model, temperature, max_tokens
    ) -> Tuple[str, gr.Dropdown, gr.CheckboxGroup]:
        """템플릿 저장"""
        try:
            if not name:
                return "템플릿 이름을 입력해주세요.", gr.update(), gr.update()
            
            config = RAGConfig(
                splitter_type=splitter_type,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                separators=separators,
                embedding_type=embedding_type,
                embedding_model=embedding_model,
                embedding_dimensions=int(embedding_dimensions),
                vector_store_type=vector_store_type,
                search_type=search_type,
                k=int(k),
                score_threshold=float(score_threshold),
                lambda_mult=float(lambda_mult),
                llm_model=llm_model,
                temperature=float(temperature),
                max_tokens=int(max_tokens)
            )
            
            success = self.template_manager.save_template(name, config, desc)
            
            if success:
                templates = self.template_manager.list_templates()
                return (
                    f"✅ 템플릿 '{name}'이(가) 저장되었습니다.",
                    gr.update(choices=templates, value=name),
                    gr.update(choices=templates)
                )
            else:
                return "❌ 템플릿 저장 실패", gr.update(), gr.update()
        
        except Exception as e:
            return f"❌ 오류: {str(e)}", gr.update(), gr.update()
    
    def handle_load_template_v2(self, template_name) -> Tuple:
        """템플릿 불러오기"""
        try:
            if not template_name:
                return ("템플릿을 선택해주세요.",) + (gr.update(),) * 15
            
            config = self.template_manager.load_template(template_name)
            
            if config:
                return (
                    f"✅ 템플릿 '{template_name}'을(를) 불러왔습니다.",
                    config.splitter_type,
                    config.chunk_size,
                    config.chunk_overlap,
                    config.separators,
                    config.embedding_type,
                    config.embedding_model,
                    config.embedding_dimensions,
                    config.vector_store_type,
                    config.search_type,
                    config.k,
                    config.score_threshold,
                    config.lambda_mult,
                    config.llm_model,
                    config.temperature,
                    config.max_tokens
                )
            else:
                return ("❌ 템플릿을 찾을 수 없습니다.",) + (gr.update(),) * 15
        
        except Exception as e:
            return (f"❌ 오류: {str(e)}",) + (gr.update(),) * 15
    
    def handle_delete_template(self, template_name) -> Tuple[str, gr.Dropdown, gr.CheckboxGroup]:
        """템플릿 삭제"""
        try:
            if not template_name:
                return "템플릿을 선택해주세요.", gr.update(), gr.update()
            
            success = self.template_manager.delete_template(template_name)
            
            if success:
                templates = self.template_manager.list_templates()
                return (
                    f"✅ 템플릿 '{template_name}'이(가) 삭제되었습니다.",
                    gr.update(choices=templates),
                    gr.update(choices=templates)
                )
            else:
                return "❌ 템플릿 삭제 실패", gr.update(), gr.update()
        
        except Exception as e:
            return f"❌ 오류: {str(e)}", gr.update(), gr.update()

    def handle_comparison_v2(self, query, selected_templates, file) -> Tuple[str, str]:
        """템플릿 비교 실행"""
        try:
            if not query.strip():
                return "질문을 입력해주세요.", ""
            
            if not selected_templates:
                return "비교할 템플릿을 선택해주세요.", ""
            
            if len(selected_templates) > 4:
                return "최대 4개의 템플릿만 선택할 수 있습니다.", ""
            
            if not file:
                return "비교할 문서를 업로드해주세요.", ""
            
            # 선택된 템플릿 로드
            templates = []
            for name in selected_templates:
                config = self.template_manager.load_template(name)
                if config:
                    templates.append((name, config))
            
            if not templates:
                return "선택된 템플릿을 불러올 수 없습니다.", ""
            
            # 비교 실행
            status_msg = f"🔍 {len(templates)}개 템플릿으로 비교 실행 중..."
            
            results = self.comparison_engine.run_comparison(
                query=query,
                templates=templates,
                current_file_path=file.name,
                doc_processor=self.doc_processor,
                embedding_manager=self.embedding_manager,
                vector_store_manager=self.vector_store_manager,
                rag_chain=self.rag_chain
            )
            
            # 결과 리포트 생성
            report = self.comparison_engine.generate_comparison_report(results)
            
            return f"✅ 비교 완료! {len(results)}개 템플릿 결과", report
        
        except Exception as e:
            return f"❌ 오류: {str(e)}", ""
    
    def handle_export_csv(self) -> str:
        """CSV 내보내기"""
        try:
            if not self.comparison_engine.results:
                return "내보낼 비교 결과가 없습니다."
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./comparison_results_{timestamp}.csv"
            
            success = self.comparison_engine.export_to_csv(
                self.comparison_engine.results,
                output_path
            )
            
            if success:
                return f"✅ CSV 파일 저장: {output_path}"
            else:
                return "❌ CSV 내보내기 실패"
        
        except Exception as e:
            return f"❌ 오류: {str(e)}"
    
    def _format_source_docs(self, docs: List) -> str:
        """소스 문서 포맷팅"""
        if not docs:
            return "검색된 문서가 없습니다."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            metadata = doc.metadata
            formatted.append(
                f"📄 문서 {i}\n"
                f"내용: {content}\n"
                f"메타데이터: {metadata}\n"
                f"{'-'*50}"
            )
        
        return "\n\n".join(formatted)

    def get_template_list(self) -> Tuple:
        """템플릿 목록 가져오기"""
        templates = self.template_manager.list_templates()
        return (
            gr.update(choices=templates),
            gr.update(choices=templates)
        )
