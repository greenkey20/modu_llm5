"""
학교생활기록부 기재요령 RAG 챗봇
- PDFPlumber 사용 (표 최적화)
- LangSmith 미사용 (간소화)
- requirements.txt 불필요
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import gradio as gr

# API 키 로드
load_dotenv()

# ==================== 설정 ====================
PDF_FOLDER = "./pdfs"
CHROMA_DB_PATH = "./chroma_db"

SCHOOL_PDFS = {
    "초등학교": "2026_학교생활기록부_기재요령_초등학교.pdf",
    "중학교": "2026_학교생활기록부_기재요령_중학교.pdf",
    "고등학교": "2026_학교생활기록부_기재요령_고등학교.pdf"
}

# ==================== 프롬프트 ====================
SINGLE_SCHOOL_PROMPT = """당신은 학교생활기록부 기재요령 전문가입니다.
제공된 문서 내용만을 바탕으로 정확하게 답변하세요.

규칙:
1. 문서에 명확히 나와있는 내용만 답변
2. 표나 목록이 있으면 그대로 구조화하여 제시
3. 문서에 없는 내용은 "해당 내용을 찾을 수 없습니다" 응답

문서 내용:
{context}

질문: {question}

답변:"""

COMPARISON_PROMPT = """당신은 학교생활기록부 기재요령 전문가입니다.
초등학교, 중학교, 고등학교의 기재요령을 비교하여 답변하세요.

규칙:
1. 각 학교급별 차이점을 명확히 구분
2. 공통점과 차이점 모두 언급
3. 가능하면 표 형식으로 정리

문서 내용:
{context}

질문: {question}

비교 답변:"""

# ==================== RAG 시스템 ====================
class SchoolRAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            max_tokens=2000
        )
        self.vector_stores = {}
        
    # 섹션 헤더 패턴: "07 출결상황", "08 수상경력" 등
    SECTION_PATTERN = re.compile(r'^(\d{2})\s+(.+?)$', re.MULTILINE)

    def _is_noise_page(self, text):
        """목차, 표지, 빈 페이지 등 노이즈 판별"""
        stripped = text.strip()
        if len(stripped) < 30:
            return True
        dot_lines = sum(1 for line in stripped.split('\n') if '···' in line or '…' in line)
        if dot_lines >= 3:
            return True
        return False

    def _tag_sections(self, documents):
        """각 페이지에 섹션 제목을 메타데이터로 태깅"""
        current_section = ""
        for doc in documents:
            first_line = doc.page_content.strip().split('\n')[0]
            match = self.SECTION_PATTERN.match(first_line)
            if match:
                current_section = match.group(2).strip()
            doc.metadata["section"] = current_section
            # 섹션 제목을 본문 앞에 추가하여 임베딩 품질 향상
            if current_section:
                doc.page_content = f"[{current_section}] {doc.page_content}"

    def initialize_vector_store(self, school_type, pdf_path):
        """벡터 스토어 초기화"""
        db_path = f"{CHROMA_DB_PATH}_{school_type}"

        if os.path.exists(db_path):
            print(f"✅ {school_type} 기존 벡터 DB 로드")
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=self.embeddings
            )
        else:
            print(f"📁 {school_type} PDF 로딩 중... (PDFPlumber)")
            loader = PDFPlumberLoader(pdf_path)
            documents = loader.load()
            print(f"   - {len(documents)}페이지 로드됨")

            # 노이즈 페이지 필터링 (목차, 표지, 빈 페이지)
            filtered = [doc for doc in documents if not self._is_noise_page(doc.page_content)]
            print(f"   - 노이즈 제거 후 {len(filtered)}페이지")

            # 섹션 태깅 (검색 품질 향상)
            self._tag_sections(filtered)
            sections = set(doc.metadata["section"] for doc in filtered if doc.metadata["section"])
            print(f"   - 감지된 섹션: {len(sections)}개")

            print(f"✂️  {school_type} 텍스트 청킹 중...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(filtered)
            print(f"   - {len(chunks)}개 청크 생성됨")

            print(f"🎯 {school_type} 임베딩 생성 중...")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=db_path
            )
            print(f"💾 {school_type} 벡터 DB 저장 완료")

        return vector_store
    
    def initialize_all(self):
        """모든 학교급 초기화"""
        print("\n" + "="*70)
        print("🎓 학교생활기록부 RAG 시스템 초기화")
        print("📊 PDFPlumber 사용 (표 최적화)")
        print("="*70 + "\n")
        
        for school_type, filename in SCHOOL_PDFS.items():
            pdf_path = os.path.join(PDF_FOLDER, filename)
            
            if not os.path.exists(pdf_path):
                print(f"⚠️  {school_type} PDF 없음: {pdf_path}")
                print(f"   건너뜁니다...\n")
                continue
            
            try:
                vector_store = self.initialize_vector_store(school_type, pdf_path)
                self.vector_stores[school_type] = vector_store
                print(f"✅ {school_type} 준비 완료\n")
                
            except Exception as e:
                print(f"❌ {school_type} 초기화 실패: {e}\n")
        
        if not self.vector_stores:
            raise Exception("PDF 파일을 확인하세요.")
        
        print("="*70)
        print(f"✅ RAG 챗봇 준비 완료!")
        print(f"📚 로드된 학교급: {', '.join(self.vector_stores.keys())}")
        print("="*70 + "\n")
    
    def _find_matching_section(self, school_type, question):
        """질문과 매칭되는 섹션을 찾아 필터로 반환"""
        vector_store = self.vector_stores[school_type]
        all_meta = vector_store._collection.get(include=["metadatas"])["metadatas"]
        sections = list(set(m.get("section", "") for m in all_meta if m.get("section")))
        # 공백 제거 후 비교 (출결 상황 → 출결상황)
        q_normalized = question.replace(" ", "")
        for section in sections:
            if section.replace(" ", "") in q_normalized:
                return {"section": section}
        return None

    def query_single_school(self, school_type, question):
        """단일 학교급 질의"""
        if school_type not in self.vector_stores:
            return f"❌ {school_type} 데이터가 없습니다."

        vector_store = self.vector_stores[school_type]

        # 섹션 필터 매칭 시도 → 실패하면 일반 검색
        section_filter = self._find_matching_section(school_type, question)
        if section_filter:
            docs = vector_store.similarity_search(question, k=5, filter=section_filter)
        else:
            docs = vector_store.similarity_search(question, k=5)

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = PromptTemplate(
            template=SINGLE_SCHOOL_PROMPT,
            input_variables=["context", "question"]
        )
        formatted = prompt.format(context=context, question=question)
        response = self.llm.invoke(formatted)

        answer = f"### 📚 {school_type} 기재요령\n\n"
        answer += response.content + "\n\n"

        if docs:
            answer += "---\n### 📄 출처\n"
            for i, doc in enumerate(docs[:3], 1):
                page = doc.metadata.get("page", "?")
                answer += f"{i}. 페이지 {page + 1}\n"

        return answer
    
    def query_all_schools(self, question):
        """전체 학교급 비교"""
        all_docs = []
        
        for school_type, vector_store in self.vector_stores.items():
            docs = vector_store.similarity_search(question, k=4)
            for doc in docs:
                doc.metadata["school_type"] = school_type
            all_docs.extend(docs)
        
        context = "\n\n".join([
            f"[{doc.metadata['school_type']}]\n{doc.page_content}"
            for doc in all_docs
        ])
        
        prompt = PromptTemplate(
            template=COMPARISON_PROMPT,
            input_variables=["context", "question"]
        )
        
        formatted_prompt = prompt.format(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)
        
        answer = "### 📊 초·중·고 비교\n\n"
        answer += response.content + "\n\n"
        answer += "---\n### 📚 참조 학교급\n"
        answer += "- " + "\n- ".join(self.vector_stores.keys())
        
        return answer

# ==================== Gradio UI ====================
rag_system = None

def initialize_system():
    global rag_system
    rag_system = SchoolRAGSystem()
    rag_system.initialize_all()

def chatbot_response(message, history, school_type):
    if rag_system is None:
        return "❌ 시스템이 초기화되지 않았습니다."
    
    if not message.strip():
        return "질문을 입력해주세요."
    
    try:
        if school_type == "전체 비교":
            return rag_system.query_all_schools(message)
        else:
            return rag_system.query_single_school(school_type, message)
    except Exception as e:
        return f"❌ 오류: {str(e)}"

CSS = """
.header {
    text-align: center;
    padding: 2rem;
    background: #a1c1a2;
    color: white;
    border-radius: 1rem;
    margin-bottom: 2rem;
}
"""

def create_gradio_interface():

  theme = gr.themes.Default().set(
    button_primary_background_fill="#a1c1a2",
    button_primary_background_fill_hover="#819e82",
    button_primary_text_color="white",
    checkbox_background_color_selected="#819e82",
    checkbox_border_color_selected="#819e82",
    checkbox_border_color_focus="#819e82",
  )

  with gr.Blocks(theme=theme) as demo:
        gr.HTML("""
            <div class="header">
                <h1>📚 학교생활기록부 기재요령 챗봇</h1>
                <p>초·중·고 생활기록부 기재요령 AI 가이드</p>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                school_selector = gr.Radio(
                    choices=["초등학교", "중학교", "고등학교", "전체 비교"],
                    value="고등학교",
                    label="🏫 학교급 선택",
                )
                
                chatbot = gr.Chatbot(height=500, label="💬 대화")
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="질문을 입력하세요...",
                        label="질문",
                        scale=4
                    )
                    send_btn = gr.Button("📤 전송", scale=1, variant="primary")
                
                clear_btn = gr.Button("🗑️ 초기화")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📝 예시 질문")
                gr.Examples(
                    examples=[
                        ["출결 상황은 어떻게 기재하나요?"],
                        ["봉사활동 특기사항 작성 방법은?"],
                        ["수상 경력 기재 시 주의사항은?"],
                    ],
                    inputs=msg
                )
                
                gr.Markdown("### 🔍 비교 질문")
                gr.Examples(
                    examples=[
                        ["초중고 출결 기재 차이점은?"],
                        ["학교급별 수상 경력 기재 차이는?"],
                    ],
                    inputs=msg
                )
        
        def respond(message, chat_history, school_type):
            bot_message = chatbot_response(message, chat_history, school_type)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_message})
            return "", chat_history
        
        msg.submit(respond, [msg, chatbot, school_selector], [msg, chatbot])
        send_btn.click(respond, [msg, chatbot, school_selector], [msg, chatbot])
        clear_btn.click(lambda: [], None, chatbot, queue=False)
    
  return demo

# ==================== 메인 ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎓 학교생활기록부 RAG 챗봇")
    print("="*70 + "\n")
    
    try:
        initialize_system()
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            css=CSS
        )
    except Exception as e:
        print(f"\n❌ 오류: {e}")
       