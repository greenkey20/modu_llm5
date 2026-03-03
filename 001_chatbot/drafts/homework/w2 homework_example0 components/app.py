"""
RAG 챗봇 메인 애플리케이션 (2탭 UI)
"""
import os
from dotenv import load_dotenv
from src.gradio_interface_v2 import GradioInterface

# 환경 변수 로드
load_dotenv()

def main():
    """메인 함수"""
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 설정해주세요.")
        print()
    
    # Gradio 인터페이스 생성 및 실행
    interface = GradioInterface()
    demo = interface.create_interface()
    
    print("🚀 RAG 챗봇 서버를 시작합니다...")
    print("   브라우저에서 http://localhost:7860 으로 접속하세요.")
    print()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
