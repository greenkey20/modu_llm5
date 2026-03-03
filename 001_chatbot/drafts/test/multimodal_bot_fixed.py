import gradio as gr
import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def convert_to_url(image_path):
    """이미지를 URL 형식으로 변환"""
    with open(image_path, "rb") as image_file:
        # 이미지를 base64로 인코딩
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

def multimodal_bot(message, history):

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    
    if isinstance(message, dict):
        # 텍스트와 파일 추출
        text = message.get("text", "")
        
        # 히스토리와 현재 메시지에서 모든 파일 경로 추출
        filepath_list = []
        
        # ========================================
        # 🔧 수정된 부분: 히스토리에서 이미지 파일 추출
        # ========================================
        print("History:", history)  # 디버깅용
        for exchange in history:
            # 새로운 Gradio 형식: exchange는 딕셔너리
            if isinstance(exchange, dict) and 'content' in exchange:
                content = exchange['content']
                # content는 리스트 형태
                if isinstance(content, list):
                    for item in content:
                        # 파일 타입 아이템 찾기
                        if isinstance(item, dict) and item.get('type') == 'file':
                            file_info = item.get('file', {})
                            if 'path' in file_info:
                                filepath_list.append(file_info['path'])
        # ========================================
        
        # 현재 메시지의 파일들도 추가
        files = message.get("files", [])
        filepath_list.extend(files)
        
        print("Filepath list:", filepath_list)  # 디버깅용
        
        if filepath_list:
            # 모든 이미지 처리
            image_urls = []
            for file_path in filepath_list:
                try:
                    image_url = convert_to_url(file_path)
                    image_urls.append({"type": "image_url", "image_url": image_url})
                except Exception as e:
                    print(f"이미지 처리 중 오류 발생: {e}")
                    continue
            
            if not image_urls:
                return "이미지 처리 중 오류가 발생했습니다."
            
            # 메시지 구성
            content = [
                {"type": "text", "text": text if text else "이 이미지들에 대해 설명해주세요."},
                *image_urls
            ]
            
            try:
                # API 호출
                response = model.invoke([
                    HumanMessage(content=content)
                ])
                return response.content
            except Exception as e:
                return f"모델 응답 생성 중 오류가 발생했습니다: {str(e)}"
        
        return text if text else "이미지를 업로드해주세요."
    
    return "텍스트나 이미지를 입력해주세요."

# Gradio 인터페이스 설정
demo = gr.ChatInterface(
    fn=multimodal_bot,
    multimodal=True,
    title="멀티모달 챗봇",
    description="텍스트와 이미지를 함께 처리할 수 있는 챗봇입니다. 이전 대화의 이미지들도 함께 고려합니다.",
    analytics_enabled=False,  
    textbox=gr.MultimodalTextbox(placeholder="텍스트를 입력하거나 이미지를 업로드해주세요.", file_count="multiple", file_types=["image"]),
)

demo.launch()
