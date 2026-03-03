import json
import sys

def clear_notebook_outputs(notebook_path):
    """노트북 파일의 모든 출력 결과를 삭제합니다."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # 모든 셀의 출력 삭제
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    
    # 수정된 노트북 저장
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print(f"✅ {notebook_path}의 모든 출력 결과가 삭제되었습니다.")

if __name__ == "__main__":
    notebook_path = "PRJ01_W1_002_OpenAI_Chat_Completion.ipynb"
    clear_notebook_outputs(notebook_path)
