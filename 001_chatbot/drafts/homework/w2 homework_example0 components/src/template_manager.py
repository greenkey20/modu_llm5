"""
TemplateManager 컴포넌트
RAG 파라미터 조합을 템플릿으로 저장하고 관리
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os


@dataclass
class RAGConfig:
    """RAG 시스템 설정 데이터 모델"""
    # 문서 처리 설정
    splitter_type: str = "RecursiveCharacterTextSplitter"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: str = "\\n\\n,\\n, ,"
    
    # 임베딩 설정
    embedding_type: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # 벡터 저장소 설정
    vector_store_type: str = "chroma"
    
    # 검색 설정
    search_type: str = "similarity"
    k: int = 4
    score_threshold: float = 0.5
    lambda_mult: float = 0.5
    
    # LLM 설정
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000


@dataclass
class Template:
    """템플릿 데이터 모델"""
    name: str
    config: RAGConfig
    created_at: str
    description: Optional[str] = None


class TemplateManager:
    """템플릿 관리 클래스"""
    
    def __init__(self, storage_path: str = "./templates.json"):
        self.templates: Dict[str, Template] = {}
        self.storage_path = storage_path
        self.load_from_disk()
    
    def save_template(
        self,
        name: str,
        config: RAGConfig,
        description: Optional[str] = None
    ) -> bool:
        """
        템플릿 저장
        
        Args:
            name: 템플릿 이름
            config: RAG 설정
            description: 템플릿 설명 (선택)
            
        Returns:
            저장 성공 여부
        """
        try:
            template = Template(
                name=name,
                config=config,
                created_at=datetime.now().isoformat(),
                description=description
            )
            self.templates[name] = template
            self.persist_to_disk()
            return True
        except Exception as e:
            print(f"템플릿 저장 실패: {str(e)}")
            return False
    
    def load_template(self, name: str) -> Optional[RAGConfig]:
        """
        템플릿 불러오기
        
        Args:
            name: 템플릿 이름
            
        Returns:
            RAGConfig 객체 또는 None
        """
        template = self.templates.get(name)
        if template:
            return template.config
        return None
    
    def delete_template(self, name: str) -> bool:
        """
        템플릿 삭제
        
        Args:
            name: 템플릿 이름
            
        Returns:
            삭제 성공 여부
        """
        try:
            if name in self.templates:
                del self.templates[name]
                self.persist_to_disk()
                return True
            return False
        except Exception as e:
            print(f"템플릿 삭제 실패: {str(e)}")
            return False
    
    def list_templates(self) -> List[str]:
        """
        저장된 템플릿 목록 반환
        
        Returns:
            템플릿 이름 리스트
        """
        return list(self.templates.keys())
    
    def get_template_info(self, name: str) -> Optional[Dict]:
        """
        템플릿 정보 반환
        
        Args:
            name: 템플릿 이름
            
        Returns:
            템플릿 정보 딕셔너리
        """
        template = self.templates.get(name)
        if template:
            return {
                "name": template.name,
                "description": template.description,
                "created_at": template.created_at,
                "config": asdict(template.config)
            }
        return None
    
    def persist_to_disk(self):
        """템플릿을 디스크에 저장"""
        try:
            data = {}
            for name, template in self.templates.items():
                data[name] = {
                    "name": template.name,
                    "config": asdict(template.config),
                    "created_at": template.created_at,
                    "description": template.description
                }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"템플릿 저장 실패: {str(e)}")
    
    def load_from_disk(self):
        """디스크에서 템플릿 로드"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for name, template_data in data.items():
                    config = RAGConfig(**template_data['config'])
                    template = Template(
                        name=template_data['name'],
                        config=config,
                        created_at=template_data['created_at'],
                        description=template_data.get('description')
                    )
                    self.templates[name] = template
        
        except Exception as e:
            print(f"템플릿 로드 실패: {str(e)}")
