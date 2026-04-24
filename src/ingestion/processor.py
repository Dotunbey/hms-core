import sys
import os
import hashlib
import json
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from src.api.schemas import MemoryObject
from src.config.settings import settings
from src.memory.vector_store import VectorStore
from src.memory.graph_store import GraphStore

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = None
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0
        )

    def _generate_id(self, content: str, source_id: str) -> str:
        unique_string = f"{source_id}-{content}"
        return hashlib.sha256(unique_string.encode()).hexdigest()

    async def _extract_entities_and_segment(self, content: str) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI data processor for the Hermes Memory System. "
                       "Your task is to analyze the text and return a JSON object with two keys: "
                       "'memory_type' (one of: 'Identity Context', 'Policies and Rules', "
                       "'Historical Actions', 'Events and Campaigns', 'Documents', 'Conversations', 'Insights and Learnings'), "
                       "and 'entities' (a list of dictionaries, each with 'label' (e.g. Person, Organization, Role, Document), "
                       "'name' (the entity value), and optionally 'relationships' (list of dicts with 'type' and 'target_name')). "
                       "Return ONLY valid JSON."),
            ("user", "Text: {text}")
        ])
        
        chain = prompt | self.llm
        try:
            response = await chain.ainvoke({"text": content[:2000]})
            content_str = response.content.strip()
            if content_str.startswith("```json"):
                content_str = content_str[7:-3]
            elif content_str.startswith("```"):
                content_str = content_str[3:-3]
                
            data = json.loads(content_str)
            return data
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"memory_type": "Documents", "entities": []}

    async def process_document(self, file_path: str, chunk_size: int, chunk_overlap: int, metadata: Dict[str, Any]) -> List[MemoryObject]:
        logger.info(f"Processing document: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = self.text_splitter.split_text(full_text)
        memory_objects = []
        source_id = metadata.get("source_id", os.path.basename(file_path))
        
        for chunk in chunks:
            chunk_id = self._generate_id(chunk, source_id)
            analysis = await self._extract_entities_and_segment(chunk)
            memory_type = analysis.get("memory_type", "Documents")
            entities = analysis.get("entities", [])
            
            meta = metadata.copy()
            meta["source_id"] = source_id
            
            memory_obj = MemoryObject(
                id=chunk_id,
                content=chunk,
                metadata=meta,
                memory_type=memory_type,
                confidence_score=1.0
            )
            memory_objects.append(memory_obj)
            
            await self.vector_store.upsert(
                id=chunk_id,
                text=chunk,
                metadata=meta,
                namespace="documents"
            )
            
            for entity in entities:
                entity_name = entity.get("name")
                entity_label = entity.get("label", "Unknown")
                if entity_name:
                    entity_id = self._generate_id(entity_name, entity_label)
                    await self.graph_store.create_entity(
                        entity_id=entity_id,
                        label=entity_label,
                        properties={"name": entity_name, "source_id": source_id}
                    )
                    for rel in entity.get("relationships", []):
                        target_name = rel.get("target_name")
                        rel_type = rel.get("type", "RELATES_TO")
                        if target_name:
                            target_id = self._generate_id(target_name, "Unknown")
                            await self.graph_store.create_entity(
                                entity_id=target_id,
                                label="Unknown",
                                properties={"name": target_name}
                            )
                            await self.graph_store.create_relationship(
                                source_id=entity_id,
                                target_id=target_id,
                                relation_type=rel_type.upper().replace(" ", "_"),
                                properties={"source_id": source_id}
                            )
            
        logger.info(f"Successfully processed and stored {len(memory_objects)} chunks.")
        return memory_objects