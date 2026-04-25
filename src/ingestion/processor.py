import os
import json
import hashlib
import asyncio
from typing import List, Dict, Any
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.db.session import SessionLocal
from src.db.models import Document as DBDocument, Section as DBSection, MemoryObject as DBMemoryObject
from src.storage.blob import BlobStorage
from src.ingestion.parser import DocumentParser
from src.memory.vector_store import VectorStore
from src.memory.graph_store import GraphStore
from src.config.settings import settings

# Use a cheaper model with higher rate limits for entity extraction
ENTITY_EXTRACTION_MODEL = "gemini-2.0-flash"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds

class DocumentProcessor:
    def __init__(self):
        self.blob_storage = BlobStorage()
        self.parser = DocumentParser()
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()
        # Main LLM for agent responses (high quality)
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0
        )
        # Separate LLM for entity extraction (higher rate limits)
        self.entity_llm = ChatGoogleGenerativeAI(
            model=ENTITY_EXTRACTION_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0
        )

    def _generate_id(self, *args) -> str:
        unique_string = "-".join(str(a) for a in args)
        return hashlib.sha256(unique_string.encode()).hexdigest()

    async def _extract_entities(self, content: str) -> Dict[str, Any]:
        """Extracts entities and relationships from a text block with retry logic."""
        if not content or len(content.strip()) < 10:
            return {"entities": []}
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract key entities (Person, Organization, Role, Event, Policy) from the text. "
                       "Return a JSON object with 'entities' (list of dicts, each with 'label', 'name', and optionally "
                       "'relationships' which is a list of dicts with 'type' and 'target_name'). "
                       "Return ONLY valid JSON."),
            ("user", "Text: {text}")
        ])
        
        chain = prompt | self.entity_llm
        
        for attempt in range(MAX_RETRIES):
            try:
                response = await chain.ainvoke({"text": content[:2000]})
                content_str = response.content.strip()
                if content_str.startswith("```json"):
                    content_str = content_str[7:-3]
                elif content_str.startswith("```"):
                    content_str = content_str[3:-3]
                    
                return json.loads(content_str)
            except Exception as e:
                error_str = str(e)
                if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Rate limited on entity extraction (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Entity extraction failed: {e}")
                    return {"entities": []}
        
        logger.warning("Entity extraction skipped after max retries")
        return {"entities": []}

    async def _resolve_and_store_graph(self, entities: List[Dict[str, Any]], source_id: str):
        """Validates and deduplicates entities before storing in Neo4j."""
        for entity in entities:
            entity_name = entity.get("name")
            entity_label = entity.get("label", "Unknown")
            if not entity_name:
                continue
                
            # Deterministic ID based on normalized name and label to prevent duplicates
            entity_id = self._generate_id(entity_name.lower().strip(), entity_label)
            
            # Upsert entity in graph
            await self.graph_store.create_entity(
                entity_id=entity_id,
                label=entity_label,
                properties={"name": entity_name, "source_id": source_id}
            )
            
            for rel in entity.get("relationships", []):
                target_name = rel.get("target_name")
                rel_type = rel.get("type", "RELATES_TO").upper().replace(" ", "_")
                
                if target_name:
                    # In a full system, we'd query neo4j to find the best match for target_name here
                    target_id = self._generate_id(target_name.lower().strip(), "Unknown")
                    await self.graph_store.create_entity(
                        entity_id=target_id,
                        label="Unknown",
                        properties={"name": target_name}
                    )
                    await self.graph_store.create_relationship(
                        source_id=entity_id,
                        target_id=target_id,
                        relation_type=rel_type,
                        properties={"source_id": source_id}
                    )

    async def process_document(self, file_path: str, metadata: Dict[str, Any]) -> int:
        """
        Full Tri-Store Ingestion Pipeline:
        1. Parse document into blocks
        2. Store facts in Postgres
        3. Extract and store identity in Neo4j
        4. Embed and store meaning in Pinecone
        """
        logger.info(f"Processing document via Tri-Store Pipeline: {file_path}")
        original_filename = metadata.get("source_id", os.path.basename(file_path))
        
        # 1. Save to Blob Storage
        with open(file_path, "rb") as f:
            storage_uri = self.blob_storage.save(f, original_filename)
            
        # 2. Parse Document
        blocks = self.parser.parse_document(file_path, strategy="auto")
        
        db = SessionLocal()
        try:
            # Create Document Record
            db_doc = DBDocument(
                filename=original_filename,
                storage_uri=storage_uri
            )
            db.add(db_doc)
            db.flush() # get ID
            
            current_section = None
            processed_count = 0
            
            for block in blocks:
                block_type = block.get("type", "Paragraph")
                block_text = block.get("text", "")
                
                # Update Section Hierarchy
                if block_type in ["Title", "Heading"]:
                    current_section = DBSection(
                        document_id=db_doc.id,
                        title=block_text[:255]
                    )
                    db.add(current_section)
                    db.flush()
                    continue # optionally store headings as memory objects too, but we skip for brevity
                    
                # Create Memory Object in Postgres
                mem_obj = DBMemoryObject(
                    document_id=db_doc.id,
                    section_id=current_section.id if current_section else None,
                    type=block_type,
                    text_content=block_text,
                    structured_content=block if block_type == "Table" else None
                )
                db.add(mem_obj)
                db.flush()
                
                # Graph Extraction & Storage (with rate-limit-friendly pacing)
                analysis = await self._extract_entities(block_text)
                await self._resolve_and_store_graph(analysis.get("entities", []), source_id=db_doc.id)
                await asyncio.sleep(1)  # pace requests to avoid bursts
                
                # Vector Embedding & Storage (Semantic Meaning)
                if block_text:
                    await self.vector_store.upsert(
                        id=mem_obj.id,
                        text=block_text,
                        metadata={
                            "memory_id": mem_obj.id,
                            "document_id": db_doc.id,
                            "type": block_type,
                            "department": metadata.get("department", "general")
                        },
                        namespace="documents"
                    )
                processed_count += 1
                
            db.commit()
            logger.info(f"Successfully processed {processed_count} memory objects for {original_filename}")
            return processed_count
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to process document: {e}")
            raise
        finally:
            db.close()