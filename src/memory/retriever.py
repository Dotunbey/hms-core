"""Hybrid retriever combining vector and graph search."""

import json
from typing import List, Dict, Any, Optional
from loguru import logger
from sqlalchemy.orm import joinedload
from src.memory.vector_store import VectorStore
from src.memory.graph_store import GraphStore
from src.api.schemas import MemoryObject, Entity, Relationship, QueryResponse
from src.db.session import SessionLocal
from src.db.models import MemoryObject as DBMemoryObject, Section as DBSection


class HybridRetriever:
    """Hybrid search combining vector, relational, and graph retrieval."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        include_graph: bool = True
    ) -> QueryResponse:
        """Search across Pinecone, Postgres, and Neo4j."""
        
        # 1. Vector Search (Pinecone) -> returns metadata with memory_ids
        vector_results = await self.vector_store.query(
            query_text=query,
            top_k=top_k,
            namespace="documents"
        )
        
        # Extract candidate memory IDs
        memory_ids = []
        for res in vector_results:
            mem_id = res.get("metadata", {}).get("memory_id")
            if mem_id and mem_id not in memory_ids:
                memory_ids.append(mem_id)
                
        # 2. Relational Search (Postgres) -> expand candidate IDs to full context
        memories = []
        if memory_ids:
            db = SessionLocal()
            try:
                # Fetch memory objects and eager load their section for hierarchical context
                db_memories = db.query(DBMemoryObject).options(
                    joinedload(DBMemoryObject.section)
                ).filter(DBMemoryObject.id.in_(memory_ids)).all()
                
                for db_mem in db_memories:
                    # Construct full context using section hierarchy if available
                    context_prefix = ""
                    if db_mem.section:
                        context_prefix = f"[{db_mem.section.title}]\n"
                        
                    content = context_prefix + (db_mem.text_content or "")
                    if db_mem.structured_content:
                        content += "\n" + json.dumps(db_mem.structured_content, indent=2)
                        
                    memories.append(
                        MemoryObject(
                            id=db_mem.id,
                            content=content.strip(),
                            metadata={
                                "document_id": db_mem.document_id,
                                "section_id": db_mem.section_id,
                                "type": db_mem.type
                            },
                            memory_type=db_mem.type,
                            confidence_score=1.0,
                            created_at=db_mem.created_at,
                            updated_at=db_mem.created_at
                        )
                    )
            except Exception as e:
                logger.error(f"Postgres retrieval failed: {e}")
            finally:
                db.close()

        # 3. Graph Search (Neo4j) -> resolve entities
        entities = []
        relationships = []

        if include_graph:
            graph_results = await self._graph_search(query)
            entities = graph_results.get("entities", [])
            relationships = graph_results.get("relationships", [])

        # 4. Assemble Results
        return QueryResponse(
            results=memories,
            entities=entities,
            relationships=relationships
        )

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity from the graph store."""
        entity_data = await self.graph_store.get_entity(entity_id)
        if entity_data:
            return Entity(
                id=entity_data.get("id", entity_id),
                label=entity_data.get("label", "Unknown"),
                properties=entity_data.get("properties", {})
            )
        return None

    async def _graph_search(self, query: str) -> Dict[str, Any]:
        """Perform graph-based search using LLM query routing."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from src.config.settings import settings
        
        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract key entities from the user's query to search in a knowledge graph. "
                       "Return a JSON object with a single key 'entities' containing a list of strings (the entity names). "
                       "Return ONLY valid JSON."),
            ("user", "Query: {query}")
        ])
        
        try:
            chain = prompt | llm
            response = await chain.ainvoke({"query": query})
            content_str = response.content.strip()
            if content_str.startswith("```json"):
                content_str = content_str[7:-3]
            elif content_str.startswith("```"):
                content_str = content_str[3:-3]
                
            data = json.loads(content_str)
            entity_names = data.get("entities", [])
        except Exception as e:
            logger.error(f"Failed to extract entities from query: {e}")
            entity_names = [word for word in query.split() if len(word) > 3]
            
        entities = []
        relationships = []
        seen_entity_ids = set()
        
        try:
            async with self.graph_store.driver.session() as session:
                for name in entity_names:
                    result = await session.run(
                        """
                        MATCH (e:Entity)
                        WHERE toLower(e.properties.name) CONTAINS toLower($name)
                        OPTIONAL MATCH (e)-[r]->(related:Entity)
                        RETURN e, type(r) as r_type, properties(r) as r_props, related
                        LIMIT 10
                        """,
                        name=name
                    )
                    records = await result.data()
                    for record in records:
                        e_node = record.get("e")
                        if e_node:
                            e_id = e_node.get("id")
                            if e_id not in seen_entity_ids:
                                entities.append(Entity(
                                    id=e_id,
                                    label=e_node.get("label", "Unknown"),
                                    properties=e_node.get("properties", {})
                                ))
                                seen_entity_ids.add(e_id)
                                
                        related_node = record.get("related")
                        r_type = record.get("r_type")
                        r_props = record.get("r_props", {})
                        
                        if related_node and r_type:
                            rel_id = related_node.get("id")
                            if rel_id not in seen_entity_ids:
                                entities.append(Entity(
                                    id=rel_id,
                                    label=related_node.get("label", "Unknown"),
                                    properties=related_node.get("properties", {})
                                ))
                                seen_entity_ids.add(rel_id)
                                
                            relationships.append(Relationship(
                                source_id=e_id,
                                target_id=rel_id,
                                relation_type=r_type,
                                properties=r_props if r_props else {}
                            ))
        except Exception as e:
            logger.error(f"Graph search query failed: {e}")

        return {
            "entities": entities,
            "relationships": relationships
        }