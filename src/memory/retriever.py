"""Hybrid retriever combining vector and graph search."""

from typing import List, Dict, Any, Optional
from loguru import logger
from src.memory.vector_store import VectorStore
from src.memory.graph_store import GraphStore
from src.api.schemas import MemoryObject, Entity, Relationship, QueryResponse


class HybridRetriever:
    """Hybrid search combining vector and graph retrieval."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        include_graph: bool = True
    ) -> QueryResponse:
        """Search both vector and graph stores."""
        # Vector search
        vector_results = await self.vector_store.query(
            query_text=query,
            top_k=top_k,
            namespace="documents"
        )

        # Convert to MemoryObject
        memories = [
            MemoryObject(
                id=result["id"],
                content=result["text"],
                metadata=result["metadata"]
            )
            for result in vector_results
        ]

        # Graph search (if enabled)
        entities = []
        relationships = []

        if include_graph:
            # Extract potential entities from query
            # For now, do a simple keyword-based graph search
            graph_results = await self._graph_search(query)
            entities = graph_results.get("entities", [])
            relationships = graph_results.get("relationships", [])

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
        """Perform graph-based search."""
        # Simple implementation: extract keywords and search
        # In production, this would use NLP to extract entities
        keywords = query.lower().split()

        entities = []
        relationships = []

        # This is a placeholder - real implementation would
        # use NER or entity extraction
        logger.debug(f"Graph search for query: {query}")

        return {
            "entities": entities,
            "relationships": relationships
        }