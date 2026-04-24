from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class MemoryObject(BaseModel):
    """Core memory entity stored in the vector store."""

    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


class Entity(BaseModel):
    """Represents an entity in the knowledge graph."""

    id: str
    label: str
    properties: Dict[str, Any] = {}
    created_at: datetime = datetime.now()


class Relationship(BaseModel):
    """Represents a relationship between entities."""

    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = {}


class QueryRequest(BaseModel):
    """Request model for querying the memory system."""

    query: str
    top_k: int = 5
    include_graph: bool = True


class QueryResponse(BaseModel):
    """Response model for query results."""

    results: List[MemoryObject]
    entities: List[Entity] = []
    relationships: List[Relationship] = []


class IngestRequest(BaseModel):
    """Request model for ingesting documents."""

    file_path: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    metadata: Dict[str, Any] = {}