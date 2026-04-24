import sys
import os
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

def get_utc_now():
    return datetime.now(timezone.utc)

class MemoryObject(BaseModel):
    """Core memory entity stored in the vector store."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Must contain 'source_id' for PRD traceability"
    )
    memory_type: str = Field(default="Document", description="Category of memory segment")
    confidence_score: float = Field(default=1.0, description="Confidence score for this memory object")
    created_at: datetime = Field(default_factory=get_utc_now)
    updated_at: datetime = Field(default_factory=get_utc_now)

class Entity(BaseModel):
    """Represents an entity in the knowledge graph."""

    id: str
    label: str
    properties: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=get_utc_now)


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

class AgentRequest(BaseModel):
    """Request model for agent queries."""
    query: str
    actor_id: str = "System"

class AgentResponse(BaseModel):
    """Response model for agent queries."""
    answer: str
    sources: List[str] = []