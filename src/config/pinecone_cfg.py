"""Pinecone configuration for vector storage."""

from src.config.settings import settings


# Index configuration
INDEX_CONFIG = {
    "name": settings.PINECONE_INDEX_NAME,
    "dimension": 768,  # Google text-embedding-004 dimension
    "metric": "cosine",
    "pod_type": "p1"
}

# Namespace configuration
NAMESPACES = {
    "documents": "documents",      # Chunked document content
    "entities": "entities",        # Extracted entities
    "memories": "memories"        # User memories
}