"""Pinecone configuration for vector storage."""

from src.config.settings import settings


# Index configuration
INDEX_CONFIG = {
    "name": settings.PINECONE_INDEX_NAME,
    "dimension": 3072,  # Google gemini-embedding-2 dimension
    "metric": "cosine",
    "pod_type": "p1"
}

# Namespace configuration
NAMESPACES = {
    "documents": "documents",      # Chunked document content
    "entities": "entities",        # Extracted entities
    "memories": "memories"        # User memories
}