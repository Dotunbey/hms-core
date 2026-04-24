"""Document processing: chunking and metadata tagging."""

import uuid
from typing import List, Dict, Any, Optional
from loguru import logger
from src.ingestion.parsers import DocumentParser
from src.ingestion.vector_ops import VectorStore


class DocumentProcessor:
    """Process documents: parse, chunk, and store in vector DB."""

    def __init__(self):
        self.parser = DocumentParser()
        self.vector_store = VectorStore()

    async def process_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Process a document and store chunks in vector store."""
        # Parse document
        text = self.parser.parse(file_path)

        # Chunk text
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)

        # Generate embeddings and store
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_metadata = {
                "source": file_path,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }

            await self.vector_store.upsert(
                id=chunk_id,
                text=chunk,
                metadata=chunk_metadata,
                namespace="documents"
            )
            chunk_ids.append(chunk_id)

        logger.info(f"Processed {len(chunks)} chunks from {file_path}")
        return chunk_ids

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk.strip())

            start += chunk_size - chunk_overlap

        return chunks