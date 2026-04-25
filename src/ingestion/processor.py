"""
Document ingestion processor — Tri-Store Pipeline.

Key design decisions:
- NO LLM calls during ingestion (zero quota burn).
- Uses RecursiveCharacterTextSplitter for smart chunking.
- Batched vector uploads with retry and dead-letter queue.
- Entity extraction is deferred to query time (in the retriever).
- Graph population is optional and done via a separate offline step.
"""

import os
import json
import hashlib
import asyncio
from typing import List, Dict, Any, Iterator
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.db.session import SessionLocal
from src.db.models import Document as DBDocument, Section as DBSection, MemoryObject as DBMemoryObject
from src.storage.blob import BlobStorage
from src.ingestion.parser import DocumentParser
from src.memory.vector_store import VectorStore
from src.config.settings import settings

# --- Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MONSTER_CHUNK_THRESHOLD = 30000  # Split any block larger than this
BATCH_SIZE = 20
MAX_UPLOAD_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


class DocumentProcessor:
    def __init__(self):
        self.blob_storage = BlobStorage()
        self.parser = DocumentParser()
        self.vector_store = VectorStore()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.monster_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=200
        )

    def _generate_id(self, *args) -> str:
        unique_string = "-".join(str(a) for a in args)
        return hashlib.sha256(unique_string.encode()).hexdigest()

    def _apply_monster_check(self, chunks: List[Document]) -> List[Document]:
        """Ensure no single chunk exceeds Pinecone size limits."""
        safe_chunks = []
        for doc in chunks:
            if len(doc.page_content) > MONSTER_CHUNK_THRESHOLD:
                sub_chunks = self.monster_splitter.split_documents([doc])
                for j, sub in enumerate(sub_chunks):
                    sub.metadata = doc.metadata.copy()
                    sub.metadata['chunk_part'] = f"Part {j + 1}"
                    safe_chunks.append(sub)
            else:
                safe_chunks.append(doc)
        return safe_chunks

    def _batch_generator(self, items: List, batch_size: int) -> Iterator[List]:
        """Yield successive batches from a list."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    async def _upload_batch_with_retry(
        self,
        chunks: List[Document],
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> tuple[int, List[Document]]:
        """
        Upload a batch of chunks to Pinecone with retry logic.
        Returns (success_count, failed_chunks).
        """
        failed = []
        success = 0

        for chunk in chunks:
            chunk_id = self._generate_id(doc_id, chunk.page_content[:100])

            for attempt in range(MAX_UPLOAD_RETRIES):
                try:
                    await self.vector_store.upsert(
                        id=chunk_id,
                        text=chunk.page_content,
                        metadata={
                            "memory_id": chunk_id,
                            "document_id": doc_id,
                            "type": chunk.metadata.get("type", "Paragraph"),
                            "section": chunk.metadata.get("section", ""),
                            "page": chunk.metadata.get("page", ""),
                            "paragraph": chunk.metadata.get("paragraph", ""),
                            "department": metadata.get("department", "general"),
                            "source": chunk.metadata.get("source", "")
                        },
                        namespace="documents"
                    )
                    success += 1
                    break
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(f"Rate limited on embed (attempt {attempt + 1}/{MAX_UPLOAD_RETRIES}). "
                                       f"Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Failed to upsert chunk: {e}")
                        failed.append(chunk)
                        break
            else:
                # All retries exhausted
                logger.error(f"Chunk permanently failed after {MAX_UPLOAD_RETRIES} retries")
                failed.append(chunk)

        return success, failed

    async def process_document(self, file_path: str, metadata: Dict[str, Any]) -> int:
        """
        Full Tri-Store Ingestion Pipeline (LLM-free):
        1. Parse document into blocks (PyMuPDF)
        2. Chunk blocks with RecursiveCharacterTextSplitter
        3. Store facts in Postgres (relational context)
        4. Embed and batch-upload to Pinecone (semantic meaning)

        Entity extraction for Neo4j is deferred to query-time.
        """
        logger.info(f"Processing document via Tri-Store Pipeline: {file_path}")
        original_filename = metadata.get("source_id", os.path.basename(file_path))

        # 1. Save to Blob Storage
        with open(file_path, "rb") as f:
            storage_uri = self.blob_storage.save(f, original_filename)

        # 2. Parse Document into blocks
        blocks = self.parser.parse_document(file_path, strategy="auto")

        # 3. Convert blocks to LangChain Documents and chunk them
        raw_docs = []
        for block in blocks:
            if block["type"] == "Heading":
                continue  # Headings are stored in Postgres sections, not as chunks
            text = block.get("text", "")
            if text and len(text.strip()) > 20:
                raw_docs.append(Document(
                    page_content=text,
                    metadata=block.get("metadata", {})
                ))

        # Smart chunking — split large blocks, keep small ones intact
        chunked_docs = []
        for doc in raw_docs:
            if len(doc.page_content) > CHUNK_SIZE:
                sub_chunks = self.splitter.split_documents([doc])
                for i, sub in enumerate(sub_chunks):
                    sub.metadata = doc.metadata.copy()
                    sub.metadata["chunk_index"] = i
                    chunked_docs.append(sub)
            else:
                doc.metadata["chunk_index"] = 0
                chunked_docs.append(doc)

        # Monster check
        chunked_docs = self._apply_monster_check(chunked_docs)

        logger.info(f"Parsed {len(blocks)} blocks → {len(chunked_docs)} chunks for {original_filename}")

        # 4. Store in Postgres (relational context)
        db = SessionLocal()
        try:
            db_doc = DBDocument(
                filename=original_filename,
                storage_uri=storage_uri
            )
            db.add(db_doc)
            db.flush()

            current_section = None
            for block in blocks:
                block_type = block.get("type", "Paragraph")
                block_text = block.get("text", "")

                if block_type in ["Title", "Heading"]:
                    current_section = DBSection(
                        document_id=db_doc.id,
                        title=block_text[:255]
                    )
                    db.add(current_section)
                    db.flush()
                    continue

                if block_text and len(block_text.strip()) > 20:
                    mem_obj = DBMemoryObject(
                        document_id=db_doc.id,
                        section_id=current_section.id if current_section else None,
                        type=block_type,
                        text_content=block_text,
                        structured_content=block if block_type == "Table" else None
                    )
                    db.add(mem_obj)

            db.commit()
            logger.info(f"Stored {len(blocks)} blocks in Postgres for {original_filename}")

        except Exception as e:
            db.rollback()
            logger.error(f"Postgres storage failed: {e}")
            raise
        finally:
            db.close()

        # 5. Batch upload to Pinecone (semantic meaning)
        total_success = 0
        total_failed = []

        for batch in self._batch_generator(chunked_docs, BATCH_SIZE):
            success, failed = await self._upload_batch_with_retry(batch, db_doc.id, metadata)
            total_success += success
            total_failed.extend(failed)

            # Small delay between batches to avoid rate limits
            if total_success > 0:
                await asyncio.sleep(0.5)

        if total_failed:
            logger.warning(f"⚠️  {len(total_failed)} chunks failed to upload (dead-letter queue)")
        
        logger.info(f"✅ Pipeline complete: {total_success}/{len(chunked_docs)} chunks uploaded "
                     f"for {original_filename}")

        return total_success