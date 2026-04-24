import os
import tempfile
import json
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from src.api.schemas import QueryRequest, QueryResponse
from src.memory.retriever import HybridRetriever
from src.ingestion.processor import DocumentProcessor
from src.utils.audit import AuditLogger

router = APIRouter()

# Initialize services
retriever = HybridRetriever()
processor = DocumentProcessor()

@router.post("/query", response_model=QueryResponse)
async def query_memory(request: QueryRequest):
    """Query the hybrid memory system."""
    try:
        results = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            include_graph=request.include_graph
        )
        
        AuditLogger.log_event(
            actor="Agent/User",
            action="RETRIEVE",
            data_accessed=f"Query: '{request.query}', Found {len(results.results)} vector results and {len(results.entities)} entities."
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    metadata_json: Optional[str] = Form(None)
):
    """Ingest a document into the memory system."""
    try:
        meta = json.loads(metadata_json) if metadata_json else {}
        meta["source_id"] = file.filename
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        processed_count = await processor.process_document(
            file_path=tmp_path,
            metadata=meta
        )
        
        os.remove(tmp_path)
        
        AuditLogger.log_event(
            actor="System",
            action="INGEST",
            data_accessed=f"File: {file.filename}, Processed {processed_count} chunks",
            approval_status="APPROVED",
            metadata={"file_name": file.filename}
        )
        
        return {"status": "success", "chunks_processed": processed_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entity/{entity_id}")
async def get_entity(entity_id: str):
    """Get an entity from the graph store."""
    try:
        entity = await retriever.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
            
        AuditLogger.log_event(
            actor="Agent/User",
            action="RETRIEVE_ENTITY",
            data_accessed=f"Entity ID: {entity_id}"
        )
        return entity
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))