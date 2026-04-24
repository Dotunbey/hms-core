from fastapi import APIRouter, HTTPException
from src.api.schemas import QueryRequest, QueryResponse, IngestRequest
from src.memory.retriever import HybridRetriever
from src.ingestion.processor import DocumentProcessor

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
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest_document(request: IngestRequest):
    """Ingest a document into the memory system."""
    try:
        result = await processor.process_document(
            file_path=request.file_path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            metadata=request.metadata
        )
        return {"status": "success", "chunks": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{entity_id}")
async def get_entity(entity_id: str):
    """Get an entity from the graph store."""
    try:
        entity = await retriever.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        return entity
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))