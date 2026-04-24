"""Vector store operations for Pinecone."""

from typing import List, Dict, Any, Optional
from loguru import logger
from src.config.settings import settings
from src.config.pinecone_cfg import INDEX_CONFIG, NAMESPACES


class VectorStore:
    """Pinecone vector store interface."""

    def __init__(self):
        self._client = None
        self._index = None

    @property
    def client(self):
        """Lazy initialization of Pinecone client."""
        if self._client is None:
            try:
                from pinecone import Pinecone
                self._client = Pinecone(
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_environment
                )
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone client: {e}")
                raise
        return self._client

    @property
    def index(self):
        """Get or create the Pinecone index."""
        if self._index is None:
            self._index = self.client.Index(INDEX_CONFIG["name"])
        return self._index

    async def upsert(
        self,
        id: str,
        text: str,
        metadata: Dict[str, Any],
        namespace: str = "documents"
    ) -> bool:
        """Upsert a vector into the index."""
        try:
            # Generate embedding
            embedding = await self._get_embedding(text)

            # Prepare vector
            vector = {
                "id": id,
                "values": embedding,
                "metadata": {"text": text, **metadata}
            }

            # Upsert to Pinecone
            self.index.upsert(vectors=[vector], namespace=namespace)
            logger.debug(f"Upserted vector {id} to namespace {namespace}")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert vector: {e}")
            raise

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        namespace: str = "documents",
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector store."""
        try:
            # Generate query embedding
            query_embedding = await self._get_embedding(query_text)

            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=True
            )

            return [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "text": match["metadata"].get("text", ""),
                    "metadata": {k: v for k, v in match["metadata"].items() if k != "text"}
                }
                for match in results.get("matches", [])
            ]

        except Exception as e:
            logger.error(f"Failed to query vector store: {e}")
            raise

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI."""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise