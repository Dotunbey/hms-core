"""Helper utilities for HMS Core."""

import hashlib
from datetime import datetime
from typing import Any, Dict


def generate_id(*args) -> str:
    """Generate a unique ID from given arguments."""
    content = "".join(str(arg) for arg in args)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.now().isoformat()


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metadata for storage."""
    return {
        k: v for k, v in metadata.items()
        if v is not None and not isinstance(v, (list, dict))
    }


def format_chunk_metadata(source: str, index: int, total: int) -> Dict[str, Any]:
    """Format chunk metadata."""
    return {
        "source": source,
        "chunk_index": index,
        "total_chunks": total,
        "ingested_at": timestamp()
    }