"""Audit logging for governance and traceability."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from loguru import logger

class AuditLogger:
    """Handles structured audit logging for HMS."""

    @staticmethod
    def log_event(
        actor: str,
        action: str,
        data_accessed: Any,
        approval_status: str = "APPROVED",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an interaction with the memory system.
        
        Args:
            actor: The agent or user making the request.
            action: The action being performed (e.g., 'RETRIEVE', 'INGEST').
            data_accessed: Summary or IDs of data accessed.
            approval_status: Status of access control.
            metadata: Any additional context.
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "data_accessed": data_accessed,
            "approval_status": approval_status,
            "metadata": metadata or {}
        }
        
        # We log to 'audit' which the loguru config directs to logs/audit.log
        logger.info(f"AUDIT_EVENT: {json.dumps(event)}")
