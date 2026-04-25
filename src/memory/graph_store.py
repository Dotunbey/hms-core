"""Graph store interface for Neo4j."""

import json
from typing import List, Dict, Any, Optional
from loguru import logger
from src.config.settings import settings


class GraphStore:
    """Neo4j graph store interface for identity context."""

    def __init__(self):
        self._driver = None

    @property
    def driver(self):
        """Lazy initialization of Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import AsyncGraphDatabase
                self._driver = AsyncGraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password)
                )
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")
                raise
        return self._driver

    async def create_entity(
        self,
        entity_id: str,
        label: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Create an entity node in the graph."""
        try:
            async with self.driver.session() as session:
                await session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.label = $label, e.properties = $properties
                    """,
                    id=entity_id,
                    label=label,
                    properties=json.dumps(properties) if isinstance(properties, dict) else str(properties)
                )
            logger.debug(f"Created entity {entity_id} with label {label}")
            return True
        except Exception as e:
            logger.error(f"Failed to create entity: {e}")
            raise

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Dict[str, Any] = {}
    ) -> bool:
        """Create a relationship between entities."""
        try:
            async with self.driver.session() as session:
                await session.run(
                    """
                    MATCH (s:Entity {id: $source_id})
                    MATCH (t:Entity {id: $target_id})
                    MERGE (s)-[r:RELATES {type: $relation_type}]->(t)
                    SET r.properties = $properties
                    """,
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    properties=json.dumps(properties) if isinstance(properties, dict) else str(properties)
                )
            logger.debug(f"Created relationship {source_id} -> {target_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            raise

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity by ID."""
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    "MATCH (e:Entity {id: $id}) RETURN e",
                    id=entity_id
                )
                record = await result.single()
                if record:
                    return dict(record["e"])
                return None
        except Exception as e:
            logger.error(f"Failed to get entity: {e}")
            raise

    async def get_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get entities related to a given entity."""
        try:
            async with self.driver.session() as session:
                if relation_type:
                    query = """
                    MATCH (e:Entity {id: $id})-[r:RELATES {type: $relation_type}]->(related)
                    RETURN related
                    """
                    result = await session.run(
                        query,
                        id=entity_id,
                        relation_type=relation_type
                    )
                else:
                    query = """
                    MATCH (e:Entity {id: $id})-[r:RELATES]->(related)
                    RETURN related
                    """
                    result = await session.run(query, id=entity_id)

                return [dict(record["related"]) async for record in result]
        except Exception as e:
            logger.error(f"Failed to get related entities: {e}")
            raise

    async def close(self):
        """Close the driver connection."""
        if self._driver:
            await self._driver.close()