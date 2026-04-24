from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Pinecone
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "hms-core-index"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # OpenAI
    openai_api_key: str

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()