import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamically find the absolute path to the hms-core root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
ENV_PATH = os.path.join(ROOT_DIR, '.env')

class Settings(BaseSettings):
    PINECONE_API_KEY: str
    OPENAI_API_KEY: str
    PINECONE_INDEX_NAME: str = "hmscore"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Neo4j Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # App Settings
    log_level: str = "INFO"
    
    # Point directly to the absolute path of the .env file
    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8", extra="ignore")

settings = Settings()