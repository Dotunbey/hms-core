from sqlalchemy import Column, String, Text, ForeignKey, DateTime, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from src.db.session import Base
import uuid

def get_utc_now():
    return datetime.now(timezone.utc)

def generate_uuid():
    return str(uuid.uuid4())

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    storage_uri = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=get_utc_now)
    
    sections = relationship("Section", back_populates="document", cascade="all, delete-orphan")
    memory_objects = relationship("MemoryObject", back_populates="document", cascade="all, delete-orphan")


class Section(Base):
    __tablename__ = "sections"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    title = Column(String, nullable=False)
    hierarchy_level = Column(String)  # e.g., "1.1", "2.0"
    created_at = Column(DateTime(timezone=True), default=get_utc_now)
    
    document = relationship("Document", back_populates="sections")
    memory_objects = relationship("MemoryObject", back_populates="section")


class MemoryObject(Base):
    __tablename__ = "memory_objects"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    section_id = Column(String, ForeignKey("sections.id"), nullable=True)
    
    # Routing / Type information
    type = Column(String, nullable=False)  # e.g., "paragraph", "table", "image", "list"
    
    # Split content representation
    text_content = Column(Text, nullable=True)
    structured_content = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=get_utc_now)

    document = relationship("Document", back_populates="memory_objects")
    section = relationship("Section", back_populates="memory_objects")
