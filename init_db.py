from src.db.session import engine, Base
from src.db.models import Document, Section, MemoryObject

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Database tables created successfully!")
