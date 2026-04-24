import os
import shutil
import uuid
from loguru import logger
from typing import Optional

class BlobStorage:
    """
    Interface for blob storage (MVP uses local filesystem).
    Future extension: swap this with S3/MinIO using boto3.
    """
    
    def __init__(self, base_dir: str = "data/blobs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
    def save(self, file_obj, original_filename: str) -> str:
        """Saves a file-like object and returns the blob URI."""
        ext = os.path.splitext(original_filename)[1]
        blob_id = str(uuid.uuid4())
        filename = f"{blob_id}{ext}"
        filepath = os.path.join(self.base_dir, filename)
        
        try:
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file_obj, buffer)
            uri = f"local://{filepath}"
            logger.info(f"Saved blob: {uri}")
            return uri
        except Exception as e:
            logger.error(f"Failed to save blob {original_filename}: {e}")
            raise
            
    def get_path(self, uri: str) -> str:
        """Returns the local file path for a given URI."""
        if uri.startswith("local://"):
            return uri[8:]
        raise ValueError("Unsupported URI scheme.")
