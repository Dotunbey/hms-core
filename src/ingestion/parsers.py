import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

class DocumentParser:
    """Handles the extraction of text from raw enterprise files."""
    
    @staticmethod
    def parse(file_path: str) -> List[Document]:
        """
        Takes a file path, determines the type, and returns LangChain Documents.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found at: {file_path}")
            
        extension = os.path.splitext(file_path)[1].lower()
        
        print(f"Parsing document: {file_path}")
        
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()
            
        elif extension == ".txt":
            loader = TextLoader(file_path)
            return loader.load()
            
        elif extension == ".docx":
            # For DOCX, you would typically use Docx2txtLoader
            # Requires: pip install docx2txt
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            return loader.load()
            
        else:
            raise ValueError(f"Unsupported file type '{extension}'. MVP supports PDF, TXT, DOCX.")