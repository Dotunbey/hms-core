import os
from typing import List, Dict, Any
from loguru import logger

class DocumentParser:
    """
    Pluggable document parser supporting fallback strategies.
    """
    
    def __init__(self):
        # We can initialize fallback models (like layoutLM) here in the future
        pass

    def parse_document(self, file_path: str, strategy: str = "auto") -> List[Dict[str, Any]]:
        """
        Parses a document into semantic blocks.
        Supported strategies: 'auto', 'unstructured'.
        Returns a list of dicts representing parsed blocks.
        """
        logger.info(f"Parsing document {file_path} using strategy {strategy}")
        
        if strategy in ["auto", "unstructured"]:
            try:
                return self._parse_with_unstructured(file_path)
            except Exception as e:
                logger.error(f"Unstructured parsing failed: {e}")
                logger.warning("No fallback strategy implemented yet. Failing.")
                raise
        
        raise ValueError(f"Unknown parsing strategy: {strategy}")

    def _parse_with_unstructured(self, file_path: str) -> List[Dict[str, Any]]:
        """Uses the 'unstructured' library to extract blocks."""
        try:
            from unstructured.partition.auto import partition
            
            elements = partition(filename=file_path)
            
            blocks = []
            for element in elements:
                # unstructured elements have a 'category' like Title, NarrativeText, Table, etc.
                block = {
                    "type": element.category,
                    "text": element.text,
                    "metadata": element.metadata.to_dict() if hasattr(element, "metadata") and element.metadata else {}
                }
                
                # If it's a table, try to extract HTML representation if available
                if block["type"] == "Table" and hasattr(element.metadata, "text_as_html"):
                    block["html"] = element.metadata.text_as_html
                    
                blocks.append(block)
                
            return blocks
        except ImportError:
            logger.error("The 'unstructured' package is not installed.")
            raise
        except Exception as e:
            logger.error(f"Error partitioning with unstructured: {e}")
            raise
