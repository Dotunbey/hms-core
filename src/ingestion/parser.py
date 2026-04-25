"""
Lightweight document parser using PyMuPDF.
No LLM calls, no heavy ML dependencies.
"""

import re
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger


class DocumentParser:
    """
    Adaptive document parser.
    - Detects numbered/structured documents and splits on paragraph boundaries.
    - Falls back to page-based extraction for unstructured documents.
    """

    # Pattern for numbered paragraphs (e.g., "1.", "E-1", "12:", "§3")
    PARA_PATTERN = re.compile(r'^\s*(E-\d+|\d+|§\d+)(?:\.|:)?\s+')
    # Pattern for section headings (ALL CAPS lines, common heading patterns)
    HEADING_PATTERN = re.compile(r'^(?:SECTION|CHAPTER|ARTICLE|PART)\s+\w+', re.IGNORECASE)

    def parse_document(self, file_path: str, strategy: str = "auto") -> List[Dict[str, Any]]:
        """
        Parses a document into semantic blocks using PyMuPDF.
        Returns a list of dicts with 'type', 'text', and 'metadata'.
        """
        logger.info(f"Parsing document {file_path} using PyMuPDF (strategy: {strategy})")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF (fitz) is not installed. Run: pip install PyMuPDF")
            raise

        try:
            doc = fitz.open(file_path)
            pages_text = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    pages_text.append({"page": page_num + 1, "text": text})
            doc.close()
        except Exception as e:
            logger.error(f"Failed to read PDF {file_path}: {e}")
            raise

        if not pages_text:
            logger.warning(f"No text extracted from {file_path}")
            return []

        full_text = "\n".join([p["text"] for p in pages_text])
        lines = full_text.split('\n')

        # Detect if the document is numbered/structured
        number_matches = sum(1 for line in lines if self.PARA_PATTERN.match(line))
        is_numbered = number_matches > 5

        if is_numbered:
            blocks = self._split_by_paragraphs(lines, file_path)
        else:
            blocks = self._split_by_pages(pages_text, file_path)

        logger.info(f"Extracted {len(blocks)} blocks from {Path(file_path).name} "
                     f"(strategy: {'paragraph' if is_numbered else 'page'})")
        return blocks

    def _split_by_paragraphs(self, lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Split document on numbered paragraph boundaries."""
        blocks = []
        current_para_num = "Intro"
        current_text_buffer = []
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headings
            if self.HEADING_PATTERN.match(line) or (line.isupper() and len(line) > 5 and len(line) < 100):
                # Save current buffer
                if current_text_buffer:
                    text = " ".join(current_text_buffer)
                    if len(text) > 20:
                        blocks.append({
                            "type": "Paragraph",
                            "text": text,
                            "metadata": {
                                "paragraph": current_para_num,
                                "section": current_section,
                                "source": Path(file_path).name
                            }
                        })
                    current_text_buffer = []

                current_section = line
                blocks.append({
                    "type": "Heading",
                    "text": line,
                    "metadata": {"source": Path(file_path).name}
                })
                continue

            # Check for numbered paragraph start
            match = self.PARA_PATTERN.match(line)
            if match:
                # Save previous paragraph
                if current_text_buffer:
                    text = " ".join(current_text_buffer)
                    if len(text) > 20:
                        blocks.append({
                            "type": "Paragraph",
                            "text": text,
                            "metadata": {
                                "paragraph": current_para_num,
                                "section": current_section,
                                "source": Path(file_path).name
                            }
                        })
                # Start new paragraph
                current_para_num = match.group(1)
                current_text_buffer = [line]
            else:
                current_text_buffer.append(line)

        # Save tail
        if current_text_buffer:
            text = " ".join(current_text_buffer)
            if len(text) > 20:
                blocks.append({
                    "type": "Paragraph",
                    "text": text,
                    "metadata": {
                        "paragraph": current_para_num,
                        "section": current_section,
                        "source": Path(file_path).name
                    }
                })

        return blocks

    def _split_by_pages(self, pages_text: List[Dict], file_path: str) -> List[Dict[str, Any]]:
        """Split document by pages, detecting headings within each page."""
        blocks = []

        for page_data in pages_text:
            page_num = page_data["page"]
            text = page_data["text"].strip()

            if not text or len(text) < 20:
                continue

            # Try to detect headings at the top of the page
            lines = text.split('\n')
            heading = None
            body_lines = []

            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if i < 3 and (self.HEADING_PATTERN.match(line_stripped) or
                              (line_stripped.isupper() and 5 < len(line_stripped) < 100)):
                    heading = line_stripped
                    blocks.append({
                        "type": "Heading",
                        "text": heading,
                        "metadata": {"page": page_num, "source": Path(file_path).name}
                    })
                else:
                    body_lines.append(line_stripped)

            body_text = " ".join(body_lines)
            if body_text and len(body_text) > 20:
                blocks.append({
                    "type": "Paragraph",
                    "text": body_text,
                    "metadata": {
                        "page": page_num,
                        "section": heading,
                        "source": Path(file_path).name
                    }
                })

        return blocks
