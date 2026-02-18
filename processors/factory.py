"""
Processor Factory — routes files to the appropriate processor.
"""

import logging
from typing import Optional

from processors.base import FileProcessor
from processors.pdf_processor import PDFProcessor
from processors.docx_processor import DOCXProcessor
from processors.pptx_processor import PPTXProcessor
from processors.xlsx_processor import XLSXProcessor
from processors.image_processor import ImageProcessor
from processors.data_processors import CSVProcessor, JSONProcessor, XMLProcessor
from processors.text_processors import (
    TXTProcessor, HTMLProcessor, MarkdownProcessor, RTFProcessor, ODTProcessor
)

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """Factory for creating appropriate file processors.

    Processor order matters: more specific types come before generic ones.
    TXT is last as a catch-all for plain text files.
    """

    def __init__(self):
        self.processors = [
            PDFProcessor(),
            DOCXProcessor(),
            PPTXProcessor(),
            XLSXProcessor(),
            ImageProcessor(),
            CSVProcessor(),
            JSONProcessor(),
            XMLProcessor(),
            HTMLProcessor(),
            MarkdownProcessor(),
            RTFProcessor(),
            ODTProcessor(),
            TXTProcessor(),  # Last — catch-all for text/plain
        ]

    def get_processor(self, mime_type: str, extension: str) -> Optional[FileProcessor]:
        """Get the appropriate processor for a file type."""
        for processor in self.processors:
            if processor.can_process(mime_type, extension):
                return processor
        return None

    def get_supported_formats(self) -> dict:
        """Return a summary of all supported formats."""
        return {
            "document": ["PDF", "DOCX", "DOC", "PPTX", "XLSX", "XLS"],
            "text": ["TXT", "HTML", "Markdown", "RTF", "ODT"],
            "image": ["JPG", "JPEG", "PNG", "TIFF", "BMP", "WEBP"],
            "data": ["CSV", "JSON", "XML"],
        }
