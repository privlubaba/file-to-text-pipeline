"""
File Processors Package — State-of-the-art document ingestion pipeline.

Supports 13+ file formats with ML-powered OCR, structural extraction,
and proper encoding detection.
"""

from processors.base import (
    FileProcessor,
    TextExtractionResult,
    QualityMetrics,
    ArabicTextNormalizer,
)
from processors.factory import ProcessorFactory

__all__ = [
    "ProcessorFactory",
    "TextExtractionResult",
    "FileProcessor",
    "QualityMetrics",
    "ArabicTextNormalizer",
]
