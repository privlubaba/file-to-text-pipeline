"""
Base classes and shared utilities for file processors.
Contains FileProcessor ABC, TextExtractionResult, QualityMetrics, ArabicTextNormalizer.
"""

import io
import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Language detection (optional)
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class ArabicTextNormalizer:
    """Utilities for normalizing Arabic text"""

    ARABIC_DIACRITICS = [
        '\u064B',  # Fathatan
        '\u064C',  # Dammatan
        '\u064D',  # Kasratan
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u0653',  # Maddah
        '\u0654',  # Hamza above
        '\u0655',  # Hamza below
        '\u0656',  # Subscript alef
        '\u0657',  # Inverted damma
        '\u0658',  # Mark noon ghunna
        '\u0670',  # Superscript alef
    ]

    @staticmethod
    def remove_diacritics(text: str) -> str:
        """Remove Arabic diacritics (harakat) from text"""
        for diacritic in ArabicTextNormalizer.ARABIC_DIACRITICS:
            text = text.replace(diacritic, '')
        return text

    @staticmethod
    def normalize_arabic(text: str) -> str:
        """Normalize Arabic text for better readability"""
        text = ArabicTextNormalizer.remove_diacritics(text)
        text = re.sub('[إأآا]', 'ا', text)
        text = text.replace('ة', 'ه')
        text = re.sub('[ىي]', 'ي', text)
        return text

    @staticmethod
    def clean_spacing(text: str) -> str:
        """Clean up spacing issues in Arabic text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?،؛])', r'\1', text)
        return text.strip()


class QualityMetrics:
    """Quality assessment metrics for extracted content"""

    def __init__(self):
        self.total_text_length = 0
        self.total_tables = 0
        self.total_images = 0
        self.average_confidence = 0.0
        self.language_consistency = 1.0
        self.structure_score = 0.0
        self.completeness_score = 0.0

    def calculate_overall_score(self) -> float:
        """Calculate overall quality score (0-100)"""
        scores = []
        if self.total_text_length > 100:
            scores.append(min(100, (self.total_text_length / 1000) * 50))
        scores.append(self.structure_score)
        scores.append(self.average_confidence)
        scores.append(self.completeness_score)
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.calculate_overall_score(), 2),
            "total_text_length": self.total_text_length,
            "total_tables_extracted": self.total_tables,
            "total_images_processed": self.total_images,
            "average_ocr_confidence": round(self.average_confidence, 2),
            "language_consistency": round(self.language_consistency, 2),
            "structure_preservation_score": round(self.structure_score, 2),
            "completeness_score": round(self.completeness_score, 2),
            "quality_rating": self._get_quality_rating()
        }

    def _get_quality_rating(self) -> str:
        score = self.calculate_overall_score()
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Very Poor"


class TextExtractionResult:
    """Container for text extraction results"""

    def __init__(self, file_id: str, filename: str, file_type: str):
        self.file_id = file_id
        self.filename = filename
        self.file_type = file_type
        self.pages: List[Dict[str, Any]] = []
        self.total_pages = 0
        self.extraction_method = ""
        self.metadata: Dict[str, Any] = {}
        self.quality_metrics = QualityMetrics()
        self.tables: List[Dict[str, Any]] = []
        self.images: List[Dict[str, Any]] = []

    def add_page(self, page_number: int, text: str, method: str, **kwargs):
        """Add a page of extracted text"""
        has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))

        page_data = {
            "page_number": page_number,
            "raw_text": text,
            "char_count": len(text),
            "extraction_method": method,
            **kwargs
        }

        if has_arabic:
            normalized_text = ArabicTextNormalizer.remove_diacritics(text)
            cleaned_text = ArabicTextNormalizer.clean_spacing(normalized_text)
            page_data["normalized_text"] = cleaned_text
            page_data["has_diacritics"] = text != normalized_text
            page_data["language"] = "arabic"

        self.pages.append(page_data)
        self.total_pages = len(self.pages)
        self.quality_metrics.total_text_length += len(text)

    def add_table(self, page_number: int, table_data: List[List[str]], table_index: int = 0):
        """Add extracted table with structure"""
        self.tables.append({
            "page_number": page_number,
            "table_index": table_index,
            "rows": len(table_data),
            "columns": len(table_data[0]) if table_data else 0,
            "data": table_data
        })
        self.quality_metrics.total_tables += 1
        self.quality_metrics.structure_score = min(100, self.quality_metrics.total_tables * 20)

    def add_image(self, page_number: int, image_index: int, image_info: Dict[str, Any]):
        """Add extracted image information"""
        self.images.append({
            "page_number": page_number,
            "image_index": image_index,
            **image_info
        })
        self.quality_metrics.total_images += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "total_pages": self.total_pages,
            "extraction_method": self.extraction_method,
            "metadata": self.metadata,
            "quality_assessment": self.quality_metrics.to_dict(),
            "pages": self.pages,
            "tables": self.tables if self.tables else [],
            "images": self.images if self.images else []
        }


class FileProcessor(ABC):
    """Abstract base class for file processors with shared utilities"""

    @abstractmethod
    def can_process(self, mime_type: str, extension: str) -> bool:
        pass

    @abstractmethod
    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        pass

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language from text. Returns ISO 639-1 code or None."""
        if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 20:
            return None
        try:
            return detect(text)
        except (LangDetectException, Exception):
            return None

    def _has_arabic_diacritics(self, text: str) -> bool:
        """Check if text contains Arabic diacritical marks."""
        return any(d in text for d in ArabicTextNormalizer.ARABIC_DIACRITICS)

    def _detect_encoding(self, file_bytes: bytes) -> str:
        """Detect encoding using charset-normalizer."""
        try:
            from charset_normalizer import from_bytes
            result = from_bytes(file_bytes).best()
            return result.encoding if result else 'utf-8'
        except ImportError:
            logger.warning("charset-normalizer not available, falling back to utf-8")
            return 'utf-8'
        except Exception:
            return 'utf-8'

    def _decode_bytes(self, file_bytes: bytes) -> str:
        """Decode bytes with proper encoding detection."""
        encoding = self._detect_encoding(file_bytes)
        try:
            return file_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            return file_bytes.decode('utf-8', errors='replace')
