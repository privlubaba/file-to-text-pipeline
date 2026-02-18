"""
PDF Processor — State-of-the-art PDF text extraction.
Uses PyMuPDF (fitz) for text/table/image extraction.
Per-page hybrid: text pages via PyMuPDF, scanned pages via Tesseract with OpenCV preprocessing.
Detects garbled font encodings and falls back to OCR automatically.
"""

import io
import re
import logging
import subprocess
from typing import List, Dict, Any, Optional

import numpy as np

from processors.base import FileProcessor, TextExtractionResult

logger = logging.getLogger(__name__)

# Characters that should NOT appear in Arabic/English PDF text.
# Their presence signals broken ToUnicode font mappings.
_GARBLE_CHARS = re.compile(
    r'[\u0370-\u03FF'   # Greek
    r'\u0400-\u04FF'    # Cyrillic
    r'\u0300-\u036F'    # Combining diacritical marks (excessive)
    r'\u2100-\u214F'    # Letterlike symbols
    r'\u0180-\u024F'    # Latin Extended-B
    r'\u1E00-\u1EFF'    # Latin Extended Additional
    r']'
)

# Tesseract language codes (used directly by pytesseract)
TESSERACT_LANGUAGES = [
    'ara', 'eng', 'fra', 'spa', 'deu', 'rus', 'jpn', 'kor',
    'chi_sim', 'chi_tra', 'hin', 'urd', 'tur', 'ita', 'por',
]


class PDFProcessor(FileProcessor):
    """Process PDF files with PyMuPDF + per-page hybrid Tesseract OCR."""

    def __init__(self):
        self._ocr_pipeline = None

    @property
    def ocr_pipeline(self):
        """Lazy-load OCR pipeline only when needed (scanned pages detected)."""
        if self._ocr_pipeline is None:
            from processors.ocr_pipeline import OCRPipeline
            self._ocr_pipeline = OCRPipeline()
        return self._ocr_pipeline

    def can_process(self, mime_type: str, extension: str) -> bool:
        return mime_type == 'application/pdf' or extension.lower() == '.pdf'

    @staticmethod
    def _is_garbled_text(text: str) -> bool:
        """
        Detect if PyMuPDF-extracted text is garbled due to broken font
        encoding (corrupt ToUnicode CMap).

        Garbled Arabic PDFs typically contain Greek/Cyrillic/exotic Latin
        characters that should never appear in Arabic+English documents.
        We flag the text as garbled if more than 5% of alphabetic chars
        are from these unexpected scripts.
        """
        if not text or len(text.strip()) < 20:
            return False

        garble_hits = len(_GARBLE_CHARS.findall(text))
        alpha_count = sum(1 for c in text if c.isalpha())

        if alpha_count == 0:
            return False

        garble_ratio = garble_hits / alpha_count
        if garble_ratio > 0.05:
            logger.info(
                f"Garbled text detected: {garble_hits}/{alpha_count} "
                f"({garble_ratio:.1%}) unexpected characters"
            )
            return True
        return False

    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        import fitz  # PyMuPDF

        result = TextExtractionResult(file_id, filename, "pdf")
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        result.metadata = {
            "page_count": len(doc),
            "pdf_metadata": {
                "title": doc.metadata.get("title"),
                "author": doc.metadata.get("author"),
                "subject": doc.metadata.get("subject"),
                "creator": doc.metadata.get("creator"),
            }
        }

        has_scanned_pages = False
        has_text_pages = False
        confidences = []

        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            page_number = page_num + 1

            # Extract text via PyMuPDF
            text = page.get_text("text")

            # Calculate image area ratio for scanned detection
            image_area_ratio = self._calculate_image_ratio(page)

            # Decide extraction method:
            # 1. Scanned page (little/no embedded text) → OCR
            # 2. Garbled text (broken font encoding) → OCR fallback
            # 3. Clean text → PyMuPDF zone extraction
            is_scanned = self.ocr_pipeline.is_scanned_page(text, image_area_ratio)
            is_garbled = not is_scanned and self._is_garbled_text(text)

            if is_scanned or is_garbled:
                has_scanned_pages = True
                reason = "scanned" if is_scanned else "garbled font encoding, falling back to OCR"
                logger.info(f"Page {page_number}/{total_pages}: {reason}, running OCR...")
                ocr_text, confidence, zones = self._ocr_page(page, page_number, total_pages)
                method = "tesseract_ocr_fallback" if is_garbled else "tesseract"
                result.add_page(page_number, ocr_text, method,
                                ocr_confidence=round(confidence, 2),
                                is_scanned=is_scanned,
                                is_garbled_fallback=is_garbled,
                                zones=zones)
                confidences.append(confidence)
            else:
                has_text_pages = True
                zone_text, zones = self._extract_text_zones(page)
                result.add_page(page_number, zone_text, "pymupdf",
                                is_scanned=False, zones=zones)
                confidences.append(100.0)
                logger.info(f"Page {page_number}/{total_pages}: text extracted ({len(zone_text)} chars, {len(zones)} zones)")

            # Extract tables (PyMuPDF built-in, v1.23+)
            self._extract_tables(page, page_number, result)

            # Extract embedded image metadata
            self._extract_image_info(page, page_number, result)

        # Set extraction method based on what was encountered
        if has_scanned_pages and has_text_pages:
            result.extraction_method = "pymupdf+tesseract_hybrid"
        elif has_scanned_pages:
            result.extraction_method = "tesseract"
        else:
            result.extraction_method = "pymupdf"

        # Update quality metrics
        if confidences:
            result.quality_metrics.average_confidence = sum(confidences) / len(confidences)
        result.quality_metrics.completeness_score = min(
            100, result.quality_metrics.total_text_length / 100
        )

        result.metadata["has_scanned_pages"] = has_scanned_pages
        result.metadata["has_text_pages"] = has_text_pages

        doc.close()

        logger.info(
            f"Extracted {result.total_pages} pages from PDF: {filename} "
            f"({result.quality_metrics.total_text_length} chars, "
            f"method: {result.extraction_method})"
        )
        return result

    def _calculate_image_ratio(self, page) -> float:
        """Calculate the ratio of image area to page area."""
        try:
            page_area = page.rect.width * page.rect.height
            if page_area == 0:
                return 0.0

            images = page.get_images(full=True)
            if not images:
                return 0.0

            # Estimate image area from image dimensions
            # Note: actual rendered area may differ, but this is a good heuristic
            total_image_area = 0
            for img in images:
                xref = img[0]
                try:
                    img_info = page.parent.extract_image(xref)
                    if img_info:
                        total_image_area += img_info.get("width", 0) * img_info.get("height", 0)
                except Exception:
                    # Fallback: assume image covers significant area
                    total_image_area += page_area * 0.3

            return min(1.0, total_image_area / page_area)
        except Exception:
            return 0.0

    def _ocr_page(self, page, page_number: int = 0, total_pages: int = 0) -> tuple:
        """Render a page to image and run OCR with preprocessing."""
        import time

        t0 = time.time()

        # Render at 300 DPI (needed for Arabic text with dots/diacritics)
        pix = page.get_pixmap(dpi=300)
        logger.info(f"  OCR page {page_number}/{total_pages}: rendered {pix.w}x{pix.h}px ({time.time()-t0:.1f}s)")

        # Convert pixmap to numpy array
        if pix.n == 4:  # RGBA
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4)
            img_array = img_array[:, :, :3]
        elif pix.n == 3:  # RGB
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        else:  # Grayscale
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)

        # Preprocess with OpenCV pipeline (will resize if too large)
        t1 = time.time()
        preprocessed = self.ocr_pipeline.preprocess_image(img_array)
        logger.info(f"  OCR page {page_number}/{total_pages}: preprocessed ({time.time()-t1:.1f}s)")

        # Run OCR with zone detection
        t2 = time.time()
        text, confidence, zones = self.ocr_pipeline.ocr_image(preprocessed)
        logger.info(f"  OCR page {page_number}/{total_pages}: OCR done, {len(text)} chars, {confidence:.0f}% conf, {len(zones)} zones ({time.time()-t2:.1f}s)")

        return text, confidence, zones

    def _extract_text_zones(self, page) -> tuple:
        """
        Extract text from a native-text PDF page using zone-based grouping.

        Uses PyMuPDF's get_text("dict") which returns structured blocks
        with bounding boxes, then groups them into spatial zones sorted
        top-to-bottom, left-to-right. This preserves multi-column and
        scattered layouts instead of reading line-by-line across the page.

        Returns (combined_text, zones_list).
        """
        page_dict = page.get_text("dict", sort=True)
        blocks = page_dict.get("blocks", [])

        zones = []
        for block in blocks:
            # Only process text blocks (type 0), skip image blocks (type 1)
            if block.get("type", 0) != 0:
                continue

            lines = block.get("lines", [])
            if not lines:
                continue

            # Reconstruct text from spans within each line
            line_texts = []
            for line in lines:
                spans = line.get("spans", [])
                line_str = "".join(span.get("text", "") for span in spans)
                if line_str.strip():
                    line_texts.append(line_str.strip())

            if not line_texts:
                continue

            zone_text = "\n".join(line_texts)

            bbox = block.get("bbox", (0, 0, 0, 0))  # (x0, y0, x1, y1)
            zone = {
                "zone_id": len(zones) + 1,
                "text": zone_text,
                "bbox": {
                    "x": round(bbox[0], 1),
                    "y": round(bbox[1], 1),
                    "w": round(bbox[2] - bbox[0], 1),
                    "h": round(bbox[3] - bbox[1], 1),
                },
                "confidence": 100.0,
                "line_count": len(line_texts),
                "word_count": sum(len(lt.split()) for lt in line_texts),
            }
            zones.append(zone)

        # Sort zones spatially: top-to-bottom, then left-to-right
        zones.sort(key=lambda z: (z["bbox"]["y"], z["bbox"]["x"]))

        # Re-assign zone_ids after sorting
        for idx, zone in enumerate(zones):
            zone["zone_id"] = idx + 1

        # Combine zone texts with double newline separator
        combined_text = "\n\n".join(z["text"] for z in zones if z["text"].strip())

        return combined_text, zones

    def _extract_tables(self, page, page_number: int, result: TextExtractionResult):
        """Extract tables using PyMuPDF's built-in table finder."""
        try:
            tables = page.find_tables()
            if tables and tables.tables:
                for table_idx, table in enumerate(tables.tables):
                    table_data = table.extract()
                    if table_data:
                        # Clean None values
                        cleaned = []
                        for row in table_data:
                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                            cleaned.append(cleaned_row)
                        result.add_table(page_number, cleaned, table_idx)
        except Exception as e:
            logger.debug(f"Table extraction failed for page {page_number}: {e}")

    def _extract_image_info(self, page, page_number: int, result: TextExtractionResult):
        """Extract metadata about embedded images."""
        try:
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images):
                xref = img[0]
                try:
                    img_info = page.parent.extract_image(xref)
                    result.add_image(page_number, img_idx, {
                        "width": img_info.get("width", 0),
                        "height": img_info.get("height", 0),
                        "colorspace": img_info.get("cs-name", "unknown"),
                        "bpc": img_info.get("bpc", 0),
                        "type": "embedded_image"
                    })
                except Exception:
                    result.add_image(page_number, img_idx, {"type": "embedded_image"})
        except Exception as e:
            logger.debug(f"Image extraction failed for page {page_number}: {e}")
