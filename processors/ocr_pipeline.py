"""
Shared OCR pipeline with OpenCV preprocessing and Tesseract OCR.
Used by PDFProcessor and ImageProcessor.
"""

import re
import logging
import unicodedata
from typing import Tuple, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Arabic Unicode ranges
_ARABIC_RANGE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
# Stray Latin in Arabic context (short Latin words surrounded by Arabic/spaces/punctuation)
_STRAY_LATIN = re.compile(r'(?<=[\u0600-\u06FF\s.،؟!])\s*[A-Za-z]{1,5}\s*(?=[\u0600-\u06FF\s.،؟!])')
# Unicode control/formatting characters that shouldn't appear in output
_CONTROL_CHARS = re.compile(r'[\u200e\u200f\u200b\u200c\u200d\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff]')


class OCRPipeline:
    """OCR pipeline with image preprocessing and Tesseract."""

    def __init__(self, languages: Optional[List[str]] = None):
        self._languages = languages or ['ara', 'eng']

    @property
    def lang_string(self) -> str:
        """Tesseract language string — Arabic first for priority."""
        return '+'.join(self._languages)

    def preprocess_image(self, image: np.ndarray, max_dimension: int = 4000) -> np.ndarray:
        """
        OpenCV preprocessing pipeline for optimal OCR accuracy.
        Steps: resize (if too large) -> grayscale -> deskew -> denoise -> adaptive binarization -> cleanup.
        """
        import cv2

        # Resize if image is too large (major speed improvement)
        h, w = image.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Deskew
        gray = self._deskew(gray)

        # Light denoise (bilateral filter is much faster than fastNlMeans)
        gray = cv2.bilateralFilter(gray, 5, 75, 75)

        # Adaptive binarization (Otsu)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup (remove tiny specks/noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew image using Hough line detection."""
        import cv2

        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

        median_angle = np.median(angles)

        # Only correct if skew is significant but not too extreme
        if abs(median_angle) < 0.5 or abs(median_angle) > 45:
            return image

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def ocr_image(self, image: np.ndarray) -> Tuple[str, float, list]:
        """
        Run OCR on an image using Tesseract with zone-based extraction.
        Returns (text, confidence_percentage, zones).

        Uses --psm 3 (fully automatic page segmentation) so Tesseract
        detects text zones/blocks rather than treating the page as a
        single uniform block.  Each zone gets its own bounding box,
        text, and confidence score.
        """
        import pytesseract
        from PIL import Image

        pil_image = Image.fromarray(image)

        # PSM 3 = fully automatic page segmentation (detects zones)
        config = '--psm 3'

        # Get word-level data with positions and block assignments
        data = pytesseract.image_to_data(pil_image, lang=self.lang_string,
                                         config=config,
                                         output_type=pytesseract.Output.DICT)

        # Group words into spatial zones by block
        zones = self._group_into_zones(data)

        # Build combined text from zones (separated by double newline)
        zone_texts = [z['text'] for z in zones if z['text'].strip()]
        combined_text = '\n\n'.join(zone_texts)

        # Overall confidence = weighted average across all zones
        total_words = sum(z['word_count'] for z in zones if z['word_count'] > 0)
        if total_words > 0:
            avg_confidence = sum(
                z['confidence'] * z['word_count']
                for z in zones if z['word_count'] > 0
            ) / total_words
        else:
            avg_confidence = 0.0

        return combined_text, avg_confidence, zones

    def _group_into_zones(self, data: dict) -> list:
        """
        Group Tesseract image_to_data output into spatial text zones.

        Tesseract assigns each word a block_num, par_num, line_num, word_num.
        We group by block_num to form zones, reconstruct lines within each
        block, compute bounding boxes, and sort zones spatially
        (top-to-bottom, then left-to-right).
        """
        from collections import defaultdict

        # Collect words per block -> paragraph -> line
        blocks = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        block_coords = defaultdict(lambda: {'left': [], 'top': [], 'right': [], 'bottom': [], 'confs': []})

        n = len(data['text'])
        for i in range(n):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            block = data['block_num'][i]
            par = data['par_num'][i]
            line = data['line_num'][i]
            word = data['word_num'][i]

            if not text or conf < 0:
                continue

            blocks[block][par][line].append((word, text))

            # Track bounding box coordinates
            left = data['left'][i]
            top = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            block_coords[block]['left'].append(left)
            block_coords[block]['top'].append(top)
            block_coords[block]['right'].append(left + w)
            block_coords[block]['bottom'].append(top + h)
            if conf > 0:
                block_coords[block]['confs'].append(conf)

        # Build zone list
        zones = []
        for block_num in sorted(blocks.keys()):
            pars = blocks[block_num]
            par_texts = []
            line_count = 0

            for par_num in sorted(pars.keys()):
                lines = pars[par_num]
                for line_num in sorted(lines.keys()):
                    words = lines[line_num]
                    words.sort(key=lambda w: w[0])  # sort by word_num
                    line_text = ' '.join(w[1] for w in words)
                    par_texts.append(line_text)
                    line_count += 1

            zone_text = '\n'.join(par_texts)
            zone_text = self._clean_ocr_text(zone_text)

            if not zone_text.strip():
                continue

            coords = block_coords[block_num]
            confs = coords['confs']

            zone = {
                'zone_id': len(zones) + 1,
                'text': zone_text,
                'bbox': {
                    'x': min(coords['left']) if coords['left'] else 0,
                    'y': min(coords['top']) if coords['top'] else 0,
                    'w': (max(coords['right']) - min(coords['left'])) if coords['left'] else 0,
                    'h': (max(coords['bottom']) - min(coords['top'])) if coords['top'] else 0,
                },
                'confidence': round(sum(confs) / len(confs), 2) if confs else 0.0,
                'line_count': line_count,
                'word_count': len(confs),
            }
            zones.append(zone)

        # Sort zones spatially: top-to-bottom, then left-to-right
        zones.sort(key=lambda z: (z['bbox']['y'], z['bbox']['x']))

        # Re-assign zone_ids after sorting
        for idx, zone in enumerate(zones):
            zone['zone_id'] = idx + 1

        return zones

    def _clean_ocr_text(self, text: str) -> str:
        """Clean up OCR artifacts from Tesseract output."""
        # Remove Unicode control/formatting characters
        text = _CONTROL_CHARS.sub('', text)

        # Detect if text is predominantly Arabic
        arabic_chars = len(_ARABIC_RANGE.findall(text))
        total_alpha = sum(1 for c in text if c.isalpha())

        if total_alpha > 0 and arabic_chars / total_alpha > 0.5:
            # Predominantly Arabic text — clean up stray Latin artifacts
            text = _STRAY_LATIN.sub(' - ', text)

        # Normalize Arabic question marks
        text = text.replace('?', '؟') if arabic_chars > total_alpha * 0.5 else text

        # Clean up excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # Clean up excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip trailing whitespace per line
        text = '\n'.join(line.strip() for line in text.split('\n'))

        return text.strip()

    def is_scanned_page(self, page_text: str, page_image_area_ratio: float) -> bool:
        """
        Per-page scanned detection using text-to-image-area ratio.
        More robust than checking only char count.
        """
        text_density = len(page_text.strip())

        # If very little text and images dominate the page
        if text_density < 50 and page_image_area_ratio > 0.3:
            return True

        # If almost no text at all
        if text_density < 20:
            return True

        return False

    def detect_languages(self, image: np.ndarray) -> List[str]:
        """
        Detect languages present in an image by running a quick OCR pass.
        Returns list of detected language codes.
        """
        try:
            import pytesseract
            from PIL import Image
            from langdetect import detect

            pil_image = Image.fromarray(image)
            text = pytesseract.image_to_string(pil_image, lang=self.lang_string)
            if len(text.strip()) > 20:
                lang = detect(text)
                return [lang]
        except Exception:
            pass
        return ['en']
