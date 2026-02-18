"""
Image Processor — OCR for standalone image files.
Supports JPG, PNG, TIFF (multi-page), BMP, WEBP.
Reuses the shared OCR pipeline with OpenCV preprocessing.
"""

import io
import logging
from typing import List, Dict, Any

import numpy as np
from PIL import Image

from processors.base import FileProcessor, TextExtractionResult

logger = logging.getLogger(__name__)


class ImageProcessor(FileProcessor):
    """Process standalone image files with OCR."""

    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp']
    SUPPORTED_MIMES = [
        'image/jpeg', 'image/png', 'image/tiff', 'image/bmp', 'image/webp',
        'image/gif'
    ]

    def __init__(self):
        self._ocr_pipeline = None

    @property
    def ocr_pipeline(self):
        if self._ocr_pipeline is None:
            from processors.ocr_pipeline import OCRPipeline
            self._ocr_pipeline = OCRPipeline()
        return self._ocr_pipeline

    def can_process(self, mime_type: str, extension: str) -> bool:
        return (mime_type in self.SUPPORTED_MIMES
                or extension.lower() in self.SUPPORTED_EXTENSIONS)

    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        result = TextExtractionResult(file_id, filename, "image")
        result.extraction_method = "tesseract"

        img = Image.open(io.BytesIO(file_bytes))

        # Auto-rotate based on EXIF orientation
        img = self._auto_rotate(img)

        # Handle multi-page images (TIFF)
        pages = self._get_all_frames(img)

        confidences = []

        for page_num, page_img in enumerate(pages, start=1):
            # Convert to RGB numpy array
            if page_img.mode not in ('RGB', 'L'):
                page_img = page_img.convert('RGB')

            img_array = np.array(page_img)

            # Preprocess with OpenCV pipeline
            preprocessed = self.ocr_pipeline.preprocess_image(img_array)

            # Run OCR with zone detection
            text, confidence, zones = self.ocr_pipeline.ocr_image(preprocessed)
            confidences.append(confidence)

            result.add_page(page_num, text, "tesseract",
                            ocr_confidence=round(confidence, 2),
                            zones=zones)

            result.add_image(page_num, 0, {
                "width": page_img.width,
                "height": page_img.height,
                "type": "source_image",
                "mode": page_img.mode,
                "ocr_confidence": round(confidence, 2)
            })

        # Quality metrics
        if confidences:
            result.quality_metrics.average_confidence = sum(confidences) / len(confidences)
        result.quality_metrics.completeness_score = min(
            100, result.quality_metrics.total_text_length / 100
        )

        result.metadata = {
            "image_count": len(pages),
            "format": img.format or "unknown",
            "original_size": f"{img.width}x{img.height}",
            "mode": img.mode
        }

        logger.info(
            f"Extracted text from image: {filename} "
            f"({len(pages)} pages, {result.quality_metrics.total_text_length} chars)"
        )
        return result

    def _auto_rotate(self, img: Image.Image) -> Image.Image:
        """Rotate image based on EXIF orientation tag."""
        try:
            exif = img.getexif()
            orientation = exif.get(274)  # 274 = Orientation tag
            rotations = {
                3: 180,
                6: 270,
                8: 90
            }
            if orientation in rotations:
                img = img.rotate(rotations[orientation], expand=True)
            elif orientation == 2:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 4:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
            elif orientation == 7:
                img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        except Exception:
            pass
        return img

    def _get_all_frames(self, img: Image.Image) -> List[Image.Image]:
        """Extract all frames from a potentially multi-frame image (e.g. TIFF)."""
        frames = []
        try:
            frame_idx = 0
            while True:
                img.seek(frame_idx)
                frames.append(img.copy())
                frame_idx += 1
        except EOFError:
            pass

        if not frames:
            frames = [img]

        return frames
