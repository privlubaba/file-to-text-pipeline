"""
Text-based file processors: TXT, HTML, Markdown, RTF, ODT.
All fixed to use the correct TextExtractionResult pattern.
"""

import io
import re
import logging
from typing import Dict, Any

from processors.base import FileProcessor, TextExtractionResult, ArabicTextNormalizer

logger = logging.getLogger(__name__)


class TXTProcessor(FileProcessor):
    """Process plain text files with proper encoding detection"""

    def can_process(self, mime_type: str, extension: str) -> bool:
        return mime_type == 'text/plain' or extension.lower() == '.txt'

    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        result = TextExtractionResult(file_id, filename, "txt")
        result.extraction_method = "plain_text"

        encoding = self._detect_encoding(file_bytes)
        text = self._decode_bytes(file_bytes)

        result.add_page(1, text, "plain_text")
        result.quality_metrics.average_confidence = 100.0
        result.quality_metrics.completeness_score = min(100, len(text) / 100)

        result.metadata = {
            "encoding": encoding,
            "file_size": len(file_bytes),
            "line_count": text.count('\n') + 1
        }

        logger.info(f"Extracted text from TXT: {filename} ({len(text)} chars, encoding: {encoding})")
        return result


class HTMLProcessor(FileProcessor):
    """Process HTML files with BeautifulSoup or regex fallback"""

    def can_process(self, mime_type: str, extension: str) -> bool:
        return mime_type == 'text/html' or extension.lower() in ['.html', '.htm']

    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        result = TextExtractionResult(file_id, filename, "html")
        result.extraction_method = "html_parser"

        html_content = self._decode_bytes(file_bytes)

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract title
            title = soup.title.string if soup.title else None

            # Remove non-content elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Extract tables as structured data
            for table_idx, table in enumerate(soup.find_all('table')):
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                if rows:
                    result.add_table(1, rows, table_idx)

            # Extract links
            links = []
            for a in soup.find_all('a', href=True):
                link_text = a.get_text(strip=True)
                if link_text:
                    links.append({"text": link_text, "href": a['href']})

            text = soup.get_text(separator='\n')
        except ImportError:
            # Regex fallback
            text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '', text)
            title = None
            links = []

        # Clean whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text).strip()

        result.add_page(1, text, "html_parser")
        result.quality_metrics.average_confidence = 100.0
        result.quality_metrics.completeness_score = min(100, len(text) / 100)

        result.metadata = {
            "file_size": len(file_bytes),
            "title": title,
            "link_count": len(links) if links else 0
        }

        logger.info(f"Extracted text from HTML: {filename} ({len(text)} chars)")
        return result


class MarkdownProcessor(FileProcessor):
    """Process Markdown files preserving structure"""

    def can_process(self, mime_type: str, extension: str) -> bool:
        return mime_type == 'text/markdown' or extension.lower() in ['.md', '.markdown']

    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        result = TextExtractionResult(file_id, filename, "markdown")
        result.extraction_method = "markdown_parser"

        text = self._decode_bytes(file_bytes)
        raw_text = text  # Keep original for reference

        # Strip markdown formatting
        # Remove headers markers but keep text
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove images
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # Remove horizontal rules
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\*\*\*+$', '', text, flags=re.MULTILINE)

        text = text.strip()

        result.add_page(1, text, "markdown_parser")
        result.quality_metrics.average_confidence = 100.0
        result.quality_metrics.completeness_score = min(100, len(text) / 100)

        result.metadata = {
            "file_size": len(file_bytes),
            "heading_count": len(re.findall(r'^#{1,6}\s+', raw_text, flags=re.MULTILINE))
        }

        logger.info(f"Extracted text from Markdown: {filename} ({len(text)} chars)")
        return result


class RTFProcessor(FileProcessor):
    """Process RTF (Rich Text Format) files"""

    def can_process(self, mime_type: str, extension: str) -> bool:
        return mime_type in ['text/rtf', 'application/rtf'] or extension.lower() == '.rtf'

    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        result = TextExtractionResult(file_id, filename, "rtf")
        result.extraction_method = "rtf_parser"

        rtf_content = self._decode_bytes(file_bytes)

        try:
            from striprtf.striprtf import rtf_to_text
            text = rtf_to_text(rtf_content)
        except ImportError:
            # Basic RTF parsing fallback
            text = re.sub(r'\\[a-z]+\d*\s?', '', rtf_content)
            text = re.sub(r'[{}]', '', text)
            text = text.strip()
            logger.warning("striprtf not available, using basic RTF fallback")

        result.add_page(1, text, "rtf_parser")
        result.quality_metrics.average_confidence = 100.0
        result.quality_metrics.completeness_score = min(100, len(text) / 100)

        result.metadata = {"file_size": len(file_bytes)}

        logger.info(f"Extracted text from RTF: {filename} ({len(text)} chars)")
        return result


class ODTProcessor(FileProcessor):
    """Process ODT (OpenDocument Text) files"""

    def can_process(self, mime_type: str, extension: str) -> bool:
        return mime_type == 'application/vnd.oasis.opendocument.text' or extension.lower() == '.odt'

    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        result = TextExtractionResult(file_id, filename, "odt")
        result.extraction_method = "odt_parser"

        import zipfile
        from xml.etree import ElementTree as ET

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as odt_file:
                content_xml = odt_file.read('content.xml')
                root = ET.fromstring(content_xml)

                # Define ODF namespaces
                ns = {
                    'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0',
                    'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0',
                }

                # Extract paragraphs
                text_parts = []
                for para in root.iter('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p'):
                    para_text = ''.join(para.itertext())
                    if para_text.strip():
                        text_parts.append(para_text)

                # Extract headings
                for heading in root.iter('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}h'):
                    heading_text = ''.join(heading.itertext())
                    level = heading.get('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}outline-level', '1')
                    if heading_text.strip():
                        text_parts.append(f"{'#' * int(level)} {heading_text}")

                # Extract tables
                for table_idx, table_elem in enumerate(
                    root.iter('{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table')
                ):
                    rows = []
                    for row_elem in table_elem.iter(
                        '{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-row'
                    ):
                        cells = []
                        for cell_elem in row_elem.iter(
                            '{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-cell'
                        ):
                            cell_text = ''.join(cell_elem.itertext()).strip()
                            cells.append(cell_text)
                        if cells:
                            rows.append(cells)
                    if rows:
                        result.add_table(1, rows, table_idx)
                        table_text = "\n[TABLE]\n" + "\n".join(" | ".join(row) for row in rows)
                        text_parts.append(table_text)

                text = '\n'.join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting ODT: {e}")
            text = f"Error extracting ODT content: {str(e)}"

        result.add_page(1, text, "odt_parser")
        result.quality_metrics.average_confidence = 100.0
        result.quality_metrics.completeness_score = min(100, len(text) / 100)
        result.metadata = {"file_size": len(file_bytes)}

        logger.info(f"Extracted text from ODT: {filename} ({len(text)} chars)")
        return result
