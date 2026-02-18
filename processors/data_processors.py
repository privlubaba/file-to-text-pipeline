"""
Data format processors: CSV, JSON, XML.
Structured extraction with encoding detection and format-aware parsing.
"""

import io
import csv
import json
import logging
from typing import List, Dict, Any

from processors.base import FileProcessor, TextExtractionResult

logger = logging.getLogger(__name__)


class CSVProcessor(FileProcessor):
    """Process CSV files with encoding detection and delimiter sniffing."""

    def can_process(self, mime_type: str, extension: str) -> bool:
        return mime_type in ['text/csv', 'application/csv'] or extension.lower() == '.csv'

    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        result = TextExtractionResult(file_id, filename, "csv")
        result.extraction_method = "csv_parser"

        # Detect encoding
        text = self._decode_bytes(file_bytes)

        # Sniff delimiter
        delimiter = ','
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(text[:8192])
            delimiter = dialect.delimiter
        except csv.Error:
            # Try common delimiters
            for d in [',', '\t', ';', '|']:
                if d in text[:1000]:
                    delimiter = d
                    break

        # Parse CSV — normalize line endings to avoid "new-line character seen in unquoted field"
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = []
        for row in reader:
            rows.append(row)

        # Detect if first row is a header
        has_header = False
        if len(rows) > 1:
            try:
                has_header = csv.Sniffer().has_header(text[:8192])
            except csv.Error:
                pass

        # Create text representation
        if rows:
            # Calculate column widths for alignment
            num_cols = max(len(row) for row in rows) if rows else 0
            col_widths = [0] * num_cols
            for row in rows[:50]:  # Sample first 50 rows for widths
                for i, cell in enumerate(row):
                    if i < num_cols:
                        col_widths[i] = max(col_widths[i], min(len(cell), 30))
            col_widths = [max(w, 3) for w in col_widths]

            lines = []
            for row_idx, row in enumerate(rows):
                formatted_cells = []
                for i, cell in enumerate(row):
                    width = col_widths[i] if i < len(col_widths) else 10
                    formatted_cells.append(cell.ljust(width)[:width])
                lines.append(" | ".join(formatted_cells))

                # Add separator after header
                if row_idx == 0 and has_header:
                    lines.append("-+-".join("-" * w for w in col_widths[:len(row)]))

            csv_text = "\n".join(lines)
        else:
            csv_text = "(empty CSV file)"

        result.add_page(1, csv_text, "csv_parser")

        # Add as structured table
        if rows:
            result.add_table(1, rows, 0)

        result.quality_metrics.average_confidence = 100.0
        result.quality_metrics.completeness_score = min(100, len(csv_text) / 100)

        result.metadata = {
            "row_count": len(rows),
            "column_count": max(len(row) for row in rows) if rows else 0,
            "delimiter": repr(delimiter),
            "has_header": has_header,
            "file_size": len(file_bytes)
        }

        logger.info(f"Extracted text from CSV: {filename} ({len(rows)} rows, delimiter={repr(delimiter)})")
        return result


class JSONProcessor(FileProcessor):
    """Process JSON files with structure preservation and flattened representation."""

    def can_process(self, mime_type: str, extension: str) -> bool:
        return mime_type == 'application/json' or extension.lower() == '.json'

    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        result = TextExtractionResult(file_id, filename, "json")
        result.extraction_method = "json_parser"

        text = self._decode_bytes(file_bytes)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Try to salvage — maybe it's JSONL (one JSON per line)
            lines = text.strip().split('\n')
            data = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if not data:
                result.add_page(1, f"(Invalid JSON: {e})", "json_parser")
                return result

        # Pretty-printed structure
        pretty_text = json.dumps(data, indent=2, ensure_ascii=False, default=str)

        # Flattened key-path representation for better RAG chunking
        flat_lines = self._flatten_json(data)
        flat_text = "\n".join(flat_lines)

        combined = f"[JSON STRUCTURE]\n{pretty_text}"
        if flat_lines:
            combined += f"\n\n[FLATTENED CONTENT]\n{flat_text}"

        result.add_page(1, combined, "json_parser")

        # If it's an array of objects, also add as table
        if isinstance(data, list) and data and isinstance(data[0], dict):
            headers = list(data[0].keys())
            table_data = [headers]
            for item in data[:1000]:  # Cap at 1000 rows
                row = [str(item.get(h, "")) for h in headers]
                table_data.append(row)
            result.add_table(1, table_data, 0)

        result.quality_metrics.average_confidence = 100.0
        result.quality_metrics.completeness_score = min(100, len(combined) / 100)

        result.metadata = {
            "type": type(data).__name__,
            "file_size": len(file_bytes),
            "element_count": len(data) if isinstance(data, (list, dict)) else 1
        }

        logger.info(f"Extracted text from JSON: {filename} ({len(combined)} chars)")
        return result

    def _flatten_json(self, data, prefix: str = "", max_depth: int = 10) -> List[str]:
        """Flatten nested JSON to key-path: value lines."""
        if max_depth <= 0:
            return [f"{prefix}: (max depth reached)"]

        lines = []
        if isinstance(data, dict):
            for k, v in data.items():
                path = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, (dict, list)):
                    lines.extend(self._flatten_json(v, path, max_depth - 1))
                else:
                    lines.append(f"{path}: {v}")
        elif isinstance(data, list):
            for i, item in enumerate(data[:100]):  # Cap array expansion
                path = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    lines.extend(self._flatten_json(item, path, max_depth - 1))
                else:
                    lines.append(f"{path}: {item}")
            if len(data) > 100:
                lines.append(f"{prefix}[...]: ({len(data) - 100} more items)")
        else:
            lines.append(f"{prefix}: {data}")
        return lines


class XMLProcessor(FileProcessor):
    """Process XML files with namespace-aware parsing."""

    def can_process(self, mime_type: str, extension: str) -> bool:
        return (mime_type in ['application/xml', 'text/xml']
                or extension.lower() == '.xml')

    def extract_text(self, file_bytes: bytes, filename: str, file_id: str) -> TextExtractionResult:
        from xml.etree import ElementTree as ET

        result = TextExtractionResult(file_id, filename, "xml")
        result.extraction_method = "xml_parser"

        text = self._decode_bytes(file_bytes)

        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            result.add_page(1, f"(Invalid XML: {e})", "xml_parser")
            return result

        # Extract text content with tag context
        text_parts = []
        self._walk_xml(root, text_parts)

        content = "\n".join(text_parts)

        # Also extract any table-like structures (repeated sibling elements)
        tables = self._detect_xml_tables(root)
        for table_idx, table_data in enumerate(tables):
            result.add_table(1, table_data, table_idx)

        result.add_page(1, content, "xml_parser")
        result.quality_metrics.average_confidence = 100.0
        result.quality_metrics.completeness_score = min(100, len(content) / 100)

        # Strip namespace from root tag for metadata
        root_tag = root.tag.split('}')[-1] if '}' in root.tag else root.tag
        result.metadata = {
            "root_tag": root_tag,
            "element_count": len(list(root.iter())),
            "file_size": len(file_bytes),
            "has_namespaces": '{' in root.tag
        }

        logger.info(f"Extracted text from XML: {filename} ({len(content)} chars, root: {root_tag})")
        return result

    def _walk_xml(self, element, text_parts: List[str], depth: int = 0):
        """Walk the XML tree extracting text with tag context."""
        # Strip namespace
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

        if element.text and element.text.strip():
            indent = "  " * depth
            text_parts.append(f"{indent}[{tag}]: {element.text.strip()}")

        for child in element:
            self._walk_xml(child, text_parts, depth + 1)

        if element.tail and element.tail.strip():
            indent = "  " * depth
            text_parts.append(f"{indent}{element.tail.strip()}")

    def _detect_xml_tables(self, root) -> List[List[List[str]]]:
        """
        Detect table-like structures in XML:
        repeated sibling elements with the same tag that have child elements.
        """
        tables = []

        for elem in root.iter():
            children = list(elem)
            if len(children) < 2:
                continue

            # Check if children have the same tag (repeated rows)
            child_tags = [c.tag for c in children]
            if len(set(child_tags)) == 1 and len(child_tags) >= 2:
                # Looks like a table — each child is a row
                first_child = children[0]
                sub_children = list(first_child)
                if not sub_children:
                    continue

                # Extract headers from first row's child tags
                headers = [
                    c.tag.split('}')[-1] if '}' in c.tag else c.tag
                    for c in sub_children
                ]

                table_data = [headers]
                for child in children:
                    row = []
                    for sub in child:
                        row.append(sub.text.strip() if sub.text else "")
                    if row:
                        table_data.append(row)

                if len(table_data) > 1:
                    tables.append(table_data)

        return tables
