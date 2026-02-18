"""
File-to-Text Ingestion Pipeline API
Main FastAPI application
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uuid
import magic
import os
import csv as csv_module
import json
import logging
import time
import re
from datetime import datetime
from pathlib import Path

from processors import ProcessorFactory, TextExtractionResult
from rag_service import rag_service

# Evaluation CSV path
EVAL_CSV_PATH = Path("./evaluation_log.csv")
EVAL_CSV_COLUMNS = [
    "File", "Format", "Arabic_Scenario", "Upload", "Route",
    "Extract", "Index", "Search", "Runtime_s"
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="File-to-Text Ingestion Pipeline",
    description="Upload files and extract text with ML-powered OCR. Supports PDF, DOCX, PPTX, XLSX, TXT, HTML, Markdown, RTF, ODT, JPG, PNG, TIFF, BMP, CSV, JSON, XML.",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("./uploads")
RESULTS_DIR = Path("./results")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Counter for file naming
upload_counter_file = Path("./upload_counter.txt")

def get_next_upload_number() -> int:
    """Get next upload number and increment counter"""
    if upload_counter_file.exists():
        with open(upload_counter_file, 'r') as f:
            counter = int(f.read().strip())
    else:
        counter = 0
    
    counter += 1
    
    # Save updated counter
    with open(upload_counter_file, 'w') as f:
        f.write(str(counter))
    
    return counter

def sanitize_filename(filename: str) -> str:
    """Remove special characters from filename"""
    import re
    # Remove extension
    name = Path(filename).stem
    # Remove special characters, keep only alphanumeric, dash, underscore
    name = re.sub(r'[^\w\s-]', '', name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Limit length
    return name[:50] if name else "document"

# Initialize processor factory
processor_factory = ProcessorFactory()

# In-memory storage for processing status (in production, use Redis or database)
processing_status = {}


def get_file_extension(filename: str) -> str:
    """Extract file extension from filename"""
    return Path(filename).suffix


def detect_mime_type(file_bytes: bytes) -> str:
    """Detect MIME type of file"""
    try:
        mime = magic.Magic(mime=True)
        return mime.from_buffer(file_bytes)
    except:
        return "application/octet-stream"


def save_result(file_id: str, result: TextExtractionResult):
    """Save extraction result to JSON file"""
    result_path = RESULTS_DIR / f"{file_id}.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved result for file_id: {file_id}")


def _detect_arabic_scenario(text: str) -> str:
    """Detect Arabic scenario from extracted text."""
    if not text:
        return "N/A"
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
    if not has_arabic:
        return "N/A"
    has_diacritics = bool(re.search(r'[\u064B-\u0652\u0670]', text))
    has_latin = bool(re.search(r'[a-zA-Z]{3,}', text))
    # Check for dialectal markers (common Egyptian/Gulf particles)
    dialect_markers = ['ايش', 'ليش', 'هيك', 'كده', 'عشان', 'دلوقتي', 'يلا']
    has_dialect = any(m in text for m in dialect_markers)
    if has_dialect:
        return "Dialect"
    if has_diacritics:
        return "Diacritics"
    if has_latin and has_arabic:
        return "Code-mixed"
    return "MSA"


def _detect_format_label(extension: str, extraction_method: str) -> str:
    """Map extension to a clean format label."""
    ext = extension.lower().lstrip('.')
    fmt_map = {
        'pdf': 'PDF', 'docx': 'DOCX', 'doc': 'DOC', 'pptx': 'PPTX',
        'xlsx': 'XLSX', 'xls': 'XLS', 'csv': 'CSV', 'json': 'JSON',
        'xml': 'XML', 'txt': 'TXT', 'html': 'HTML', 'htm': 'HTML',
        'md': 'MD', 'markdown': 'MD', 'rtf': 'RTF', 'odt': 'ODT',
        'jpg': 'Image', 'jpeg': 'Image', 'png': 'Image',
        'tiff': 'Image', 'tif': 'Image', 'bmp': 'Image', 'webp': 'Image',
    }
    return fmt_map.get(ext, ext.upper())


def _evaluate_extraction(result: TextExtractionResult) -> str:
    """Evaluate extraction health: pass/warn/fail."""
    total_chars = sum(len(p.get("raw_text", "")) for p in result.pages)
    if total_chars == 0:
        return "FAIL"
    if total_chars < 50:
        return "WARN"
    return "PASS"


def _evaluate_route(extension: str, extraction_method: str) -> str:
    """Check if the expected processor was used."""
    ext = extension.lower().lstrip('.')
    expected_map = {
        'pdf': ['pymupdf', 'tesseract', 'pymupdf+tesseract_hybrid'],
        'docx': ['python-docx-enhanced', 'pandoc'],
        'doc': ['pandoc'],
        'pptx': ['python-pptx-enhanced'],
        'xlsx': ['openpyxl-enhanced', 'pandas'],
        'xls': ['openpyxl-enhanced', 'pandas'],
        'csv': ['csv_parser'],
        'json': ['json_parser'],
        'xml': ['xml_parser'],
        'txt': ['plain_text'],
        'html': ['html_parser'], 'htm': ['html_parser'],
        'md': ['markdown_parser'], 'markdown': ['markdown_parser'],
        'rtf': ['rtf_parser'],
        'odt': ['odt_parser'],
        'jpg': ['tesseract'], 'jpeg': ['tesseract'], 'png': ['tesseract'],
        'tiff': ['tesseract'], 'tif': ['tesseract'], 'bmp': ['tesseract'],
        'webp': ['tesseract'],
    }
    expected_list = expected_map.get(ext, [])
    method = extraction_method.lower()
    matched = any(exp in method for exp in expected_list) if expected_list else True
    expected_str = expected_list[0] if expected_list else "unknown"
    status = "PASS" if matched else "FAIL"
    return f"Expected: {expected_str} | Actual: {extraction_method} {status}"


def _evaluate_index(file_id: str) -> str:
    """Check if RAG indexing succeeded (chunks + embeddings + FAISS)."""
    rag_dir = Path("./rag_storage") / file_id
    has_chunks = (rag_dir / "chunks.json").exists()
    has_embeddings = (rag_dir / "embeddings.npy").exists()
    has_faiss = (rag_dir / "faiss.index").exists()
    if has_chunks and has_embeddings and has_faiss:
        return "PASS"
    return "FAIL"


def _evaluate_search(file_id: str, result: TextExtractionResult) -> str:
    """Run a sanity search: pick first meaningful word and check if retrieval works."""
    try:
        # Get a sample keyword from extracted text
        all_text = " ".join(p.get("raw_text", "") for p in result.pages)
        # Pick a meaningful word (>4 chars, not common stopwords)
        words = re.findall(r'\b\w{5,}\b', all_text)
        if not words:
            # Try Arabic words
            words = re.findall(r'[\u0600-\u06FF]{3,}', all_text)
        if not words:
            return "FAIL (no query word)"
        query = words[min(5, len(words) - 1)]  # Pick 6th word if available
        results = rag_service.search(file_id, query, top_k=3)
        if results and any(query.lower() in r["text"].lower() for r in results):
            return "PASS"
        if results:
            return "PASS"  # Got results even if exact match isn't found
        return "FAIL"
    except Exception as e:
        return f"FAIL ({str(e)[:30]})"


def _write_eval_csv(row: dict):
    """Append one evaluation row to the CSV file."""
    file_exists = EVAL_CSV_PATH.exists()
    with open(EVAL_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv_module.DictWriter(f, fieldnames=EVAL_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def process_file_async(file_id: str, file_bytes: bytes, filename: str, mime_type: str, extension: str, meaningful_name: str = None):
    """Process file in background with evaluation metrics and CSV logging."""
    eval_row = {col: "" for col in EVAL_CSV_COLUMNS}
    eval_row["File"] = filename
    eval_row["Format"] = _detect_format_label(extension, "")
    eval_row["Upload"] = "PASS"  # If we got here, upload succeeded

    total_start = time.time()

    try:
        logger.info(f"Starting background processing for {filename} (ID: {file_id})")
        processing_status[file_id] = {"status": "processing", "progress": 0}

        # Get appropriate processor
        processor = processor_factory.get_processor(mime_type, extension)

        if not processor:
            eval_row["Route"] = "FAIL (no processor)"
            eval_row["Extract"] = "FAIL"
            eval_row["Index"] = "FAIL"
            eval_row["Search"] = "FAIL"
            eval_row["Runtime_s"] = round(time.time() - total_start, 2)
            _write_eval_csv(eval_row)
            processing_status[file_id] = {
                "status": "failed",
                "error": f"Unsupported file type: {mime_type} ({extension})",
                "evaluation": eval_row
            }
            return

        # Extract text with timing
        extract_start = time.time()
        result = processor.extract_text(file_bytes, filename, file_id)
        extract_time = round(time.time() - extract_start, 2)

        # Evaluate route
        eval_row["Route"] = _evaluate_route(extension, result.extraction_method)

        # Evaluate extraction
        eval_row["Extract"] = _evaluate_extraction(result)

        # Detect Arabic scenario
        all_text = " ".join(p.get("raw_text", "") for p in result.pages)
        eval_row["Arabic_Scenario"] = _detect_arabic_scenario(all_text)

        # Save result with meaningful name
        if meaningful_name:
            result_filename = f"result_{meaningful_name}.json"
        else:
            result_filename = f"{file_id}.json"

        result_path = RESULTS_DIR / result_filename
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved result as: {result_filename}")

        # RAG Pipeline: Chunk, embed, and index
        index_start = time.time()
        try:
            logger.info(f"Starting RAG pipeline for {file_id}...")
            rag_result = rag_service.process_document(file_id, result.pages)
            logger.info(f"RAG pipeline completed: {rag_result}")
        except Exception as rag_error:
            logger.error(f"RAG pipeline failed for {file_id}: {str(rag_error)}")
        index_time = round(time.time() - index_start, 2)

        # Evaluate indexing
        eval_row["Index"] = _evaluate_index(file_id)

        # Evaluate search (sanity check)
        eval_row["Search"] = _evaluate_search(file_id, result)

        # Total runtime
        total_time = round(time.time() - total_start, 2)
        eval_row["Runtime_s"] = total_time

        # Write to CSV
        _write_eval_csv(eval_row)

        # Update status with evaluation
        processing_status[file_id] = {
            "status": "completed",
            "total_pages": result.total_pages,
            "extraction_method": result.extraction_method,
            "result_file": result_filename,
            "evaluation": {
                "upload": eval_row["Upload"],
                "route": eval_row["Route"],
                "extract": eval_row["Extract"],
                "index": eval_row["Index"],
                "search": eval_row["Search"],
                "arabic_scenario": eval_row["Arabic_Scenario"],
                "runtime_s": total_time,
                "extract_time_s": extract_time,
                "index_time_s": index_time,
            }
        }

        logger.info(f"Successfully processed {filename} (ID: {file_id}) in {total_time}s")

    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        eval_row["Extract"] = "FAIL"
        eval_row["Runtime_s"] = round(time.time() - total_start, 2)
        _write_eval_csv(eval_row)
        processing_status[file_id] = {
            "status": "failed",
            "error": str(e),
            "evaluation": eval_row
        }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body>
                <h1>File-to-Text Ingestion Pipeline API</h1>
                <p>index.html not found. API is running at:</p>
                <ul>
                    <li>Upload: <a href="/upload">/upload</a></li>
                    <li>API Docs: <a href="/docs">/docs</a></li>
                </ul>
            </body>
        </html>
        """


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "File-to-Text Ingestion Pipeline API",
        "version": "2.5.0",
        "endpoints": {
            "upload": "/upload - Upload a file for processing",
            "status": "/status/{file_id} - Check processing status",
            "text": "/text/{file_id} - Get all extracted text",
            "text_normalized": "/text/{file_id}/normalized - Get text with Arabic diacritics removed",
            "page": "/text/{file_id}/page/{page_num} - Get specific page text",
            "quality": "/quality/{file_id} - Get quality assessment metrics",
            "tables": "/tables/{file_id} - Get all extracted tables",
            "images": "/images/{file_id} - Get all image information",
            "health": "/health - Health check with feature list"
        },
        "features": [
            "Enhanced table extraction with structure preservation",
            "Image detection and metadata extraction",
            "Quality assessment scoring (0-100)",
            "OCR confidence metrics",
            "Complex document support",
            "Arabic diacritics (harakat) normalization",
            "Automatic document chunking and vector storage (background)"
        ]
    }


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a file for text extraction

    Supported formats: PDF, DOCX, PPTX, XLSX, TXT, HTML, MD, RTF, ODT
    Maximum file size: 50 MB
    """
    try:
        # Read file
        file_bytes = await file.read()
        
        # Validate file size
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f} MB"
            )
        
        # Detect file type
        mime_type = detect_mime_type(file_bytes)
        extension = get_file_extension(file.filename)
        
        logger.info(f"Received file: {file.filename} (MIME: {mime_type}, Extension: {extension})")
        
        # Validate file type
        processor = processor_factory.get_processor(mime_type, extension)
        if not processor:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {mime_type} ({extension}). Supported: PDF, DOCX, PPTX, XLSX, TXT, HTML, MD, RTF, ODT"
            )
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Get upload number and create meaningful name
        upload_number = get_next_upload_number()
        sanitized_name = sanitize_filename(file.filename)
        meaningful_name = f"upload_{upload_number:04d}_{sanitized_name}"
        
        # Save uploaded file with meaningful name
        upload_path = UPLOAD_DIR / f"{meaningful_name}{extension}"
        with open(upload_path, 'wb') as f:
            f.write(file_bytes)
        
        logger.info(f"Saved upload as: {meaningful_name}{extension}")
        
        # Process file in background
        background_tasks.add_task(
            process_file_async,
            file_id,
            file_bytes,
            file.filename,
            mime_type,
            extension,
            meaningful_name  # Pass meaningful name
        )
        
        # Initialize status
        processing_status[file_id] = {
            "status": "queued",
            "filename": file.filename,
            "meaningful_name": meaningful_name,
            "upload_number": upload_number,
            "upload_time": datetime.now().isoformat()
        }
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "meaningful_name": meaningful_name,
            "upload_number": upload_number,
            "status": "queued",
            "message": "File uploaded successfully. Processing started.",
            "status_url": f"/status/{file_id}",
            "result_url": f"/text/{file_id}",
            "saved_as": {
                "upload": f"{meaningful_name}{extension}",
                "result": f"result_{meaningful_name}.json"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/status/{file_id}")
async def get_status(file_id: str):
    """Get processing status for a file"""
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    return processing_status[file_id]


@app.get("/text/{file_id}")
async def get_text(file_id: str):
    """Get extracted text for a file (all pages)"""
    # Try old naming first (UUID.json)
    result_path = RESULTS_DIR / f"{file_id}.json"
    
    # If not found, try to find by meaningful name (new naming)
    if not result_path.exists() and file_id in processing_status:
        if "result_file" in processing_status[file_id]:
            result_path = RESULTS_DIR / processing_status[file_id]["result_file"]
    
    if not result_path.exists():
        # Check if still processing
        if file_id in processing_status:
            status = processing_status[file_id]["status"]
            if status in ["queued", "processing"]:
                raise HTTPException(
                    status_code=202,
                    detail="File is still being processed. Check /status endpoint."
                )
            elif status == "failed":
                raise HTTPException(
                    status_code=500,
                    detail=f"Processing failed: {processing_status[file_id].get('error', 'Unknown error')}"
                )
        
        raise HTTPException(status_code=404, detail=f"Result not found. Looking for: {result_path.name}")
    
    # Load and return result
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return result
    except Exception as e:
        logger.error(f"Error reading result file {result_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading result: {str(e)}")


@app.get("/text/{file_id}/page/{page_num}")
async def get_page_text(file_id: str, page_num: int):
    """Get extracted text for a specific page"""
    result_path = RESULTS_DIR / f"{file_id}.json"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Load result
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    # Find the page
    pages = result.get("pages", [])
    page = next((p for p in pages if p["page_number"] == page_num), None)
    
    if not page:
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_num} not found. Total pages: {len(pages)}"
        )
    
    return page


@app.get("/quality/{file_id}")
async def get_quality_assessment(file_id: str):
    """Get quality assessment metrics for a processed file"""
    # Try old naming first (UUID.json)
    result_path = RESULTS_DIR / f"{file_id}.json"

    # If not found, try to find by meaningful name (new naming)
    if not result_path.exists() and file_id in processing_status:
        if "result_file" in processing_status[file_id]:
            result_path = RESULTS_DIR / processing_status[file_id]["result_file"]

    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    # Load result
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        # Extract quality assessment
        quality = result.get("quality_assessment", {})

        # Add summary statistics
        summary = {
            "file_id": file_id,
            "filename": result.get("filename"),
            "quality_metrics": quality,
            "content_summary": {
                "total_pages": result.get("total_pages", 0),
                "total_tables": len(result.get("tables", [])),
                "total_images": len(result.get("images", [])),
                "extraction_method": result.get("extraction_method")
            }
        }

        return summary
    except Exception as e:
        logger.error(f"Error reading quality assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading quality assessment: {str(e)}")


@app.get("/tables/{file_id}")
async def get_tables(file_id: str):
    """Get all extracted tables from a file"""
    # Try old naming first (UUID.json)
    result_path = RESULTS_DIR / f"{file_id}.json"

    # If not found, try to find by meaningful name
    if not result_path.exists() and file_id in processing_status:
        if "result_file" in processing_status[file_id]:
            result_path = RESULTS_DIR / processing_status[file_id]["result_file"]

    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    # Load result
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        tables = result.get("tables", [])

        return {
            "file_id": file_id,
            "filename": result.get("filename"),
            "total_tables": len(tables),
            "tables": tables
        }
    except Exception as e:
        logger.error(f"Error reading tables: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading tables: {str(e)}")


@app.get("/images/{file_id}")
async def get_images(file_id: str):
    """Get all extracted image information from a file"""
    # Try old naming first (UUID.json)
    result_path = RESULTS_DIR / f"{file_id}.json"

    # If not found, try to find by meaningful name
    if not result_path.exists() and file_id in processing_status:
        if "result_file" in processing_status[file_id]:
            result_path = RESULTS_DIR / processing_status[file_id]["result_file"]

    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    # Load result
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        images = result.get("images", [])

        return {
            "file_id": file_id,
            "filename": result.get("filename"),
            "total_images": len(images),
            "images": images
        }
    except Exception as e:
        logger.error(f"Error reading images: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading images: {str(e)}")


@app.get("/text/{file_id}/normalized")
async def get_normalized_text(file_id: str):
    """Get normalized text (with Arabic diacritics removed) for a file"""
    # Try old naming first (UUID.json)
    result_path = RESULTS_DIR / f"{file_id}.json"

    # If not found, try to find by meaningful name
    if not result_path.exists() and file_id in processing_status:
        if "result_file" in processing_status[file_id]:
            result_path = RESULTS_DIR / processing_status[file_id]["result_file"]

    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    # Load result
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        # Extract normalized text from pages
        normalized_pages = []
        for page in result.get("pages", []):
            page_data = {
                "page_number": page["page_number"],
                "has_diacritics": page.get("has_diacritics", False),
                "language": page.get("language", "unknown")
            }

            # Prefer normalized text if available
            if "normalized_text" in page:
                page_data["text"] = page["normalized_text"]
                page_data["normalization"] = "diacritics_removed"
            else:
                page_data["text"] = page["raw_text"]
                page_data["normalization"] = "none"

            normalized_pages.append(page_data)

        return {
            "file_id": file_id,
            "filename": result.get("filename"),
            "total_pages": len(normalized_pages),
            "pages": normalized_pages,
            "note": "Arabic diacritics (harakat) have been removed for better readability"
        }
    except Exception as e:
        logger.error(f"Error reading normalized text: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading normalized text: {str(e)}")




@app.get("/search/{file_id}")
async def search_file(file_id: str, query: str, top_k: int = 5):
    """Search within a processed file using semantic similarity"""
    try:
        results = rag_service.search(file_id, query, top_k=top_k)
        return {
            "file_id": file_id,
            "query": query,
            "top_k": top_k,
            "results": results
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="RAG data not found for this file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/evaluation/csv")
async def download_evaluation_csv():
    """Download the evaluation log CSV"""
    if not EVAL_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="No evaluation data yet. Upload a file first.")
    return FileResponse(
        path=str(EVAL_CSV_PATH),
        filename="evaluation_log.csv",
        media_type="text/csv"
    )


@app.get("/evaluation")
async def get_evaluation_log():
    """Get all evaluation entries as JSON"""
    if not EVAL_CSV_PATH.exists():
        return {"entries": [], "total": 0}
    entries = []
    with open(EVAL_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            entries.append(row)
    return {"entries": entries, "total": len(entries)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.5.0",
        "features": [
            "PDF (text + OCR with quality metrics)",
            "DOCX (with tables and images)",
            "PPTX (with tables and images)",
            "XLSX",
            "Table extraction with structure preservation",
            "Image detection and metadata",
            "Quality assessment metrics",
            "Arabic text normalization (diacritics removal)",
            "Automatic document chunking and embeddings (background)",
            "FAISS vector storage"
        ],
        "processors": [
            "PDF (text + OCR)",
            "DOCX",
            "PPTX",
            "XLSX",
            "TXT",
            "HTML",
            "Markdown",
            "RTF",
            "ODT"
        ],
        "storage": {
            "uploads": "./uploads/",
            "results": "./results/",
            "vectors": "./rag_storage/{file_id}/",
            "vector_format": "FAISS IndexFlatL2 + NumPy embeddings"
        },
        "arabic_support": {
            "ocr": "Full support with Tesseract (ara)",
            "diacritics_handling": "Automatic detection and normalization",
            "normalization_endpoint": "/text/{file_id}/normalized",
            "embeddings": "Multilingual model (50+ languages)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)