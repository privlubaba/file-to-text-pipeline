# File-to-Text Ingestion Pipeline with Automatic Vector Storage

A production-ready multi-format document ingestion pipeline that accepts PDF, DOCX, PPTX, and XLSX files, extracts text using OCR, and automatically chunks documents and stores them as vector embeddings using Sentence-Transformers and FAISS for future retrieval capabilities.

## Features

### Document Processing
✅ **Multiple File Formats**
- PDF (text-based and scanned with OCR)
- Microsoft Word (DOCX)
- PowerPoint (PPTX)
- Excel (XLSX)
- Plain Text (TXT)
- HTML/HTM
- Markdown (MD)
- Rich Text Format (RTF)
- OpenDocument Text (ODT)

✅ **Intelligent Text Extraction**
- Automatic detection of scanned vs. text-based PDFs
- OCR support using Tesseract with quality metrics
- Layout-aware extraction using Docling
- Per-page/per-slide/per-sheet storage
- Table structure preservation
- Image detection and metadata extraction
- Arabic diacritics (harakat) normalization

### Automatic Vector Processing (Background)
✅ **Document Chunking**
- Hybrid chunking strategy to preserve context (512 chars, 50 overlap)
- Metadata enrichment for each chunk
- Page-level tracking and references

✅ **Vector Embeddings**
- Sentence-Transformers multilingual model
- Automatic embedding generation (384-dimensional vectors)
- Supports 50+ languages including Arabic

✅ **Vector Storage**
- FAISS IndexFlatL2 for exact similarity search
- Persistent disk storage
- Ready for future retrieval features

### API & Interface
✅ **RESTful API**
- File upload endpoint
- Status checking
- Text retrieval (full or by page)
- Quality assessment metrics
- Table and image extraction endpoints
- Normalized text endpoint (Arabic support)

✅ **Interactive Web Interface**
- Drag-and-drop file upload
- Real-time processing status with polling
- Extracted text preview per page
- Clean and simple UI

## Architecture

```
┌─────────────┐
│ File Upload │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│  Validation  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Format Router │
└──────┬───────┘
       │
    ┌──┴────┐
    │       │
    ▼       ▼
┌─────┐  ┌────────┐
│ PDF │  │ Office │
└──┬──┘  └───┬────┘
   │         │
   │  ┌──────┴─────┐
   │  │    │    │  │
   │  DOCX PPTX XLSX
   │
   ├─Text → pdfplumber
   └─Scan → Tesseract OCR
        │
        ▼
  ┌──────────────┐
  │   Docling    │
  │ Unification  │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │Hybrid Chunker│
  └──────┬───────┘
         │
         ▼
  ┌─────────────────┐
  │Sentence-Transf. │
  │   Embeddings    │
  └──────┬──────────┘
         │
         ▼
  ┌─────────────┐
  │FAISS Vector │
  │   Storage   │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Similarity │
  │   Retrieval │
  └─────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR
- Poppler utilities
- (Optional) LibreOffice, Pandoc

### Quick Setup

```bash
# Run automated setup
./setup.sh

# Or manual installation:

# 1. Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ara poppler-utils

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python packages
pip install -r requirements.txt
```

### macOS Installation

```bash
# Install system dependencies
brew install tesseract tesseract-lang poppler

# Continue with Python setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the API server
python main.py

# Server will run on http://localhost:8000
```

### Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

The interface provides:
1. **Upload Section**: Drag-and-drop or click to upload files
2. **Processing Status**: Real-time status updates with extraction methods
3. **Extracted Text**: View all pages with character counts
4. **Document Chunks**: Browse intelligent chunks with metadata (pending backend)
5. **Embedding Info**: View embedding model details and vector statistics (pending backend)
6. **Similarity Search**: Query interface with relevance-scored results (pending backend)

### API Documentation

Interactive API docs available at:
```
http://localhost:8000/docs
```

### API Endpoints

#### Core Endpoints (Implemented)

**1. Upload File**
```bash
POST /upload
```

**Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "file_id": "a1b2c3d4-5678-90ef-ghij-klmnopqrstuv",
  "filename": "document.pdf",
  "meaningful_name": "upload_0001_document",
  "upload_number": 1,
  "status": "queued",
  "message": "File uploaded successfully. Processing started.",
  "status_url": "/status/a1b2c3d4-5678-90ef-ghij-klmnopqrstuv",
  "result_url": "/text/a1b2c3d4-5678-90ef-ghij-klmnopqrstuv"
}
```

**2. Check Status**
```bash
GET /status/{file_id}
```

**3. Get Extracted Text**
```bash
GET /text/{file_id}
```

**4. Get Specific Page**
```bash
GET /text/{file_id}/page/{page_num}
```

**5. Get Normalized Text (Arabic support)**
```bash
GET /text/{file_id}/normalized
```

**6. Get Quality Assessment**
```bash
GET /quality/{file_id}
```

**7. Get Tables**
```bash
GET /tables/{file_id}
```

**8. Get Images**
```bash
GET /images/{file_id}
```


### Python Client Example

```python
import requests
import time

# Upload file
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f}
    )

result = response.json()
file_id = result['file_id']

# Poll for completion
while True:
    status = requests.get(f'http://localhost:8000/status/{file_id}').json()
    print(f"Status: {status['status']}")

    if status['status'] == 'completed':
        break
    elif status['status'] == 'failed':
        print(f"Error: {status['error']}")
        exit(1)

    time.sleep(2)

# Get extracted text
text_result = requests.get(f'http://localhost:8000/text/{file_id}').json()

# Print all pages
for page in text_result['pages']:
    print(f"\n--- Page {page['page_number']} ---")
    print(page['raw_text'][:500])  # First 500 chars

# Check storage location
print(f"\nVector storage location:")
print(f"./rag_storage/{file_id}/")
```

## File Format Support

### PDF
- **Text-based PDFs**: Direct text extraction using pdfplumber
- **Scanned PDFs**: Automatic OCR using Tesseract with quality metrics
- **Hybrid PDFs**: Intelligent detection and processing
- **Features**: Table extraction, image detection, quality assessment
- **Output**: Per-page text with page numbers

### DOCX (Microsoft Word)
- **Extraction**: python-docx with pandoc fallback
- **Features**: Preserves paragraphs, tables, and images
- **Output**: Sequential text with metadata

### PPTX (PowerPoint)
- **Extraction**: python-pptx
- **Features**: Slide-by-slide extraction, includes speaker notes, tables, and images
- **Output**: Per-slide text with slide numbers

### XLSX (Excel)
- **Extraction**: pandas + openpyxl
- **Features**: Sheet-by-sheet extraction, includes cell values
- **Output**: Per-sheet structured text

## Current Status

### ✅ Fully Implemented
- Multi-format document upload (PDF, DOCX, PPTX, XLSX)
- OCR processing with Tesseract
- Text extraction with quality metrics
- Table and image extraction
- Arabic text normalization
- Web interface with drag-and-drop
- Real-time status polling
- **Automatic document chunking (background)**
- **Automatic embedding generation (background)**
- **FAISS vector storage (background)**

### 📁 Storage Locations

All processed data is automatically stored:

```
/Users/lubaba_raed/Downloads/files/
├── uploads/                    # Original uploaded files
│   └── upload_XXXX_filename.ext
│
├── results/                    # Extraction results (JSON)
│   └── result_upload_XXXX_filename.json
│       ├── pages[]             # Per-page text
│       ├── tables[]            # Extracted tables
│       ├── images[]            # Image metadata
│       └── metadata{}          # Quality metrics
│
└── rag_storage/               # Vector embeddings (automatic)
    └── {file_id}/
        ├── chunks.json         # Document chunks with metadata
        ├── embeddings.npy      # 384-dim vectors (NumPy)
        ├── faiss.index         # FAISS IndexFlatL2
        └── metadata.json       # Processing info
```

### 🔄 Processing Pipeline

1. **Upload** → File saved to `uploads/`
2. **Extract** → Text/tables/images extracted
3. **Save** → Results saved to `results/`
4. **Chunk** → Document split into 512-char chunks (background)
5. **Embed** → Multilingual embeddings generated (background)
6. **Store** → Vectors saved to `rag_storage/` (background)

Everything happens automatically - no user action required!

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Maximum file size (bytes)
MAX_FILE_SIZE=52428800  # 50 MB

# Upload directory
UPLOAD_DIR=./uploads

# Results directory
RESULTS_DIR=./results

# Server settings
HOST=0.0.0.0
PORT=8000

# RAG Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
FAISS_INDEX_TYPE=IndexFlatL2
```

## Performance Optimization

### For Large Files
- Increase timeout in background processing
- Consider queue-based architecture (Celery + Redis)
- Use chunked processing for very large documents

### For High Volume
- Deploy multiple worker processes
- Add Redis for job queue
- Use PostgreSQL instead of JSON files
- Add caching layer (Redis)

### OCR Optimization
- Adjust DPI in pdf2image (lower = faster, higher = better quality)
- Pre-process images before OCR
- Use GPU-accelerated Tesseract if available

### Vector Search Optimization
- Use FAISS IVF indices for large datasets (>100k vectors)
- Consider GPU acceleration for similarity search
- Implement batch embedding generation

## Troubleshooting

### Tesseract Not Found
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara

# macOS
brew install tesseract tesseract-lang
```

### Poor OCR Quality
- Increase DPI in `PDFProcessor._extract_with_ocr()` (default: 300)
- Pre-process images (deskew, denoise)
- Ensure correct language pack is installed

### Memory Issues
- Process files in chunks
- Limit concurrent uploads
- Increase system RAM or use swap
- Use FAISS memory-mapped indices for large datasets

## Security Considerations

### Production Deployment

1. **Authentication**: Add API key or JWT authentication
2. **Rate Limiting**: Implement request throttling
3. **File Validation**: Strict MIME type checking
4. **Sandboxing**: Process files in isolated containers
5. **Malware Scanning**: Integrate antivirus scanning
6. **HTTPS**: Use TLS/SSL in production
7. **Input Sanitization**: Validate all file inputs

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn

```bash
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

## Installation

### Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies (includes RAG libraries)
pip install -r requirements.txt

# 3. Start server
python main.py

# 4. Open browser
http://localhost:8000
```

### First-Time Setup

The first time you upload a file, the system will automatically download the multilingual embedding model (~400MB). This happens once and is cached for future use.

## Team Update (3 Lines)

Enhanced the document ingestion pipeline with automatic background processing that chunks documents (512 chars with 50-char overlap using LangChain) and generates multilingual vector embeddings (384-dim using Sentence-Transformers) for all uploaded files. Implemented persistent FAISS vector storage that saves chunks, embeddings, and indices to disk at `rag_storage/{file_id}/` for future retrieval capabilities. System supports full Arabic language processing through multilingual models and maintains all existing features (OCR, table/image extraction, quality metrics) while seamlessly adding vector storage in the background.

## License

Proprietary - See LICENSE file for details

## Support

For issues or questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- View web interface at `http://localhost:8000`

## What Gets Stored

Every uploaded document creates three storage artifacts:

1. **Original File** (`uploads/upload_XXXX_filename.ext`)
   - Preserved exactly as uploaded
   - Numbered sequentially for tracking

2. **Extraction Results** (`results/result_upload_XXXX_filename.json`)
   - Full text per page
   - Extracted tables with structure
   - Image metadata
   - Quality metrics and OCR confidence
   - Arabic text (normalized and raw)

3. **Vector Embeddings** (`rag_storage/{file_id}/`)
   - `chunks.json` - Text chunks with page references
   - `embeddings.npy` - 384-dimensional vectors
   - `faiss.index` - FAISS search index
   - `metadata.json` - Model and processing info

All data persists across server restarts and is ready for future search/retrieval features.
