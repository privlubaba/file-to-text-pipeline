# File-to-Text Ingestion Pipeline

A FastAPI-powered document processing API that extracts text from 13+ file formats using OCR, then automatically chunks and indexes the content for semantic search using a RAG pipeline.

## Features

### Document Processing
- **Multi-format support** — PDF, DOCX, PPTX, XLSX, TXT, HTML, Markdown, RTF, ODT, and images (JPG, PNG, TIFF, BMP)
- **Intelligent PDF extraction** — automatic detection of scanned vs. text-based pages using PyMuPDF
- **OCR** — Tesseract with Arabic (`ara`) and English (`eng`) language packs, OpenCV preprocessing (deskew, denoise, binarization)
- **Arabic language support** — diacritics detection, normalization, dialect/MSA/code-mixed classification
- **Structured extraction** — table structure preservation, image metadata, quality metrics per page

### RAG Pipeline (automatic background processing)
- **Chunking** — LangChain `RecursiveCharacterTextSplitter`, 512 chars / 50 overlap, preserves page references
- **Embeddings** — `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, supports 50+ languages including Arabic)
- **Vector storage** — FAISS `IndexFlatL2`, persisted to disk per file

### API & Interface
- REST API with FastAPI — upload, status, text retrieval, search, quality, tables, images
- Interactive web UI — drag-and-drop upload, real-time status polling, per-page text preview
- Evaluation logging — extraction quality, routing correctness, runtime per file in CSV

## Architecture

```
User Upload
    │
    ▼
FastAPI (main.py)
    │  validates file, detects MIME type, saves original
    │
    ├──────────────────────────────────────────────────┐
    │  Extraction (processors/)                        │
    │                                                  │
    │  ├─ PDF  → PyMuPDF (text) + Tesseract (scanned)  │
    │  ├─ DOCX → python-docx                           │
    │  ├─ PPTX → python-pptx                           │
    │  ├─ XLSX → openpyxl / pandas                     │
    │  ├─ TXT / HTML / MD / RTF / ODT → text parsers   │
    │  └─ Images → OpenCV + Tesseract OCR              │
    │                                                  │
    │  Output → results/result_upload_XXXX.json        │
    └──────────────────────────────────────────────────┘
    │
    ▼  (background task)
RAG Pipeline (rag_service.py)
    │
    ├─ Chunk  → chunks.json
    ├─ Embed  → embeddings.npy
    └─ Index  → faiss.index
               rag_storage/{file_id}/
```

## Quick Start (Docker)

```bash
# 1. Build the image
docker build -t file-to-text-pipeline .

# 2. Run the container
docker run -d -p 8000:8000 --name pipeline file-to-text-pipeline

# 3. Open in browser
http://localhost:8000
```

## Manual Setup

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara poppler-utils libmagic1

# macOS
brew install tesseract tesseract-lang poppler libmagic

# Python setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start server
python main.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a file for processing |
| GET | `/status/{file_id}` | Check processing status |
| GET | `/text/{file_id}` | Get all extracted text |
| GET | `/text/{file_id}/page/{n}` | Get a specific page |
| GET | `/text/{file_id}/normalized` | Arabic diacritics-removed text |
| GET | `/quality/{file_id}` | Quality assessment metrics |
| GET | `/tables/{file_id}` | Extracted tables |
| GET | `/images/{file_id}` | Image metadata |
| GET | `/search/{file_id}?query=...` | Semantic search within a file |
| GET | `/evaluation` | Evaluation log (JSON) |
| GET | `/evaluation/csv` | Download evaluation CSV |

Interactive API docs: `http://localhost:8000/docs`

## Storage Layout

Every uploaded file produces three artifacts:

```
uploads/
└── upload_XXXX_filename.ext        ← original file

results/
└── result_upload_XXXX_filename.json
    ├── pages[]                     ← per-page extracted text
    ├── tables[]                    ← structured tables
    ├── images[]                    ← image metadata
    └── metadata{}                  ← quality + processing info

rag_storage/{file_id}/
    ├── chunks.json                 ← text chunks with page references
    ├── embeddings.npy              ← 384-dim vectors
    ├── faiss.index                 ← FAISS search index
    └── metadata.json               ← model and processing info
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| PDF extraction | PyMuPDF |
| OCR | Tesseract (`ara` + `eng`) + OpenCV |
| Office docs | python-docx, python-pptx, openpyxl |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Sentence-Transformers (multilingual) |
| Vector index | FAISS IndexFlatL2 |
| Containerization | Docker |
