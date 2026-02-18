┌──────────────────────────────────────────────────────────────────────────┐
│        MY APP: Document Upload → Extraction → RAG Storage (Auto)         │
└──────────────────────────────────────────────────────────────────────────┘

USER (Browser)
│
│  Upload file (PDF / DOCX / PPTX / XLSX / TXT / HTML / Markdown / RTF / ODT)
▼
WEB UI  (index.html)
│  • Drag & drop
│  • Shows status + extracted text
│  • Vanilla JS + HTML/CSS
▼
FASTAPI SERVER  (main.py)
│  • Receives upload
│  • Saves original file
│  • Chooses correct processor by file type
│  • Returns extraction result to UI
│  • Triggers background RAG processing
│
├─────────────────────────────────────────────────────────────────────────┐
│  1) STORAGE: ORIGINAL FILE                                               │
└─────────────────────────────────────────────────────────────────────────┘
▼
uploads/
upload_XXXX_filename.ext

├─────────────────────────────────────────────────────────────────────────┐
│  2) EXTRACTION LAYER (processors.py)  ← user-visible results             │
└─────────────────────────────────────────────────────────────────────────┘
│
├─ PDF Processor
│   • pdfplumber → text/tables (when PDF has real text)
│   • Tesseract OCR (ara) → scanned pages / low-text pages
│   • Detect tables + image metadata
│   • Compute quality metrics
│
├─ DOCX Processor
│   • python-docx → paragraphs/runs text
│   • pages[] represents sections/blocks (not literal “pages”)
│
├─ PPTX Processor
│   • python-pptx → slide text/shapes
│   • pages[] represents slides
│
└─ XLSX Processor
• openpyxl → sheets/cells text
• pages[] represents sheets (or sheet blocks)

│
├─ TXT Processor
│   • Read plain text (UTF-8; fallback encoding if needed)
│   • pages[] represents whole file or split blocks
│
├─ HTML Processor
│   • Extract visible text (strip tags, keep basic structure)
│   • pages[] represents sections/blocks
│
├─ Markdown Processor
│   • Extract text (optionally strip markdown symbols)
│   • pages[] represents sections/blocks
│
├─ RTF Processor
│   • Convert/extract text from RTF
│   • pages[] represents sections/blocks
│
└─ ODT Processor
• Extract text from OpenDocument
• pages[] represents sections/blocks (tables/images if supported)

▼
results/
result_upload_XXXX_filename.json
• pages[]     (PDF pages OR DOCX/PPTX/XLSX/TXT/HTML/MD/RTF/ODT units)
• tables[]    (structured tables when available)
• images[]    (image metadata when available)
• metadata{}  (quality + processing info)

├─────────────────────────────────────────────────────────────────────────┐
│  3) RAG LAYER (rag_service.py)  ← automatic BACKGROUND processing        │
└─────────────────────────────────────────────────────────────────────────┘
│  Input: extracted text from results JSON
│
├─ Chunking (LangChain-style splitting)
│   • 512 characters per chunk
│   • 50-character overlap
│   • Keeps page/source references
│   → rag_storage/{file-id}/chunks.json
│
├─ Embeddings (Sentence-Transformers)
│   • Model: paraphrase-multilingual-MiniLM-L12-v2
│   • Output: 384-d vectors
│   • Arabic + multilingual support
│   → rag_storage/{file-id}/embeddings.npy   (NumPy)
│
└─ Vector Index (FAISS)
• IndexFlatL2
→ rag_storage/{file-id}/faiss.index
→ rag_storage/{file-id}/metadata.json

┌─────────────────────────────────────────────────────────────────────────┐
│  FINAL OUTPUTS I GET (PERSISTENT ON DISK)                                │
└─────────────────────────────────────────────────────────────────────────┘
• uploads/   → original file (traceability)
• results/   → extraction JSON (what UI shows)
• rag_storage/ → chunks + embeddings + FAISS (ready for future search/RAG)

┌─────────────────────────────────────────────────────────────────────────┐
│  TECHNOLOGY USED                                                        │
└─────────────────────────────────────────────────────────────────────────┘
Frontend:
• HTML/CSS + Vanilla JavaScript (index.html)

Backend:
• FastAPI (main.py)

Extraction:
• pdfplumber (PDF)
• Tesseract OCR + ara language pack (Arabic OCR)
• python-docx (DOCX)
• python-pptx (PPTX)
• openpyxl (XLSX)
• Plain text reading (TXT)
• HTML text extraction (HTML)
• Markdown text extraction (Markdown)
• RTF text extraction (RTF)
• ODT text extraction (ODT)

RAG / Vectors:
• LangChain-style chunking (512 / overlap 50)
• Sentence-Transformers (embeddings)
• NumPy (.npy storage)
• FAISS (IndexFlatL2)

Deployment:
• requirements.txt
• Dockerfile + docker-compose.yml
