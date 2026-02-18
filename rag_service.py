"""
RAG Service Module
Handles chunking, embedding generation, and FAISS vector storage
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle

# Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer

# FAISS for vector storage
import faiss

# LangChain for chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    chunk_id: int
    text: str
    page_number: Optional[int]
    char_count: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EmbeddingInfo:
    """Embedding model and index information"""
    model: str
    dimension: int
    total_vectors: int
    index_type: str

    def to_dict(self) -> Dict:
        return asdict(self)


class RAGService:
    """Service for chunking, embedding, and vector search"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        storage_dir: str = "./rag_storage"
    ):
        """
        Initialize RAG service

        Args:
            model_name: Sentence-Transformers model (multilingual for Arabic support)
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            storage_dir: Directory to store FAISS indices and metadata
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Initialize embedding model (lazy loading)
        self._model = None

        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "。", "؟", "!", "?", "،", ",", " ", ""]
        )

        logger.info(f"RAG Service initialized with model: {model_name}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self._model

    def chunk_document(
        self,
        pages: List[Dict[str, Any]],
        file_id: str
    ) -> List[Chunk]:
        """
        Chunk a document into smaller pieces with metadata

        Args:
            pages: List of page dictionaries from extraction result
            file_id: Unique file identifier

        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_id = 0

        for page in pages:
            page_number = page.get("page_number", None)
            raw_text = page.get("raw_text", "")

            # Skip empty pages
            if not raw_text or not raw_text.strip():
                continue

            # Create LangChain document for splitting
            doc = Document(
                page_content=raw_text,
                metadata={
                    "page_number": page_number,
                    "extraction_method": page.get("extraction_method", "unknown"),
                    "char_count": page.get("char_count", 0)
                }
            )

            # Split the page into chunks
            split_docs = self.text_splitter.split_documents([doc])

            # Convert to Chunk objects
            for split_doc in split_docs:
                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=split_doc.page_content,
                    page_number=page_number,
                    char_count=len(split_doc.page_content),
                    metadata=split_doc.metadata
                )
                chunks.append(chunk)
                chunk_id += 1

        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages for file {file_id}")
        return chunks

    def generate_embeddings(
        self,
        chunks: List[Chunk]
    ) -> np.ndarray:
        """
        Generate embeddings for chunks

        Args:
            chunks: List of Chunk objects

        Returns:
            NumPy array of embeddings (n_chunks x embedding_dim)
        """
        texts = [chunk.text for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True  # L2 normalization for better similarity
        )

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def create_faiss_index(
        self,
        embeddings: np.ndarray
    ) -> faiss.IndexFlatL2:
        """
        Create FAISS index from embeddings

        Args:
            embeddings: NumPy array of embeddings

        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]

        # Create flat L2 index (exact search)
        index = faiss.IndexFlatL2(dimension)

        # Add embeddings to index
        index.add(embeddings.astype('float32'))

        logger.info(f"Created FAISS index with {index.ntotal} vectors, dimension {dimension}")
        return index

    def save_rag_data(
        self,
        file_id: str,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        index: faiss.IndexFlatL2
    ):
        """
        Save chunks, embeddings, and FAISS index to disk

        Args:
            file_id: Unique file identifier
            chunks: List of Chunk objects
            embeddings: NumPy array of embeddings
            index: FAISS index
        """
        file_dir = self.storage_dir / file_id
        file_dir.mkdir(exist_ok=True)

        # Save chunks as JSON
        chunks_path = file_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump([chunk.to_dict() for chunk in chunks], f, indent=2, ensure_ascii=False)

        # Save embeddings as NumPy array
        embeddings_path = file_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)

        # Save FAISS index
        index_path = file_dir / "faiss.index"
        faiss.write_index(index, str(index_path))

        # Save metadata
        metadata = {
            "file_id": file_id,
            "model": self.model_name,
            "dimension": embeddings.shape[1],
            "total_chunks": len(chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
        metadata_path = file_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved RAG data for file {file_id} to {file_dir}")

    def load_rag_data(
        self,
        file_id: str
    ) -> Tuple[List[Chunk], np.ndarray, faiss.IndexFlatL2, Dict]:
        """
        Load chunks, embeddings, and FAISS index from disk

        Args:
            file_id: Unique file identifier

        Returns:
            Tuple of (chunks, embeddings, index, metadata)
        """
        file_dir = self.storage_dir / file_id

        if not file_dir.exists():
            raise FileNotFoundError(f"RAG data not found for file {file_id}")

        # Load chunks
        chunks_path = file_dir / "chunks.json"
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            chunks = [Chunk(**chunk) for chunk in chunks_data]

        # Load embeddings
        embeddings_path = file_dir / "embeddings.npy"
        embeddings = np.load(embeddings_path)

        # Load FAISS index
        index_path = file_dir / "faiss.index"
        index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = file_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        logger.info(f"Loaded RAG data for file {file_id}")
        return chunks, embeddings, index, metadata

    def search(
        self,
        file_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity

        Args:
            file_id: Unique file identifier
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of search results with scores
        """
        # Load RAG data
        chunks, embeddings, index, metadata = self.load_rag_data(file_id)

        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')

        # Search in FAISS index
        distances, indices = index.search(query_embedding, min(top_k, len(chunks)))

        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(chunks):  # Valid index
                chunk = chunks[idx]
                result = {
                    "rank": i + 1,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "score": float(distance),  # L2 distance (lower is better)
                    "similarity": float(1 / (1 + distance)),  # Convert to similarity score
                    "char_count": chunk.char_count,
                    "metadata": chunk.metadata
                }
                results.append(result)

        logger.info(f"Found {len(results)} results for query: {query[:50]}...")
        return results

    def process_document(
        self,
        file_id: str,
        pages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: chunk -> embed -> index -> save

        Args:
            file_id: Unique file identifier
            pages: List of page dictionaries from extraction result

        Returns:
            Summary dictionary with statistics
        """
        # Step 1: Chunk the document
        chunks = self.chunk_document(pages, file_id)

        if not chunks:
            logger.warning(f"No chunks created for file {file_id}")
            return {
                "success": False,
                "message": "No text content to chunk",
                "total_chunks": 0
            }

        # Step 2: Generate embeddings
        embeddings = self.generate_embeddings(chunks)

        # Step 3: Create FAISS index
        index = self.create_faiss_index(embeddings)

        # Step 4: Save everything
        self.save_rag_data(file_id, chunks, embeddings, index)

        return {
            "success": True,
            "file_id": file_id,
            "total_chunks": len(chunks),
            "embedding_dimension": embeddings.shape[1],
            "model": self.model_name,
            "index_type": "IndexFlatL2"
        }

    def get_chunks(self, file_id: str) -> Dict[str, Any]:
        """Get all chunks for a file"""
        chunks, _, _, metadata = self.load_rag_data(file_id)
        return {
            "file_id": file_id,
            "total_chunks": len(chunks),
            "chunks": [chunk.to_dict() for chunk in chunks]
        }

    def get_embedding_info(self, file_id: str) -> Dict[str, Any]:
        """Get embedding information for a file"""
        _, embeddings, index, metadata = self.load_rag_data(file_id)

        return {
            "file_id": file_id,
            "model": metadata.get("model", self.model_name),
            "dimension": metadata.get("dimension", embeddings.shape[1]),
            "total_vectors": metadata.get("total_chunks", index.ntotal),
            "num_chunks": index.ntotal,
            "index_type": "IndexFlatL2",
            "chunk_size": metadata.get("chunk_size", self.chunk_size),
            "chunk_overlap": metadata.get("chunk_overlap", self.chunk_overlap)
        }


# Global RAG service instance
rag_service = RAGService(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Supports Arabic + 50+ languages
    chunk_size=512,
    chunk_overlap=50
)
