"""
RAG (Retrieval Augmented Generation) Engine
Handles embedding, vector storage, and semantic search for medical documents
"""
import os
import re
from typing import List, Dict, Any, Optional
import structlog
import asyncio
from dataclasses import dataclass

# Vector Store
import chromadb
from chromadb.config import Settings as ChromaSettings

# Embeddings
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.core.document_processor import ExtractedChunk, ProcessedDocument

logger = structlog.get_logger()


class RecursiveCharacterTextSplitter:
    """Simple text splitter that recursively splits text by characters"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self._split_text(text, self.separators)
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators"""
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        splits = text.split(separator) if separator else list(text)
        
        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    other_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(other_chunks)
                else:
                    final_chunks.append(split)
        
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks with overlap"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            if current_length + split_len > self.chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))
                # Keep overlap
                while current_length > self.chunk_overlap and current_chunk:
                    current_length -= len(current_chunk[0])
                    current_chunk = current_chunk[1:]
            
            current_chunk.append(split)
            current_length += split_len
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks


@dataclass
class RetrievedContext:
    """Represents a retrieved context from vector search"""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class RAGEngine:
    """
    RAG Engine for medical document retrieval and context augmentation.
    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    """
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        embedding_model: str = None
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        
        # Initialize embedding model (use configured device)
        logger.info("Loading embedding model", model=self.embedding_model_name, device=settings.EMBEDDING_DEVICE)
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=settings.EMBEDDING_DEVICE)
        logger.info("Embedding model loaded successfully", device=settings.EMBEDDING_DEVICE)
        
        # Initialize ChromaDB
        self._init_chroma()
    
    def _init_chroma(self):
        """Initialize ChromaDB client and collection"""
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(
            "ChromaDB initialized",
            collection=self.collection_name,
            count=self.collection.count()
        )
    
    async def add_document(
        self,
        processed_doc: ProcessedDocument
    ) -> int:
        """
        Add a processed document to the vector store.
        
        Args:
            processed_doc: ProcessedDocument with chunks to add
        
        Returns:
            Number of chunks added
        """
        if not processed_doc.chunks:
            logger.warning("No chunks to add", document_id=processed_doc.document_id)
            return 0
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk in processed_doc.chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            
            # Create metadata (filter out None values for ChromaDB compatibility)
            metadata = {
                'document_id': processed_doc.document_id,
                'filename': processed_doc.filename,
                'file_type': processed_doc.file_type,
                'page_number': chunk.page_number or 0,
                'section_title': chunk.section_title or '',
                **{k: str(v) if isinstance(v, (list, dict)) else v 
                   for k, v in chunk.metadata.items() 
                   if v is not None and not isinstance(v, (list, dict))}
            }
            # Remove None values from metadata
            metadata = {k: v for k, v in metadata.items() if v is not None}
            metadatas.append(metadata)
        
        # Generate embeddings in batches
        logger.info(
            "Generating embeddings",
            document_id=processed_doc.document_id,
            num_chunks=len(documents)
        )
        
        embeddings = self.embedding_model.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(
            "Document added to vector store",
            document_id=processed_doc.document_id,
            num_chunks=len(ids)
        )
        
        return len(ids)
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> List[RetrievedContext]:
        """
        Search for relevant contexts using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            document_ids: Optional filter by document IDs
            threshold: Minimum similarity threshold
        
        Returns:
            List of RetrievedContext objects
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Build where filter if document_ids provided
        where_filter = None
        if document_ids:
            where_filter = {"document_id": {"$in": document_ids}}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to RetrievedContext objects
        contexts = []
        
        if results and results['ids']:
            for i, chunk_id in enumerate(results['ids'][0]):
                # ChromaDB returns distances, convert to similarity
                distance = results['distances'][0][i]
                similarity = 1 - distance  # For cosine distance
                
                if similarity >= threshold:
                    contexts.append(RetrievedContext(
                        chunk_id=chunk_id,
                        document_id=results['metadatas'][0][i].get('document_id', ''),
                        content=results['documents'][0][i],
                        score=similarity,
                        metadata=results['metadatas'][0][i]
                    ))
        
        return contexts
    
    async def search_for_question_generation(
        self,
        topics: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        num_contexts: int = 10
    ) -> List[RetrievedContext]:
        """
        Retrieve diverse contexts for question generation.
        
        Args:
            topics: Optional list of topics to focus on
            document_ids: Optional filter by document IDs
            num_contexts: Number of contexts to retrieve
        
        Returns:
            List of diverse contexts for question generation
        """
        all_contexts = []
        
        if topics:
            # Search for each topic
            contexts_per_topic = max(2, num_contexts // len(topics))
            
            for topic in topics:
                topic_contexts = await self.search(
                    query=topic,
                    top_k=contexts_per_topic,
                    document_ids=document_ids,
                    threshold=0.3
                )
                all_contexts.extend(topic_contexts)
        else:
            # Get diverse sample by using medical keywords
            medical_queries = [
                "chẩn đoán và điều trị",
                "triệu chứng lâm sàng",
                "xét nghiệm cận lâm sàng",
                "phác đồ điều trị",
                "biến chứng và tiên lượng",
                "diagnosis and treatment",
                "clinical symptoms",
                "laboratory findings",
                "treatment protocol",
                "complications"
            ]
            
            contexts_per_query = max(1, num_contexts // len(medical_queries))
            
            for query in medical_queries[:num_contexts]:
                query_contexts = await self.search(
                    query=query,
                    top_k=contexts_per_query,
                    document_ids=document_ids,
                    threshold=0.2
                )
                all_contexts.extend(query_contexts)
        
        # Remove duplicates based on chunk_id
        seen_ids = set()
        unique_contexts = []
        for ctx in all_contexts:
            if ctx.chunk_id not in seen_ids:
                seen_ids.add(ctx.chunk_id)
                unique_contexts.append(ctx)
        
        return unique_contexts[:num_contexts]
    
    async def get_document_chunks(
        self,
        document_id: str
    ) -> List[RetrievedContext]:
        """Get all chunks for a specific document"""
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )
        
        contexts = []
        if results and results['ids']:
            for i, chunk_id in enumerate(results['ids']):
                contexts.append(RetrievedContext(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=results['documents'][i],
                    score=1.0,
                    metadata=results['metadatas'][i]
                ))
        
        return contexts
    
    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document"""
        # Get chunks for this document
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        count = self.collection.count()
        
        return {
            'total_chunks': count,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model_name
        }


# Global RAG engine instance
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the global RAG engine instance"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
