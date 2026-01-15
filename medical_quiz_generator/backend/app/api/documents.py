"""
Document API Routes
Handles document upload, processing, and management
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import uuid
import aiofiles
from datetime import datetime
import structlog

from app.config import settings
from app.models import (
    DocumentCreate, DocumentResponse, APIResponse,
    DocumentStats
)
from app.core.document_processor import DocumentProcessor
from app.core.rag_engine import get_rag_engine
from app.flows import create_document_processing_flow

logger = structlog.get_logger()
router = APIRouter(prefix="/documents", tags=["Documents"])

# In-memory document storage (replace with database in production)
documents_db = {}


async def process_document_background(
    document_id: str,
    file_path: str,
    metadata: dict
):
    """Background task for document processing"""
    try:
        # Update status
        if document_id in documents_db:
            documents_db[document_id]['status'] = 'processing'
        
        # Create and run the document processing flow
        flow = create_document_processing_flow()
        
        result = await flow.run({
            'file_path': file_path,
            'document_id': document_id,
            'metadata': metadata
        })
        
        # Update document with results
        if document_id in documents_db:
            documents_db[document_id]['status'] = 'completed'
            documents_db[document_id]['num_chunks'] = result.get('chunks_embedded', 0)
            documents_db[document_id]['num_pages'] = result.get('processed_document', {})
            if hasattr(result.get('processed_document'), 'total_pages'):
                documents_db[document_id]['num_pages'] = result['processed_document'].total_pages
        
        logger.info("Document processing completed", document_id=document_id)
        
    except Exception as e:
        logger.error("Document processing failed", document_id=document_id, error=str(e))
        if document_id in documents_db:
            documents_db[document_id]['status'] = 'failed'
            documents_db[document_id]['error'] = str(e)


@router.post("/upload", response_model=APIResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    specialty: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)  # Comma-separated tags
):
    """
    Upload a document for processing.
    Supports PDF, PPTX, DOCX, and TXT files.
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Save file
    file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}{file_ext}")
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            
            # Check file size
            if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
                )
            
            await f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File save failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to save file")
    
    # Parse tags
    tag_list = [t.strip() for t in tags.split(',')] if tags else []
    
    # Create document record
    doc_record = {
        'id': document_id,
        'filename': file.filename,
        'title': title or file.filename,
        'description': description,
        'specialty': specialty,
        'tags': tag_list,
        'file_type': file_ext[1:],
        'file_size': len(content),
        'file_path': file_path,
        'num_pages': None,
        'num_chunks': 0,
        'status': 'pending',
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': None,
        'error': None
    }
    
    documents_db[document_id] = doc_record
    
    # Start background processing
    metadata = {
        'title': title or file.filename,
        'specialty': specialty,
        'tags': tag_list
    }
    background_tasks.add_task(
        process_document_background,
        document_id,
        file_path,
        metadata
    )
    
    return APIResponse(
        success=True,
        message="Document uploaded successfully. Processing started.",
        data=DocumentResponse(
            id=document_id,
            filename=file.filename,
            title=title or file.filename,
            description=description,
            specialty=specialty,
            tags=tag_list,
            file_type=file_ext[1:],
            file_size=len(content),
            status='pending',
            created_at=datetime.utcnow()
        )
    )


@router.get("/", response_model=APIResponse)
async def list_documents(
    specialty: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List all uploaded documents with optional filtering"""
    docs = list(documents_db.values())
    
    # Apply filters
    if specialty:
        docs = [d for d in docs if d.get('specialty') == specialty]
    if status:
        docs = [d for d in docs if d.get('status') == status]
    
    # Pagination
    total = len(docs)
    docs = docs[offset:offset + limit]
    
    return APIResponse(
        success=True,
        message=f"Found {total} documents",
        data={
            'documents': docs,
            'total': total,
            'limit': limit,
            'offset': offset
        }
    )


@router.get("/{document_id}", response_model=APIResponse)
async def get_document(document_id: str):
    """Get a specific document by ID"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return APIResponse(
        success=True,
        data=documents_db[document_id]
    )


@router.delete("/{document_id}", response_model=APIResponse)
async def delete_document(document_id: str):
    """Delete a document and its embeddings"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_db[document_id]
    
    # Delete file
    if os.path.exists(doc['file_path']):
        os.remove(doc['file_path'])
    
    # Delete embeddings
    rag_engine = get_rag_engine()
    deleted_chunks = await rag_engine.delete_document(document_id)
    
    # Remove from database
    del documents_db[document_id]
    
    return APIResponse(
        success=True,
        message=f"Document deleted. {deleted_chunks} chunks removed from vector store."
    )


@router.get("/{document_id}/chunks", response_model=APIResponse)
async def get_document_chunks(document_id: str):
    """Get all chunks for a document"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    rag_engine = get_rag_engine()
    chunks = await rag_engine.get_document_chunks(document_id)
    
    return APIResponse(
        success=True,
        data={
            'document_id': document_id,
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'content': c.content,
                    'score': c.score,
                    'metadata': c.metadata
                }
                for c in chunks
            ],
            'total': len(chunks)
        }
    )


@router.get("/stats/overview", response_model=APIResponse)
async def get_document_stats():
    """Get document statistics"""
    docs = list(documents_db.values())
    
    # Count by specialty
    by_specialty = {}
    for doc in docs:
        spec = doc.get('specialty') or 'Unknown'
        by_specialty[spec] = by_specialty.get(spec, 0) + 1
    
    # Count by status
    by_status = {}
    for doc in docs:
        status = doc.get('status', 'unknown')
        by_status[status] = by_status.get(status, 0) + 1
    
    # Total chunks
    total_chunks = sum(doc.get('num_chunks', 0) for doc in docs)
    
    return APIResponse(
        success=True,
        data=DocumentStats(
            total_documents=len(docs),
            total_chunks=total_chunks,
            total_questions=0,  # Would need to query questions DB
            documents_by_specialty=by_specialty,
            documents_by_status=by_status
        )
    )
