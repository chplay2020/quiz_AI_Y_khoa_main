"""
Medical Quiz Generator - FastAPI Application
Main entry point for the backend API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import structlog
import os

from app.config import settings
from app.api import documents_router, questions_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting Medical Quiz Generator API", version=settings.APP_VERSION)
    
    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Medical Quiz Generator API")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Medical Quiz Generator API
    
    An AI-powered system for generating medical education quiz questions from documents.
    
    ### Features:
    - **Document Processing**: Upload and process PDF, PPTX, DOCX files
    - **RAG-based Retrieval**: Semantic search through medical content
    - **AI Question Generation**: Generate high-quality MCQs using LLMs
    - **Multiple Export Formats**: JSON, PDF, DOCX, Excel
    
    ### Supported Medical Specialties:
    - Internal Medicine, Surgery, Pediatrics
    - Cardiology, Neurology, Oncology
    - And many more...
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents_router, prefix=settings.API_PREFIX)
app.include_router(questions_router, prefix=settings.API_PREFIX)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Medical Quiz Generator API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get(f"{settings.API_PREFIX}/specialties")
async def list_specialties():
    """List available medical specialties"""
    return {
        "specialties": settings.MEDICAL_SPECIALTIES
    }


@app.get(f"{settings.API_PREFIX}/config")
async def get_config():
    """Get public configuration"""
    return {
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "allowed_extensions": settings.ALLOWED_EXTENSIONS,
        "max_questions_per_request": settings.MAX_QUESTIONS_PER_REQUEST,
        "difficulty_levels": settings.QUESTION_DIFFICULTY_LEVELS,
        "default_num_questions": settings.DEFAULT_NUM_QUESTIONS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
