"""
Configuration settings for Medical Quiz Generator
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # App Settings
    APP_NAME: str = "Medical Quiz Generator"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # CORS Settings
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # Default LLM Provider: openai, anthropic, google
    DEFAULT_LLM_PROVIDER: str = "openai"
    DEFAULT_MODEL: str = "gpt-4-turbo-preview"
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DEVICE: str = "cpu"  # Force CPU to avoid CUDA compatibility issues with older GPUs
    
    # Vector Store Settings
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    COLLECTION_NAME: str = "medical_documents"
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE_MB: int = 50
    
    # Upload Settings
    UPLOAD_DIR: str = "./data/uploads"
    ALLOWED_EXTENSIONS: list = [".pdf", ".pptx", ".ppt", ".docx", ".doc", ".txt"]
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./data/medical_quiz.db"
    
    # Redis Settings (optional, for caching)
    REDIS_URL: Optional[str] = None
    
    # Question Generation Settings
    DEFAULT_NUM_QUESTIONS: int = 5  # Reduced from 10 to avoid rate limits
    MAX_QUESTIONS_PER_REQUEST: int = 50
    QUESTION_DIFFICULTY_LEVELS: list = ["easy", "medium", "hard"]
    
    # Medical Specialties
    MEDICAL_SPECIALTIES: list = [
        "Internal Medicine",
        "Surgery", 
        "Pediatrics",
        "Obstetrics & Gynecology",
        "Cardiology",
        "Neurology",
        "Oncology",
        "Emergency Medicine",
        "Radiology",
        "Pathology",
        "Pharmacology",
        "Anatomy",
        "Physiology",
        "Biochemistry",
        "Microbiology",
        "Public Health",
        "Dermatology",
        "Ophthalmology",
        "Psychiatry",
        "Orthopedics"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)
