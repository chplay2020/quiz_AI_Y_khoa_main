"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(str, Enum):
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    CASE_BASED = "case_based"


class ExportFormat(str, Enum):
    JSON = "json"
    PDF = "pdf"
    DOCX = "docx"
    WORD = "word"
    EXCEL = "excel"


# Document Models
class DocumentBase(BaseModel):
    title: str
    description: Optional[str] = None
    specialty: Optional[str] = None
    tags: Optional[List[str]] = []


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: str
    filename: str
    file_type: str
    file_size: int
    num_pages: Optional[int] = None
    num_chunks: int = 0
    status: str = "pending"
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Question Models
class QuestionOption(BaseModel):
    id: str
    text: str
    is_correct: bool = False


class QuestionBase(BaseModel):
    question_text: str
    question_type: QuestionType = QuestionType.SINGLE_CHOICE
    options: List[QuestionOption]
    correct_answer: str
    explanation: str
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    topic: Optional[str] = None
    keywords: Optional[List[str]] = []
    source_chunk_id: Optional[str] = None
    reference_text: Optional[str] = None


class QuestionCreate(QuestionBase):
    document_id: str


class QuestionResponse(QuestionBase):
    id: str
    document_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Quiz Models
class QuizBase(BaseModel):
    title: str
    description: Optional[str] = None
    specialty: Optional[str] = None
    difficulty: Optional[DifficultyLevel] = None
    time_limit_minutes: Optional[int] = None


class QuizCreate(QuizBase):
    question_ids: List[str]


class QuizResponse(QuizBase):
    id: str
    questions: List[QuestionResponse]
    total_questions: int
    created_at: datetime

    class Config:
        from_attributes = True


# Generation Request Models
class QuestionGenerationRequest(BaseModel):
    document_ids: List[str]
    num_questions: int = Field(default=5, ge=1, le=50)  # Reduced from 10
    difficulty: Optional[DifficultyLevel] = None
    question_types: Optional[List[QuestionType]] = None
    topics: Optional[List[str]] = None
    focus_areas: Optional[List[str]] = None
    include_case_based: bool = False
    include_explanations: bool = True
    enable_double_check: bool = False  # Disabled by default to avoid rate limits


class AIReviewResult(BaseModel):
    """Result from AI double-check"""
    status: str  # approved, needs_revision, reject, skipped, error
    accuracy_score: Optional[int] = None
    clarity_score: Optional[int] = None
    educational_value: Optional[int] = None
    issues: List[str] = []
    suggestions: List[str] = []
    corrected_answer: Optional[str] = None
    corrected_explanation: Optional[str] = None
    reviewed: bool = False
    error: Optional[str] = None


class ReviewStats(BaseModel):
    """Statistics from AI review process"""
    total_questions: int = 0
    reviewed: int = 0
    high_accuracy: int = 0
    needs_revision: int = 0
    review_rate: float = 0.0


class QuestionGenerationResponse(BaseModel):
    task_id: str
    status: str
    message: str


class GenerationStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    total_questions: int = 0
    generated_questions: int = 0
    questions: Optional[List[QuestionResponse]] = None
    review_stats: Optional[ReviewStats] = None
    error: Optional[str] = None


# Export Models
class ExportRequest(BaseModel):
    question_ids: List[str]
    format: ExportFormat = ExportFormat.JSON
    include_answers: bool = True
    include_explanations: bool = True
    shuffle_questions: bool = False
    shuffle_options: bool = False


class ExportResponse(BaseModel):
    file_url: str
    filename: str
    format: ExportFormat


# Search Models
class SearchRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


# Statistics Models
class DocumentStats(BaseModel):
    total_documents: int
    total_chunks: int
    total_questions: int
    documents_by_specialty: Dict[str, int]
    documents_by_status: Dict[str, int]


class QuestionStats(BaseModel):
    total_questions: int
    questions_by_difficulty: Dict[str, int]
    questions_by_type: Dict[str, int]
    questions_by_topic: Dict[str, int]


# API Response wrapper
class APIResponse(BaseModel):
    success: bool = True
    message: str = "Success"
    data: Optional[Any] = None
    error: Optional[str] = None
