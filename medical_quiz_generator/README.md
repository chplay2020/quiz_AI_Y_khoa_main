# Medical Quiz Generator - AI Tạo Câu Hỏi Trắc Nghiệm Y Khoa 🏥

Hệ thống AI tạo tự động câu hỏi trắc nghiệm y khoa từ tài liệu (PDF, PowerPoint, Word) sử dụng RAG (Retrieval Augmented Generation) và PocketFlow.

## 🎯 Tính năng chính

- ✅ **Upload & xử lý tài liệu**: PDF, PPTX, DOCX
- ✅ **RAG Engine**: ChromaDB + Sentence Transformers cho semantic search
- ✅ **Multi-LLM Support**: OpenAI GPT-4, Anthropic Claude, Google Gemini
- ✅ **PocketFlow**: Workflow orchestration với các nodes tùy chỉnh
- ✅ **AI Double-Check**: LLM tự động kiểm tra độ chính xác y khoa
- ✅ **Question Types**: Single choice, Multiple choice, True/False, Case-based
- ✅ **Export**: JSON, Excel, PDF, DOCX
- ✅ **Frontend**: React + TypeScript + TailwindCSS

---

## 📁 Cấu trúc dự án

```
medical_quiz_generator/
├── backend/                              # Backend sử dụng FastAPI
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                       # Điểm khởi chạy ứng dụng FastAPI
│   │   ├── config.py                     # Cấu hình hệ thống và biến môi trường
│   │   ├── models.py                     # Các mô hình dữ liệu (Pydantic) cho API
│   │   │
│   │   ├── api/                          # Các tuyến API (API Routes)
│   │   │   ├── documents.py              # API tải lên, liệt kê và xoá tài liệu
│   │   │   └── questions.py              # API sinh câu hỏi, truy vấn và xuất dữ liệu
│   │   │
│   │   ├── core/                         # Lõi xử lý nghiệp vụ (Business Logic)
│   │   │   ├── document_processor.py     # Trích xuất nội dung từ PDF/PPTX/DOCX
│   │   │   ├── rag_engine.py              # Hệ thống RAG sử dụng ChromaDB và embeddings
│   │   │   └── llm_provider.py            # Lớp trừu tượng quản lý nhiều mô hình LLM
│   │   │
│   │   └── flows/                        # Các luồng xử lý PocketFlow
│   │       └── pocketflow_nodes.py        # Các node workflow cho pipeline AI
│   │
│   ├── data/
│   │   ├── uploads/                      # Thư mục lưu trữ tài liệu người dùng tải lên
│   │   └── chroma_db/                    # Cơ sở dữ liệu vector (ChromaDB)
│   │
│   ├── requirements.txt                  # Danh sách thư viện Python
│   ├── Dockerfile                        # Cấu hình Docker cho backend
│   └── .env                              # Tập tin biến môi trường
│
├── frontend/                             # Frontend sử dụng React
│   ├── src/
│   │   ├── main.tsx                      # Điểm khởi chạy ứng dụng React
│   │   ├── App.tsx                       # Thành phần ứng dụng chính
│   │   │
│   │   ├── api/                          # Lớp giao tiếp API
│   │   │   └── index.ts                  # Cấu hình Axios và các hàm gọi API
│   │   │
│   │   ├── components/                   # Các thành phần tái sử dụng
│   │   │   ├── Layout.tsx                # Bố cục chính (sidebar, header)
│   │   │   ├── FileUpload.tsx            # Thành phần tải tệp kéo–thả
│   │   │   └── QuestionCard.tsx          # Hiển thị câu hỏi và đánh giá AI
│   │   │
│   │   ├── pages/                        # Các trang chức năng
│   │   │   ├── Dashboard.tsx             # Trang tổng quan và thống kê
│   │   │   ├── Documents.tsx             # Quản lý tài liệu
│   │   │   ├── Generate.tsx              # Sinh câu hỏi trắc nghiệm
│   │   │   ├── Questions.tsx             # Ngân hàng câu hỏi
│   │   │   └── QuizPreview.tsx           # Giao diện làm bài kiểm tra
│   │   │
│   │   └── store/                        # Quản lý trạng thái toàn cục
│   │       └── index.ts                  # Zustand store
│   │
│   ├── package.json                     # Danh sách phụ thuộc npm
│   ├── vite.config.ts                   # Cấu hình Vite
│   ├── tailwind.config.js               # Cấu hình Tailwind CSS
│   └── Dockerfile                       # Cấu hình Docker cho frontend
│
├── docker-compose.yml                   # Điều phối các dịch vụ Docker
├── setup.sh                             # Script cài đặt nhanh hệ thống
└── README.md                            # Tài liệu mô tả dự án (this file)
```

---

## 🔧 Chức năng từng file chính

### Backend

#### `app/main.py`
- FastAPI application entry point
- CORS middleware configuration
- Mount API routers (`/documents`, `/questions`)
- Health check endpoint: `GET /health`

#### `app/config.py`
- Load environment variables từ `.env`
- Cấu hình LLM providers (API keys, models)
- Cấu hình embedding model
- File upload limits, database paths

#### `app/models.py`
Pydantic models cho validation:
- `Document`: Metadata tài liệu
- `Question`, `QuestionOption`: Câu hỏi và đáp án
- `GenerationRequest`: Request tạo câu hỏi
- `GenerationStatus`: Progress tracking
- `AIReviewResult`: Kết quả AI review
- `ReviewStats`: Thống kê review

#### `app/core/document_processor.py`
**Chức năng**: Extract text từ tài liệu
- `process()`: Entry point xử lý file
- **PDF**: `PyPDF2` + `pdfplumber` để extract text
- **PPTX**: `python-pptx` đọc slides và shapes
- **DOCX**: `python-docx` đọc paragraphs và tables
- Chunk text thành các đoạn nhỏ với metadata (page, section)
- Medical entity recognition (optional)

**Output**: `ProcessedDocument` với list of `ExtractedChunk`

#### `app/core/rag_engine.py`
**Chức năng**: Vector search và semantic retrieval
- `RAGEngine.__init__()`: Load embedding model (sentence-transformers)
- `add_document()`: 
  - Embed từng chunk thành vector
  - Lưu vào ChromaDB với metadata
- `search()`: 
  - Encode query thành vector
  - Semantic search với cosine similarity
  - Return top_k relevant chunks
- `RecursiveCharacterTextSplitter`: Tự implement text splitting algorithm

**Technology**: ChromaDB (persistent) + `paraphrase-multilingual-MiniLM-L12-v2`

#### `app/core/llm_provider.py`
**Chức năng**: Unified interface cho multiple LLMs
- `LLMProvider.generate()`: 
  - Gọi LLM với prompt
  - Support OpenAI GPT-4, Anthropic Claude, Google Gemini
  - Retry logic với exponential backoff
  - Token counting và error handling
- Auto-select provider dựa trên config
- Streaming support (future)

#### `app/flows/pocketflow_nodes.py`
**Chức năng**: PocketFlow workflow nodes

**Base**: `BaseNode` với `prep()`, `exec()`, `post()` lifecycle

**Nodes:**

1. **DocumentIngestionNode**
   - Load document từ file path
   - Parse metadata
   - Output: Raw document object

2. **EmbeddingNode**
   - Process document → chunks
   - Embed và index vào ChromaDB
   - Output: Embedding stats

3. **ContextRetrievalNode**
   - Search relevant chunks từ RAG
   - Filter by document IDs
   - Output: Retrieved contexts

4. **QuestionGenerationNode**
   - LLM generate questions từ context
   - Parse JSON response
   - Support multiple question types

5. **CaseBasedQuestionNode**
   - Generate clinical case scenarios
   - Multi-step reasoning questions
   - Patient case + questions

6. **QuestionValidationNode**
   - Validate structure (required fields)
   - Check options count >= 2
   - Verify correct answer exists
   - Fix is_correct flags

7. **AIDoubleCheckNode** ⭐ NEW
   - LLM review medical accuracy
   - Score: Accuracy, Clarity, Educational value
   - Detect issues & suggest improvements
   - Verdict: APPROVED / NEEDS_REVISION / REJECT

**Flows:**
- `create_question_generation_flow()`: 
  - Retrieval → Generation → Validation → AI Check
- `create_document_processing_flow()`: 
  - Ingestion → Embedding
- `create_full_pipeline_flow()`: 
  - End-to-end từ document đến questions

#### `app/api/documents.py`
**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Upload file (multipart/form-data) |
| GET | `/documents/` | List documents với filters |
| GET | `/documents/{id}` | Get document details |
| DELETE | `/documents/{id}` | Delete document + chunks |
| GET | `/documents/{id}/chunks` | Get document chunks từ RAG |
| GET | `/documents/stats/overview` | Statistics dashboard |

**Upload flow:**
1. Validate file type & size
2. Save file → `data/uploads/`
3. DocumentProcessor.process()
4. RAGEngine.add_document()
5. Return document metadata

#### `app/api/questions.py`
**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/questions/generate` | Start generation (background task) |
| GET | `/questions/generate/{task_id}/status` | Check progress + results |
| GET | `/questions/` | List questions với filters |
| GET | `/questions/{id}` | Get question details |
| PUT | `/questions/{id}` | Update question |
| DELETE | `/questions/{id}` | Delete question |
| POST | `/questions/export` | Export to JSON/Excel/PDF/DOCX |
| POST | `/questions/search` | Semantic search questions |
| GET | `/questions/stats/overview` | Statistics dashboard |

**Generation flow:**
1. Create background task với UUID
2. Run PocketFlow (async)
3. Client polling `/status` endpoint
4. Return questions + review_stats

### Frontend

#### `src/api/index.ts`
- **Axios client** với base URL
- **TypeScript interfaces**:
  - `Document`, `Question`, `QuestionOption`
  - `AIReview`, `ReviewStats`
  - `GenerationRequest`, `GenerationStatus`
- **API functions**:
  - `documentsApi`: upload, list, get, delete, getChunks, getStats
  - `questionsApi`: generate, getGenerationStatus, list, update, delete, export, search
  - `configApi`: getSpecialties, getConfig

#### `src/store/index.ts`
Zustand state management:
- `selectedDocuments`: Array of selected doc IDs
- `currentTaskId`: Current generation task
- `setCurrentTaskId()`, `clearDocumentSelection()`
- Global app state

#### `src/components/Layout.tsx`
- Sidebar navigation với icons
- Routes: Dashboard, Documents, Generate, Questions, Quiz
- Page wrapper với responsive layout
- Active link highlighting

#### `src/components/FileUpload.tsx`
- **Drag & drop** file upload zone
- File validation (type: pdf/pptx/docx, size < 50MB)
- Metadata input form:
  - Title (required)
  - Description (optional)
  - Specialty dropdown
  - Tags input
- Upload progress bar
- Success/error toast notifications

#### `src/components/QuestionCard.tsx`
**Props:**
- `question`: Question object
- `mode`: 'preview' | 'quiz' | 'review'
- `showAnswer`: Boolean
- `showAIReview`: Boolean (NEW)

**Features:**
- Display question với difficulty badge
- Options với color-coded states
- Show/hide answers
- Explanation section (expandable)
- Reference text (collapsible)
- **AI Review section** ⭐:
  - Status badge (approved/needs_revision/reject)
  - Accuracy + Clarity scores (1-10)
  - Issues list
  - Suggestions list
  - Color-coded by status

#### `src/pages/Dashboard.tsx`
Overview page:
- Statistics cards (documents, questions, topics)
- Recent documents list
- Recent questions list
- Quick actions

#### `src/pages/Documents.tsx`
Document management:
- FileUpload component
- Documents table với filters
- Status badges (pending/processing/completed/failed)
- Actions: View chunks, Delete
- Pagination

#### `src/pages/Generate.tsx`
Question generation interface:

**Left panel:**
- Document selection (checkboxes)
- Configuration form:
  - Số lượng câu hỏi (1-50)
  - Độ khó (easy/medium/hard)
  - Ngôn ngữ (vi/en)
  - Include case-based questions (checkbox)
  - Include explanations (checkbox)
  - **AI Double-Check** (checkbox) ⭐

**Right panel:**
- **AI Review Stats** (NEW):
  - Total questions
  - High accuracy count
  - Needs revision count
  - Review rate %
- Progress bar
- Generated questions list
- Actions: View bank, Generate more

**Polling logic:**
- Every 2s check task status
- Update progress bar
- Show toast when complete
- Display review stats

#### `src/pages/Questions.tsx`
Question bank:
- Search bar (semantic search)
- Filters: difficulty, type, topic
- Questions grid với QuestionCard
- Bulk actions: Select, Export, Delete
- Edit modal
- AI review status indicators

#### `src/pages/QuizPreview.tsx`
Quiz taking interface:
- Timer (optional)
- Question navigation
- Answer selection
- Submit quiz
- Review answers với scores
- Retry option

---

## 🔄 Cách hoạt động (Flow Chi Tiết)

### 1. Upload Document Flow
```
User chọn file → FileUpload component
    ↓
Validate file type (PDF/PPTX/DOCX) & size (< 50MB)
    ↓
FormData với file + metadata (title, specialty, tags)
    ↓
Frontend → POST /api/v1/documents/upload
    ↓
Backend: Save file → data/uploads/{uuid}_{filename}
    ↓
DocumentProcessor.process():
    ├─ PDF: PyPDF2 + pdfplumber → extract text by page
    ├─ PPTX: python-pptx → extract slides + shapes
    └─ DOCX: python-docx → extract paragraphs + tables
    ↓
Chunk text (RecursiveCharacterTextSplitter):
    - Chunk size: 1000 chars
    - Overlap: 200 chars
    - Preserve sentences
    ↓
Create ExtractedChunk[] với metadata:
    - chunk_id, content, page_number, section_title
    ↓
RAGEngine.add_document():
    ├─ Embedding model encode chunks → vectors
    ├─ ChromaDB.add(ids, embeddings, documents, metadatas)
    └─ Persist to disk
    ↓
Return Document object với:
    - id, filename, num_chunks, status='completed'
    ↓
Frontend: Update documents list, show success toast
```

### 2. Generate Questions Flow (Với AI Double-Check)
```
User:
    ├─ Chọn documents (checkboxes)
    ├─ Config: num_questions=10, difficulty=medium
    ├─ Enable AI Double-Check ✅
    └─ Click "Tạo câu hỏi"
    ↓
Frontend → POST /api/v1/questions/generate
Body: {
    document_ids: ["doc1"],
    num_questions: 10,
    difficulty: "medium",
    enable_double_check: true
}
    ↓
Backend: Create background task
    - task_id = uuid4()
    - status = 'pending'
    - Start async flow
    ↓
Frontend: Start polling GET /generate/{task_id}/status
    - Every 2 seconds
    - Update progress bar
    ↓
Backend PocketFlow execution:

1️⃣ ContextRetrievalNode.prep()
    - Get document_ids từ shared_state
    
   ContextRetrievalNode.exec()
    - RAGEngine.search(query="Generate medical questions", document_ids)
    - Retrieve top 20 relevant chunks
    - Output: List[RetrievedContext]
    
   ContextRetrievalNode.post()
    - shared_state['retrieved_context'] = contexts

2️⃣ QuestionGenerationNode.prep()
    - Get retrieved_context, num_questions, difficulty
    
   QuestionGenerationNode.exec()
    - Build LLM prompt với:
      * Context chunks
      * Question requirements (type, difficulty, language)
      * JSON output format
    - LLMProvider.generate() → GPT-4
    - Parse JSON response
    - Output: List[Dict] questions
    
   QuestionGenerationNode.post()
    - shared_state['generated_questions'] = questions

3️⃣ QuestionValidationNode.prep()
    - Get generated_questions
    
   QuestionValidationNode.exec()
    - For each question:
      * Check required fields (question_text, options, correct_answer)
      * Validate options count >= 2
      * Verify correct_answer in options
      * Fix is_correct flags
    - Filter invalid questions
    - Output: validated_questions
    
   QuestionValidationNode.post()
    - shared_state['validated_questions'] = validated

4️⃣ AIDoubleCheckNode.prep() ⭐ NEW
    - Get validated_questions, context, enable_double_check
    
   AIDoubleCheckNode.exec()
    - If enable_double_check == false:
      * Return questions as-is với ai_review.status='skipped'
    
    - Process in batches of 5:
      For each batch:
        * Build review prompt với medical criteria
        * LLMProvider.generate() → Review request
        * Parse JSON response:
          {
            "reviews": [{
              "question_index": 1,
              "accuracy_score": 8,
              "clarity_score": 9,
              "educational_value": 7,
              "issues": ["Minor terminology"],
              "suggestions": ["Use 'myocardial infarction' instead of 'heart attack'"],
              "verdict": "APPROVED",
              "corrected_answer": null
            }]
          }
        * Attach ai_review to each question
    
    - Output: reviewed_questions với ai_review metadata
    
   AIDoubleCheckNode.post()
    - shared_state['reviewed_questions'] = reviewed
    - Calculate review_stats:
      * total_questions
      * reviewed (count)
      * high_accuracy (accuracy >= 8)
      * needs_revision (accuracy < 6)
      * review_rate (reviewed / total)
    - shared_state['review_stats'] = stats

    ↓
Store questions in database:
    - For each question:
      * Generate question_id = uuid4()
      * Add created_at timestamp
      * questions_db[id] = question
    ↓
Update task status:
    - status = 'completed'
    - questions = stored_questions
    - review_stats = stats
    - progress = 1.0
    ↓
Frontend polling receives status:
    - Show success toast với review stats
    - Display review stats cards:
      * 10 tổng câu hỏi
      * 8 đạt chuẩn (accuracy >= 8)
      * 2 cần sửa (accuracy < 6)
      * 100% đã kiểm tra
    ↓
Render QuestionCard[] với AI review:
    - Each question shows:
      * Green badge: "AI Double-Check: Đạt chuẩn"
      * Scores: Accuracy 8/10, Clarity 9/10
      * Issues & suggestions (if any)
```

### 3. AI Double-Check Flow (Chi Tiết)
```
Input: validated_questions, context

AIDoubleCheckNode:
    ↓
Build review prompt:
"""
You are a medical education expert reviewing quiz questions.

Review each question for:
1. Accuracy Score (1-10): Medical info correct?
2. Clarity Score (1-10): Question clear?
3. Educational Value (1-10): Tests important knowledge?
4. Issues: List problems
5. Suggestions: How to improve
6. Verdict: APPROVED / NEEDS_REVISION / REJECT

Questions:
Q1: [question_text]
Options: A. [...] B. [...] C. [...] D. [...]
Correct: A
Explanation: [...]

Respond in JSON format:
{
  "reviews": [
    {
      "question_index": 1,
      "accuracy_score": 8,
      "clarity_score": 9,
      ...
    }
  ]
}
"""
    ↓
LLM (GPT-4) processes:
    - Analyze medical accuracy
    - Check for dangerous misinformation
    - Verify terminology (Vietnamese medical terms)
    - Assess educational value
    - Generate scores + feedback
    ↓
Parse JSON response → ai_review object:
{
    "status": "approved",  # or needs_revision, reject
    "accuracy_score": 8,
    "clarity_score": 9,
    "educational_value": 7,
    "issues": ["Minor terminology issue"],
    "suggestions": ["Use standard medical term"],
    "corrected_answer": null,
    "corrected_explanation": null,
    "reviewed": true
}
    ↓
Attach to question.ai_review
    ↓
Calculate stats:
    - high_accuracy: count(accuracy >= 8)
    - needs_revision: count(accuracy < 6)
    - review_rate: reviewed / total
    ↓
Return reviewed_questions + review_stats
```

### 4. RAG Search Flow
```
Query: "Generate medical questions"
Document IDs: ["doc1", "doc2"]
    ↓
RAGEngine.search():
    ↓
Encode query:
    embedding_model.encode(query) → vector [768 dims]
    ↓
ChromaDB query:
    collection.query(
        query_embeddings=[vector],
        n_results=20,
        where={"document_id": {"$in": ["doc1", "doc2"]}},
        include=["documents", "metadatas", "distances"]
    )
    ↓
Compute similarity:
    similarity = 1 - cosine_distance
    ↓
Filter by threshold (>= 0.5)
    ↓
Return top_k results:
[
    RetrievedContext(
        chunk_id="chunk_uuid",
        document_id="doc1",
        content="Myocardial infarction is...",
        score=0.87,
        metadata={"page": 5, "section": "Cardiology"}
    ),
    ...
]
    ↓
Use as context for question generation
```

---

## 🚀 Cài đặt & Chạy

### Yêu cầu hệ thống
- **Python**: 3.12+
- **Node.js**: 20+
- **RAM**: 8GB+ (để load embedding model)
- **Disk**: 5GB+ (cho models và vector DB)
- **API Key**: OpenAI / Anthropic / Google (ít nhất 1)

### Cách 1: Chạy thủ công (Development)

#### Bước 1: Clone & Setup Backend
```bash
cd medical_quiz_generator/backend

# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r requirements.txt

# Tạo thư mục data
mkdir -p data/uploads data/chroma_db
```

#### Bước 2: Cấu hình Backend
```bash
# Sửa file .env và thêm API key
nano backend/.env

# Thêm vào:
OPENAI_API_KEY=sk-your-openai-key-here
# hoặc
ANTHROPIC_API_KEY=your-anthropic-key
# hoặc
GOOGLE_API_KEY=your-google-key
```

#### Bước 3: Chạy Backend
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend chạy tại: `http://localhost:8000`
API Docs: `http://localhost:8000/docs`

#### Bước 4: Setup Frontend
```bash
cd ../frontend

# Cài dependencies (Linux WSL: dùng npm của Linux, không phải Windows)
npm install
```

#### Bước 5: Chạy Frontend
```bash
npm run dev
```

Frontend chạy tại: `http://localhost:3000`

### Cách 2: Docker Compose (Production-ready)

```bash
# Build và chạy tất cả services
docker-compose up -d

# Hoặc chạy quick setup script
chmod +x setup.sh
./setup.sh
```

Services:
- Backend API: `http://localhost:8000`
- Frontend: `http://localhost:3000`

### Cách 3: Quick Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

Script sẽ:
1. Kiểm tra dependencies (Python 3.12+, Node.js 20+)
2. Tạo virtual environment
3. Cài backend packages
4. Cài frontend packages
5. Tạo thư mục cần thiết
6. Chạy backend và frontend

---

## ⚙️ Cấu hình

### Backend `.env`
```bash
# App Settings
APP_NAME="Medical Quiz Generator"
APP_VERSION="1.0.0"
DEBUG=true

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v1

# CORS - Frontend URLs
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# LLM Provider API Keys (chọn ít nhất 1)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key

# Default LLM Settings
DEFAULT_LLM_PROVIDER=openai  # openai / anthropic / google
DEFAULT_MODEL=gpt-4-turbo-preview

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Vector Store
CHROMA_PERSIST_DIR=./data/chroma_db
COLLECTION_NAME=medical_documents

# Upload Limits
MAX_FILE_SIZE=52428800  # 50MB in bytes
UPLOAD_DIRECTORY=./data/uploads

# Question Generation
MAX_QUESTIONS_PER_REQUEST=50
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
```

### Frontend `.env` (optional)
```bash
VITE_API_URL=http://localhost:8000/api/v1
```

---

## 📊 API Documentation

### Swagger UI
Truy cập `http://localhost:8000/docs` để xem interactive API docs với Swagger UI.

### Key Endpoints

#### Health Check
```bash
GET /health
Response: {
  "status": "healthy",
  "app": "Medical Quiz Generator",
  "version": "1.0.0"
}
```

#### Documents
```bash
# Upload document
POST /api/v1/documents/upload
Content-Type: multipart/form-data
Body: 
  - file: <binary>
  - title: "Cardiology Guidelines 2024"
  - description: "ESC guidelines"
  - specialty: "Cardiology"
  - tags: "guidelines,cardiology,2024"

Response: {
  "success": true,
  "data": {
    "document": {
      "id": "doc-uuid",
      "filename": "cardiology.pdf",
      "status": "completed",
      "num_chunks": 150
    }
  }
}

# List documents
GET /api/v1/documents/?status=completed&specialty=cardiology&limit=10

# Get document chunks (từ vector DB)
GET /api/v1/documents/{id}/chunks

# Delete document
DELETE /api/v1/documents/{id}

# Statistics
GET /api/v1/documents/stats/overview
```

#### Questions
```bash
# Generate questions
POST /api/v1/questions/generate
Body: {
  "document_ids": ["doc-uuid"],
  "num_questions": 10,
  "difficulty": "medium",
  "question_types": ["single_choice", "case_based"],
  "topics": ["Cardiology"],
  "language": "vi",
  "include_case_based": true,
  "include_explanations": true,
  "enable_double_check": true  # AI review
}

Response: {
  "task_id": "task-uuid",
  "status": "pending",
  "message": "Generation started"
}

# Check generation status (polling)
GET /api/v1/questions/generate/{task_id}/status

Response: {
  "task_id": "task-uuid",
  "status": "completed",
  "progress": 1.0,
  "total_questions": 10,
  "generated_questions": 10,
  "review_stats": {
    "total_questions": 10,
    "reviewed": 10,
    "high_accuracy": 8,
    "needs_revision": 2,
    "review_rate": 1.0
  },
  "questions": [
    {
      "id": "q-uuid",
      "question_text": "What is the first-line treatment for STEMI?",
      "options": [...],
      "correct_answer": "A",
      "explanation": "...",
      "ai_review": {
        "status": "approved",
        "accuracy_score": 9,
        "clarity_score": 8,
        "issues": [],
        "suggestions": []
      }
    }
  ]
}

# List questions
GET /api/v1/questions/?difficulty=medium&topic=Cardiology&limit=20

# Update question
PUT /api/v1/questions/{id}
Body: {
  "question_text": "Updated question...",
  "explanation": "Updated explanation..."
}

# Delete question
DELETE /api/v1/questions/{id}

# Export questions
POST /api/v1/questions/export
Body: {
  "question_ids": ["q1", "q2", "q3"],
  "format": "excel",  # json / pdf / docx / excel
  "include_answers": true,
  "include_explanations": true,
  "shuffle_questions": false,
  "shuffle_options": false
}

Response: {
  "download_url": "/exports/quiz_20241226.xlsx"
}

# Semantic search
POST /api/v1/questions/search
Body: {
  "query": "myocardial infarction treatment",
  "document_ids": ["doc1"],
  "top_k": 5
}

# Statistics
GET /api/v1/questions/stats/overview
```

---

## 🧪 Testing

### Backend Tests
```bash
cd backend
source venv/bin/activate
pytest tests/ -v

# Test specific module
pytest tests/test_rag_engine.py -v

# Coverage
pytest --cov=app tests/
```

### Frontend Tests
```bash
cd frontend
npm test

# E2E tests
npm run test:e2e
```

### Manual Testing Flow
1. **Upload tài liệu**:
   - Upload PDF guideline y khoa (VD: ESC Cardiology 2024)
   - Verify status = 'completed'
   - Check num_chunks > 0

2. **Generate câu hỏi**:
   - Chọn document vừa upload
   - Config: 10 câu, medium, enable AI double-check
   - Click "Tạo câu hỏi"
   - Verify polling works, progress updates

3. **Review AI check**:
   - Verify review_stats hiển thị
   - Check từng câu hỏi có ai_review
   - Verify scores (accuracy, clarity)
   - Check issues & suggestions

4. **Edit câu hỏi**:
   - Sửa câu cần revision
   - Apply suggestions từ AI
   - Save changes

5. **Export**:
   - Select questions
   - Export Excel
   - Verify file download

---

## 🔍 AI Double-Check Criteria

LLM đánh giá câu hỏi dựa trên các tiêu chí y khoa:

### 1. Accuracy Score (1-10)
- ✅ **9-10**: Thông tin y khoa hoàn toàn chính xác
- ✅ **7-8**: Chính xác nhưng có thể cải thiện thuật ngữ
- ⚠️ **5-6**: Có một số sai sót nhỏ
- ❌ **1-4**: Thông tin sai lệch nguy hiểm

**Kiểm tra:**
- Đáp án đúng có chính xác 100%?
- Có thông tin sai lệch nguy hiểm?
- Thuật ngữ y khoa chuẩn?
- Guidelines/evidence-based?

### 2. Clarity Score (1-10)
- ✅ **9-10**: Câu hỏi rõ ràng, không mơ hồ
- ✅ **7-8**: Rõ ràng nhưng có thể cải thiện wording
- ⚠️ **5-6**: Hơi mơ hồ hoặc phức tạp
- ❌ **1-4**: Khó hiểu, confusing

**Kiểm tra:**
- Câu hỏi có duy nhất 1 ý?
- Options phân biệt rõ ràng?
- Không có trick questions?
- Ngôn ngữ phù hợp với level?

### 3. Educational Value (1-10)
- ✅ **9-10**: Test kiến thức quan trọng, clinical relevance cao
- ✅ **7-8**: Hữu ích nhưng không critical
- ⚠️ **5-6**: Kiến thức ít quan trọng
- ❌ **1-4**: Trivial, không có giá trị học tập

**Kiểm tra:**
- Test high-yield concepts?
- Có clinical application?
- Phù hợp với mục tiêu học tập?
- Độ khó appropriate?

### 4. Issues Detection
**Common issues:**
- ❌ Đáp án sai
- ❌ Thuật ngữ không chuẩn
- ❌ Thông tin lỗi thời (outdated guidelines)
- ❌ Options overlap hoặc trùng lặp
- ❌ Câu hỏi mơ hồ
- ❌ Giải thích thiếu hoặc sai

### 5. Suggestions
**Improvement suggestions:**
- 💡 Sửa thuật ngữ y khoa
- 💡 Clarify wording
- 💡 Cập nhật guidelines mới
- 💡 Thêm context cho case
- 💡 Improve distractors
- 💡 Add better explanation

### Verdict Logic
```python
if accuracy_score >= 8 and no critical_issues:
    verdict = "APPROVED"  # ✅ Đạt chuẩn
elif accuracy_score >= 6:
    verdict = "NEEDS_REVISION"  # ⚠️ Cần sửa
else:
    verdict = "REJECT"  # ❌ Không đạt
```

---

## 📚 Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.12+ | Language |
| FastAPI | 0.104+ | Web framework |
| Pydantic | 2.5+ | Validation |
| ChromaDB | 0.4+ | Vector database |
| Sentence-Transformers | 2.2+ | Embeddings |
| PyPDF2 | 3.0+ | PDF parsing |
| python-pptx | 0.6+ | PowerPoint parsing |
| python-docx | 1.1+ | Word parsing |
| pdfplumber | 0.10+ | Enhanced PDF extraction |
| OpenAI | 1.6+ | GPT-4 API |
| Anthropic | 0.8+ | Claude API |
| Google GenAI | 0.3+ | Gemini API |
| Uvicorn | 0.24+ | ASGI server |
| SQLAlchemy | 2.0+ | ORM (future) |
| Structlog | 23.2+ | Logging |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18 | UI framework |
| TypeScript | 5 | Type safety |
| Vite | 5 | Build tool |
| TailwindCSS | 3 | Styling |
| Zustand | 4 | State management |
| TanStack Query | 5 | Data fetching |
| Axios | 1.6+ | HTTP client |
| React Router | 6 | Routing |
| Heroicons | 2 | Icons |
| Framer Motion | 11 | Animations |
| React Hot Toast | 2 | Notifications |

### DevOps
- Docker + Docker Compose
- Nginx (for frontend)
- Git

---

## 🐛 Troubleshooting

### Backend không start được

**Lỗi: Port 8000 đã được sử dụng**
```bash
# Kiểm tra process đang dùng port
lsof -i :8000

# Kill process
kill -9 <PID>
```

**Lỗi: Python version không đúng**
```bash
# Kiểm tra version
python3 --version  # Cần >= 3.12

# Cài Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```

**Lỗi: Thiếu API key**
```bash
# Kiểm tra .env
cat backend/.env | grep API_KEY

# Thêm API key
echo "OPENAI_API_KEY=sk-your-key" >> backend/.env
```

**Lỗi: ModuleNotFoundError**
```bash
# Activate venv
source backend/venv/bin/activate

# Reinstall packages
pip install -r backend/requirements.txt
```

### Frontend build lỗi

**Lỗi: npm install failed**
```bash
# Clear cache
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

**Lỗi: Node version không đúng**
```bash
# Kiểm tra Node version
node --version  # Cần >= 20

# Cài Node 20 (Ubuntu/WSL)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

**Lỗi: TypeScript errors**
```bash
# Kiểm tra tsconfig.json
# Đảm bảo có vite-env.d.ts trong src/

# Recreate vite-env.d.ts nếu thiếu
echo '/// <reference types="vite/client" />' > src/vite-env.d.ts
```

### ChromaDB lỗi

**Lỗi: Database corrupted**
```bash
# Xóa và tạo lại
rm -rf backend/data/chroma_db
mkdir -p backend/data/chroma_db

# Re-upload documents để rebuild index
```

**Lỗi: Out of memory khi embedding**
```bash
# Reduce batch size trong rag_engine.py
# Line: embeddings = self.embedding_model.encode(..., batch_size=8)
```

### LLM API lỗi

**Lỗi: Rate limit exceeded**
```bash
# Wait và retry
# Hoặc giảm num_questions trong request
```

**Lỗi: Invalid API key**
```bash
# Verify API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Lỗi: Context length exceeded**
```bash
# Reduce chunk_size trong config.py
DEFAULT_CHUNK_SIZE = 500  # thay vì 1000
```

### Import lỗi

**Lỗi: ModuleNotFoundError: No module named 'pypdf2'**
```bash
# PyPDF2 case-sensitive
pip uninstall pypdf2 PyPDF2
pip install PyPDF2
```

**Lỗi: Property 'env' does not exist on type 'ImportMeta'**
```bash
# Tạo vite-env.d.ts
cat > frontend/src/vite-env.d.ts << 'EOF'
/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_API_URL: string
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}
EOF
```

---

## 🎓 Use Cases

### 1. Đào tạo sinh viên y khoa
- Upload bài giảng PDF/PowerPoint
- Generate 50 câu hỏi ôn tập
- AI double-check đảm bảo accuracy
- Export Excel cho Moodle import

### 2. Thi chứng chỉ chuyên khoa
- Upload clinical guidelines (ESC, AHA, etc.)
- Generate case-based questions
- Filter high accuracy questions (score >= 8)
- Export PDF đề thi

### 3. Self-study cho resident
- Upload journal articles
- Generate diverse question types
- Quiz mode với timer
- Review answers với explanations

### 4. Assessment nhanh cho giảng viên
- Upload lecture notes
- Generate 10 câu easy + 10 medium + 10 hard
- Review & edit với AI suggestions
- Export DOCX để in

### 5. Knowledge verification
- Upload guideline mới
- Generate questions về key points
- AI review phát hiện gaps trong tài liệu
- Supplement với references

---

## 📝 Roadmap

### Phase 1 (Current) ✅
- [x] Document upload & processing
- [x] RAG với ChromaDB
- [x] Multi-LLM support
- [x] Question generation
- [x] AI Double-Check
- [x] Basic frontend

### Phase 2 (Q1 2026)
- [ ] User authentication (JWT)
- [ ] PostgreSQL database
- [ ] Question rating & feedback
- [ ] Collaborative editing
- [ ] Version control cho questions

### Phase 3 (Q2 2026)
- [ ] Spaced repetition algorithm
- [ ] Learning analytics dashboard
- [ ] Performance tracking
- [ ] Adaptive difficulty
- [ ] Mobile app (React Native)

### Phase 4 (Q3 2026)
- [ ] LMS integration (Moodle, Canvas)
- [ ] Real-time collaboration
- [ ] Question marketplace
- [ ] Video/image support
- [ ] Multi-language (English full support)

### Future Ideas
- AI-generated explanations với medical images
- Voice input cho case scenarios
- Interactive clinical simulations
- Integration với PubMed/UpToDate
- Blockchain verified certificates

---

## 👨‍💻 Contributing

Contributions welcome! Please:
1. Fork repo
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 for Python
- Use TypeScript strict mode
- Write tests for new features
- Update documentation
- Add AI review criteria cho medical accuracy

---

## 📧 Contact & Support

- **Issues**: GitHub Issues
- **Email**: [your-email@example.com]
- **Documentation**: This README + API docs at `/docs`

---

## 🙏 Acknowledgments

- **OpenAI** - GPT-4 API
- **Anthropic** - Claude API
- **Google** - Gemini API
- **ChromaDB** team - Excellent vector database
- **Sentence-Transformers** - Multilingual embeddings
- **FastAPI** & **React** communities

Special thanks to all medical educators who provide feedback! 🏥

---

## 📄 License

MIT License - Free for educational use.

For commercial use in healthcare, please contact for licensing.

---

**Built with ❤️ for medical education**

**Happy Learning! 🎓🏥**
