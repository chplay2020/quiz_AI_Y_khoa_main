# Medical Quiz Generator 🏥📚

Hệ thống AI tạo sinh câu hỏi trắc nghiệm từ tài liệu y khoa (slides, guidelines, PDF) sử dụng PocketFlow, RAG và các công nghệ AI tiên tiến.

## 🌟 Tính năng

- **Xử lý đa định dạng**: Hỗ trợ PDF, PowerPoint (PPTX), Word (DOCX), Text
- **RAG-based**: Sử dụng Retrieval Augmented Generation để tạo câu hỏi chính xác
- **PocketFlow Workflow**: Quy trình xử lý modular, dễ mở rộng
- **Đa dạng câu hỏi**: Single choice, Multiple choice, True/False, Case-based
- **Hỗ trợ đa ngôn ngữ**: Tiếng Việt và Tiếng Anh
- **Chuyên khoa y khoa**: Hỗ trợ 20+ chuyên khoa
- **Export đa định dạng**: JSON, PDF, DOCX, Excel

## 🏗️ Kiến trúc

```
medical_quiz_generator/
├── backend/                    # Python FastAPI Backend
│   ├── app/
│   │   ├── api/               # API Routes
│   │   ├── core/              # Core modules (RAG, LLM, Document Processing)
│   │   ├── flows/             # PocketFlow workflows
│   │   ├── models.py          # Pydantic models
│   │   ├── config.py          # Configuration
│   │   └── main.py            # FastAPI app
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                   # React + TypeScript Frontend
│   ├── src/
│   │   ├── api/               # API client
│   │   ├── components/        # React components
│   │   ├── pages/             # Page components
│   │   ├── store/             # Zustand store
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- Node.js 18+
- OpenAI API key (hoặc Anthropic/Google)

### Backend Setup

```bash
cd backend

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Copy và cấu hình environment
cp .env.example .env
# Chỉnh sửa .env với API keys của bạn

# Chạy server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Cài đặt dependencies
npm install

# Chạy development server
npm run dev
```

## 📖 Sử dụng

### 1. Upload tài liệu

1. Truy cập http://localhost:3000
2. Vào trang "Tài liệu"
3. Kéo thả hoặc chọn file PDF/PPTX/DOCX
4. Nhập thông tin metadata (tùy chọn)
5. Đợi hệ thống xử lý

### 2. Tạo câu hỏi

1. Vào trang "Tạo câu hỏi"
2. Chọn tài liệu đã upload
3. Cấu hình:
   - Số lượng câu hỏi
   - Độ khó
   - Ngôn ngữ
   - Bao gồm case-based questions
4. Nhấn "Tạo câu hỏi"
5. Đợi AI tạo câu hỏi

### 3. Quản lý & Export

1. Vào "Ngân hàng câu hỏi"
2. Tìm kiếm, lọc theo tiêu chí
3. Chỉnh sửa câu hỏi nếu cần
4. Export theo định dạng mong muốn

## 🔧 API Endpoints

### Documents
- `POST /api/v1/documents/upload` - Upload tài liệu
- `GET /api/v1/documents/` - Danh sách tài liệu
- `GET /api/v1/documents/{id}` - Chi tiết tài liệu
- `DELETE /api/v1/documents/{id}` - Xóa tài liệu

### Questions
- `POST /api/v1/questions/generate` - Tạo câu hỏi
- `GET /api/v1/questions/generate/{task_id}/status` - Trạng thái tạo câu hỏi
- `GET /api/v1/questions/` - Danh sách câu hỏi
- `PUT /api/v1/questions/{id}` - Cập nhật câu hỏi
- `POST /api/v1/questions/export` - Export câu hỏi
- `POST /api/v1/questions/search` - Tìm kiếm ngữ nghĩa

## 🧠 PocketFlow Workflow

```
Document → Ingestion → Embedding → Retrieval → Generation → Validation
    ↓           ↓           ↓           ↓            ↓           ↓
  Upload    Extract     Store in    Search      LLM Call    Quality
  File      Text       ChromaDB    Context      GPT-4      Check
```

### Nodes:
1. **DocumentIngestionNode**: Xử lý PDF/PPTX/DOCX
2. **EmbeddingNode**: Tạo embeddings với Sentence Transformers
3. **ContextRetrievalNode**: RAG search với ChromaDB
4. **QuestionGenerationNode**: Tạo MCQ với LLM
5. **CaseBasedQuestionNode**: Tạo câu hỏi tình huống
6. **QuestionValidationNode**: Kiểm tra chất lượng

## ⚙️ Cấu hình

### Environment Variables

```env
# LLM Provider
OPENAI_API_KEY=sk-...
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4-turbo-preview

# Embedding
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Limits
MAX_FILE_SIZE_MB=50
MAX_QUESTIONS_PER_REQUEST=50
```

## 🔒 Bảo mật

- API keys được lưu trong environment variables
- CORS được cấu hình cho frontend
- File upload có giới hạn kích thước
- Input validation với Pydantic

## 🛣️ Roadmap

- [ ] Database persistence (PostgreSQL)
- [ ] User authentication
- [ ] PDF/DOCX export
- [ ] Quiz mode với scoring
- [ ] Spaced repetition
- [ ] Team collaboration
- [ ] Mobile app

## 📄 License

MIT License

## 👥 Contributing

Pull requests are welcome! Vui lòng tạo issue trước khi submit PR lớn.

## 📞 Support

- Issues: GitHub Issues
- Email: support@medquiz.ai
