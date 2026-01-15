# Medical Quiz Generator - AI Tạo Câu Hỏi Trắc Nghiệm Y Khoa 🏥

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3+-3178C6.svg)](https://www.typescriptlang.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hệ thống AI tự động tạo câu hỏi trắc nghiệm y khoa từ tài liệu với RAG & LLM**

[Tính Năng](#-tính-năng-chính) • [Công Nghệ](#-công-nghệ-sử-dụng) • [Cài Đặt](#-cài-đặt) • [Sử Dụng](#-hướng-dẫn-sử-dụng) • [Roadmap](#-hướng-phát-triển-tương-lai)

</div>

---

## 📋 Giới thiệu

**Medical Quiz Generator** là một hệ thống AI tiên tiến giúp tự động hóa quá trình tạo câu hỏi trắc nghiệm y khoa từ tài liệu học thuật. Hệ thống kết hợp công nghệ **RAG (Retrieval Augmented Generation)** với các **Large Language Models** tiên tiến để đảm bảo câu hỏi được tạo ra có độ chính xác y khoa cao và phù hợp với nội dung tài liệu.

### 🎯 Vấn đề giải quyết

- ⏱️ **Tiết kiệm thời gian**: Tự động hóa việc soạn câu hỏi trắc nghiệm từ tài liệu dày hàng trăm trang
- 🎓 **Chất lượng cao**: Câu hỏi được AI kiểm tra kỹ lưỡng về độ chính xác y khoa
- 📚 **Đa dạng format**: Hỗ trợ PDF, PowerPoint, Word - các định dạng phổ biến trong y học
- 🔍 **Semantic Search**: Tìm kiếm ngữ nghĩa giúp câu hỏi bám sát nội dung tài liệu
- ✅ **AI Double-Check**: LLM tự động đánh giá và gắn nhãn độ tin cậy cho từng câu hỏi

---

## 🚀 Tính năng chính

### 1. **Xử lý Tài liệu Thông minh**
- 📄 Upload và phân tích các định dạng: **PDF, PPTX, DOCX**
- 🔍 Trích xuất nội dung văn bản, bảng biểu, danh sách
- 📊 Chia nhỏ tài liệu thành các chunk có ngữ nghĩa (chunking)
- 💾 Lưu trữ và quản lý kho tài liệu y khoa

### 2. **RAG Engine - Tìm kiếm Ngữ nghĩa**
- 🧠 **Vector Database**: ChromaDB để lưu trữ embeddings
- 🎯 **Semantic Search**: Sentence Transformers cho tìm kiếm ngữ nghĩa
- 🔗 **Context Retrieval**: Truy xuất ngữ cảnh liên quan cho mỗi câu hỏi
- 📈 **Similarity Ranking**: Xếp hạng độ liên quan của các đoạn văn bản

### 3. **Multi-LLM Support - Đa mô hình AI**
- 🤖 **OpenAI GPT-4/GPT-4-Turbo**: Mô hình mạnh mẽ cho y khoa
- 🧪 **Anthropic Claude**: Claude-3-Opus, Claude-3-Sonnet
- 🌟 **Google Gemini**: Gemini-Pro cho đa dạng hóa
- ⚙️ **Tùy chỉnh**: Dễ dàng thêm các LLM provider khác

### 4. **PocketFlow - Workflow Orchestration**
- 🔄 **Pipeline tự động**: Từ tài liệu → RAG → LLM → Câu hỏi
- 📋 **Custom Nodes**: 
  - `DocumentIngestionNode`: Xử lý và nhúng tài liệu
  - `ContextRetrievalNode`: Tìm kiếm ngữ cảnh liên quan
  - `QuestionGenerationNode`: Sinh câu hỏi từ ngữ cảnh
  - `AIReviewNode`: **AI Double-Check** đánh giá chất lượng
- 🎭 **Batch Processing**: Xử lý hàng loạt câu hỏi hiệu quả
- 🔀 **Conditional Logic**: Điều hướng workflow dựa trên kết quả

### 5. **AI Double-Check - Kiểm tra Chất lượng**
- ✅ **Accuracy Check**: Kiểm tra tính chính xác y khoa
- 🎯 **Relevance Check**: Đánh giá mức độ liên quan với tài liệu
- 💡 **Clarity Check**: Kiểm tra độ rõ ràng câu hỏi và đáp án
- 🏷️ **Confidence Score**: Gắn nhãn mức độ tin cậy (High/Medium/Low)
- 📝 **Suggestions**: Đề xuất cải thiện cho câu hỏi

### 6. **Đa dạng Loại Câu hỏi**
- ✅ **Single Choice**: Trắc nghiệm một đáp án đúng
- ☑️ **Multiple Choice**: Trắc nghiệm nhiều đáp án đúng
- ⭕ **True/False**: Câu hỏi đúng/sai
- 🏥 **Case-based**: Câu hỏi tình huống lâm sàng (vignette)

### 7. **Export đa định dạng**
- 📊 **JSON**: Dữ liệu thô cho xử lý tiếp
- 📗 **Excel**: Dễ dàng chỉnh sửa và quản lý
- 📄 **PDF**: In ấn và phân phối
- 📝 **DOCX**: Tích hợp vào tài liệu Word

### 8. **Giao diện Web hiện đại**
- 🎨 **React + TypeScript**: Giao diện responsive, type-safe
- 💅 **TailwindCSS**: UI đẹp mắt, nhất quán
- 🔥 **Real-time Updates**: Cập nhật tiến trình tạo câu hỏi trực tiếp
- 📱 **Mobile Friendly**: Tương thích đa thiết bị
- 🌙 **Dark Mode Ready**: Sẵn sàng cho chế độ tối

---

## 🛠️ Công nghệ sử dụng

### **Backend Stack**

| Công nghệ | Phiên bản | Vai trò |
|-----------|-----------|---------|
| **Python** | 3.11+ | Ngôn ngữ lập trình chính |
| **FastAPI** | 0.104+ | Web framework, REST API |
| **Pydantic** | 2.5+ | Validation & serialization |
| **ChromaDB** | 0.4+ | Vector database, embeddings storage |
| **Sentence-Transformers** | 2.2+ | Embedding model (semantic search) |
| **PyPDF2 & pdfplumber** | 3.0+, 0.10+ | Xử lý PDF |
| **python-pptx** | 0.6+ | Xử lý PowerPoint |
| **python-docx** | 1.1+ | Xử lý Word |
| **OpenAI SDK** | 1.6+ | Tích hợp GPT-4 |
| **Anthropic SDK** | 0.8+ | Tích hợp Claude |
| **Google Generative AI** | 0.3+ | Tích hợp Gemini |
| **Structlog** | 23.2+ | Structured logging |
| **Uvicorn** | 0.24+ | ASGI server |

### **Frontend Stack**

| Công nghệ | Phiên bản | Vai trò |
|-----------|-----------|---------|
| **React** | 18.2+ | UI library |
| **TypeScript** | 5.3+ | Type-safe JavaScript |
| **Vite** | 5.0+ | Build tool & dev server |
| **TailwindCSS** | 3.4+ | Utility-first CSS framework |
| **React Router** | 6.21+ | Client-side routing |
| **Zustand** | 4.4+ | State management |
| **React Query** | 5.17+ | Server state management |
| **Axios** | 1.6+ | HTTP client |
| **React Hook Form** | 7.49+ | Form handling |
| **Framer Motion** | 10.18+ | Animation library |
| **Recharts** | 2.10+ | Data visualization |
| **Headless UI** | 1.7+ | Accessible UI components |

### **DevOps & Infrastructure**

- 🐳 **Docker & Docker Compose**: Containerization
- 📦 **Multi-stage Builds**: Tối ưu kích thước image
- 🔧 **Nginx**: Reverse proxy cho frontend
- 🌍 **CORS**: Cross-origin resource sharing
- 📝 **Environment Variables**: Quản lý cấu hình

### **AI & ML Components**

- 🧠 **RAG Architecture**: Retrieval Augmented Generation
- 🎯 **Embedding Models**: `all-MiniLM-L6-v2` (Sentence-BERT)
- 🤖 **LLM Providers**: OpenAI, Anthropic, Google
- 🔀 **Workflow Engine**: Custom PocketFlow implementation
- 📊 **Token Management**: Tiktoken cho token counting

---

## 📦 Cài đặt

### **Yêu cầu hệ thống**

- 🐳 Docker & Docker Compose
- 🔑 API Keys: OpenAI / Anthropic / Google (tối thiểu 1)
- 💾 Dung lượng: ~5GB (bao gồm models, images, data)
- 🧠 RAM: Tối thiểu 4GB khuyến nghị 8GB+

### **Cài đặt nhanh với Docker Compose**

```bash
# 1. Clone repository
git clone https://github.com/yourusername/quiz_AI_Y_khoa_main.git
cd quiz_AI_Y_khoa_main/medical_quiz_generator

# 2. Tạo file .env từ template
cp backend/.env.example backend/.env

# 3. Cấu hình API keys trong backend/.env
nano backend/.env
# Thêm ít nhất một trong các API key sau:
# OPENAI_API_KEY=sk-xxx
# ANTHROPIC_API_KEY=sk-ant-xxx
# GOOGLE_API_KEY=xxx

# 4. Build và chạy
docker-compose up --build

# 5. Truy cập ứng dụng
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### **Cài đặt thủ công (Development)**

<details>
<summary>📖 Xem hướng dẫn chi tiết</summary>

#### **Backend**
```bash
cd backend

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Cấu hình .env
cp .env.example .env
nano .env

# Chạy server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### **Frontend**
```bash
cd frontend

# Cài đặt dependencies
npm install

# Chạy dev server
npm run dev
```

</details>

---

## 📖 Hướng dẫn sử dụng

### **1. Upload tài liệu**
1. Truy cập trang **Documents**
2. Kéo thả file PDF/PPTX/DOCX hoặc click để chọn
3. Hệ thống tự động xử lý và tạo embeddings

### **2. Tạo câu hỏi**
1. Vào trang **Generate**
2. Chọn tài liệu nguồn
3. Cấu hình:
   - Số lượng câu hỏi
   - Độ khó (Easy/Medium/Hard)
   - Loại câu hỏi (Single/Multiple/True-False/Case-based)
   - Chủ đề cụ thể (optional)
   - Bật/tắt **AI Double-Check**
4. Click **Generate Questions**
5. Theo dõi tiến trình real-time

### **3. Quản lý câu hỏi**
1. Xem danh sách câu hỏi tại trang **Questions**
2. Lọc theo:
   - Confidence level (High/Medium/Low)
   - Loại câu hỏi
   - Chủ đề
3. Chỉnh sửa câu hỏi nếu cần
4. Export theo định dạng mong muốn

### **4. Export**
- Chọn câu hỏi cần export
- Chọn format: JSON / Excel / PDF / DOCX
- Download file

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│              (React + TypeScript + TailwindCSS)                 │
└────────────────┬────────────────────────────────────────────────┘
                 │ HTTP/REST API
┌────────────────▼────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              API ROUTES LAYER                             │  │
│  │  • /documents  • /questions  • /export                   │  │
│  └──────┬───────────────────────────────────────────────────┘  │
│         │                                                        │
│  ┌──────▼───────────────────────────────────────────────────┐  │
│  │           POCKETFLOW ORCHESTRATION                        │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │  │
│  │  │Ingest  │→ │Retrieve│→ │Generate│→ │Review  │         │  │
│  │  │  Node  │  │  Node  │  │  Node  │  │  Node  │         │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘         │  │
│  └──────┬───────────────────────────────────────────────────┘  │
│         │                                                        │
│  ┌──────▼───────────────────────────────────────────────────┐  │
│  │               CORE BUSINESS LOGIC                         │  │
│  │  ┌────────────┐  ┌──────────┐  ┌─────────────┐          │  │
│  │  │ Document   │  │   RAG    │  │    LLM      │          │  │
│  │  │ Processor  │  │  Engine  │  │  Provider   │          │  │
│  │  └────────────┘  └──────────┘  └─────────────┘          │  │
│  └──────┬───────────────────────────────────────────────────┘  │
└─────────┼──────────────────────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────────┐
│                    DATA & AI SERVICES                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ChromaDB  │  │ OpenAI   │  │Anthropic │  │  Google  │       │
│  │(Vectors) │  │  GPT-4   │  │  Claude  │  │  Gemini  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔮 Hướng phát triển tương lai

### **Phase 1: Nâng cao tính năng AI (Q2 2026)**
- [ ] 🧪 **Advanced RAG**: Hybrid search (BM25 + Vector), Re-ranking
- [ ] 🎯 **Fine-tuned Models**: Fine-tune LLM trên dataset y khoa Việt Nam
- [ ] 🔗 **Multi-hop Reasoning**: Câu hỏi yêu cầu suy luận nhiều bước
- [ ] 📊 **Difficulty Calibration**: AI tự động đánh giá độ khó thực tế
- [ ] 🌐 **Multilingual**: Hỗ trợ Tiếng Anh, Tiếng Việt song song

### **Phase 2: Tích hợp Database & Persistence (Q3 2026)**
- [ ] 🗄️ **PostgreSQL**: Thay thế in-memory storage
- [ ] 👥 **User Authentication**: Đăng nhập, phân quyền (JWT)
- [ ] 📈 **Analytics Dashboard**: Thống kê sử dụng, hiệu suất
- [ ] 🔄 **Version Control**: Theo dõi lịch sử thay đổi câu hỏi
- [ ] 💾 **Backup & Restore**: Sao lưu tự động

### **Phase 3: Cộng tác & Chia sẻ (Q4 2026)**
- [ ] 👫 **Multi-user**: Nhiều người dùng cùng làm việc
- [ ] 🔗 **Question Bank Sharing**: Chia sẻ ngân hàng câu hỏi
- [ ] 💬 **Comments & Reviews**: Góp ý, đánh giá câu hỏi
- [ ] 🏆 **Quality Voting**: Cộng đồng vote câu hỏi chất lượng
- [ ] 📚 **Public Repository**: Kho câu hỏi y khoa mở

### **Phase 4: Mở rộng chức năng (Q1 2027)**
- [ ] 📝 **Tự động tạo Flashcards**: Từ tài liệu
- [ ] 🎓 **Quiz Taking Mode**: Giao diện làm bài thi thực tế
- [ ] 📊 **Performance Tracking**: Theo dõi kết quả học tập
- [ ] 🤖 **Adaptive Learning**: Gợi ý câu hỏi dựa trên năng lực
- [ ] 🔊 **Audio Questions**: Câu hỏi nghe hiểu (radiology, sounds)

### **Phase 5: Enterprise Features (Q2 2027)**
- [ ] 🏢 **White-label**: Tùy chỉnh thương hiệu cho tổ chức
- [ ] 📜 **Compliance**: HIPAA, GDPR compliance
- [ ] 🔐 **SSO Integration**: Single Sign-On
- [ ] 📊 **Advanced Analytics**: Power BI, Tableau integration
- [ ] ☁️ **Cloud Deployment**: AWS, GCP, Azure templates

### **Phase 6: AI & Research (Ongoing)**
- [ ] 🧬 **Specialized Domains**: Chuyên khoa (tim mạch, thần kinh, etc.)
- [ ] 🔬 **Evidence-based**: Liên kết với PubMed, UpToDate
- [ ] 🎨 **Image Questions**: OCR, medical image analysis
- [ ] 🧠 **Explanation Generation**: Tự động tạo đáp án giải thích
- [ ] 📖 **Citation Tracking**: Trích dẫn nguồn cho mỗi câu hỏi

---

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp từ cộng đồng! Vui lòng:

1. Fork repository
2. Tạo branch: `git checkout -b feature/AmazingFeature`
3. Commit: `git commit -m 'Add some AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Tạo Pull Request

---

## 📄 License

Dự án được phân phối dưới giấy phép MIT. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 👨‍💻 Tác giả

**Medical Quiz Generator Team**

- 📧 Email: your.email@example.com
- 🌐 Website: https://yourwebsite.com
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/quiz_AI_Y_khoa_main/discussions)

---

## 🙏 Lời cảm ơn

- Cảm ơn OpenAI, Anthropic, Google vì các LLM APIs tuyệt vời
- Cảm ơn cộng đồng open-source: LangChain, ChromaDB, FastAPI, React
- Cảm ơn các chuyên gia y khoa đã đóng góp ý kiến

---

<div align="center">

**⭐ Nếu project hữu ích, hãy cho chúng tôi một star! ⭐**

Made with ❤️ and 🤖 AI

</div>