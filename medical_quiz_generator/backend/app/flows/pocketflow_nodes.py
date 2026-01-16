# """
# PocketFlow Nodes for Medical Quiz Generation
# Implements the workflow nodes for document processing, RAG, and question generation
# """
# import re
# from typing import Dict, Any, List, Optional
# from dataclasses import dataclass
# import asyncio
# import json
# import structlog

# # PocketFlow-style base classes (simplified implementation)
# from abc import ABC, abstractmethod

# from app.core.document_processor import DocumentProcessor, ProcessedDocument
# from app.core.rag_engine import RAGEngine, RetrievedContext, get_rag_engine
# from app.core.llm_provider import LLMProvider, get_llm_provider
# from app.config import settings

# logger = structlog.get_logger()


# # ============================================
# # Base Node Classes (PocketFlow-style)
# # ============================================

# class BaseNode(ABC):
#     """Base class for all PocketFlow nodes"""
    
#     def __init__(self, name: str = None):
#         self.name = name or self.__class__.__name__
#         self.successors: Dict[str, 'BaseNode'] = {}
    
#     def add_successor(self, node: 'BaseNode', condition: str = "default") -> 'BaseNode':
#         """Add a successor node with an optional condition"""
#         self.successors[condition] = node
#         return node
    
#     @abstractmethod
#     async def prep(self, shared_state: Dict[str, Any]) -> Any:
#         """Prepare data for execution"""
#         pass
    
#     @abstractmethod
#     async def exec(self, prep_result: Any) -> Any:
#         """Execute the node's main logic"""
#         pass
    
#     @abstractmethod
#     async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: Any) -> str:
#         """Post-process and determine next node"""
#         pass
    
#     async def run(self, shared_state: Dict[str, Any]) -> str:
#         """Run the complete node lifecycle"""
#         logger.info(f"Running node: {self.name}")
        
#         prep_result = await self.prep(shared_state)
#         exec_result = await self.exec(prep_result)
#         next_action = await self.post(shared_state, prep_result, exec_result)
        
#         return next_action


# class BatchNode(BaseNode):
#     """Node that processes items in batches"""
    
#     @abstractmethod
#     async def prep(self, shared_state: Dict[str, Any]) -> List[Any]:
#         """Return a list of items to process"""
#         pass
    
#     @abstractmethod
#     async def exec(self, item: Any) -> Any:
#         """Process a single item"""
#         pass
    
#     async def run(self, shared_state: Dict[str, Any]) -> str:
#         """Run batch processing"""
#         logger.info(f"Running batch node: {self.name}")
        
#         items = await self.prep(shared_state)
#         results = []
        
#         for item in items:
#             result = await self.exec(item)
#             results.append(result)
        
#         next_action = await self.post(shared_state, items, results)
#         return next_action


# class Flow:
#     """Flow orchestrator that runs a sequence of nodes"""
    
#     def __init__(self, start_node: BaseNode):
#         self.start_node = start_node
    
#     async def run(self, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
#         """Run the flow from start to end"""
#         shared_state = initial_state or {}
#         current_node = self.start_node
        
#         while current_node:
#             next_action = await current_node.run(shared_state)
            
#             if next_action in current_node.successors:
#                 current_node = current_node.successors[next_action]
#             else:
#                 current_node = None
        
#         return shared_state


# # ============================================
# # Document Processing Nodes
# # ============================================

# class DocumentIngestionNode(BaseNode):
#     """Node for ingesting and procBaseNodeessing documents"""
    
#     def __init__(self):
#         super().__init__("DocumentIngestion")
#         self.processor = DocumentProcessor(
#             chunk_size=settings.CHUNK_SIZE,
#             chunk_overlap=settings.CHUNK_OVERLAP
#         )
    
#     async def prep(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
#         """Extract file info from shared state"""
#         return {
#             'file_path': shared_state.get('file_path'),
#             'document_id': shared_state.get('document_id'),
#             'metadata': shared_state.get('metadata', {})
#         }
    
#     async def exec(self, prep_result: Dict[str, Any]) -> ProcessedDocument:
#         """Process the document"""
#         processed = await self.processor.process_file(
#             file_path=prep_result['file_path'],
#             document_id=prep_result['document_id'],
#             metadata=prep_result['metadata']
#         )
#         return processed
    
#     async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: ProcessedDocument) -> str:
#         """Store processed document in shared state"""
#         shared_state['processed_document'] = exec_result
#         shared_state['num_chunks'] = len(exec_result.chunks)
        
#         logger.info(
#             "Document processed",
#             document_id=exec_result.document_id,
#             chunks=len(exec_result.chunks)
#         )
        
#         return "default"


# class EmbeddingNode(BaseNode):
#     """Node for generating embeddings and storing in vector DB"""
    
#     def __init__(self):
#         super().__init__("Embedding")
#         self.rag_engine = None
    
#     async def prep(self, shared_state: Dict[str, Any]) -> ProcessedDocument:
#         """Get processed document from shared state"""
#         if self.rag_engine is None:
#             self.rag_engine = get_rag_engine()
#         return shared_state.get('processed_document')
    
#     async def exec(self, processed_doc: ProcessedDocument) -> int:
#         """Generate embeddings and store in vector DB"""
#         num_stored = await self.rag_engine.add_document(processed_doc)
#         return num_stored
    
#     async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: int) -> str:
#         """Update shared state with embedding info"""
#         shared_state['chunks_embedded'] = exec_result
        
#         logger.info("Embeddings stored", count=exec_result)
        
#         return "default"


# # ============================================
# # Retrieval Nodes
# # ============================================

# class ContextRetrievalNode(BaseNode):
#     """Node for retrieving relevant contexts for question generation"""
    
#     def __init__(self):
#         super().__init__("ContextRetrieval")
#         self.rag_engine = None
    
#     async def prep(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
#         """Extract retrieval parameters"""
#         if self.rag_engine is None:
#             self.rag_engine = get_rag_engine()
        
#         return {
#             'document_ids': shared_state.get('document_ids', []),
#             'topics': shared_state.get('topics', []),
#             'num_contexts': shared_state.get('num_questions', 10) * 2,
#             'focus_areas': shared_state.get('focus_areas', [])
#         }
    
#     async def exec(self, prep_result: Dict[str, Any]) -> List[RetrievedContext]:
#         """Retrieve relevant contexts"""
#         contexts = await self.rag_engine.search_for_question_generation(
#             topics=prep_result['topics'] or prep_result['focus_areas'],
#             document_ids=prep_result['document_ids'],
#             num_contexts=prep_result['num_contexts']
#         )
#         return contexts
    
#     async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: List[RetrievedContext]) -> str:
#         """Store retrieved contexts"""

#         language = shared_state.get('language', 'vi')

#         # 🔒 FIX 1: LỌC CONTEXT THEO NGÔN NGỮ
#         if language == "vi":
#             filtered_contexts = []
#             for ctx in exec_result:
#                 text = ctx.content.lower()
#                 # heuristic đơn giản để nhận diện tiếng Việt
#                 if any(ch in text for ch in "ăâđêôơưáàảãạíìỉĩịúùủũụýỳỷỹỵ"):
#                     filtered_contexts.append(ctx)

#             exec_result = filtered_contexts

#         shared_state['retrieved_contexts'] = exec_result

#         logger.info(
#             "Contexts retrieved (after language filter)",
#             language=language,
#             count=len(exec_result)
#         )
        
#         if len(exec_result) == 0:
#             return "no_contexts"
        
#         return "default"


# # ============================================
# # Question Generation Nodes
# # ============================================

# class QuestionGenerationNode(BatchNode):
#     """Node for generating questions from contexts"""
    
#     def __init__(self, llm_provider: Optional[LLMProvider] = None):
#         super().__init__("QuestionGeneration")
#         self.llm = llm_provider or get_llm_provider()
#         self.questions_per_context = 1
    
#     async def prep(self, shared_state: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Prepare contexts for question generation"""
#         contexts = shared_state.get('retrieved_contexts', [])
#         num_questions = shared_state.get('num_questions', 10)
#         difficulty = shared_state.get('difficulty', 'medium')
#         question_types = shared_state.get('question_types', ['single_choice'])
#         language = shared_state.get('language', 'vi')
        
#         # Calculate how many contexts to use and questions per context
#         num_contexts = min(len(contexts), num_questions)
#         self.questions_per_context = 1
        
#         logger.info(
#             "Question generation strategy",
#             total_contexts_available=len(contexts),
#             num_contexts_to_use=num_contexts,
#             questions_per_context=self.questions_per_context,
#             target_questions=num_questions
#         )
        
#         # Prepare batch items
#         items = []
#         for ctx in contexts[:num_contexts]:
#             items.append({
#                 'context': ctx,
#                 'difficulty': difficulty,
#                 'question_types': question_types,
#                 'language': language
#             })
        
#         return items
    
#     async def exec(self, item: Dict[str, Any]) -> Dict[str, Any]:
#         """Generate questions for a single context"""
#         context = item['context']
#         difficulty = item['difficulty']
#         language = item['language']
        
#         # Build the prompt
#         prompt = self._build_question_prompt(
#             context=context.content[:1500],
#             difficulty=difficulty,
#             language=language,
#             num_questions=self.questions_per_context
#         )
        
#         system_prompt = self._get_system_prompt(language)
        
#         try:
#             result = await self.llm.generate_structured(
#                 prompt=prompt,
#                 system_prompt=system_prompt,
#                 temperature=0.4
#             )
            
#             # Handle case where result is a string instead of dict
#             if isinstance(result, str):
#                 import json
#                 try:
#                     result = json.loads(result)
#                 except json.JSONDecodeError:
#                     logger.error("Failed to parse LLM response as JSON", response=result[:200])
#                     return {'questions': [], 'error': 'Invalid JSON response'}
            
#             # Ensure result is a dict
#             if not isinstance(result, dict):
#                 return {'questions': [], 'error': 'Invalid response format'}
            
#             # Add context reference
#             if 'questions' in result:
#                 for q in result['questions']:
#                     q['source_chunk_id'] = context.chunk_id
#                     q['document_id'] = context.document_id
#                     q['reference_text'] = context.content[:500]
            
#             return result
            
#         except Exception as e:
#             logger.error("Question generation failed", error=str(e))
#             return {'questions': [], 'error': str(e)}
    
#     async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: List[Dict[str, Any]]) -> str:
#         """Aggregate generated questions"""
#         all_questions = []
        
#         logger.info("Aggregating question generation results", 
#                    num_results=len(exec_result),
#                    num_contexts_processed=len(prep_result) if isinstance(prep_result, list) else 1)
        
#         for i, result in enumerate(exec_result):
#             questions = result.get('questions', [])
#             error = result.get('error')
            
#             if error:
#                 logger.warning(f"Context {i} had error", error=error)
            
#             if isinstance(questions, list):
#                 all_questions.extend(questions)
#                 logger.info(f"Context {i} generated {len(questions)} questions")
#             else:
#                 logger.error(
#                     "Invalid questions format from LLM",
#                     context_index=i,
#                     questions_type=type(questions).__name__,
#                     questions_value=str(questions)[:200]
#                 )
        
#         target = shared_state.get("num_questions", len(all_questions))

#         if len(all_questions) > target:
#             all_questions = all_questions[:target]

#         missing = target - len(all_questions)

#         shared_state["generated_questions"] = all_questions
#         shared_state["missing_questions"] = missing


#         logger.info(
#             "Question generation summary",
#             target=target,
#             generated=len(all_questions),
#             missing=missing
#         )

#         if missing > 0:
#             logger.warning(
#                 "Not enough questions generated, missing questions",
#                 missing=missing
#             )
        
#         return "default"
    
#     def _get_system_prompt(self, language: str) -> str:
#         """Get system prompt based on language"""
#         if language == 'vi':
#             return """Bạn là một chuyên gia y khoa giàu kinh nghiệm trong việc tạo câu hỏi trắc nghiệm cho đào tạo y khoa.

#                         ⚠️ QUY TẮC BẮT BUỘC (KHÔNG ĐƯỢC VI PHẠM):
#                         - BẮT BUỘC viết 100% nội dung bằng TIẾNG VIỆT
#                         - TUYỆT ĐỐI KHÔNG sử dụng tiếng Anh, kể cả thuật ngữ
#                         - Nếu nội dung gốc có tiếng Anh, PHẢI dịch sang tiếng Việt chuẩn y khoa
#                         - Nếu vi phạm, câu trả lời bị coi là KHÔNG HỢP LỆ

#                         YÊU CẦU CHUYÊN MÔN:
#                             1. Câu hỏi phải dựa trên nội dung được cung cấp
#                             2. Đáp án đúng phải được hỗ trợ bởi nội dung gốc
#                             3. Các đáp án nhiễu phải hợp lý nhưng rõ ràng là sai
#                             4. Giải thích phải chi tiết và mang tính giáo dục
#                             5. Sử dụng thuật ngữ y khoa chuẩn tiếng Việt
#                         LUÔN LUÔN trả về JSON hợp lệ và KHÔNG kèm markdown."""
#         else:
#             return """You are an expert medical educator specializing in creating high-quality multiple choice questions for medical training.
#                         You create accurate, clinically relevant questions that follow medical education standards.

#                     Rules:
#                         1. Questions must be based on the provided content
#                         2. Correct answers must be supported by the source material
#                         3. Distractors must be plausible but clearly incorrect
#                         4. Explanations must be detailed and educational
#                         5. Use standard medical terminology
#                     Always return valid JSON."""
    
#     def _build_question_prompt(
#         self,
#         context: str,
#         difficulty: str,
#         language: str,
#         num_questions: int = 1
#     ) -> str:
#         """Build the question generation prompt"""
#         difficulty_instruction = {
#             'easy': 'Câu hỏi đơn giản, kiểm tra kiến thức cơ bản' if language == 'vi' else 'Simple questions testing basic knowledge',
#             'medium': 'Câu hỏi trung bình, yêu cầu hiểu và áp dụng kiến thức' if language == 'vi' else 'Moderate questions requiring understanding and application',
#             'hard': 'Câu hỏi khó, yêu cầu phân tích và tổng hợp kiến thức' if language == 'vi' else 'Difficult questions requiring analysis and synthesis'
#         }
        
#         if language == 'vi':
#             prompt = f"""Dựa trên nội dung y khoa sau đây, hãy tạo {num_questions} câu hỏi trắc nghiệm.

#             NỘI DUNG:
#             {context}

#             YÊU CẦU:
#                 - Độ khó: {difficulty_instruction.get(difficulty, difficulty_instruction['medium'])}
#                 - Mỗi câu hỏi có 4 lựa chọn (A, B, C, D)
#                 - Chỉ có 1 đáp án đúng
#                 - Cung cấp giải thích chi tiết cho đáp án đúng

#                 Trả về JSON theo format sau:
# {{
#     "questions": [
#         {{
#             "question_text": "Nội dung câu hỏi",
#             "question_type": "single_choice",
#             "difficulty": "{difficulty}",
#             "options": [
#                 {{"id": "A", "text": "Lựa chọn A", "is_correct": false}},
#                 {{"id": "B", "text": "Lựa chọn B", "is_correct": true}},
#                 {{"id": "C", "text": "Lựa chọn C", "is_correct": false}},
#                 {{"id": "D", "text": "Lựa chọn D", "is_correct": false}}
#             ],
#             "correct_answer": "B",
#             "explanation": "Giải thích chi tiết tại sao B là đáp án đúng...",
#             "topic": "Chủ đề của câu hỏi",
#             "keywords": ["từ khóa 1", "từ khóa 2"]
#         }}
#     ]
# }}"""
#         else:
#             prompt = f"""Based on the following medical content, create {num_questions} multiple choice questions.

# CONTENT:
# {context}

# REQUIREMENTS:
# - Difficulty: {difficulty_instruction.get(difficulty, difficulty_instruction['medium'])}
# - Each question has 4 options (A, B, C, D)
# - Only 1 correct answer
# - Provide detailed explanation for the correct answer

# Return JSON in this format:
# {{
#     "questions": [
#         {{
#             "question_text": "Question content",
#             "question_type": "single_choice",
#             "difficulty": "{difficulty}",
#             "options": [
#                 {{"id": "A", "text": "Option A", "is_correct": false}},
#                 {{"id": "B", "text": "Option B", "is_correct": true}},
#                 {{"id": "C", "text": "Option C", "is_correct": false}},
#                 {{"id": "D", "text": "Option D", "is_correct": false}}
#             ],
#             "correct_answer": "B",
#             "explanation": "Detailed explanation why B is correct...",
#             "topic": "Topic of the question",
#             "keywords": ["keyword1", "keyword2"]
#         }}
#     ]
# }}"""
        
#         return prompt


# class CaseBasedQuestionNode(BaseNode):
#     """Node for generating case-based clinical questions"""
    
#     def __init__(self, llm_provider: Optional[LLMProvider] = None):
#         super().__init__("CaseBasedQuestion")
#         self.llm = llm_provider or get_llm_provider()
    
#     async def prep(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
#         """Prepare for case-based question generation"""
#         return {
#             'contexts': shared_state.get('retrieved_contexts', []),
#             'language': shared_state.get('language', 'vi'),
#             'num_cases': shared_state.get('num_case_questions', 2)
#         }
    
#     async def exec(self, prep_result: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Generate case-based questions"""
#         contexts = prep_result['contexts']
#         language = prep_result['language']
#         num_cases = prep_result['num_cases']
        
#         if not contexts:
#             return []
        
#         # Combine multiple contexts for richer case scenarios
#         combined_context = "\n\n".join([ctx.content for ctx in contexts[:5]])
        
#         prompt = self._build_case_prompt(combined_context, language, num_cases)
        
#         try:
#             result = await self.llm.generate_structured(
#                 prompt=prompt,
#                 system_prompt=self._get_system_prompt(language),
#                 temperature=0.5
#             )
            
#             # Handle case where result is a string instead of dict
#             if isinstance(result, str):
#                 import json
#                 try:
#                     result = json.loads(result)
#                 except json.JSONDecodeError:
#                     logger.error("Failed to parse case-based response as JSON")
#                     return []
            
#             if not isinstance(result, dict):
#                 return []
                
#             return result.get('cases', [])
#         except Exception as e:
#             logger.error("Case-based question generation failed", error=str(e))
#             return []
    
#     async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: List[Dict[str, Any]]) -> str:
#         """Store case-based questions"""
#         existing_questions = shared_state.get('generated_questions', [])
        
#         # Convert cases to question format
#         for case in exec_result:
#             if 'questions' in case:
#                 for q in case['questions']:
#                     q['question_type'] = 'case_based'
#                     q['case_scenario'] = case.get('scenario', '')
#                 existing_questions.extend(case['questions'])
        
#         shared_state['generated_questions'] = existing_questions
        
#         return "default"
    
#     def _get_system_prompt(self, language: str) -> str:
#         if language == 'vi':
#             return """Bạn là một bác sĩ lâm sàng giàu kinh nghiệm. Hãy tạo các tình huống lâm sàng thực tế 
# với các câu hỏi trắc nghiệm liên quan. Các tình huống phải giống như gặp trong thực hành lâm sàng."""
#         return """You are an experienced clinical physician. Create realistic clinical case scenarios 
# with related multiple choice questions. Scenarios should be similar to real clinical practice."""
    
#     def _build_case_prompt(self, context: str, language: str, num_cases: int) -> str:
#         if language == 'vi':
#             return f"""Dựa trên kiến thức y khoa sau, tạo {num_cases} tình huống lâm sàng với câu hỏi.

# KIẾN THỨC:
# {context}

# Trả về JSON:
# {{
#     "cases": [
#         {{
#             "scenario": "Mô tả bệnh nhân: tuổi, giới, lý do đến khám, triệu chứng...",
#             "questions": [
#                 {{
#                     "question_text": "Câu hỏi về chẩn đoán/điều trị...",
#                     "options": [...],
#                     "correct_answer": "A",
#                     "explanation": "..."
#                 }}
#             ]
#         }}
#     ]
# }}"""
#         else:
#             return f"""Based on this medical knowledge, create {num_cases} clinical case scenarios with questions.

# KNOWLEDGE:
# {context}

# Return JSON:
# {{
#     "cases": [
#         {{
#             "scenario": "Patient description: age, gender, chief complaint, symptoms...",
#             "questions": [
#                 {{
#                     "question_text": "Question about diagnosis/treatment...",
#                     "options": [...],
#                     "correct_answer": "A",
#                     "explanation": "..."
#                 }}
#             ]
#         }}
#     ]
# }}"""


# class QuestionValidationNode(BaseNode):
#     """Node for validating generated questions"""

#     def __init__(self):
#         super().__init__("QuestionValidation")

#     async def prep(self, shared_state: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Get questions to validate"""
#         return shared_state.get('generated_questions', [])

#     async def exec(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         validated = []

#         language = self.shared_state.get("language", "vi") \
#             if hasattr(self, "shared_state") else "vi"

#         for q in questions:
#             if not isinstance(q, dict):
#                 continue

#             # 🔒 FIX 3: CHẶN TIẾNG ANH
#             if language == "vi":
#                 text = (q.get("question_text", "") + " " + q.get("explanation", "")).lower()
#                 if re.search(r"\b(the|is|are|which|what|based on|according to)\b", text):
#                     logger.warning(
#                         "Filtered English question",
#                         question=q.get("question_text", "")[:50]
#                     )
#                     continue

#             if self._is_valid_question(q):
#                 validated.append(q)
#             else:
#                 logger.warning(
#                     "Invalid question filtered",
#                     question=q.get("question_text", "")[:50]
#                 )
#         return validated


#     async def post(
#         self,
#         shared_state: Dict[str, Any],
#         prep_result: Any,
#         exec_result: List[Dict[str, Any]]
#     ) -> str:
#         """Store validated questions"""
#         shared_state['validated_questions'] = exec_result
#         shared_state['validation_stats'] = {
#             'total': len(prep_result),
#             'valid': len(exec_result),
#             'filtered': len(prep_result) - len(exec_result)
#         }
#         return "default"

#     def _is_valid_question(self, question: Dict[str, Any]) -> bool:
#         """Check if a question is valid"""

#         # Required fields
#         required = ['question_text', 'options', 'correct_answer']
#         if not all(key in question for key in required):
#             return False

#         # Question text checks
#         question_text = question.get('question_text')
#         if not isinstance(question_text, str) or len(question_text.strip()) < 10:
#             return False

#         # Options checks
#         options = question.get('options')
#         if not isinstance(options, list) or len(options) < 2:
#             return False

#         # Validate option structure (FIX CHÍNH)
#         option_ids = []
#         for opt in options:
#             if not isinstance(opt, dict):
#                 return False

#             opt_id = opt.get('id')
#             if not isinstance(opt_id, str) or not opt_id:
#                 return False

#             option_ids.append(opt_id)

#         # Check that correct answer exists in options
#         correct = question.get('correct_answer')
#         if not isinstance(correct, str) or correct not in option_ids:
#             return False

#         # Check that exactly one option is marked correct
#         correct_count = sum(1 for opt in options if opt.get('is_correct', False))

#         if correct_count != 1:
#             # Auto-fix: normalize is_correct flags
#             for opt in options:
#                 opt['is_correct'] = (opt.get('id') == correct)

#         return True



# class AIDoubleCheckNode(BaseNode):
#     """
#     Node for AI-powered double-checking of generated questions.
#     Uses LLM to verify medical accuracy, clarity, and educational value.
#     """
    
#     def __init__(self, llm_provider: Optional[LLMProvider] = None):
#         super().__init__("AIDoubleCheck")
#         self.llm = llm_provider or LLMProvider()
    
#     async def prep(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
#         """Get validated questions and context for review"""
#         return {
#             'questions': shared_state.get('validated_questions', []),
#             'context': shared_state.get('retrieved_context', []),
#             'topic': shared_state.get('topic', ''),
#             'enable_double_check': shared_state.get('enable_double_check', True)
#         }
    
#     async def exec(self, prep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Use AI to double-check questions for accuracy and quality"""
#         questions = prep_data['questions']
        
#         if not prep_data.get('enable_double_check', True) or not questions:
#             # Return questions as-is with default review
#             for q in questions:
#                 q['ai_review'] = {
#                     'status': 'skipped',
#                     'accuracy_score': None,
#                     'clarity_score': None,
#                     'suggestions': [],
#                     'reviewed': False
#                 }
#             return questions
        
#         logger.info("Starting AI double-check", num_questions=len(questions))
        
#         reviewed_questions = []
        
#         # Process in batches of 5 for efficiency
#         batch_size = 5
#         for i in range(0, len(questions), batch_size):
#             batch = questions[i:i + batch_size]
            
#             prompt = self._build_review_prompt(batch, prep_data.get('context', []))
            
#             try:
#                 response = await self.llm.generate(
#                     prompt,
#                     max_tokens=2000,
#                     temperature=0.3
#                 )
                
#                 reviews = self._parse_review_response(response)
                
#                 for j, q in enumerate(batch):
#                     if j < len(reviews):
#                         q['ai_review'] = reviews[j]
#                     else:
#                         q['ai_review'] = self._default_review()
#                     reviewed_questions.append(q)
                    
#             except Exception as e:
#                 logger.error("AI review failed", error=str(e))
#                 for q in batch:
#                     q['ai_review'] = self._default_review(error=str(e))
#                     reviewed_questions.append(q)
        
#         return reviewed_questions
    
#     async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: List[Dict[str, Any]]) -> str:
#         """Store reviewed questions and generate report"""
#         shared_state['reviewed_questions'] = exec_result
        
#         # Calculate review statistics
#         total = len(exec_result)
#         reviewed = sum(1 for q in exec_result if q.get('ai_review', {}).get('reviewed', False))
        
#         high_accuracy = sum(1 for q in exec_result 
#                           if (q.get('ai_review', {}).get('accuracy_score') or 0) >= 8)
#         needs_revision = sum(1 for q in exec_result 
#                            if (q.get('ai_review', {}).get('accuracy_score') or 10) < 6)
        
#         shared_state['review_stats'] = {
#             'total_questions': total,
#             'reviewed': reviewed,
#             'high_accuracy': high_accuracy,
#             'needs_revision': needs_revision,
#             'review_rate': reviewed / total if total > 0 else 0
#         }
        
#         logger.info(
#             "AI double-check completed",
#             total=total,
#             reviewed=reviewed,
#             high_accuracy=high_accuracy,
#             needs_revision=needs_revision
#         )
        
#         return "default"
    
#     def _build_review_prompt(self, questions: List[Dict], context: List) -> str:
#         """Build prompt for AI review"""
#         questions_text = ""
#         for i, q in enumerate(questions, 1):
#             options_text = "\n".join([
#                 f"   {opt.get('id', chr(64+j))}. {opt.get('text', '')}" 
#                 for j, opt in enumerate(q.get('options', []), 1)
#             ])
#             questions_text += f"""
# Question {i}:
# {q.get('question_text', '')}
# Options:
# {options_text}
# Correct Answer: {q.get('correct_answer', '')}
# Explanation: {q.get('explanation', 'N/A')}
# ---
# """
        
#         return f"""You are a medical education expert reviewing quiz questions for accuracy and quality.

# Review each question below and provide:
# 1. Accuracy Score (1-10): Is the medical information correct?
# 2. Clarity Score (1-10): Is the question clear and unambiguous?
# 3. Educational Value (1-10): Does it test important medical knowledge?
# 4. Issues: List any problems found (incorrect info, ambiguous wording, etc.)
# 5. Suggestions: How to improve the question
# 6. Verdict: APPROVED, NEEDS_REVISION, or REJECT

# Questions to review:
# {questions_text}

# Respond in JSON format:
# {{
#     "reviews": [
#         {{
#             "question_index": 1,
#             "accuracy_score": 8,
#             "clarity_score": 9,
#             "educational_value": 7,
#             "issues": ["Minor issue description"],
#             "suggestions": ["Suggestion to improve"],
#             "verdict": "APPROVED",
#             "corrected_answer": null,
#             "corrected_explanation": null
#         }}
#     ]
# }}

# Important:
# - Be strict about medical accuracy
# - Flag any potentially dangerous misinformation
# - If the correct answer is wrong, provide the corrected answer
# - Vietnamese medical terminology should be accurate"""
    
#     def _parse_review_response(self, response: str) -> List[Dict[str, Any]]:
#         """Parse AI review response"""
#         import json
#         import re
        
#         reviews = []
        
#         try:
#             # Extract JSON from response
#             json_match = re.search(r'\{[\s\S]*\}', response)
#             if json_match:
#                 data = json.loads(json_match.group())
                
#                 for review in data.get('reviews', []):
#                     reviews.append({
#                         'status': review.get('verdict', 'APPROVED').lower(),
#                         'accuracy_score': review.get('accuracy_score'),
#                         'clarity_score': review.get('clarity_score'),
#                         'educational_value': review.get('educational_value'),
#                         'issues': review.get('issues', []),
#                         'suggestions': review.get('suggestions', []),
#                         'corrected_answer': review.get('corrected_answer'),
#                         'corrected_explanation': review.get('corrected_explanation'),
#                         'reviewed': True
#                     })
#         except json.JSONDecodeError as e:
#             logger.error("Failed to parse AI review", error=str(e))
        
#         return reviews
    
#     def _default_review(self, error: str = None) -> Dict[str, Any]:
#         """Return default review when AI check is skipped or failed"""
#         return {
#             'status': 'error' if error else 'skipped',
#             'accuracy_score': None,
#             'clarity_score': None,
#             'educational_value': None,
#             'issues': [f"Review failed: {error}"] if error else [],
#             'suggestions': [],
#             'reviewed': False,
#             'error': error
#         }


# # ============================================
# # Flow Builder
# # ============================================

# def create_document_processing_flow() -> Flow:
#     """Create a flow for document ingestion and embedding"""
#     ingestion_node = DocumentIngestionNode()
#     embedding_node = EmbeddingNode()
    
#     ingestion_node.add_successor(embedding_node)
    
#     return Flow(ingestion_node)


# def create_question_generation_flow(
#     llm_provider: Optional[LLMProvider] = None,
#     include_case_based: bool = False,
#     enable_double_check: bool = True
# ) -> Flow:
#     """Create a flow for question generation with optional AI double-check"""
#     retrieval_node = ContextRetrievalNode()
#     question_node = QuestionGenerationNode(llm_provider)
#     validation_node = QuestionValidationNode()
#     double_check_node = AIDoubleCheckNode(llm_provider)
    
#     retrieval_node.add_successor(question_node)
    
#     if include_case_based:
#         case_node = CaseBasedQuestionNode(llm_provider)
#         question_node.add_successor(case_node)
#         case_node.add_successor(validation_node)
#     else:
#         question_node.add_successor(validation_node)
    
#     # Add AI double-check as final step
#     if enable_double_check:
#         validation_node.add_successor(double_check_node)
    
#     return Flow(retrieval_node)


# def create_full_pipeline_flow(
#     llm_provider: Optional[LLMProvider] = None,
#     enable_double_check: bool = True
# ) -> Flow:
#     """Create a complete pipeline from document to questions with AI review"""
#     # Document processing
#     ingestion_node = DocumentIngestionNode()
#     embedding_node = EmbeddingNode()
    
#     # Question generation
#     retrieval_node = ContextRetrievalNode()
#     question_node = QuestionGenerationNode(llm_provider)
#     validation_node = QuestionValidationNode()
#     double_check_node = AIDoubleCheckNode(llm_provider)
    
#     # Connect the flow
#     ingestion_node.add_successor(embedding_node)
#     embedding_node.add_successor(retrieval_node)
#     retrieval_node.add_successor(question_node)
#     question_node.add_successor(validation_node)
    
#     # Add AI double-check as final step
#     if enable_double_check:
#         validation_node.add_successor(double_check_node)
    
#     return Flow(ingestion_node)



"""
PocketFlow Nodes for Medical Quiz Generation
Implements the workflow nodes for document processing, RAG, and question generation
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio
import json

from requests import options
import structlog

# PocketFlow-style base classes (simplified implementation)
from abc import ABC, abstractmethod

from app.core.document_processor import DocumentProcessor, ProcessedDocument
from app.core.rag_engine import RAGEngine, RetrievedContext, get_rag_engine
from app.core.llm_provider import LLMProvider, get_llm_provider
from app.config import settings

logger = structlog.get_logger()


# ============================================
# Base Node Classes (PocketFlow-style)
# ============================================

class BaseNode(ABC):
    """Base class for all PocketFlow nodes"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.successors: Dict[str, 'BaseNode'] = {}
    
    def add_successor(self, node: 'BaseNode', condition: str = "default") -> 'BaseNode':
        """Add a successor node with an optional condition"""
        self.successors[condition] = node
        return node
    
    @abstractmethod
    async def prep(self, shared_state: Dict[str, Any]) -> Any:
        """Prepare data for execution"""
        pass
    
    @abstractmethod
    async def exec(self, prep_result: Any) -> Any:
        """Execute the node's main logic"""
        pass
    
    @abstractmethod
    async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: Any) -> str:
        """Post-process and determine next node"""
        pass
    
    async def run(self, shared_state: Dict[str, Any]) -> str:
        """Run the complete node lifecycle"""
        logger.info(f"Running node: {self.name}")
        
        prep_result = await self.prep(shared_state)
        exec_result = await self.exec(prep_result)
        next_action = await self.post(shared_state, prep_result, exec_result)
        
        return next_action


class BatchNode(BaseNode):
    """Node that processes items in batches"""
    
    @abstractmethod
    async def prep(self, shared_state: Dict[str, Any]) -> List[Any]:
        """Return a list of items to process"""
        pass
    
    @abstractmethod
    async def exec(self, item: Any) -> Any:
        """Process a single item"""
        pass
    
    async def run(self, shared_state: Dict[str, Any]) -> str:
        """Run batch processing"""
        logger.info(f"Running batch node: {self.name}")
        
        items = await self.prep(shared_state)
        results = []
        
        for item in items:
            result = await self.exec(item)
            results.append(result)
        
        next_action = await self.post(shared_state, items, results)
        return next_action


class Flow:
    """Flow orchestrator that runs a sequence of nodes"""
    
    def __init__(self, start_node: BaseNode):
        self.start_node = start_node
    
    async def run(self, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the flow from start to end"""
        shared_state = initial_state or {}
        current_node = self.start_node
        
        while current_node:
            next_action = await current_node.run(shared_state)
            
            if next_action in current_node.successors:
                current_node = current_node.successors[next_action]
            else:
                current_node = None
        
        return shared_state


# ============================================
# Document Processing Nodes
# ============================================

class DocumentIngestionNode(BaseNode):
    """Node for ingesting and processing documents"""
    
    def __init__(self):
        super().__init__("DocumentIngestion")
        self.processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    async def prep(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract file info from shared state"""
        return {
            'file_path': shared_state.get('file_path'),
            'document_id': shared_state.get('document_id'),
            'metadata': shared_state.get('metadata', {})
        }
    
    async def exec(self, prep_result: Dict[str, Any]) -> ProcessedDocument:
        """Process the document"""
        processed = await self.processor.process_file(
            file_path=prep_result['file_path'],
            document_id=prep_result['document_id'],
            metadata=prep_result['metadata']
        )
        return processed
    
    async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: ProcessedDocument) -> str:
        """Store processed document in shared state"""
        shared_state['processed_document'] = exec_result
        shared_state['num_chunks'] = len(exec_result.chunks)
        
        logger.info(
            "Document processed",
            document_id=exec_result.document_id,
            chunks=len(exec_result.chunks)
        )
        
        return "default"


class EmbeddingNode(BaseNode):
    """Node for generating embeddings and storing in vector DB"""
    
    def __init__(self):
        super().__init__("Embedding")
        self.rag_engine = None
    
    async def prep(self, shared_state: Dict[str, Any]) -> ProcessedDocument:
        """Get processed document from shared state"""
        if self.rag_engine is None:
            self.rag_engine = get_rag_engine()
        return shared_state.get('processed_document')
    
    async def exec(self, processed_doc: ProcessedDocument) -> int:
        """Generate embeddings and store in vector DB"""
        num_stored = await self.rag_engine.add_document(processed_doc)
        return num_stored
    
    async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: int) -> str:
        """Update shared state with embedding info"""
        shared_state['chunks_embedded'] = exec_result
        
        logger.info("Embeddings stored", count=exec_result)
        
        return "default"


# ============================================
# Retrieval Nodes
# ============================================

class ContextRetrievalNode(BaseNode):
    """Node for retrieving relevant contexts for question generation"""
    
    def __init__(self):
        super().__init__("ContextRetrieval")
        self.rag_engine = None
    
    async def prep(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract retrieval parameters"""
        if self.rag_engine is None:
            self.rag_engine = get_rag_engine()
        
        return {
            'document_ids': shared_state.get('document_ids', []),
            'topics': shared_state.get('topics', []),
            'num_contexts': shared_state.get('num_questions', 10) * 2,
            'focus_areas': shared_state.get('focus_areas', [])
        }
    
    async def exec(self, prep_result: Dict[str, Any]) -> List[RetrievedContext]:
        """Retrieve relevant contexts"""
        contexts = await self.rag_engine.search_for_question_generation(
            topics=prep_result['topics'] or prep_result['focus_areas'],
            document_ids=prep_result['document_ids'],
            num_contexts=prep_result['num_contexts']
        )
        return contexts
    
    async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: List[RetrievedContext]) -> str:
        """Store retrieved contexts (and optionally filter by output language)."""
        # language = shared_state.get('language', 'vi')
        # contexts = exec_result or []

        # # FIX 1: lọc chunk nhiễu/watermark và (tuỳ chọn) giữ đúng ngôn ngữ yêu cầu
        # if language == 'vi':
        #     import re

        #     def _looks_vietnamese(s: str) -> bool:
        #         if not s:
        #             return False
        #         if re.search(r"[À-ỹ]", s):
        #             return True
        #         s2 = f" {s.lower()} "
        #         hits = sum(w in s2 for w in [" và ", " của ", " không ", " được ", " bệnh ", " điều trị ", " cấp cứu "])
        #         return hits >= 2

        #     filtered: List[RetrievedContext] = []
        #     for c in contexts:
        #         txt = getattr(c, 'content', '') or ''
        #         txt_wo_mark = re.sub(r"(?i)\byhocdata\.com\b", " ", txt).strip()
        #         if len(txt_wo_mark) < 80:
        #             continue
        #         if _looks_vietnamese(txt_wo_mark):
        #             filtered.append(c)

        #     if filtered:
        #         contexts = filtered

        # shared_state['retrieved_contexts'] = contexts
        # logger.info("Contexts retrieved", count=len(contexts))

        # if len(contexts) == 0:
        #     return "no_contexts"

        # return "default"

        shared_state["retrieved_contexts"] = exec_result

        logger.warning(
            "CONTEXT RETRIEVAL RESULT",
            count=len(exec_result),
            sample=exec_result[0].content[:200] if exec_result else "EMPTY"
        )

        if len(exec_result) == 0:
            logger.error("NO CONTEXT RETRIEVED – FLOW STUCK")
            return "no_contexts"

        return "default"


class QuestionGenerationNode(BatchNode):
    """Node for generating questions from contexts"""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        super().__init__("QuestionGeneration")
        self.llm = llm_provider or get_llm_provider()
        self.questions_per_context = 1
    
    async def prep(self, shared_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare contexts for question generation"""
        contexts = shared_state.get('retrieved_contexts', [])
        target_questions = int(shared_state.get('num_questions', 10))
        difficulty = shared_state.get('difficulty', 'medium')
        question_types = shared_state.get('question_types', ['single_choice'])
        language = shared_state.get('language', 'vi')

        # FIX 2: tạo buffer (đề phòng JSON lỗi / bị filter ở Validation) rồi sẽ cắt đúng target ở post/validation
        buffer_target = target_questions + max(2, target_questions // 3)

        if not contexts:
            return []

        # Dùng tối đa `buffer_target` contexts (mỗi context thường tạo 1 câu), nếu thiếu context thì phân bổ nhiều câu/context
        num_contexts_to_use = min(len(contexts), buffer_target)
        selected_contexts = contexts[:num_contexts_to_use]

        base = buffer_target // num_contexts_to_use
        remainder = buffer_target % num_contexts_to_use

        logger.info(
            "Question generation strategy",
            total_contexts_available=len(contexts),
            num_contexts_to_use=num_contexts_to_use,
            target_questions=target_questions,
            buffer_target=buffer_target,
            base_per_context=base,
            remainder=remainder,
            language=language
        )

        # items: List[Dict[str, Any]] = []
        # for i, ctx in enumerate(selected_contexts):
        #     n_q = base + (1 if i < remainder else 0)
        #     if n_q <= 0:
        #         continue
        #     items.append({
        #         'context': ctx,
        #         'difficulty': difficulty,
        #         'question_types': question_types,
        #         'language': language,
        #         'num_questions': n_q,
        #     })

        # ===== FIX 4: GỘP CONTEXT THÀNH 1 PROMPT =====
        combined_context = "\n\n".join(
            f"[CONTEXT {i+1}]\n{ctx.content[:500]}"
            for i, ctx in enumerate(selected_contexts)
        )

        items = [{
            'context': combined_context,   # ⚠️ STRING, KHÔNG PHẢI ctx
            'original_contexts': selected_contexts,  # Store originals for metadata
            'difficulty': difficulty,
            'question_types': question_types,
            'language': language,
            'num_questions': buffer_target,
        }]


        shared_state['unused_contexts'] = contexts[num_contexts_to_use:]
        shared_state['question_target'] = target_questions
        shared_state['question_buffer_target'] = buffer_target

        return items

    async def exec(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate questions for a single context"""
        context = item['context']
        difficulty = item['difficulty']
        language = item['language']
        
        # Build the prompt
        prompt = self._build_question_prompt(
            context=context,  # context is already a string after FIX 4 combined contexts
            difficulty=difficulty,
            language=language,
            num_questions=item.get('num_questions', 1)
        )
        
        system_prompt = self._get_system_prompt(language)
        
        try:
            await asyncio.sleep(20)  # 🔥 FIX RATE LIMIT
            result = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4
            )
            
            # Handle case where result is a string instead of dict
            if isinstance(result, str):
                import json
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM response as JSON", response=result[:200])
                    return {'questions': [], 'error': 'Invalid JSON response'}
            
            # Ensure result is a dict
            if not isinstance(result, dict):
                return {'questions': [], 'error': 'Invalid response format'}
            
            # Add context reference (use first original context for metadata)
            original_contexts = item.get('original_contexts', [])
            if 'questions' in result and original_contexts:
                first_ctx = original_contexts[0]
                for q in result['questions']:
                    q['source_chunk_id'] = getattr(first_ctx, 'chunk_id', 'combined')
                    q['document_id'] = getattr(first_ctx, 'document_id', 'unknown')
                    q['reference_text'] = context[:500] if isinstance(context, str) else str(context)[:500]
            
            return result
            
        except Exception as e:
            logger.error("Question generation failed", error=str(e))
            return {'questions': [], 'error': str(e)}
    
    async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: List[Dict[str, Any]]) -> str:
        """Aggregate generated questions"""
        all_questions = []
        
        logger.info("Aggregating question generation results", 
                   num_results=len(exec_result),
                   num_contexts_processed=len(prep_result) if isinstance(prep_result, list) else 1)
        
        for i, result in enumerate(exec_result):
            questions = result.get('questions', [])
            error = result.get('error')
            
            if error:
                logger.warning(f"Context {i} had error", error=error)
            
            if isinstance(questions, list):
                all_questions.extend(questions)
                logger.info(f"Context {i} generated {len(questions)} questions")
            else:
                logger.error(
                    "Invalid questions format from LLM",
                    context_index=i,
                    questions_type=type(questions).__name__,
                    questions_value=str(questions)[:200]
                )
        
        # shared_state['generated_questions'] = all_questions
        # logger.info("Total questions generated", total=len(all_questions))

        # =========================
        # FIX 4: CẮT ĐÚNG SỐ LƯỢNG CÂU HỎI CUỐI
        # =========================
        target = int(shared_state.get("question_target", len(all_questions)))

        if len(all_questions) > target:
            all_questions = all_questions[:target]

        shared_state["generated_questions"] = all_questions
        shared_state["generated_count"] = len(all_questions)
        shared_state["missing_questions"] = max(0, target - len(all_questions))

        logger.info(
            "Question count finalized",
            target=target,
            generated=len(all_questions),
            missing=shared_state["missing_questions"]
        )



        if not all_questions:
            logger.warning("No valid questions generated from any context")
        
        logger.info("Questions generated", count=len(all_questions))
        
        return "default"
    
    def _get_system_prompt(self, language: str) -> str:
        """Get system prompt - always Vietnamese"""
        return """
Bạn là chuyên gia y khoa.

⚠️ QUY ĐỊNH BẮT BUỘC:
- BẮT BUỘC sử dụng TIẾNG VIỆT
- TUYỆT ĐỐI KHÔNG dùng tiếng Anh
- KHÔNG dịch sang tiếng Anh
- KHÔNG giải thích bằng tiếng Anh
- Mọi câu hỏi, đáp án, giải thích, chủ đề, keywords đều phải bằng tiếng Việt

Nếu vi phạm → câu trả lời bị coi là SAI.

Chỉ trả về JSON thuần, không markdown, không ```.

"""
    
    def _build_question_prompt(
        self,
        context: str,
        difficulty: str,
        language: str,
        num_questions: int = 1
    ) -> str:
        """Build the question generation prompt - always Vietnamese"""
        difficulty_instruction = {
            'easy': 'Câu hỏi đơn giản, kiểm tra kiến thức cơ bản',
            'medium': 'Câu hỏi trung bình, yêu cầu hiểu và áp dụng kiến thức',
            'hard': 'Câu hỏi khó, yêu cầu phân tích và tổng hợp kiến thức'
        }
        
        prompt = f"""Dựa trên nội dung y khoa sau đây, hãy tạo {num_questions} câu hỏi trắc nghiệm.

            NỘI DUNG:
            {context}

            YÊU CẦU:
                - Độ khó: {difficulty_instruction.get(difficulty, difficulty_instruction['medium'])}
                - Mỗi câu hỏi có 4 lựa chọn (A, B, C, D)
                - Chỉ có 1 đáp án đúng
                - Cung cấp giải thích chi tiết cho đáp án đúng
                - Tất cả nội dung phải bằng TIẾNG VIỆT
                - Không dùng markdown, không bọc bằng ```

                Trả về JSON theo format sau:
{{
    "questions": [
        {{
            "question_text": "Nội dung câu hỏi",
            "question_type": "single_choice",
            "difficulty": "{difficulty}",
            "options": [
                {{"id": "A", "text": "Lựa chọn A", "is_correct": false}},
                {{"id": "B", "text": "Lựa chọn B", "is_correct": true}},
                {{"id": "C", "text": "Lựa chọn C", "is_correct": false}},
                {{"id": "D", "text": "Lựa chọn D", "is_correct": false}}
            ],
            "correct_answer": "B",
            "explanation": "Giải thích chi tiết tại sao B là đáp án đúng...",
            "topic": "Chủ đề của câu hỏi",
            "keywords": ["từ khóa 1", "từ khóa 2"]
        }}
    ]
}}"""
        
        return prompt


class CaseBasedQuestionNode(BaseNode):
    """Node for generating case-based clinical questions"""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        super().__init__("CaseBasedQuestion")
        self.llm = llm_provider or get_llm_provider()
    
    async def prep(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for case-based question generation"""
        return {
            'contexts': shared_state.get('retrieved_contexts', []),
            'language': shared_state.get('language', 'vi'),
            'num_cases': shared_state.get('num_case_questions', 2)
        }
    
    async def exec(self, prep_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate case-based questions"""
        contexts = prep_result['contexts']
        language = prep_result['language']
        num_cases = prep_result['num_cases']
        
        if not contexts:
            return []
        
        # Combine multiple contexts for richer case scenarios
        combined_context = "\n\n".join([ctx.content for ctx in contexts[:5]])
        
        prompt = self._build_case_prompt(combined_context, language, num_cases)
        
        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=self._get_system_prompt(language),
                temperature=0.5
            )
            
            # Handle case where result is a string instead of dict
            if isinstance(result, str):
                import json
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    logger.error("Failed to parse case-based response as JSON")
                    return []
            
            if not isinstance(result, dict):
                return []
                
            return result.get('cases', [])
        except Exception as e:
            logger.error("Case-based question generation failed", error=str(e))
            return []
    
    async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: List[Dict[str, Any]]) -> str:
        """Store case-based questions"""
        existing_questions = shared_state.get('generated_questions', [])
        
        # Convert cases to question format
        for case in exec_result:
            if 'questions' in case:
                for q in case['questions']:
                    q['question_type'] = 'case_based'
                    q['case_scenario'] = case.get('scenario', '')
                existing_questions.extend(case['questions'])
        
        shared_state['generated_questions'] = existing_questions
        
        return "default"
    
    def _get_system_prompt(self, language: str) -> str:
        return """Bạn là một bác sĩ lâm sàng giàu kinh nghiệm. Hãy tạo các tình huống lâm sàng thực tế 
với các câu hỏi trắc nghiệm liên quan. Các tình huống phải giống như gặp trong thực hành lâm sàng.
TẤT CẢ NỘI DUNG PHẢI BẰNG TIẾNG VIỆT."""
    
    def _build_case_prompt(self, context: str, language: str, num_cases: int) -> str:
        return f"""Dựa trên kiến thức y khoa sau, tạo {num_cases} tình huống lâm sàng với câu hỏi.

KIẾN THỨC:
{context}

Trả về JSON:
{{
    "cases": [
        {{
            "scenario": "Mô tả bệnh nhân: tuổi, giới, lý do đến khám, triệu chứng...",
            "questions": [
                {{
                    "question_text": "Câu hỏi về chẩn đoán/điều trị...",
                    "options": [...],
                    "correct_answer": "A",
                    "explanation": "..."
                }}
            ]
        }}
    ]
}}"""


class QuestionValidationNode(BaseNode):
    """Node for validating generated questions"""

    def __init__(self):
        super().__init__("QuestionValidation")

    async def prep(self, shared_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get questions to validate"""
        return shared_state.get('generated_questions', [])

    async def exec(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate questions"""
        validated = []

        for q in questions:
            if not isinstance(q, dict):
                logger.error(
                    "Invalid question item",
                    type=type(q).__name__,
                    value=str(q)[:100]
                )
                continue

            if self._is_valid_question(q):
                validated.append(q)
            else:
                logger.warning(
                    "Invalid question filtered",
                    question=q.get('question_text', '')[:50]
                )

        return validated

    async def post(
        self,
        shared_state: Dict[str, Any],
        prep_result: Any,
        exec_result: List[Dict[str, Any]]
    ) -> str:
        """Store validated questions"""
        shared_state['validated_questions'] = exec_result
        shared_state['validation_stats'] = {
            'total': len(prep_result),
            'valid': len(exec_result),
            'filtered': len(prep_result) - len(exec_result)
        }
        return "default"

    def _is_valid_question(self, question: Dict[str, Any]) -> bool:
        """Check if a question is valid"""

        # Required fields
        required = ['question_text', 'options', 'correct_answer']
        if not all(key in question for key in required):
            return False

        # Question text checks
        question_text = question.get('question_text')
        if not isinstance(question_text, str) or len(question_text.strip()) < 10:
            return False

        # Options checks
        options = question.get('options')
        if not isinstance(options, list) or len(options) < 2:
            return False

        # Validate option structure (FIX CHÍNH)
        option_ids = []
        for opt in options:
            if not isinstance(opt, dict):
                return False

            opt_id = opt.get('id')
            if not isinstance(opt_id, str) or not opt_id:
                return False

            option_ids.append(opt_id)

        # Check that correct answer exists in options
        correct = question.get('correct_answer')
        if not isinstance(correct, str) or correct not in option_ids:
            return False

        # Check that exactly one option is marked correct
        correct_count = sum(1 for opt in options if opt.get('is_correct', False))

        if correct_count != 1:
            # Auto-fix: normalize is_correct flags
            for opt in options:
                opt['is_correct'] = (opt.get('id') == correct)

        return True



class AIDoubleCheckNode(BaseNode):
    """
    Node for AI-powered double-checking of generated questions.
    Uses LLM to verify medical accuracy, clarity, and educational value.
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        super().__init__("AIDoubleCheck")
        self.llm = llm_provider or LLMProvider()
    
    async def prep(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get validated questions and context for review"""
        return {
            'questions': shared_state.get('validated_questions', []),
            'context': shared_state.get('retrieved_context', []),
            'topic': shared_state.get('topic', ''),
            'enable_double_check': shared_state.get('enable_double_check', True)
        }
    
    async def exec(self, prep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to double-check questions for accuracy and quality"""
        questions = prep_data['questions']
        
        if not prep_data.get('enable_double_check', True) or not questions:
            # Return questions as-is with default review
            for q in questions:
                q['ai_review'] = {
                    'status': 'skipped',
                    'accuracy_score': None,
                    'clarity_score': None,
                    'suggestions': [],
                    'reviewed': False
                }
            return questions
        
        logger.info("Starting AI double-check", num_questions=len(questions))
        
        reviewed_questions = []
        
        # Process in batches of 5 for efficiency
        batch_size = 5
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            
            prompt = self._build_review_prompt(batch, prep_data.get('context', []))
            
            try:
                response = await self.llm.generate(
                    prompt,
                    max_tokens=2000,
                    temperature=0.3
                )
                
                reviews = self._parse_review_response(response)
                
                for j, q in enumerate(batch):
                    if j < len(reviews):
                        q['ai_review'] = reviews[j]
                    else:
                        q['ai_review'] = self._default_review()
                    reviewed_questions.append(q)
                    
            except Exception as e:
                logger.error("AI review failed", error=str(e))
                for q in batch:
                    q['ai_review'] = self._default_review(error=str(e))
                    reviewed_questions.append(q)
        
        return reviewed_questions
    
    async def post(self, shared_state: Dict[str, Any], prep_result: Any, exec_result: List[Dict[str, Any]]) -> str:
        """Store reviewed questions and generate report"""
        shared_state['reviewed_questions'] = exec_result
        
        # Calculate review statistics
        total = len(exec_result)
        reviewed = sum(1 for q in exec_result if q.get('ai_review', {}).get('reviewed', False))
        
        high_accuracy = sum(1 for q in exec_result 
                          if (q.get('ai_review', {}).get('accuracy_score') or 0) >= 8)
        needs_revision = sum(1 for q in exec_result 
                           if (q.get('ai_review', {}).get('accuracy_score') or 10) < 6)
        
        shared_state['review_stats'] = {
            'total_questions': total,
            'reviewed': reviewed,
            'high_accuracy': high_accuracy,
            'needs_revision': needs_revision,
            'review_rate': reviewed / total if total > 0 else 0
        }
        
        logger.info(
            "AI double-check completed",
            total=total,
            reviewed=reviewed,
            high_accuracy=high_accuracy,
            needs_revision=needs_revision
        )
        
        return "default"
    
    def _build_review_prompt(self, questions: List[Dict], context: List) -> str:
        """Build prompt for AI review"""
        questions_text = ""
        for i, q in enumerate(questions, 1):
            options_text = "\n".join([
                f"   {opt.get('id', chr(64+j))}. {opt.get('text', '')}" 
                for j, opt in enumerate(q.get('options', []), 1)
            ])
            questions_text += f"""
Question {i}:
{q.get('question_text', '')}
Options:
{options_text}
Correct Answer: {q.get('correct_answer', '')}
Explanation: {q.get('explanation', 'N/A')}
---
"""
        
        return f"""You are a medical education expert reviewing quiz questions for accuracy and quality.

Review each question below and provide:
1. Accuracy Score (1-10): Is the medical information correct?
2. Clarity Score (1-10): Is the question clear and unambiguous?
3. Educational Value (1-10): Does it test important medical knowledge?
4. Issues: List any problems found (incorrect info, ambiguous wording, etc.)
5. Suggestions: How to improve the question
6. Verdict: APPROVED, NEEDS_REVISION, or REJECT

Questions to review:
{questions_text}

Respond in JSON format:
{{
    "reviews": [
        {{
            "question_index": 1,
            "accuracy_score": 8,
            "clarity_score": 9,
            "educational_value": 7,
            "issues": ["Minor issue description"],
            "suggestions": ["Suggestion to improve"],
            "verdict": "APPROVED",
            "corrected_answer": null,
            "corrected_explanation": null
        }}
    ]
}}

Important:
- Be strict about medical accuracy
- Flag any potentially dangerous misinformation
- If the correct answer is wrong, provide the corrected answer
- Vietnamese medical terminology should be accurate"""
    
    def _parse_review_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse AI review response"""
        import json
        import re
        
        reviews = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                for review in data.get('reviews', []):
                    reviews.append({
                        'status': review.get('verdict', 'APPROVED').lower(),
                        'accuracy_score': review.get('accuracy_score'),
                        'clarity_score': review.get('clarity_score'),
                        'educational_value': review.get('educational_value'),
                        'issues': review.get('issues', []),
                        'suggestions': review.get('suggestions', []),
                        'corrected_answer': review.get('corrected_answer'),
                        'corrected_explanation': review.get('corrected_explanation'),
                        'reviewed': True
                    })
        except json.JSONDecodeError as e:
            logger.error("Failed to parse AI review", error=str(e))
        
        return reviews
    
    def _default_review(self, error: str = None) -> Dict[str, Any]:
        """Return default review when AI check is skipped or failed"""
        return {
            'status': 'error' if error else 'skipped',
            'accuracy_score': None,
            'clarity_score': None,
            'educational_value': None,
            'issues': [f"Review failed: {error}"] if error else [],
            'suggestions': [],
            'reviewed': False,
            'error': error
        }


# ============================================
# Flow Builder
# ============================================

def create_document_processing_flow() -> Flow:
    """Create a flow for document ingestion and embedding"""
    ingestion_node = DocumentIngestionNode()
    embedding_node = EmbeddingNode()
    
    ingestion_node.add_successor(embedding_node)
    
    return Flow(ingestion_node)


def create_question_generation_flow(
    llm_provider: Optional[LLMProvider] = None,
    include_case_based: bool = False,
    enable_double_check: bool = True
) -> Flow:
    """Create a flow for question generation with optional AI double-check"""
    retrieval_node = ContextRetrievalNode()
    question_node = QuestionGenerationNode(llm_provider)
    validation_node = QuestionValidationNode()
    double_check_node = AIDoubleCheckNode(llm_provider)
    
    retrieval_node.add_successor(question_node)
    
    if include_case_based:
        case_node = CaseBasedQuestionNode(llm_provider)
        question_node.add_successor(case_node)
        case_node.add_successor(validation_node)
    else:
        question_node.add_successor(validation_node)
    
    # Add AI double-check as final step
    if enable_double_check:
        validation_node.add_successor(double_check_node)
    
    return Flow(retrieval_node)


def create_full_pipeline_flow(
    llm_provider: Optional[LLMProvider] = None,
    enable_double_check: bool = True
) -> Flow:
    """Create a complete pipeline from document to questions with AI review"""
    # Document processing
    ingestion_node = DocumentIngestionNode()
    embedding_node = EmbeddingNode()
    
    # Question generation
    retrieval_node = ContextRetrievalNode()
    question_node = QuestionGenerationNode(llm_provider)
    validation_node = QuestionValidationNode()
    double_check_node = AIDoubleCheckNode(llm_provider)
    
    # Connect the flow
    ingestion_node.add_successor(embedding_node)
    embedding_node.add_successor(retrieval_node)
    retrieval_node.add_successor(question_node)
    question_node.add_successor(validation_node)
    
    # Add AI double-check as final step
    if enable_double_check:
        validation_node.add_successor(double_check_node)
    
    return Flow(ingestion_node)