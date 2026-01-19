"""
LLM Provider Module
Handles interactions with various LLM providers (OpenAI, Anthropic, Google)
"""
import os
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import structlog
import json
import asyncio
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    before_sleep_log
)

from app.config import settings

logger = structlog.get_logger()
py_logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[str] = None
    ) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Generate a structured JSON response"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        from openai import AsyncOpenAI
        
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.DEFAULT_MODEL
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.fallback_provider = None
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[str] = None
    ) -> str:
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}
            
            # Add small delay to avoid rate limits
            await asyncio.sleep(1)
            
            response = await self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_type = type(e).__name__
            error_str = str(e)
            
            # Check for rate limit error
            if "rate_limit" in error_str.lower() or error_type == "RateLimitError":
                logger.warning(
                    "OpenAI rate limit hit, falling back to Google provider",
                    error=error_str[:200]
                )
                
                # Use Google as fallback
                if self.fallback_provider is None:
                    self.fallback_provider = GoogleProvider()
                
                return await self.fallback_provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
            
            logger.error("OpenAI API error", error=error_str, error_type=error_type)
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        system = system_prompt or "You are a helpful assistant that outputs JSON."
        system += "\n\nAlways respond with valid JSON."
        
        if schema:
            system += f"\n\nFollow this JSON schema: {json.dumps(schema)}"
        
        response = await self.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature,
            response_format="json"
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", error=str(e))
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM Provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        from anthropic import AsyncAnthropic
        
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.model = model
        self.client = AsyncAnthropic(api_key=self.api_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[str] = None
    ) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = await self.client.messages.create(**kwargs)
        
        return response.content[0].text
    
    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        system = system_prompt or "You are a helpful assistant that outputs JSON."
        system += "\n\nAlways respond with valid JSON only, no markdown formatting."
        
        if schema:
            system += f"\n\nFollow this JSON schema: {json.dumps(schema)}"
        
        response = await self.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature
        )
        
        # Extract JSON from response (handle potential markdown code blocks)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", error=str(e), response=response[:200])
            raise


class GoogleProvider(LLMProvider):
    """Google Gemini LLM Provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        import google.generativeai as genai
        
        self.api_key = api_key or settings.GOOGLE_API_KEY
        self.model = model or settings.DEFAULT_MODEL
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[str] = None
    ) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = await self.client.generate_content_async(
            full_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        
        return response.text
    
    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        """Generate a JSON object and parse it robustly (Gemini often wraps JSON in markdown fences)."""
        import json
        import re

        system = system_prompt or "You are a helpful assistant that outputs JSON."
        system += "\n\nReturn JSON ONLY. Do not include markdown or code fences like ```."
        system += "\n\nLanguage enforcement: ALL output MUST be in Vietnamese. Do NOT use English."


        if schema:
            system += f"\n\nFollow this JSON schema: {json.dumps(schema, ensure_ascii=False)}"

        response = await self.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = (response or "").strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Remove opening fence: ``` or ```json
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            # Remove closing fence
            text = re.sub(r"\s*```\s*$", "", text)
            text = text.strip()

        # If model added a leading `json` token, drop it
        if text.lower().startswith("json") and "{" in text:
            text = text[text.find("{"):].strip()

        # Try parse directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Salvage: take the largest {...} block in the text
            start_i = text.find("{")
            end_i = text.rfind("}")
            if start_i != -1 and end_i != -1 and end_i > start_i:
                candidate = text[start_i:end_i+1]
            else:
                candidate = text

            # Remove trailing commas (common LLM mistake)
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                logger.error(
                    "Invalid JSON from LLM",
                    response_preview=(text[:250] + ("..." if len(text) > 250 else "")),
                    error=str(e),
                )
                raise ValueError("LLM did not return valid JSON")


class GoogleGeminiThinkingProvider(LLMProvider):
    """
    Google Gemini Provider with Thinking Mode and Advanced Features
    Uses the new google-genai SDK with support for:
    - Thinking mode (deep reasoning)
    - PDF/Document upload
    - Google Search integration
    - URL context
    - Structured JSON output with schemas
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        from google import genai
        from google.genai import types
        
        self.api_key = api_key or settings.GOOGLE_API_KEY
        # Valid models: gemini-3-pro-preview, gemini-2.0-flash-thinking-exp, gemini-2.0-flash-exp, gemini-1.5-pro
        self.model = model or settings.DEFAULT_MODEL
        self.client = genai.Client(api_key=self.api_key)
        self.types = types
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[str] = None
    ) -> str:
        """Generate text response (basic method for compatibility)"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        contents = [
            self.types.Content(
                role="user",
                parts=[self.types.Part.from_text(text=full_prompt)],
            ),
        ]
        
        config = self.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        
        return response.text
    
    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Generate structured JSON response (basic method)"""
        system = system_prompt or "You are a helpful assistant that outputs JSON."
        system += "\n\nAlways respond with valid JSON."
        
        if schema:
            system += f"\n\nFollow this JSON schema: {json.dumps(schema)}"
        
        response = await self.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature,
            response_format="json"
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", error=str(e))
            raise
    
    async def generate_quiz_with_thinking(
        self,
        pdf_path: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        prompt: str = None,
        num_questions: int = 10,
        difficulty: str = "medium",
        use_thinking: bool = True,
        use_google_search: bool = False,
        temperature: float = 0.3,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate quiz questions with advanced features
        
        Args:
            pdf_path: Path to PDF file
            pdf_bytes: PDF file as bytes
            prompt: Custom prompt (default: generate medical quiz)
            num_questions: Number of questions to generate
            difficulty: easy, medium, hard
            use_thinking: Enable thinking mode for deeper reasoning
            use_google_search: Enable Google Search for fact-checking
            temperature: Sampling temperature
            schema: Custom JSON schema (uses default quiz schema if None)
        
        Returns:
            Dict with questions array and metadata
        """
        from google.genai import types
        import asyncio
        
        # Prepare content parts
        parts = []
        
        # Add PDF if provided
        if pdf_path:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        
        if pdf_bytes:
            parts.append(
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                )
            )
        
        # Add prompt
        if not prompt:
            prompt = f"""Dựa trên tài liệu được cung cấp, hãy tạo {num_questions} câu hỏi trắc nghiệm y khoa chất lượng cao.

Yêu cầu:
- Độ khó: {difficulty}
- Câu hỏi phải chính xác về mặt y khoa
- Đáp án phải có giải thích rõ ràng
- Sử dụng tiếng Việt cho tất cả nội dung
- Mỗi câu hỏi có 4 đáp án (A, B, C, D)
- Đánh dấu rõ đáp án đúng
- Thêm keywords và topic cho mỗi câu hỏi
"""
        
        parts.append(types.Part.from_text(text=prompt))
        
        contents = [
            types.Content(
                role="user",
                parts=parts,
            ),
        ]
        
        # Configure tools
        tools = []
        if use_google_search:
            tools.append(types.Tool(googleSearch=types.GoogleSearch()))
            tools.append(types.Tool(url_context=types.UrlContext()))
        
        # Use provided schema or default quiz schema
        if not schema:
            schema = self._get_default_quiz_schema()
        
        # Build config
        config_kwargs = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        
        if tools:
            config_kwargs["tools"] = tools
        
        if use_thinking and "thinking" in self.model.lower():
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level="HIGH",
            )
        
        generate_content_config = types.GenerateContentConfig(**config_kwargs)
        
        # Generate with streaming for better responsiveness
        output = ""
        
        # Run in thread pool to avoid blocking
        def _generate():
            nonlocal output
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    output += chunk.text
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _generate)
        
        # Parse JSON response
        try:
            result = json.loads(output)
            logger.info(
                "Generated quiz questions with thinking mode",
                num_questions=len(result.get("questions", [])),
                thinking_enabled=use_thinking,
                search_enabled=use_google_search
            )
            return result
        except json.JSONDecodeError as e:
            logger.error("Failed to parse quiz JSON", error=str(e), output_preview=output[:500])
            raise ValueError(f"Failed to parse quiz JSON: {e}")
    
    def _get_default_quiz_schema(self):
        """Get default quiz JSON schema"""
        from google import genai
        
        return genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["questions", "format", "total"],
            properties={
                "questions": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["id", "question_text", "options", "correct_answer"],
                        properties={
                            "id": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="UUID of the question",
                            ),
                            "question_text": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "question_type": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                enum=["single_choice", "multiple_choice"],
                            ),
                            "difficulty": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                enum=["easy", "medium", "hard"],
                            ),
                            "options": genai.types.Schema(
                                type=genai.types.Type.ARRAY,
                                items=genai.types.Schema(
                                    type=genai.types.Type.OBJECT,
                                    required=["id", "text", "is_correct"],
                                    properties={
                                        "id": genai.types.Schema(
                                            type=genai.types.Type.STRING,
                                        ),
                                        "text": genai.types.Schema(
                                            type=genai.types.Type.STRING,
                                        ),
                                        "is_correct": genai.types.Schema(
                                            type=genai.types.Type.BOOLEAN,
                                        ),
                                    },
                                ),
                            ),
                            "correct_answer": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "explanation": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "topic": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "keywords": genai.types.Schema(
                                type=genai.types.Type.ARRAY,
                                items=genai.types.Schema(
                                    type=genai.types.Type.STRING,
                                ),
                            ),
                            "source_chunk_id": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "document_id": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="UUID of the source document",
                            ),
                            "reference_text": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "created_at": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                        },
                    ),
                ),
                "format": genai.types.Schema(
                    type=genai.types.Type.STRING,
                ),
                "total": genai.types.Schema(
                    type=genai.types.Type.INTEGER,
                ),
            },
        )


# ============================================
# LLM Provider Factory
# ============================================

def get_llm_provider(provider: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory function to create an LLM provider instance.
    """
    provider = provider or settings.DEFAULT_LLM_PROVIDER
    providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'google': GoogleProvider,
        'google-thinking': GoogleGeminiThinkingProvider,
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {list(providers.keys())}"
        )

    return providers[provider](**kwargs)