# """
# LLM Provider Module
# Handles interactions with various LLM providers (OpenAI, Anthropic, Google)
# """
# from email.mime import text
# import os
# from pydoc import text
# import logging
# import re
# from typing import Optional, Dict, Any, List
# from abc import ABC, abstractmethod
# from urllib import response
# import structlog
# import json
# import asyncio
# from tenacity import (
#     retry, 
#     stop_after_attempt, 
#     wait_exponential,
#     before_sleep_log
# )

# from app.config import settings

# logger = structlog.get_logger()
# py_logger = logging.getLogger(__name__)


# class LLMProvider(ABC):
#     """Abstract base class for LLM providers"""
    
#     @abstractmethod
#     async def generate(
#         self,
#         prompt: str,
#         system_prompt: Optional[str] = None,
#         temperature: float = 0.7,
#         max_tokens: int = 4096,
#         response_format: Optional[str] = None
#     ) -> str:
#         """Generate a response from the LLM"""
#         pass
    
#     @abstractmethod
#     async def generate_structured(
#         self,
#         prompt: str,
#         system_prompt: Optional[str] = None,
#         schema: Optional[Dict[str, Any]] = None,
#         temperature: float = 0.3
#     ) -> Dict[str, Any]:
#         """Generate a structured JSON response"""
#         pass


# class OpenAIProvider(LLMProvider):
#     """OpenAI LLM Provider"""
    
#     def __init__(self, api_key: Optional[str] = None, model: str = None):
#         from openai import AsyncOpenAI
        
#         self.api_key = api_key or settings.OPENAI_API_KEY
#         self.model = model or settings.DEFAULT_MODEL
#         self.client = AsyncOpenAI(api_key=self.api_key)
    
#     @retry(
#         stop=stop_after_attempt(5),
#         wait=wait_exponential(multiplier=2, min=4, max=60),
#         before_sleep=before_sleep_log(py_logger, logging.INFO)
#     )
#     async def generate(
#         self,
#         prompt: str,
#         system_prompt: Optional[str] = None,
#         temperature: float = 0.7,
#         max_tokens: int = 4096,
#         response_format: Optional[str] = None
#     ) -> str:
#         try:
#             messages = []
            
#             if system_prompt:
#                 messages.append({"role": "system", "content": system_prompt})
            
#             messages.append({"role": "user", "content": prompt})
            
#             kwargs = {
#                 "model": self.model,
#                 "messages": messages,
#                 "temperature": temperature,
#                 "max_tokens": max_tokens
#             }
            
#             if response_format == "json":
#                 kwargs["response_format"] = {"type": "json_object"}
            
#             # Add small delay to avoid rate limits
#             await asyncio.sleep(1)
            
#             response = await self.client.chat.completions.create(**kwargs)
            
#             return response.choices[0].message.content
            
#         except Exception as e:
#             logger.error("OpenAI API error", error=str(e), error_type=type(e).__name__)
#             raise
    
#     async def generate_structured(
#         self,
#         prompt: str,
#         system_prompt: Optional[str] = None,
#         schema: Optional[Dict[str, Any]] = None,
#         temperature: float = 0.3
#     ) -> Dict[str, Any]:
#         system = system_prompt or "You are a helpful assistant that outputs JSON."
#         system += "\n\nAlways respond with valid JSON."
        
#         if schema:
#             system += f"\n\nFollow this JSON schema: {json.dumps(schema)}"
        
#         response = await self.generate(
#             prompt=prompt,
#             system_prompt=system,
#             temperature=temperature,
#             response_format="json"
#         )
        
#         try:
#             return json.loads(response)
#         except json.JSONDecodeError as e:
#             logger.error("Failed to parse JSON response", error=str(e))
#             raise


# class AnthropicProvider(LLMProvider):
#     """Anthropic Claude LLM Provider"""
    
#     def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
#         from anthropic import AsyncAnthropic
        
#         self.api_key = api_key or settings.ANTHROPIC_API_KEY
#         self.model = model
#         self.client = AsyncAnthropic(api_key=self.api_key)
    
#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
#     async def generate(
#         self,
#         prompt: str,
#         system_prompt: Optional[str] = None,
#         temperature: float = 0.7,
#         max_tokens: int = 4096,
#         response_format: Optional[str] = None
#     ) -> str:
#         kwargs = {
#             "model": self.model,
#             "max_tokens": max_tokens,
#             "temperature": temperature,
#             "messages": [{"role": "user", "content": prompt}]
#         }
        
#         if system_prompt:
#             kwargs["system"] = system_prompt
        
#         response = await self.client.messages.create(**kwargs)
        
#         return response.content[0].text
    
#     async def generate_structured(
#         self,
#         prompt: str,
#         system_prompt: Optional[str] = None,
#         schema: Optional[Dict[str, Any]] = None,
#         temperature: float = 0.3
#     ) -> Dict[str, Any]:
#         system = system_prompt or "You are a helpful assistant that outputs JSON."
#         system += "\n\nAlways respond with valid JSON only, no markdown formatting."
        
#         if schema:
#             system += f"\n\nFollow this JSON schema: {json.dumps(schema)}"
        
#         response = await self.generate(
#             prompt=prompt,
#             system_prompt=system,
#             temperature=temperature
#         )
        
#         # Extract JSON from response (handle potential markdown code blocks)
#         response = response.strip()
#         if response.startswith("```"):
#             response = response.split("```")[1]
#             if response.startswith("json"):
#                 response = response[4:]
        
#         try:
#             return json.loads(response.strip())
#         except json.JSONDecodeError as e:
#             logger.error("Failed to parse JSON response", error=str(e), response=response[:200])
#             raise


# class GoogleProvider(LLMProvider):
#     """Google Gemini LLM Provider"""
    
#     def __init__(self, api_key: Optional[str] = None, model: str = "models/gemini-2.5-flash"):
#         import google.generativeai as genai
        
#         self.api_key = api_key or settings.GOOGLE_API_KEY
#         self.model = model
#         genai.configure(api_key=self.api_key)
#         self.client = genai.GenerativeModel(model)
    
#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
#     async def generate(
#         self,
#         prompt: str,
#         system_prompt: Optional[str] = None,
#         temperature: float = 0.7,
#         max_tokens: int = 4096,
#         response_format: Optional[str] = None
#     ) -> str:
#         full_prompt = prompt
#         if system_prompt:
#             full_prompt = f"{system_prompt}\n\n{prompt}"
        
#         response = await self.client.generate_content_async(
#             full_prompt,
#             generation_config={
#                 "temperature": temperature,
#                 "max_output_tokens": max_tokens
#             }
#         )
        
#         return response.text
    
#     # async def generate_structured(
#     #     self,
#     #     prompt: str,
#     #     system_prompt: Optional[str] = None,
#     #     schema: Optional[Dict[str, Any]] = None,
#     #     temperature: float = 0.3
#     # ) -> Dict[str, Any]:
        
#     #     import re

#     #     system = system_prompt or "You are a helpful assistant that outputs JSON."
#     #     system += "\n\nAlways respond with valid JSON only."

#     #     if schema:
#     #         system += f"\n\nFollow this JSON schema: {json.dumps(schema)}"
            
#     #     # ⚠️ FIX 1: GIỚI HẠN OUTPUT
#     #     response = await self.generate(
#     #         prompt=prompt,
#     #         system_prompt=system,
#     #         temperature=temperature,
#     #         max_tokens=800   # RẤT QUAN TRỌNG
#     #     )

#     #     text = response.strip()

#     #     # ⚠️ FIX 2: TRÍCH JSON AN TOÀN
#     #     try:
#     #         return json.loads(text)
#     #     except json.JSONDecodeError:
#     #         # salvage JSON đầu tiên hợp lệ
#     #         match = re.search(r'\{[\s\S]*?\}\s*$', text)
#     #         if match:
#     #             return json.loads(match.group())

#     #         raise ValueError(f"Invalid JSON from LLM: {text[:300]}")

#     async def generate_structured(
#         self,
#         prompt: str,
#         system_prompt: Optional[str] = None,
#         schema: Optional[Dict[str, Any]] = None,
#         temperature: float = 0.3
#     ) -> Dict[str, Any]:
#         import json, re

#         system = system_prompt or "You are a helpful assistant that outputs JSON."
#         system += "\n\nAlways respond with valid JSON only."

#         if schema:
#             system += f"\n\nFollow this JSON schema: {json.dumps(schema)}"

#         response = await self.generate(
#             prompt=prompt,
#             system_prompt=system,
#             temperature=temperature,
#             max_tokens=1200
#         )

#         text = response.strip()

#         # ✅ FIX: STRIP MARKDOWN ```json ... ```
#         if text.startswith("```"):
#             # lấy phần bên trong code block
#             text = text.split("```", 2)[1]
#             if text.lstrip().startswith("json"):
#                 text = text.lstrip()[4:]
#             text = text.strip()

#         # 1️⃣ THỬ parse thẳng
#         try:
#             return json.loads(text)
#         except json.JSONDecodeError:
#             pass

#         # 2️⃣ BẮT JSON object ĐẦU TIÊN (KHÔNG CẦN KẾT THÚC Ở CUỐI)
#         json_match = re.search(r'\{[\s\S]*\}', text)
#         if json_match:
#             json_text = json_match.group()
#             try:
#                 return json.loads(json_text)
#             except json.JSONDecodeError:
#                 pass

#         # 3️⃣ BẮT JSON array (phòng trường hợp model trả [])
#         array_match = re.search(r'\[[\s\S]*\]', text)
#         if array_match:
#             array_text = array_match.group()
#             try:
#                 return json.loads(array_text)
#             except json.JSONDecodeError:
#                 pass

#         # 4️⃣ FAIL CÓ LOG RÕ
#         logger.error(
#             "Invalid JSON from LLM",
#             response_preview=text[:500]
#         )
#         raise ValueError("LLM did not return valid JSON")




# def get_llm_provider(
#     provider: Optional[str] = None,
#     **kwargs
# ) -> LLMProvider:
#     """
#     Factory function to get an LLM provider instance.
    
#     Args:
#         provider: Provider name ('openai', 'anthropic', 'google')
#         **kwargs: Additional arguments passed to the provider
    
#     Returns:
#         LLMProvider instance
#     """
#     provider = provider or settings.DEFAULT_LLM_PROVIDER
    
#     providers = {
#         'openai': OpenAIProvider,
#         'anthropic': AnthropicProvider,
#         'google': GoogleProvider
#     }
    
#     if provider not in providers:
#         raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
#     return providers[provider](**kwargs)


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
    
    def __init__(self, api_key: Optional[str] = None, model: str = "models/gemini-2.5-flash"):
        import google.generativeai as genai
        
        self.api_key = api_key or settings.GOOGLE_API_KEY
        self.model = model
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model)
    
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


# ============================================
# LLM Provider Factory
# ============================================

def get_llm_provider(provider: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory function to create an LLM provider instance.
    """
    provider = provider or getattr(settings, "LLM_PROVIDER", "openai")
    providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'google': GoogleProvider
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {list(providers.keys())}"
        )

    return providers[provider](**kwargs)