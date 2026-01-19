#!/usr/bin/env python3
"""
Test script for Google Gemini Thinking Provider
Demonstrates how to use the new GoogleGeminiThinkingProvider with PDF upload
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.llm_provider import GoogleGeminiThinkingProvider, get_llm_provider
from app.config import settings


async def test_basic_generation():
    """Test basic text generation"""
    print("🧪 Test 1: Basic Text Generation")
    print("=" * 60)
    
    provider = GoogleGeminiThinkingProvider()
    
    response = await provider.generate(
        prompt="Giải thích ngắn gọn về bệnh tiểu đường type 2",
        temperature=0.7
    )
    
    print(f"Response: {response[:300]}...")
    print()


async def test_quiz_generation_no_pdf():
    """Test quiz generation without PDF"""
    print("🧪 Test 2: Quiz Generation (No PDF)")
    print("=" * 60)
    
    provider = GoogleGeminiThinkingProvider(
        model="gemini-2.0-flash-exp"  # Use non-thinking model for faster testing
    )
    
    result = await provider.generate_quiz_with_thinking(
        prompt="""Tạo 3 câu hỏi trắc nghiệm về bệnh tiểu đường type 2.
        
Yêu cầu:
- Mỗi câu có 4 đáp án (A, B, C, D)
- Đánh dấu đáp án đúng
- Có giải thích chi tiết
- Sử dụng tiếng Việt
""",
        num_questions=3,
        difficulty="medium",
        use_thinking=False,  # Disable thinking for speed
        use_google_search=False,
        temperature=0.3
    )
    
    print(f"Generated {result.get('total', 0)} questions")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


async def test_quiz_with_pdf():
    """Test quiz generation with PDF file"""
    print("🧪 Test 3: Quiz Generation with PDF Upload")
    print("=" * 60)
    
    # Check if test PDF exists
    test_pdf = "./data/uploads/test.pdf"
    if not os.path.exists(test_pdf):
        print(f"⚠️  PDF file not found: {test_pdf}")
        print("Please upload a PDF file to test this feature")
        print()
        return
    
    provider = GoogleGeminiThinkingProvider(
        model=settings.GEMINI_THINKING_MODEL  # Use thinking model
    )
    
    result = await provider.generate_quiz_with_thinking(
        pdf_path=test_pdf,
        num_questions=5,
        difficulty="medium",
        use_thinking=True,  # Enable thinking mode
        use_google_search=False,  # Disable search to avoid quota issues
        temperature=0.3
    )
    
    print(f"Generated {result.get('total', 0)} questions from PDF")
    
    # Save to file
    output_file = "quiz_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to {output_file}")
    
    # Print first question as sample
    if result.get('questions'):
        print("\nSample Question:")
        q = result['questions'][0]
        print(f"Q: {q.get('question_text', 'N/A')}")
        for opt in q.get('options', []):
            marker = "✓" if opt.get('is_correct') else " "
            print(f"  [{marker}] {opt.get('id', '?')}) {opt.get('text', 'N/A')}")
        print(f"Explanation: {q.get('explanation', 'N/A')[:200]}...")
    print()


async def test_with_factory():
    """Test using factory function"""
    print("🧪 Test 4: Using Factory Function")
    print("=" * 60)
    
    # Get provider through factory
    provider = get_llm_provider("google-thinking")
    
    response = await provider.generate(
        prompt="Liệt kê 3 triệu chứng chính của bệnh cao huyết áp",
        temperature=0.5
    )
    
    print(f"Response: {response}")
    print()


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 Google Gemini Thinking Provider - Test Suite")
    print("="*60 + "\n")
    
    try:
        # Run tests
        await test_basic_generation()
        await test_quiz_generation_no_pdf()
        await test_quiz_with_pdf()
        await test_with_factory()
        
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    
    asyncio.run(main())
