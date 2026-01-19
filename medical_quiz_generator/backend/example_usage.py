#!/usr/bin/env python3
"""
Example script showing how to use the GoogleGeminiThinkingProvider
This is the cleaned-up version of your original code
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.llm_provider import GoogleGeminiThinkingProvider


async def generate_quiz_from_pdf():
    """
    Generate quiz questions from a PDF file using Gemini Thinking Mode
    This replicates your original code functionality
    """
    
    # Initialize provider with API key from environment
    provider = GoogleGeminiThinkingProvider(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        model="gemini-3-pro-preview"  # Using Gemini 3 Pro Preview
    )
    
    # Path to your PDF file
    pdf_file = "tailieuhoctap.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ PDF file not found: {pdf_file}")
        print("Please place your PDF file in the backend directory")
        return
    
    print(f"📄 Processing PDF: {pdf_file}")
    print("🤔 Thinking mode enabled - this may take a while...")
    
    # Generate quiz with all advanced features
    result = await provider.generate_quiz_with_thinking(
        pdf_path=pdf_file,
        num_questions=10,  # Adjust as needed
        difficulty="medium",
        use_thinking=True,        # Enable deep reasoning
        use_google_search=True,   # Enable Google Search for fact-checking
        temperature=0.3
    )
    
    # Write to file to avoid console encoding issues
    output_file = "output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Output saved to {output_file}")
    print(f"📊 Generated {result.get('total', 0)} questions")
    
    return result


if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    asyncio.run(generate_quiz_from_pdf())
