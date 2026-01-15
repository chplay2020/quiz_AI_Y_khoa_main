#!/bin/bash

# Medical Quiz Generator - Quick Start Script

echo "🏥 Medical Quiz Generator - Setup"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1)
if [[ $? -ne 0 ]]; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi
echo "✅ Found $python_version"

# Check Node.js version
node_version=$(node --version 2>&1)
if [[ $? -ne 0 ]]; then
    echo "❌ Node.js is required but not found"
    exit 1
fi
echo "✅ Found Node.js $node_version"

# Setup Backend
echo ""
echo "📦 Setting up Backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit backend/.env and add your API keys!"
fi

cd ..

# Setup Frontend
echo ""
echo "📦 Setting up Frontend..."
cd frontend
npm install
cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo ""
echo "1. Backend (in one terminal):"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   uvicorn app.main:app --reload --port 8000"
echo ""
echo "2. Frontend (in another terminal):"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "🏥 Happy quiz generating!"
