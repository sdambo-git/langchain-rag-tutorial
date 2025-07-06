#!/bin/bash

# Activate virtual environment script for langchain-rag-tutorial
echo "Activating virtual environment..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating new venv..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ Virtual environment activated successfully!"
    echo "Virtual environment path: $VIRTUAL_ENV"
    echo "Python version: $(python --version)"
    echo ""
    echo "To deactivate later, run: deactivate"
    echo ""
    echo "Ready to run your scripts:"
    echo "  python compare_embeddings.py"
    echo "  python create_database.py"
    echo "  python query_data.py \"your question\""
else
    echo "❌ Failed to activate virtual environment"
fi
