#!/bin/bash

# Check if script is being sourced (correct way) or executed (wrong way)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ ERROR: This script must be SOURCED, not executed!"
    echo ""
    echo "🔧 CORRECT usage:"
    echo "   source activate_venv.sh"
    echo "   # OR"
    echo "   . activate_venv.sh"
    echo ""
    echo "❌ WRONG usage:"
    echo "   ./activate_venv.sh  # This won't work!"
    echo ""
    echo "💡 Why? Shell scripts run in subshells and can't modify the parent environment."
    echo "   Sourcing runs the commands in your current shell."
    exit 1
fi

# Activate virtual environment script for langchain-rag-tutorial
echo "🔧 Activating virtual environment..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment not found. Creating new venv..."
    python3 -m venv venv
    echo "✅ Virtual environment created!"
fi

# Activate the virtual environment
source venv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ Virtual environment activated successfully!"
    echo "📍 Virtual environment path: $VIRTUAL_ENV"
    echo "🐍 Python version: $(python --version)"
    echo ""
    echo "🎯 Ready to use your RAG system:"
    echo "   python create_database.py --status"
    echo "   python create_database.py"
    echo "   python query_data.py \"What is BlueField-3?\""
    echo ""
    echo "🚪 To deactivate later, run: deactivate"
else
    echo "❌ Failed to activate virtual environment"
    echo "💡 Try running: python3 -m venv venv"
fi
