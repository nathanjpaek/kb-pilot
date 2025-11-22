#!/bin/bash
# Quick start script for the web interface

cd "$(dirname "$0")"

echo "🚀 Starting Kernel Generation Agent Web Interface..."
echo ""
echo "Make sure you have:"
echo "  1. Installed dependencies: pip install -r requirements.txt"
echo "  2. Set API keys (e.g., export OPENAI_API_KEY=...)"
echo "  3. Installed all kernel generation dependencies from ../requirements.txt"
echo ""
read -p "Press Enter to continue..."

python app.py

