#!/bin/bash
# Run the CuTe documentation summarization pipeline

set -e

echo "🚀 Starting CuTe DSL Documentation Summarization Pipeline"
echo "============================================================"
echo ""
echo "This will use OpenAI API to hierarchically summarize all CuTe docs."
echo "Make sure your OPENAI_API_KEY is set in your environment or .env file."
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to repo root
cd "$REPO_ROOT"

# Check if .env exists
if [ -f ".env" ]; then
    echo "✅ Found .env file"
else
    echo "⚠️  No .env file found. Make sure OPENAI_API_KEY is in your environment."
fi

# Run the summarization pipeline
echo ""
echo "Running summarization pipeline..."
echo ""

python -m scripts.cute_guideline_prompt

echo ""
echo "============================================================"
echo "✅ Summarization complete!"
echo ""
echo "Results saved to: scripts/.cute_summary_cache/"
echo "Final guideline: scripts/.cute_summary_cache/CUTE_GUIDELINE_PROMPT_FINAL.txt"
echo ""
echo "You can now use CUTE_GUIDELINE_PROMPT in generate_and_eval_rag_modal.py"

