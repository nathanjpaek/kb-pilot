# Kernel Generation Agent Web Interface

Interactive web interface for the kernel generation agent with real-time streaming of the agent's reasoning process.

## Features

- **Real-time Streaming**: Watch the agent's "train of thought" as it generates kernels
- **Interactive Chat**: Ask follow-up questions about generated kernels
- **Modern UI**: Clean, responsive interface with syntax highlighting
- **Evaluation Results**: See compilation status, correctness, and performance metrics

## Setup

1. **Activate the conda environment** (if you have one):
```bash
conda activate kb-pilot
# or whatever environment has your kernel generation dependencies
```

2. Install web interface dependencies:
```bash
cd web_interface
pip install -r requirements.txt
```

3. Make sure you have all the kernel generation dependencies installed (PyTorch, Modal, DSPy, etc.)

3. Set environment variables for API keys:
```bash
export OPENAI_API_KEY=your_key_here
# or for other providers:
export DEEPSEEK_API_KEY=your_key_here
```

## Running

**Make sure you're in the correct conda environment** (e.g., `kb-pilot`), then from the `web_interface` directory:

```bash
python app.py
```

Or use the quick start script:
```bash
./run.sh
```

Then open your browser to `http://localhost:5000`

## Usage

1. **Enter PyTorch Code**: Paste your PyTorch reference implementation in the text area
2. **Configure Settings**: Choose target DSL (CuTe, TileLang, ThunderKittens), GPU architecture, and RAG parameters
3. **Generate**: Click "Generate Kernel" and watch the agent's reasoning stream in real-time
4. **Review Results**: See the generated kernel, evaluation results (compilation, correctness, speedup)
5. **Ask Questions**: Use the chat interface to ask follow-up questions about the generated kernel

## Architecture

- **Backend** (`app.py`): Flask API with Server-Sent Events (SSE) for streaming
- **Agent Wrapper** (`kernel_agent_wrapper.py`): Captures print statements and streams them
- **Frontend** (`templates/index.html`, `static/app.js`, `static/style.css`): Modern web interface

## API Endpoints

- `POST /api/generate`: Start kernel generation (returns SSE stream)
- `POST /api/chat`: Ask follow-up questions about generated kernels
- `GET /api/session/<session_id>`: Get session status
- `GET /api/health`: Health check

## Notes

- The interface captures all print statements from the kernel agent CLI
- Generation runs in a background thread to avoid blocking
- Streams are closed automatically when generation completes
- Session state is maintained in memory (not persistent across server restarts)

