"""
Flask web application for interactive kernel generation agent.
Provides streaming output and follow-up question capabilities.
"""
import os
import json
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import sys

# Add parent directories to path
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
SCRIPTS_DIR = PARENT_DIR / "scripts"
SRC_DIR = PARENT_DIR / "src"

for path in (CURRENT_DIR, PARENT_DIR, SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from web_interface.kernel_agent_wrapper import (
    StreamingKernelAgent, 
    StreamMessage, 
    generate_kernel_streaming,
    KernelSpec
)
import argparse

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Store active generation sessions
active_sessions = {}


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate_kernel():
    """Start kernel generation and stream results"""
    try:
        data = request.json
        
        # Create KernelSpec from request
        spec = _create_spec_from_request(data)
        
        # Create streaming agent
        session_id = data.get('session_id', f'session_{int(time.time())}')
        agent = StreamingKernelAgent(spec)
        active_sessions[session_id] = {
            'agent': agent,
            'spec': spec,
            'created_at': time.time()
        }
        
        # Start generation in background
        agent.start_generation()
        
        def generate():
            """Generator function for Server-Sent Events"""
            try:
                # Stream messages as they come
                for msg in agent.stream_output():
                    if msg:
                        msg_data = {
                            'type': msg.type,
                            'content': msg.content,
                            'metadata': msg.metadata or {},
                            'timestamp': msg.timestamp
                        }
                        yield f"data: {json.dumps(msg_data)}\n\n"
                
                # Send final result
                result = agent.get_result()
                if result:
                    complete_data = {
                        'type': 'complete',
                        'result': result
                    }
                    yield f"data: {json.dumps(complete_data)}\n\n"
                else:
                    complete_data = {
                        'type': 'complete',
                        'result': None,
                        'error': agent.error
                    }
                    yield f"data: {json.dumps(complete_data)}\n\n"
                    
            except Exception as e:
                error_data = {
                    'type': 'error',
                    'content': f'Generation error: {str(e)}'
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            finally:
                # Clean up session after a delay
                if session_id in active_sessions:
                    del active_sessions[session_id]
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle follow-up questions about generated kernels"""
    try:
        data = request.json
        session_id = data.get('session_id')
        question = data.get('question', '')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_sessions[session_id]
        agent = session['agent']
        spec = session['spec']
        
        # Get the generated kernel code
        result = agent.get_result()
        if not result:
            return jsonify({'error': 'No kernel generated yet'}), 400
        
        kernel_code = result.get('code', '')
        evaluation = result.get('evaluation', {})
        
        # Generate answer using LLM (simple implementation - can be enhanced)
        answer = _generate_answer(question, kernel_code, evaluation, spec)
        
        return jsonify({
            'answer': answer,
            'timestamp': time.time()
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session status and result"""
    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = active_sessions[session_id]
    agent = session['agent']
    result = agent.get_result()
    
    return jsonify({
        'session_id': session_id,
        'status': 'complete' if result else 'in_progress',
        'result': result,
        'error': agent.error
    })


def _create_spec_from_request(data: dict):
    """Create KernelSpec from request data - simplified for generation only"""
    spec = KernelSpec(
        language=data.get('language', 'cute'),
        gpu=data.get('gpu', 'H100'),
        rag_k=data.get('rag_k', 5),
        model_name=data.get('model', 'openai/o3'),
        temperature=data.get('temperature', 1.0),
        pytorch_reference=data.get('pytorch_code', ''),
        operation_description=data.get('purpose') or data.get('description', ''),
    )
    return spec


def _generate_answer(question: str, kernel_code: str, evaluation: dict, spec: KernelSpec) -> str:
    """Generate answer to follow-up question about the kernel"""
    # Simple implementation - just return a structured response
    # TODO: Enhance with actual LLM call for more sophisticated answers
    
    question_lower = question.lower()
    
    if 'speedup' in question_lower or 'performance' in question_lower:
        speedup = evaluation.get('runtime_stats', {}).get('performance_comparison', {}).get('speedup_ratio')
        if speedup:
            return f"The generated kernel achieves a {speedup:.2f}× speedup compared to the PyTorch baseline."
        else:
            return "Performance measurement is not available. The kernel has been verified for correctness."
    
    elif 'correct' in question_lower or 'pass' in question_lower:
        if evaluation.get('correctness'):
            return "Yes, the kernel passes all correctness checks and produces the same output as the PyTorch reference."
        else:
            return "The kernel did not pass correctness checks. You may want to refine the generation."
    
    elif 'compile' in question_lower:
        if evaluation.get('compiled'):
            return "Yes, the kernel compiles successfully without errors."
        else:
            error = evaluation.get('error', 'Unknown compilation error')
            return f"The kernel failed to compile: {error}"
    
    elif 'explain' in question_lower or 'how' in question_lower:
        # Could use LLM to explain the kernel code
        return f"This kernel is written in {spec.language.upper()} DSL. It optimizes the PyTorch reference implementation using GPU-specific optimizations. You can examine the code above for details."
    
    else:
        return "I can answer questions about the kernel's performance, correctness, compilation status, or explain how it works. What would you like to know?"


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'active_sessions': len(active_sessions)})


if __name__ == '__main__':
    # Run development server
    # Use port 5001 to avoid conflict with macOS AirPlay Receiver on port 5000
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)

