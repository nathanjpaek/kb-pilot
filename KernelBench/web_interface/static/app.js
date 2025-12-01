/**
 * Frontend JavaScript for Kernel Generation Agent Interface
 * Handles real-time streaming, UI updates, and chat functionality
 */

let currentSessionId = null;
let generatedKernelCode = null;
let evaluationResults = null;
let eventSource = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
});

function initializeEventListeners() {
    const form = document.getElementById('generation-form');
    if (form) {
        form.addEventListener('submit', handleGenerateSubmit);
        console.log('Form submit listener attached');
    } else {
        console.error('Form not found!');
    }
    
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleChatSend();
            }
        });
    }
    
    if (chatSendBtn) {
        chatSendBtn.addEventListener('click', handleChatSend);
    }
    
    const copyBtn = document.getElementById('copy-kernel-btn');
    const downloadBtn = document.getElementById('download-kernel-btn');
    
    if (copyBtn) copyBtn.addEventListener('click', copyKernel);
    if (downloadBtn) downloadBtn.addEventListener('click', downloadKernel);
}

async function handleGenerateSubmit(e) {
    e.preventDefault();
    console.log('Form submitted!');
    
    // Get form data - language and gpu are fixed in the UI (not inputs)
    const formData = {
        session_id: `session_${Date.now()}`,
        language: 'cute',  // Fixed to CuTe-DSL as shown in UI
        gpu: 'H100',  // Fixed to H100 as shown in UI
        rag_k: parseInt(document.getElementById('rag_k')?.value) || 5,
        pytorch_code: document.getElementById('pytorch_code')?.value || '',
        purpose: document.getElementById('purpose')?.value || '',
        description: document.getElementById('purpose')?.value || '',
        test_time_scaling: false,  // Disabled - we just want single generation
        measure_performance: false,  // Disabled - no evaluation
    };
    
    console.log('Form data:', formData);
    
    if (!formData.pytorch_code.trim()) {
        alert('Please provide PyTorch reference code');
        return;
    }
    
    // Reset UI
    resetUI();
    currentSessionId = formData.session_id;
    
    // Update button state
    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = true;
    document.getElementById('generate-btn-text').style.display = 'none';
    document.getElementById('generate-btn-loading').style.display = 'inline';
    
    // Update status
    updateStatus('generating', 'Generating...');
    
    // Clear reasoning output
    const reasoningOutput = document.getElementById('reasoning-output');
    reasoningOutput.innerHTML = '';
    
    // Start streaming
    startStreaming(formData);
}

function startStreaming(formData) {
    // Close existing event source if any
    if (eventSource) {
        eventSource.close();
    }
    
    console.log('Starting generation with formData:', formData);
    
    // Create new EventSource for Server-Sent Events
    fetch('/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    }).then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
            return response.text().then(text => {
                throw new Error(`HTTP error! status: ${response.status}, body: ${text}`);
            });
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        function readStream() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    console.log('Stream complete');
                    handleStreamComplete();
                    return;
                }
                
                const chunk = decoder.decode(value, { stream: true });
                console.log('Received chunk:', chunk);
                buffer += chunk;
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            console.log('Parsed message:', data);
                            handleStreamMessage(data);
                        } catch (e) {
                            console.error('Error parsing stream message:', e, 'Line:', line);
                        }
                    }
                }
                
                readStream();
            }).catch(error => {
                console.error('Stream read error:', error);
                handleStreamError(error);
            });
        }
        
        readStream();
    }).catch(error => {
        console.error('Fetch error:', error);
        handleStreamError(error);
    });
}

function handleStreamMessage(data) {
    const { type, content, metadata } = data;
    
    const reasoningOutput = document.getElementById('reasoning-output');
    
    switch (type) {
        case 'log':
            appendLogEntry(content, 'info');
            break;
            
        case 'code':
            // Accumulate code in a dedicated section
            const codeContainer = document.getElementById('kernel-code');
            if (codeContainer.textContent === '' || codeContainer.textContent === 'Generated kernel code will appear here...') {
                codeContainer.textContent = content;
            } else {
                codeContainer.textContent += '\n' + content;
            }
            appendLogEntry(content, 'code');
            break;
        
        case 'rag_example':
            // Display RAG example
            displayRagExample(metadata);
            appendLogEntry(content, 'info');
            break;
            
        case 'result':
            appendLogEntry(content, 'success');
            if (metadata) {
                evaluationResults = metadata;
                // displayEvaluationResults(metadata);
            }
            break;
            
        case 'error':
            appendLogEntry(content, 'error');
            updateStatus('error', 'Error occurred');
            break;
            
        case 'complete':
            handleGenerationComplete(metadata);
            break;
    }
    
    // Auto-scroll reasoning output
    reasoningOutput.scrollTop = reasoningOutput.scrollHeight;
}

function appendLogEntry(content, className = 'info') {
    const reasoningOutput = document.getElementById('reasoning-output');
    
    // Remove placeholder if exists
    const placeholder = reasoningOutput.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    const entry = document.createElement('div');
    entry.className = `log-entry ${className}`;
    entry.textContent = content;
    reasoningOutput.appendChild(entry);
}

function handleGenerationComplete(data) {
    updateStatus('success', 'Generation complete');
    
    // Re-enable generate button
    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = false;
    document.getElementById('generate-btn-text').style.display = 'inline';
    document.getElementById('generate-btn-loading').style.display = 'none';
    
    if (data && data.result) {
        // Store the code with proper formatting from the result
        generatedKernelCode = data.result.code;
        evaluationResults = data.result.evaluation || {};
        
        // Show kernel section
        document.getElementById('kernel-section').style.display = 'block';
        
        // Show chat section
        document.getElementById('chat-section').style.display = 'block';
        
        // Display evaluation results
        displayEvaluationResults(evaluationResults);
    } else if (data && data.error) {
        updateStatus('error', 'Generation failed');
        appendLogEntry(`Error: ${data.error}`, 'error');
    }
}

function displayEvaluationResults(evaluation) {
    const resultsContainer = document.getElementById('evaluation-results');
    resultsContainer.innerHTML = '<h3>Evaluation Results</h3>';
    
    const results = document.createElement('div');
    results.className = 'evaluation-results';
    
    // Compilation status
    if (evaluation.compiled !== undefined) {
        const compiledItem = createResultItem(
            evaluation.compiled ? '✓' : '✗',
            `Compiled: ${evaluation.compiled ? 'Yes' : 'No'}`,
            evaluation.compiled
        );
        results.appendChild(compiledItem);
    }
    
    // Correctness status
    if (evaluation.correctness !== undefined) {
        const correctItem = createResultItem(
            evaluation.correctness ? '✓' : '✗',
            `Correctness: ${evaluation.correctness ? 'Passed' : 'Failed'}`,
            evaluation.correctness
        );
        results.appendChild(correctItem);
    }
    
    // Speedup
    const speedup = evaluation.runtime_stats?.performance_comparison?.speedup_ratio;
    if (speedup !== undefined) {
        const speedupItem = createResultItem(
            '⚡',
            `Speedup: ${speedup.toFixed(2)}× vs PyTorch`,
            speedup > 1.0
        );
        results.appendChild(speedupItem);
    }
    
    // Error information
    if (evaluation.error) {
        const errorItem = document.createElement('div');
        errorItem.className = 'result-item';
        errorItem.innerHTML = `<span class="result-icon">⚠️</span><span>Error: ${evaluation.error}</span>`;
        results.appendChild(errorItem);
    }
    
    resultsContainer.appendChild(results);
}

function createResultItem(icon, text, success) {
    const item = document.createElement('div');
    item.className = 'result-item';
    item.innerHTML = `
        <span class="result-icon">${icon}</span>
        <span style="color: ${success ? '#4caf50' : '#f44336'}">${text}</span>
    `;
    return item;
}

function handleStreamError(error) {
    updateStatus('error', 'Connection error');
    appendLogEntry(`Stream error: ${error.message}`, 'error');
    
    // Re-enable button
    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = false;
    document.getElementById('generate-btn-text').style.display = 'inline';
    document.getElementById('generate-btn-loading').style.display = 'none';
}

function handleStreamComplete() {
    // Stream completed normally
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
}

function updateStatus(status, text) {
    const badge = document.getElementById('status-badge');
    badge.textContent = text;
    badge.className = `status-badge ${status}`;
}

function resetUI() {
    document.getElementById('kernel-code').textContent = 'Generated kernel code will appear here...';
    document.getElementById('evaluation-results').innerHTML = '';
    document.getElementById('chat-messages').innerHTML = '';
    // Clear RAG examples
    const ragContainer = document.getElementById('rag-examples-container');
    if (ragContainer) {
        ragContainer.innerHTML = '<p class="placeholder">Similar examples will appear here after generation starts...</p>';
    }
    generatedKernelCode = null;
    evaluationResults = null;
}

function displayRagExample(example) {
    const container = document.getElementById('rag-examples-container');
    if (!container) return;
    
    // Remove placeholder if exists
    const placeholder = container.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    const exampleBox = document.createElement('div');
    exampleBox.className = 'rag-example-box';
    
    const scoreDisplay = example.score ? `Score: ${example.score.toFixed(2)}` : '';
    
    exampleBox.innerHTML = `
        <div class="rag-example-header">
            <div class="example-info">
                <span class="example-number">#${example.index}</span>
                <span class="example-name">${example.problem_name || 'Example ' + example.index}</span>
                ${scoreDisplay ? `<span class="example-score">${scoreDisplay}</span>` : ''}
            </div>
            <button class="rag-example-copy-btn" onclick="copyRagExample(${example.index})">📋 Copy DSL</button>
        </div>
        <div class="rag-example-content">
            <div class="rag-example-panel">
                <h4>Original PyTorch</h4>
                <pre class="rag-example-code" id="rag-ref-${example.index}">${escapeHtml(example.reference_code || 'N/A')}</pre>
            </div>
            <div class="rag-example-panel">
                <h4>CuTe DSL Solution</h4>
                <pre class="rag-example-code" id="rag-dsl-${example.index}">${escapeHtml(example.dsl_code || 'N/A')}</pre>
            </div>
        </div>
    `;
    
    container.appendChild(exampleBox);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function copyRagExample(index) {
    const dslCode = document.getElementById(`rag-dsl-${index}`);
    if (dslCode) {
        navigator.clipboard.writeText(dslCode.textContent).then(() => {
            alert('DSL code copied to clipboard!');
        });
    }
}

// Chat functionality
async function handleChatSend() {
    const input = document.getElementById('chat-input');
    const question = input.value.trim();
    
    if (!question || !currentSessionId) {
        return;
    }
    
    // Add user message to chat
    addChatMessage(question, 'user');
    input.value = '';
    
    // Show typing indicator
    const typingId = addChatMessage('Thinking...', 'assistant');
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                question: question
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        document.getElementById(typingId).remove();
        
        if (data.error) {
            addChatMessage(`Error: ${data.error}`, 'assistant');
        } else {
            addChatMessage(data.answer, 'assistant');
        }
    } catch (error) {
        document.getElementById(typingId).remove();
        addChatMessage(`Error: ${error.message}`, 'assistant');
    }
}

function addChatMessage(message, role) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageId = `msg_${Date.now()}_${Math.random()}`;
    
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `chat-message ${role}`;
    
    const textDiv = document.createElement('div');
    textDiv.textContent = message;
    messageDiv.appendChild(textDiv);
    
    const timestampDiv = document.createElement('div');
    timestampDiv.className = 'timestamp';
    timestampDiv.textContent = new Date().toLocaleTimeString();
    messageDiv.appendChild(timestampDiv);
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageId;
}

function copyKernel() {
    // Use the stored generatedKernelCode which has proper formatting
    const kernelCode = generatedKernelCode || document.getElementById('kernel-code').textContent;
    if (!kernelCode) {
        alert('No kernel code to copy');
        return;
    }
    
    navigator.clipboard.writeText(kernelCode).then(() => {
        const btn = document.getElementById('copy-kernel-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '✓ Copied!';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    });
}

function downloadKernel() {
    const kernelCode = document.getElementById('kernel-code').textContent;
    if (!kernelCode) {
        alert('No kernel code to download');
        return;
    }
    
    const blob = new Blob([kernelCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generated_kernel_${Date.now()}.py`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

