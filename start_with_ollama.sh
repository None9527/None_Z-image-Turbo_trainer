#!/bin/bash
# ============================================================================
# None Trainer - Linux/macOS Startup Script (with Ollama)
# 先启动 Ollama 服务，再启动 Trainer
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Default Configuration
# ============================================================================
TRAINER_PORT="${TRAINER_PORT:-9198}"
TRAINER_HOST="${TRAINER_HOST:-0.0.0.0}"
DEV_MODE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port|-p)
            TRAINER_PORT="$2"
            shift 2
            ;;
        --dev)
            DEV_MODE=1
            shift
            ;;
        --help|-h)
            echo "None Trainer Startup Script (with Ollama)"
            echo ""
            echo "Usage: ./start_with_ollama.sh [options]"
            echo ""
            echo "Options:"
            echo "  --port, -p PORT    Set port (default: 9198)"
            echo "  --dev              Development mode (hot reload)"
            echo "  --help, -h         Show this help"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# ============================================================================
# Load .env Configuration
# ============================================================================
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Apply defaults
MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/zimage_models}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets}"
OUTPUT_PATH="${OUTPUT_PATH:-$SCRIPT_DIR/output}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

# Create directories
mkdir -p "$DATASET_PATH" "$OUTPUT_PATH" "$OUTPUT_PATH/logs" "logs"

# Export for Python
export MODEL_PATH DATASET_PATH OUTPUT_PATH OLLAMA_HOST

# ============================================================================
# Display Banner
# ============================================================================
clear
echo ""
echo "   _   _                    _____          _"
echo "  | \ | |                  |_   _|        (_)"
echo "  |  \| | ___  _ __   ___    | |_ __ __ _ _ _ __   ___ _ __"
echo "  | . \` |/ _ \| '_ \ / _ \   | | '__/ _\` | | '_ \ / _ \ '__|"
echo "  | |\  | (_) | | | |  __/   | | | | (_| | | | | |  __/ |"
echo "  |_| \_|\___/|_| |_|\___|   \_/_|  \__,_|_|_| |_|\___|_|"
echo ""
echo "                 (with Ollama Auto-Start)"
echo ""

# ============================================================================
# Display Configuration
# ============================================================================
echo "=================================================="
echo "   Service Configuration"
echo "=================================================="
echo "   Port:        $TRAINER_PORT"
echo "   Host:        $TRAINER_HOST"
echo "   Models:      $MODEL_PATH"
echo "   Datasets:    $DATASET_PATH"
echo "   Output:      $OUTPUT_PATH"
echo "   Ollama:      $OLLAMA_HOST"
echo "=================================================="
echo ""

# ============================================================================
# Start Ollama Service
# ============================================================================
echo "[Start Ollama]"

# Check if Ollama is already running
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo "   Ollama:  [OK] Already running"
else
    echo "   Ollama:  [-] Not running, attempting to start..."
    
    # Check if ollama command exists
    if command -v ollama &> /dev/null; then
        echo "   Found:   $(which ollama)"
        echo "   Starting Ollama service in background..."
        
        # Start Ollama serve in background
        nohup ollama serve > /dev/null 2>&1 &
        OLLAMA_PID=$!
        
        # Wait for Ollama to start (max 10 seconds)
        for i in {1..10}; do
            sleep 1
            if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
                echo "   Ollama:  [OK] Started successfully (PID: $OLLAMA_PID)"
                break
            fi
            echo "   Waiting... ($i/10)"
        done
        
        # Final check
        if ! curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
            echo "   [WARN] Ollama did not start within 10 seconds"
        fi
    else
        echo "   [WARN] Ollama executable not found!"
        echo "          Please install Ollama from: https://ollama.com/download"
        echo "          Or start Ollama manually before using tagging features."
    fi
fi
echo ""

# ============================================================================
# Check Services
# ============================================================================
echo "[Check Services]"

# Check Python
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_EXE="$SCRIPT_DIR/venv/bin/python"
    echo "   Python:  [OK] venv"
elif command -v python3 &> /dev/null; then
    PYTHON_EXE="python3"
    echo "   Python:  [OK] system"
elif command -v python &> /dev/null; then
    PYTHON_EXE="python"
    echo "   Python:  [OK] system"
else
    echo "   Python:  [X] Not found!"
    exit 1
fi

# Activate venv if exists
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Check Ollama (final check)
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo "   Ollama:  [OK] Running"
else
    echo "   Ollama:  [-] Not running (tagging unavailable)"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    if [ -n "$GPU_NAME" ]; then
        echo "   GPU:     [OK] $GPU_NAME"
    else
        echo "   GPU:     [-] Not detected"
    fi
else
    echo "   GPU:     [-] nvidia-smi not found"
fi
echo ""

# ============================================================================
# Start Server
# ============================================================================
echo "Starting Web UI..."
echo ""
echo "   URL: http://localhost:$TRAINER_PORT"
echo ""
echo "   Press Ctrl+C to stop"
echo ""
echo "=================================================="

# Change to API directory
cd "$SCRIPT_DIR/webui-vue/api"

# Start TensorBoard in background
echo "Starting TensorBoard..."
$PYTHON_EXE -m tensorboard.main --logdir "$OUTPUT_PATH/logs" --port 6006 --host $TRAINER_HOST > /dev/null 2>&1 &
echo "   TensorBoard: http://localhost:6006"
echo ""

# Open browser (Linux/macOS)
(sleep 2 && (xdg-open "http://localhost:$TRAINER_PORT" 2>/dev/null || open "http://localhost:$TRAINER_PORT" 2>/dev/null || true)) &

# Start server
if [ "$DEV_MODE" = "1" ]; then
    echo "[Dev Mode] Hot reload enabled"
    $PYTHON_EXE -m uvicorn main:app --host $TRAINER_HOST --port $TRAINER_PORT --reload --reload-dir "$SCRIPT_DIR/webui-vue/api" --log-level info
else
    $PYTHON_EXE -m uvicorn main:app --host $TRAINER_HOST --port $TRAINER_PORT --log-level warning
fi

# ============================================================================
# Shutdown
# ============================================================================
echo ""
echo "=================================================="
echo "   Server stopped."
echo "=================================================="
