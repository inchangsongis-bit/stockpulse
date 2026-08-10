#!/bin/bash
# StockPulse — Local Development Startup
# Usage: ./run.sh [backend|frontend|both]

set -e

MODE="${1:-both}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

start_backend() {
    echo "=== Starting Backend (FastAPI) ==="
    cd "$ROOT_DIR/backend"

    # Create venv if needed
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
    fi

    source venv/bin/activate
    pip install -q -r requirements.txt

    # Copy .env if not exists
    if [ ! -f "$ROOT_DIR/.env" ]; then
        cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
        echo "Created .env from .env.example — edit it to add your API keys."
    fi

    echo "Starting FastAPI on http://localhost:8000"
    echo "API docs at http://localhost:8000/docs"
    python main.py
}

start_frontend() {
    echo "=== Starting Frontend (Next.js) ==="
    cd "$ROOT_DIR/frontend"

    if [ ! -d "node_modules" ]; then
        echo "Installing dependencies..."
        npm install
    fi

    echo "Starting Next.js on http://localhost:3000"
    npm run dev
}

case $MODE in
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    both)
        echo "Starting backend and frontend..."
        echo "(Run in separate terminals for better log visibility)"
        echo ""
        echo "  Terminal 1:  ./run.sh backend"
        echo "  Terminal 2:  ./run.sh frontend"
        echo ""
        echo "Starting backend first..."
        start_backend &
        BACKEND_PID=$!
        sleep 3
        start_frontend &
        FRONTEND_PID=$!
        wait $BACKEND_PID $FRONTEND_PID
        ;;
    *)
        echo "Usage: ./run.sh [backend|frontend|both]"
        exit 1
        ;;
esac
