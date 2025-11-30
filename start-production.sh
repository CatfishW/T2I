#!/bin/bash
# Production deployment and start script for Text2Image Web Application
# Deploys and starts both backend and frontend services
# Kills existing services on the ports before starting

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FRONTEND_DIR="frontend"
BACKEND_DIR="backend"
BACKEND_PORT=${BACKEND_PORT:-21115}
FRONTEND_PORT=${FRONTEND_PORT:-21116}
TEXT2IMAGE_SERVER_URL=${TEXT2IMAGE_SERVER_URL:-"http://localhost:8010"}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Text2Image Web Application${NC}"
echo -e "${BLUE}Deploy & Start${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to kill processes on a port
kill_port() {
    local port=$1
    local service_name=$2
    
    echo -e "${YELLOW}Checking for existing services on port ${port}...${NC}"
    
    # Try different methods to find and kill processes on the port
    local pids=""
    
    # Method 1: lsof (Linux/macOS)
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti:$port 2>/dev/null || true)
    fi
    
    # Method 2: fuser (Linux)
    if [ -z "$pids" ] && command -v fuser >/dev/null 2>&1; then
        pids=$(fuser $port/tcp 2>/dev/null | awk '{print $1}' || true)
    fi
    
    # Method 3: ss/netstat (Linux)
    if [ -z "$pids" ] && command -v ss >/dev/null 2>&1; then
        pids=$(ss -tlnp 2>/dev/null | grep ":$port " | awk '{print $6}' | sed 's/.*pid=\([0-9]*\).*/\1/' | sort -u || true)
    fi
    
    if [ -z "$pids" ] && command -v netstat >/dev/null 2>&1; then
        pids=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | sort -u || true)
    fi
    
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}Found existing ${service_name} processes on port ${port}${NC}"
        for pid in $pids; do
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                echo -e "${YELLOW}  Killing process ${pid}...${NC}"
                kill $pid 2>/dev/null || kill -9 $pid 2>/dev/null || true
            fi
        done
        sleep 2
        echo -e "${GREEN}✓ Port ${port} is now free${NC}"
    else
        echo -e "${GREEN}✓ No existing services on port ${port}${NC}"
    fi
    echo ""
}

# Kill existing services
kill_port $BACKEND_PORT "Backend"
kill_port $FRONTEND_PORT "Frontend"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command_exists node; then
    echo -e "${RED}✗ Node.js is not installed. Please install Node.js 18+ first.${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}✗ npm is not installed. Please install npm first.${NC}"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}✗ Python 3 is not installed. Please install Python 3.8+ first.${NC}"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}✗ Node.js version 18+ is required. Current version: $(node -v)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"
echo ""

# Deploy Frontend
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Deploying Frontend (Next.js)${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$FRONTEND_DIR" || exit 1

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
else
    echo -e "${YELLOW}Updating frontend dependencies...${NC}"
    npm install
fi

# Set API URL environment variable
export NEXT_PUBLIC_API_URL="http://localhost:${BACKEND_PORT}"

# Build frontend
echo -e "${YELLOW}Building frontend for production...${NC}"
npm run build

echo -e "${GREEN}✓ Frontend built successfully${NC}"
echo ""

# Return to root directory
cd ..

# Deploy Backend
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Deploying Backend (API Gateway)${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$BACKEND_DIR" || exit 1

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file with default configuration...${NC}"
    cat > .env << EOF
TEXT2IMAGE_SERVER_URL=${TEXT2IMAGE_SERVER_URL}
REQUEST_TIMEOUT=300
MAX_CONCURRENT_REQUESTS=10
EOF
    echo -e "${GREEN}✓ Created .env file${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

echo -e "${GREEN}✓ Backend dependencies installed${NC}"
echo ""

# Return to root directory
cd ..

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Backend
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Services${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${YELLOW}Starting Backend API Gateway...${NC}"
cd "$BACKEND_DIR" || exit 1

# Set environment variables
export BACKEND_PORT=${BACKEND_PORT}
export NEXT_PUBLIC_API_URL="http://localhost:${BACKEND_PORT}"

# Start backend in background
python main.py &
BACKEND_PID=$!

cd ..

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend to start...${NC}"
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Backend failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Backend started on port ${BACKEND_PORT}${NC}"
echo ""

# Start Frontend
echo -e "${YELLOW}Starting Frontend...${NC}"
cd "$FRONTEND_DIR" || exit 1

# Set API URL and port
export NEXT_PUBLIC_API_URL="http://localhost:${BACKEND_PORT}"
export PORT=${FRONTEND_PORT}

# Start frontend in background
npm start &
FRONTEND_PID=$!

cd ..

echo -e "${GREEN}✓ Frontend started on port ${FRONTEND_PORT}${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Services Running${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}✓ Backend API Gateway:${NC}"
echo "   http://localhost:${BACKEND_PORT}"
echo "   API Docs: http://localhost:${BACKEND_PORT}/docs"
echo ""
echo -e "${GREEN}✓ Frontend:${NC}"
echo "   http://localhost:${FRONTEND_PORT}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for processes
wait

