#!/bin/bash
# Deployment script for Text2Image Web Application
# This script deploys both frontend and backend services

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
echo -e "${BLUE}Text2Image Web Application Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

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

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate


# Create .env file if it doesn't exist
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

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Deployment Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}✓ Frontend:${NC}"
echo "  - Built and ready in: $FRONTEND_DIR"
echo "  - Production build: $FRONTEND_DIR/.next"
echo "  - API URL: http://localhost:${BACKEND_PORT}"
echo ""
echo -e "${GREEN}✓ Backend:${NC}"
echo "  - Dependencies installed in: $BACKEND_DIR/venv"
echo "  - Configuration: $BACKEND_DIR/.env"
echo "  - Port: ${BACKEND_PORT}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Start the Text-to-Image Server (if not running):"
echo "   python text2image_server.py --port 8001"
echo ""
echo "2. Start the Backend API Gateway:"
echo "   cd backend && source venv/bin/activate && python main.py"
echo "   Or: cd backend && ./start.sh"
echo ""
echo "3. Start the Frontend:"
echo "   cd frontend && npm start"
echo "   Frontend will be available at: http://localhost:${FRONTEND_PORT}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"

