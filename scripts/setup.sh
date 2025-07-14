#!/bin/bash
# Setup script for ATLAS infrastructure

set -e

echo "🚀 Setting up ATLAS infrastructure..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/scat
mkdir -p models/checkpoints
mkdir -p results/experiments
mkdir -p results/logs

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your configuration"
fi

# Build Docker images
echo "🔨 Building Docker images..."
docker compose build

# Start services
echo "🎯 Starting services..."
docker compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service status
echo "✅ Checking service status..."
docker compose ps

echo "🎉 ATLAS infrastructure setup complete!"
echo ""
echo "Services available at:"
echo "  - API: http://localhost:8000"
echo "  - Dashboard: http://localhost:8501"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "Run 'docker compose logs -f' to view logs"