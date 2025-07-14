#!/bin/bash
# Script to build multi-platform Docker images for ATLAS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building multi-platform Docker images for ATLAS${NC}"

# Check if Docker buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    echo -e "${RED}Docker buildx is not available. Please update Docker.${NC}"
    exit 1
fi

# Create a new buildx builder instance if it doesn't exist
BUILDER_NAME="atlas-builder"
if ! docker buildx ls | grep -q $BUILDER_NAME; then
    echo -e "${YELLOW}Creating new buildx builder: $BUILDER_NAME${NC}"
    docker buildx create --name $BUILDER_NAME --use
    docker buildx inspect --bootstrap
else
    echo -e "${YELLOW}Using existing buildx builder: $BUILDER_NAME${NC}"
    docker buildx use $BUILDER_NAME
fi

# Build arguments
PLATFORMS="linux/amd64,linux/arm64"
PUSH_FLAG=""

# Check if we should push to registry
if [ "$1" == "--push" ]; then
    PUSH_FLAG="--push"
    echo -e "${YELLOW}Images will be pushed to registry${NC}"
else
    echo -e "${YELLOW}Building locally only (use --push to push to registry)${NC}"
fi

# Build API image
echo -e "${GREEN}Building API image for platforms: $PLATFORMS${NC}"
docker buildx build \
    --platform $PLATFORMS \
    -t atlas-atc:latest \
    -t atlas-atc:api \
    -f Dockerfile \
    $PUSH_FLAG \
    .

# Build Frontend image
echo -e "${GREEN}Building Frontend image for platforms: $PLATFORMS${NC}"
docker buildx build \
    --platform $PLATFORMS \
    -t atlas-atc:frontend \
    -f Dockerfile.frontend \
    $PUSH_FLAG \
    .

echo -e "${GREEN}Build complete!${NC}"

# If not pushing, show how to load for local use
if [ -z "$PUSH_FLAG" ]; then
    echo -e "${YELLOW}To load images for local use on current platform:${NC}"
    echo "docker buildx build --load -t atlas-atc:latest -f Dockerfile ."
    echo "docker buildx build --load -t atlas-atc:frontend -f Dockerfile.frontend ."
fi