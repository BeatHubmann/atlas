# Dockerfile for ATLAS API with multi-platform support
FROM python:3.12-slim

# Build arguments for platform detection
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETARCH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv package manager with platform-specific binary
# UV provides direct download URLs for different platforms
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        ARCH="x86_64"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        ARCH="aarch64"; \
    else \
        echo "Unsupported architecture: $TARGETARCH"; exit 1; \
    fi && \
    curl -LsSf "https://github.com/astral-sh/uv/releases/latest/download/uv-${ARCH}-unknown-linux-gnu.tar.gz" | tar -xz -C /usr/local/bin/ --strip-components=1 uv-${ARCH}-unknown-linux-gnu/uv && \
    chmod +x /usr/local/bin/uv

# Create non-root user early
RUN useradd -m -u 1000 atlas

# Change ownership of the app directory to atlas user
RUN chown -R atlas:atlas /app

# Copy dependency files
COPY --chown=atlas:atlas pyproject.toml uv.lock ./

# Install dependencies with uv as atlas user
USER atlas
RUN uv sync --frozen --no-cache

# Copy application code
COPY --chown=atlas:atlas src/ ./src/
COPY --chown=atlas:atlas data/ ./data/
COPY --chown=atlas:atlas models/ ./models/
COPY --chown=atlas:atlas results/ ./results/

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:${PATH}"

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uvicorn", "atlas_atc.api.main:app", "--host", "0.0.0.0", "--port", "8000"]