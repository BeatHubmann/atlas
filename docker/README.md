# ATLAS Docker Configuration

This directory contains Docker-related configuration for the ATLAS project, supporting both ARM64 (Apple Silicon) and AMD64 (x86_64) architectures.

## Multi-Platform Support

The Dockerfiles have been configured to build and run on multiple platforms:
- **ARM64/AARCH64**: For Apple Silicon Macs (M1/M2/M3/M4)
- **AMD64/x86_64**: For Intel/AMD processors and GitHub CI/CD runners

## Key Changes for Multi-Platform Support

1. **Platform-aware UV installation**: The Dockerfiles now detect the target architecture and download the appropriate UV binary
2. **Build arguments**: Using Docker's built-in `TARGETARCH` variable to determine the platform
3. **Early user creation**: The `atlas` user is created before dependency installation for better security
4. **Proper PATH handling**: UV's virtual environment is properly added to PATH

## Building Images

### Local Development (Single Platform)
For local development on your current platform:
```bash
make build
```

### Multi-Platform Build
To build for both ARM64 and AMD64:
```bash
make build-multiplatform
```

### Build and Push to Registry
To build multi-platform images and push to a registry:
```bash
make build-multiplatform-push
```

## Running Services

### Development Mode
```bash
# Start all services
make up

# View logs
make logs

# Stop services
make down
```

### Production Mode
Remove or rename `docker-compose.override.yml` to disable development-specific settings.

## Architecture Details

### UV Package Manager
UV is installed using platform-specific binaries:
- ARM64: `uv-aarch64-unknown-linux-gnu`
- AMD64: `uv-x86_64-unknown-linux-gnu`

### Build Process
1. Docker buildx is used for multi-platform builds
2. QEMU emulation allows building for different architectures
3. Images are built for both platforms simultaneously

### GitHub Actions
The `.github/workflows/docker-build.yml` workflow automatically:
- Builds multi-platform images on push to main
- Pushes images to GitHub Container Registry (ghcr.io)
- Caches layers for faster builds

## Troubleshooting

### "exec format error"
This error indicates an architecture mismatch. Ensure you're using the correct image for your platform.

### Slow builds on non-native architecture
Building ARM64 images on AMD64 (or vice versa) uses emulation and will be slower. Consider using native runners or pre-built images.

### UV installation fails
Check that the UV release URL is correct and the platform detection logic is working:
```bash
docker build --build-arg TARGETARCH=arm64 -f Dockerfile .
```

## Best Practices

1. Always test multi-platform builds before pushing to production
2. Use `docker buildx` for consistent multi-platform builds
3. Leverage GitHub Actions for automated multi-platform builds
4. Keep base images updated for security patches