# Multi-stage build for optimized image size
FROM condaforge/miniforge3:25.11.0-0 AS builder

# Set working directory
WORKDIR /build

# Copy environment specification
COPY environment.yml /build/

# Create conda environment with all dependencies
RUN conda env create -f environment.yml && \
    conda clean -afy

# Install maap-py for MAAP secrets management
RUN /opt/conda/envs/cardamom-ecmwf-downloader/bin/pip install --no-cache-dir maap-py

# ============================================================================
# Production Stage
# ============================================================================

FROM condaforge/miniforge3:25.11.0-0

# OGC and Container Metadata Labels
LABEL maintainer="CARDAMOM Team <support@maap-project.org>"
LABEL version="1.0.0"
LABEL description="CARDAMOM ECMWF ERA5 Downloader - OGC Application Package for NASA MAAP"
LABEL org.opencontainers.image.source="https://github.com/JPL-MGHG/cardamom-preprocessor"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="cardamom-preprocessor-ecmwf"

# Copy conda environment from builder stage
COPY --from=builder /opt/conda/envs/cardamom-ecmwf-downloader /opt/conda/envs/cardamom-ecmwf-downloader

# Set environment variables to use conda environment by default
ENV PATH="/opt/conda/envs/cardamom-ecmwf-downloader/bin:$PATH"
ENV CONDA_DEFAULT_ENV="cardamom-ecmwf-downloader"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED="1"

# Set working directory for application
WORKDIR /app

# Copy application source code
COPY src/ /app/src/
COPY pyproject.toml /app/
COPY ogc/ /app/ogc/

# Make wrapper script executable
RUN chmod +x /app/ogc/ecmwf/run_ecmwf.sh && \
    /opt/conda/envs/cardamom-ecmwf-downloader/bin/pip install -e .

# Health check to verify environment
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.stac_cli import main; print('OK')" || exit 1

# Default command (will be overridden by CWL)
CMD ["/bin/bash"]
