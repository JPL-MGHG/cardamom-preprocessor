#!/bin/bash
set -euo pipefail
source activate cardamom-ecmwf-downloader

# ==============================================================================
# MAAP Wrapper for CARDAMOM NOAA CO2 Downloader
#
# Purpose:
#   1. Prepare output directory
#   2. Invoke core NOAA CO2 downloader CLI
#
# Usage (via CWL):
#   /app/ogc/noaa/run_noaa.sh \
#     [--year YEAR] \
#     [--month MONTH] \
#     [--verbose]
#
# Note: NOAA data is PUBLIC - no credentials required
#
# The wrapper script handles:
#   - Output directory preparation
#   - Core CLI invocation
#   - Error handling and status reporting
# ==============================================================================

echo "========================================="
echo "CARDAMOM NOAA CO2 Downloader - MAAP Wrapper"
echo "========================================="
echo "Timestamp: $(date -Iseconds)"
echo ""

# ==============================================================================
# Step 1: Prepare Output Directory
# ==============================================================================

echo "[1/3] Preparing output directory..."

OUTPUT_DIR="/app/outputs"

# Ensure directory exists
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Verify directory is writable
if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo "ERROR: Output directory ${OUTPUT_DIR} is not writable"
    exit 1
fi

echo "  Output directory: ${OUTPUT_DIR}"
echo "  ✓ Directory ready"
echo ""

# ==============================================================================
# Step 2: Execute Core CLI
# ==============================================================================

echo "[2/3] Executing CARDAMOM NOAA CO2 downloader..."
echo ""

cd /app

# Execute stac_cli.py noaa subcommand with all arguments
python -m src.stac_cli noaa \
    --output "$OUTPUT_DIR" \
    "$@"

EXIT_CODE=$?

echo ""
echo "========================================="

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ Download completed successfully"
    echo ""
    echo "Outputs:"
    echo "  STAC catalog: ${OUTPUT_DIR}/catalog.json"
    echo "  NetCDF files: ${OUTPUT_DIR}/data/*.nc"
    echo "  STAC items: ${OUTPUT_DIR}/*/items/"
    echo ""

    # Summary statistics
    if [[ -d "${OUTPUT_DIR}/data" ]]; then
        NUM_FILES=$(find "${OUTPUT_DIR}/data" -name "*.nc" -type f | wc -l)
        echo "  Total NetCDF files: $NUM_FILES"
    fi
else
    echo "✗ Download failed with exit code ${EXIT_CODE}"
fi

echo "========================================="

exit $EXIT_CODE
