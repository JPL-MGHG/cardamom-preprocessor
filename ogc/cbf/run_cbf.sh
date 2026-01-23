#!/bin/bash
set -euo pipefail
source activate cardamom-ecmwf-downloader

# ==============================================================================
# MAAP Wrapper for CARDAMOM CBF File Generator
#
# Purpose:
#   1. Validate STAC API source (file:// or https://)
#   2. Prepare output directory
#   3. Invoke core CBF generator CLI
#
# Usage (via CWL):
#   /app/ogc/cbf/run_cbf.sh \
#     --stac-api file:///outputs/catalog.json \
#     --start 2020-01 --end 2020-12 \
#     --region conus
#
# Note: No credentials required for CBF generation (reads from STAC + files)
#
# The wrapper script handles:
#   - STAC API source validation
#   - Output directory preparation
#   - Core CLI invocation
#   - Status reporting and error handling
# ==============================================================================

echo "========================================="
echo "CARDAMOM CBF Generator - MAAP Wrapper"
echo "========================================="
echo "Timestamp: $(date -Iseconds)"
echo ""

# ==============================================================================
# Step 1: Parse Arguments and Validate Inputs
# ==============================================================================

echo "[1/3] Parsing arguments and validating STAC source..."

# Extract stac_api from arguments to validate it
STAC_API=""
CLI_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stac-api)
            STAC_API="$2"
            shift 2
            ;;
        *)
            CLI_ARGS+=("$1")
            shift
            ;;
    esac
done

# Re-add stac_api to CLI args
if [[ -n "$STAC_API" ]]; then
    CLI_ARGS=(--stac-api "$STAC_API" "${CLI_ARGS[@]}")
fi

# Validate STAC API source
if [[ -z "$STAC_API" ]]; then
    echo "ERROR: --stac-api parameter is required"
    exit 1
fi

if [[ "$STAC_API" == file://* ]]; then
    # Local file path - validate it exists
    STAC_FILE="${STAC_API#file://}"
    if [[ ! -f "$STAC_FILE" ]]; then
        echo "ERROR: STAC catalog file not found at $STAC_FILE"
        echo "       Check --stac-api parameter: $STAC_API"
        exit 1
    fi
    echo "  ✓ STAC catalog file verified at $STAC_FILE"
elif [[ "$STAC_API" == https://* ]]; then
    # Remote HTTPS URL - note: actual connectivity test happens during processing
    echo "  ✓ STAC API endpoint: $STAC_API"
else
    echo "ERROR: STAC API must be file:// URL or https:// URL"
    echo "       Provided: $STAC_API"
    exit 1
fi

echo ""

# ==============================================================================
# Step 2: Prepare Output Directory
# ==============================================================================

echo "[2/3] Preparing output directory..."

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
# Step 3: Execute CBF Generator
# ==============================================================================

echo "[3/3] Executing CARDAMOM CBF generator..."
echo ""

cd /app

# Execute stac_cli.py cbf-generate subcommand with all arguments
python -m src.stac_cli cbf-generate \
    --output "$OUTPUT_DIR" \
    "${CLI_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "========================================="

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ CBF generation completed successfully"
    echo ""
    echo "Outputs:"
    echo "  CBF files: ${OUTPUT_DIR}/site*.cbf.nc"
    echo ""

    # Summary statistics
    if [[ -d "${OUTPUT_DIR}" ]]; then
        NUM_CBF=$(find "${OUTPUT_DIR}" -name "*.cbf.nc" -type f | wc -l)
        echo "  Total CBF files generated: $NUM_CBF"
    fi
else
    echo "✗ CBF generation failed with exit code ${EXIT_CODE}"
fi

echo "========================================="

exit $EXIT_CODE
