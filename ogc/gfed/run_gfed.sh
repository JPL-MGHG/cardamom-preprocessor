#!/bin/bash
set -euo pipefail
source activate cardamom-ecmwf-downloader

# ==============================================================================
# MAAP Wrapper for CARDAMOM GFED Fire Emissions Downloader
#
# Purpose:
#   1. Retrieve GFED SFTP credentials from MAAP secrets
#   2. Set environment variables for SFTP access
#   3. Invoke core stac_cli.py GFED downloader without modification
#
# Usage (via CWL):
#   /app/ogc/gfed/run_gfed.sh \
#     --start-year 2001 --end-year 2024 \
#     [--gfed_sftp_username USERNAME] \
#     [--gfed_sftp_password PASSWORD]
#
# The wrapper script handles:
#   - MAAP secrets retrieval using maap-py
#   - Environment variable setup for SFTP credentials
#   - Output directory preparation
#   - Core CLI invocation
#   - Status reporting and error handling
# ==============================================================================

echo "========================================="
echo "CARDAMOM GFED Downloader - MAAP Wrapper"
echo "========================================="
echo "Timestamp: $(date -Iseconds)"
echo ""

# ==============================================================================
# Step 1: Parse Arguments for Credentials
# ==============================================================================

echo "[1/4] Parsing command-line arguments..."

GFED_SFTP_USERNAME=""
GFED_SFTP_PASSWORD=""
CLI_ARGS=()

# Parse arguments to separate credentials from CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gfed_sftp_username)
            GFED_SFTP_USERNAME="$2"
            shift 2
            ;;
        --gfed_sftp_password)
            GFED_SFTP_PASSWORD="$2"
            shift 2
            ;;
        *)
            CLI_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "  ✓ Arguments parsed"
echo ""

# ==============================================================================
# Step 2: Retrieve GFED SFTP Credentials
# ==============================================================================

echo "[2/4] Retrieving GFED SFTP credentials..."

if [[ -n "$GFED_SFTP_USERNAME" ]] && [[ -n "$GFED_SFTP_PASSWORD" ]]; then
    # Credentials provided via CWL input parameters
    echo "  Using credentials from CWL input parameters"
    echo "  ✓ Credentials obtained (from inputs)"
else
    # Retrieve from MAAP secrets using maap-py
    echo "  Retrieving from MAAP platform secrets..."

    MAAP_SCRIPT=$(cat <<'PYTHON'
from maap.maap import MAAP
import sys
import traceback

try:
    maap = MAAP()
    username = maap.secrets.get_secret("GFED_SFTP_USERNAME")
    password = maap.secrets.get_secret("GFED_SFTP_PASSWORD")

    if not username or not password:
        print("ERROR: GFED SFTP credentials are missing or empty", file=sys.stderr)
        sys.exit(1)

    print(f"{username}")
    print(f"{password}")

except Exception as e:
    print(f"ERROR: Failed to retrieve MAAP secrets: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYTHON
    )

    # Execute Python script to retrieve secrets
    SECRETS_OUTPUT=$(python -c "$MAAP_SCRIPT" 2>&1) || {
        EXIT_CODE=$?
        echo "ERROR: Failed to retrieve credentials from MAAP platform"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Ensure you are running on NASA MAAP platform"
        echo "  2. Configure MAAP secrets with your GFED SFTP credentials:"
        echo "     maap.secrets.create_secret('GFED_SFTP_USERNAME', 'sftp-username')"
        echo "     maap.secrets.create_secret('GFED_SFTP_PASSWORD', 'sftp-password')"
        echo "  3. Contact GFED data provider for SFTP access"
        echo ""
        echo "Script output:"
        echo "$SECRETS_OUTPUT"
        exit $EXIT_CODE
    }

    # Parse secrets output (two lines: username, password)
    GFED_SFTP_USERNAME=$(echo "$SECRETS_OUTPUT" | sed -n '1p')
    GFED_SFTP_PASSWORD=$(echo "$SECRETS_OUTPUT" | sed -n '2p')
fi

# Validate credentials are not empty
if [[ -z "$GFED_SFTP_USERNAME" ]] || [[ -z "$GFED_SFTP_PASSWORD" ]]; then
    echo "ERROR: GFED SFTP credentials are empty or missing"
    exit 1
fi

echo "  ✓ Credentials retrieved successfully"
echo ""

# ==============================================================================
# Step 3: Set Environment Variables for SFTP Access
# ==============================================================================

echo "[3/4] Setting SFTP environment variables..."

export GFED_SFTP_USERNAME
export GFED_SFTP_PASSWORD

echo "  ✓ SFTP credentials configured"
echo ""

# ==============================================================================
# Step 4: Prepare Output Directory and Execute CLI
# ==============================================================================

echo "[4/4] Executing CARDAMOM GFED downloader..."

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

cd /app

# Execute stac_cli.py gfed subcommand with all arguments
python -m src.stac_cli gfed \
    --output "$OUTPUT_DIR" \
    "${CLI_ARGS[@]}"

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
