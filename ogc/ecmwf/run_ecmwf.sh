#!/bin/bash
set -euo pipefail
source activate cardamom-ecmwf-downloader
# ==============================================================================
# MAAP Wrapper for CARDAMOM ECMWF ERA5 Downloader
#
# Purpose:
#   1. Retrieve ECMWF CDS API credentials from MAAP secrets
#   2. Configure CDS API authentication
#   3. Invoke core stac_cli.py without modification
#
# Usage (via CWL):
#   /app/ogc/ecmwf/run_ecmwf.sh \
#     --variables t2m_min,t2m_max \
#     --year 2020 --month 1 \
#     [--ecmwf_cds_uid UID --ecmwf_cds_key KEY]
#
# The wrapper script handles:
#   - MAAP secrets retrieval using maap-py
#   - .cdsapirc file creation for cdsapi library
#   - Environment setup for CARDAMOM preprocessor
#   - Output directory preparation
# ==============================================================================

echo "========================================="
echo "CARDAMOM ECMWF Downloader - MAAP Wrapper"
echo "========================================="
echo "Timestamp: $(date -Iseconds)"
echo ""

# ==============================================================================
# Step 1: Parse and Validate Arguments
# ==============================================================================

echo "[1/5] Parsing command-line arguments..."

# Extract credentials if provided as arguments (for testing/override)
ECMWF_CDS_UID=""
ECMWF_CDS_KEY=""
CLI_ARGS=()

# Parse arguments to separate credentials from CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ecmwf_cds_uid)
            ECMWF_CDS_UID="$2"
            shift 2
            ;;
        --ecmwf_cds_key)
            ECMWF_CDS_KEY="$2"
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
# Step 2: Retrieve ECMWF CDS Credentials
# ==============================================================================

echo "[2/5] Retrieving ECMWF CDS credentials..."

if [[ -n "$ECMWF_CDS_UID" ]] && [[ -n "$ECMWF_CDS_KEY" ]]; then
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
    key = maap.secrets.get_secret("ECMWF_CDS_KEY")

    if not uid or not key:
        print("ERROR: Credentials are empty", file=sys.stderr)
        sys.exit(1)

    print(f"{key}")

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
        echo "  2. Configure MAAP secrets with your ECMWF CDS credentials:"
        echo "     maap.secrets.create_secret('ECMWF_CDS_KEY', 'your-key')"
        echo "  3. Get credentials from: https://cds.climate.copernicus.eu/user"
        echo ""
        echo "Script output:"
        echo "$SECRETS_OUTPUT"
        exit $EXIT_CODE
    }

    # Parse secrets (first line = UID, second line = KEY)
    ECMWF_CDS_KEY=$(echo "$SECRETS_OUTPUT" | sed -n '1p')
fi

# Validate credentials are not empty
if [[ -z "$ECMWF_CDS_KEY" ]]; then
    echo "ERROR: ECMWF CDS credentials are empty or missing"
    exit 1
fi

echo "  ✓ Credentials retrieved successfully"
echo ""

# ==============================================================================
# Step 3: Configure ECMWF CDS API
# ==============================================================================

echo "[3/5] Configuring ECMWF CDS API..."

# Create .cdsapirc file (expected by cdsapi library)
CDSAPIRC_FILE="${HOME}/.cdsapirc"

cat > "$CDSAPIRC_FILE" <<EOF
url: https://cds.climate.copernicus.eu/api
key: ${ECMWF_CDS_KEY}
EOF

# Set restrictive permissions (user read/write only)
chmod 600 "$CDSAPIRC_FILE"

if [[ -f "$CDSAPIRC_FILE" ]]; then
    echo "  ✓ Created ${CDSAPIRC_FILE} with secure permissions"
else
    echo "ERROR: Failed to create .cdsapirc file"
    exit 1
fi

echo ""

# ==============================================================================
# Step 4: Prepare Output Directory
# ==============================================================================

echo "[4/5] Preparing output directory..."

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
# Step 5: Invoke Core CLI
# ==============================================================================

echo "[5/5] Executing CARDAMOM ECMWF downloader..."
echo ""

cd /app

# Execute stac_cli.py ecmwf subcommand with all arguments
# The CLI_ARGS are passed through from CWL input bindings
python -m src.stac_cli ecmwf \
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
    echo "  STAC collections: ${OUTPUT_DIR}/*/"
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
