#!/usr/bin/env bash

set -e

# Get the directory containing this script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

echo "Running CARDAMOM ECMWF Downloader algorithm..."
echo "Root directory: ${root_dir}"

# Activate conda environment
echo "Activating conda environment: cardamom-ecmwf-downloader"
source activate cardamom-ecmwf-downloader

# Create output directory
mkdir -p "${root_dir}/output"

# Parse MAAP DPS input parameters
# Parameters are passed as environment variables or command line arguments

# Download mode (hourly, monthly, cardamom-hourly, cardamom-monthly)
DOWNLOAD_MODE=${1:-"cardamom-monthly"}

# Output directory
OUTPUT_DIR=${2:-"${root_dir}/output"}

# Years range (e.g., "2020-2022" or single year "2020")
YEARS=${3:-"2020-2021"}

# Months range (e.g., "1-12" or single month "6")
MONTHS=${4:-"1-12"}

# Variables (comma-separated list)
VARIABLES=${5:-"2m_temperature,total_precipitation"}

# Area bounds (N,W,S,E format, optional - defaults to global/CONUS based on mode)
AREA=${6:-""}

# Grid resolution
GRID=${7:-"0.5/0.5"}

# Data format
FORMAT=${8:-"netcdf"}

echo "Parameters:"
echo "  Download Mode: ${DOWNLOAD_MODE}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Years: ${YEARS}"
echo "  Months: ${MONTHS}"
echo "  Variables: ${VARIABLES}"
echo "  Area: ${AREA:-"(using default for mode)"}"
echo "  Grid: ${GRID}"
echo "  Format: ${FORMAT}"

# Build command arguments
CMD_ARGS=()
CMD_ARGS+=("${DOWNLOAD_MODE}")
CMD_ARGS+=("-o" "${OUTPUT_DIR}")
CMD_ARGS+=("-y" "${YEARS}")

# Add mode-specific arguments
case "${DOWNLOAD_MODE}" in
    "hourly"|"monthly")
        CMD_ARGS+=("-m" "${MONTHS}")
        CMD_ARGS+=("-v" "${VARIABLES}")
        if [ -n "${AREA}" ]; then
            CMD_ARGS+=("--area" "${AREA}")
        fi
        CMD_ARGS+=("--grid" "${GRID}")
        CMD_ARGS+=("--format" "${FORMAT}")
        ;;
    "cardamom-hourly"|"cardamom-monthly")
        # These modes have predefined variables and areas
        if [ "${YEARS}" != "2020-2021" ]; then
            CMD_ARGS+=("-y" "${YEARS}")
        fi
        ;;
esac

# Run the ECMWF downloader
echo "Executing: python ${root_dir}/ecmwf/ecmwf_downloader.py ${CMD_ARGS[*]}"
cd "${root_dir}"
python ecmwf/ecmwf_downloader.py "${CMD_ARGS[@]}"

echo "ECMWF download completed successfully!"
echo "Output files located in: ${OUTPUT_DIR}"

# List output files for verification
echo "Generated files:"
find "${OUTPUT_DIR}" -name "*.nc" -type f | head -10