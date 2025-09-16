#!/usr/bin/env bash

set -e

# Get the directory containing this script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

echo "Building CARDAMOM ECMWF Downloader algorithm..."
echo "Root directory: ${root_dir}"

# Change to root directory
pushd "${root_dir}"

# Update conda environment
echo "Creating/updating conda environment from environment.yml..."
conda env update -f environment.yml

echo "Build completed successfully!"
popd