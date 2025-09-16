# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup Commands

**Python environment setup:**
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate cardamom-ecmwf-downloader

# Install additional dependencies if needed
pip install cdsapi maap-py pystac pystac-client boto3
```

**Testing and validation:**
```bash
# Test the ECMWF downloader CLI
python ecmwf/ecmwf_downloader.py -h

# Test predefined CARDAMOM configurations
python ecmwf/ecmwf_downloader.py cardamom-monthly -y 2020 -m 1

# Run MAAP algorithm locally (requires CDS credentials)
./.maap/run.sh cardamom-monthly ./test_output 2020 1-3
```

**MAAP platform commands:**
```bash
# Build MAAP algorithm environment
./.maap/build.sh

# Run MAAP wrapper for local testing
python .maap/dps_wrapper.py cardamom-monthly ./output 2020-2021 1-12
```

## Architecture Overview

This repository implements a **modular ECMWF data downloader** for the CARDAMOM carbon cycle modeling framework. It creates preprocessed meteorological datasets required for NASA MAAP platform carbon cycle analysis.

### Core Components

**Main Python Module (`ecmwf/`):**
- `ECMWFDownloader` class: Generic ECMWF CDS API interface with configurable parameters
- Command-line interface supporting both predefined CARDAMOM configurations and custom downloads
- Supports hourly and monthly ERA5 reanalysis data with flexible spatial/temporal filtering
- Built-in variable mapping system for consistent file naming conventions

**MAAP Integration (`.maap/`):**
- `algorithm_config.yaml`: NASA MAAP platform algorithm definition
- `dps_wrapper.py`: Python wrapper for MAAP DPS (Data Processing System) integration
- `run.sh`: Shell wrapper mapping MAAP parameters to CLI arguments
- `build.sh`: Conda environment setup for MAAP execution

**Environment Configuration:**
- `environment.yml`: Conda environment with cdsapi, xarray, netcdf4, maap-py dependencies
- Designed for Python 3.9 with NASA MAAP platform compatibility

### Data Flow Architecture

1. **Input Configuration**: MAAP algorithm parameters or CLI arguments specify download requirements
2. **CDS API Integration**: Authenticates with ECMWF Climate Data Store using API credentials
3. **Data Retrieval**: Downloads ERA5 reanalysis data in NetCDF format with configurable spatial/temporal bounds
4. **File Organization**: Generates consistently named files following CARDAMOM conventions
5. **MAAP Output**: Creates output manifest for NASA MAAP platform integration

### Predefined CARDAMOM Configurations

**CARDAMOM Monthly (Global):**
- Hourly averaged variables: 2m_temperature, 2m_dewpoint_temperature
- Monthly averaged variables: total_precipitation, skin_temperature, surface_solar_radiation_downwards, snowfall
- Global coverage: 89.75°N to -89.75°N, -179.75°W to 179.75°E
- Default time range: 2001-2024

**CARDAMOM Hourly (CONUS):**
- Variables: skin_temperature, surface_solar_radiation_downwards
- CONUS region: 60°N to 20°N, -130°W to -50°W
- All hourly timesteps (00:00-23:00)
- Default time range: 2015-2020

### File Naming Conventions

- **Hourly files**: `{prefix}_{variable_abbr}_{MM}{YYYY}.nc`
- **Monthly files**: `{prefix}_{variable}_{MM}{YYYY}.nc`
- **CARDAMOM prefix**: `ECMWF_CARDAMOM_DRIVER_` or `ECMWF_CARDAMOM_HOURLY_DRIVER_`

### MAAP Platform Integration

The algorithm is designed as a **NASA MAAP algorithm** with the following characteristics:
- Algorithm ID: `cardamom-ecmwf-downloader`
- Queue: `maap-dps-worker-8gb` (configurable based on data volume)
- Container: Custom MAAP base image with scientific Python stack
- Disk space: 100GB default (adjustable for large downloads)

**Key MAAP Parameters:**
- `download_mode`: Selects predefined CARDAMOM configurations or custom modes
- `years`/`months`: Temporal filtering with range support (e.g., "2020-2022", "6-8")
- `variables`: Comma-separated ECMWF variable names
- `area`: Optional spatial bounds as "N,W,S,E"
- `grid`: Spatial resolution (default: "0.5/0.5")

### Authentication Requirements

**ECMWF CDS API credentials required:**
- Local development: `.cdsapirc` file in home directory
- MAAP platform: `ECMWF_CDS_UID` and `ECMWF_CDS_KEY` environment variables

### Error Handling and Reliability

- **Duplicate detection**: Automatically skips existing files to enable resumable downloads
- **Graceful API handling**: Handles ECMWF CDS queue system and rate limiting
- **Parameter validation**: Validates spatial/temporal bounds and variable names
- **Logging integration**: Structured logging for MAAP platform monitoring

## Development Patterns

**Adding new variables:**
1. Check ERA5 variable documentation for exact names
2. Add to predefined configurations if commonly used
3. Update variable mapping dictionaries for consistent abbreviations

**Extending spatial coverage:**
1. Define new area bounds as `[North, West, South, East]` in decimal degrees
2. Consider memory implications for high-resolution global datasets
3. Test with small temporal ranges before full downloads

**MAAP algorithm updates:**
1. Modify `algorithm_config.yaml` for parameter changes
2. Update `run.sh` for new parameter mapping logic
3. Test locally with `dps_wrapper.py` before platform deployment

## Connection to CARDAMOM Framework

This preprocessor creates meteorological inputs for the main CARDAMOM framework located at `/Users/shah/Desktop/Development/ghg/CARDAMOM/`. The downloaded ERA5 data provides essential climate drivers for:

- **DALEC model simulations**: Photosynthesis, respiration, and carbon allocation processes
- **Bayesian parameter estimation**: Constraining ecosystem model parameters using observations
- **Model-data fusion**: MCMC algorithms for uncertainty quantification
- **CBF file generation**: Input format for CARDAMOM C framework execution

The preprocessor maintains compatibility with CARDAMOM's NetCDF-based data pipeline and CBF (CARDAMOM Binary Format) requirements.