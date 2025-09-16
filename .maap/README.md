# CARDAMOM ECMWF Downloader - MAAP Algorithm

This directory contains the MAAP (Multi-Mission Algorithm and Analysis Platform) configuration for the CARDAMOM ECMWF Downloader algorithm.

## Overview

The CARDAMOM ECMWF Downloader is a NASA MAAP algorithm that downloads meteorological variables from the ECMWF Climate Data Store (CDS) for carbon cycle modeling applications. It supports both predefined CARDAMOM configurations and custom downloads with flexible spatial/temporal parameters.

## Files

- `algorithm_config.yaml` - Main MAAP algorithm configuration
- `build.sh` - Build script for setting up the conda environment
- `run.sh` - Main execution script with parameter mapping
- `dps_wrapper.py` - Python wrapper for MAAP DPS integration
- `sample-algo-configs/` - Example configuration files

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| download_mode | "cardamom-monthly" | Download mode: hourly, monthly, cardamom-hourly, cardamom-monthly |
| output_dir | "./output" | Output directory for downloaded files |
| years | "2020-2021" | Years to download (single: 2020, range: 2020-2022) |
| months | "1-12" | Months to download (single: 6, range: 6-8) |
| variables | "2m_temperature,total_precipitation" | Comma-separated list of ECMWF variables |
| area | "" | Area bounds as N,W,S,E (optional - uses defaults for cardamom modes) |
| grid | "0.5/0.5" | Grid resolution (e.g., 0.5/0.5) |
| format | "netcdf" | Data format: netcdf or grib |

## Download Modes

### Predefined CARDAMOM Modes

1. **cardamom-monthly** (Global)
   - Monthly averaged reanalysis by hour: 2m_temperature, 2m_dewpoint_temperature
   - Monthly averaged reanalysis: total_precipitation, skin_temperature, surface_solar_radiation_downwards, snowfall
   - Global coverage: -89.75°N to 89.75°N, -179.75°W to 179.75°E

2. **cardamom-hourly** (CONUS)
   - Hourly data: skin_temperature, surface_solar_radiation_downwards
   - CONUS region: 60°N, -130°W, 20°N, -50°W
   - All hours (00:00-23:00)

### Custom Modes

1. **hourly** - Custom hourly data download
2. **monthly** - Custom monthly data download

## Usage Examples

### Using MAAP Python Client

```python
from maap.maap import MAAP

maap = MAAP()

# Submit CARDAMOM monthly global download
job = maap.submit_algorithm(
    algorithm_id="cardamom-ecmwf-downloader",
    parameters={
        "download_mode": "cardamom-monthly",
        "years": "2020-2022",
        "output_dir": "./monthly_global_data"
    }
)

print(f"Job submitted: {job.id}")
```

### Custom Regional Download

```python
job = maap.submit_algorithm(
    algorithm_id="cardamom-ecmwf-downloader",
    parameters={
        "download_mode": "monthly",
        "years": "2020-2021",
        "months": "6-8",
        "variables": "2m_temperature,total_precipitation",
        "area": "70,-160,40,-120",  # Alaska
        "grid": "0.25/0.25",
        "output_dir": "./alaska_summer"
    }
)
```

## Requirements

### ECMWF CDS Credentials

The algorithm requires ECMWF Climate Data Store credentials. In MAAP, these should be provided as environment variables:

- `ECMWF_CDS_UID` - Your CDS user ID
- `ECMWF_CDS_KEY` - Your CDS API key

### Resources

- **Queue**: maap-dps-worker-8gb (configurable based on data volume)
- **Disk Space**: 100GB (configurable based on download size)
- **Environment**: Python 3.9 with cdsapi, xarray, netcdf4

## Outputs

The algorithm generates NetCDF files with the following naming convention:

- **Hourly**: `ECMWF_HOURLY_{variable}_{MM}{YYYY}.nc`
- **Monthly**: `ECMWF_MONTHLY_{variable}_{MM}{YYYY}.nc`
- **CARDAMOM**: `ECMWF_CARDAMOM_DRIVER_{variable}_{MM}{YYYY}.nc`

An output manifest (`output_manifest.json`) is also created with metadata about the generated files.

## Local Testing

```bash
# Build the environment
./.maap/build.sh

# Test run (requires CDS credentials)
./.maap/run.sh cardamom-monthly ./test_output 2020 1-3

# Check outputs
ls -la ./test_output/
```

## Monitoring and Troubleshooting

- Check algorithm logs in `/tmp/algorithm.log`
- Verify CDS credentials are properly configured
- Monitor disk space usage for large downloads
- Check ECMWF CDS status for service availability