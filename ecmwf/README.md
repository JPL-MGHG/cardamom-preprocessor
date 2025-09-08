# ECMWF Data Downloader

A generic Python script for downloading ECMWF ERA5 reanalysis data for CARDAMOM preprocessing.

## Prerequisites

### 1. Install CDS API
```bash
pip install cdsapi
```

### 2. Set up ECMWF CDS API credentials
1. Create an account at [Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
2. Get your API key from your [user profile page](https://cds.climate.copernicus.eu/how-to-api)
3. Create a `.cdsapirc` file in your home directory:
```
url: https://cds.climate.copernicus.eu/api/v2
key: {your_uid}:{your_api_key}
```

## Usage

The script supports both command-line interface (CLI) and Python API usage.

### Command Line Interface

#### Quick Start - Predefined Configurations

```bash
# Download CARDAMOM hourly drivers (CONUS, 2015-2020)
python ecmwf_downloader.py cardamom-hourly

# Download CARDAMOM monthly drivers (Global, 2001-2024)  
python ecmwf_downloader.py cardamom-monthly

# Customize years for predefined configs
python ecmwf_downloader.py cardamom-hourly -y 2018-2020
python ecmwf_downloader.py cardamom-monthly -y 2020-2023
```

#### Custom Downloads

```bash
# Download monthly temperature data for 2020-2021
python ecmwf_downloader.py monthly -v 2m_temperature -y 2020-2021 -m 1-12

# Download hourly precipitation for summer 2020 (CONUS region)
python ecmwf_downloader.py hourly -v total_precipitation -y 2020 -m 6-8 --area 60,-130,20,-50

# Download multiple variables with custom output directory
python ecmwf_downloader.py monthly -v "2m_temperature,total_precipitation" -y 2023 -m 1-3 -o ./my_data

# Use different grid resolution and format
python ecmwf_downloader.py hourly -v skin_temperature -y 2022 -m 6 --grid 0.25/0.25 --format grib
```

#### Command Help

```bash
# General help
python ecmwf_downloader.py -h

# Help for specific commands
python ecmwf_downloader.py hourly -h
python ecmwf_downloader.py monthly -h
```

### Python API Usage

For programmatic use, you can import and use the classes directly:

```python
from ecmwf_downloader import ECMWFDownloader

# Initialize downloader with custom parameters
downloader = ECMWFDownloader(
    area=[60, -130, 20, -50],  # [North, West, South, East]
    grid=["0.5/0.5"],          # Grid resolution
    output_dir="./my_data"     # Output directory
)

# Download hourly data
downloader.download_hourly_data(
    variables=["2m_temperature", "total_precipitation"],
    years=[2020, 2021],
    months=[1, 2, 3],
    file_prefix="MY_HOURLY_DATA"
)

# Download monthly data
downloader.download_monthly_data(
    variables=["skin_temperature"],
    years=[2020],
    months=[6, 7, 8],
    product_type="monthly_averaged_reanalysis"
)
```

### Variable Mapping Files

For complex variable mappings, create a JSON file:

```json
{
    "skin_temperature": "SKT",
    "surface_solar_radiation_downwards": "SSRD",
    "2m_temperature": "T2M"
}
```

Use with CLI:
```bash
python ecmwf_downloader.py hourly -v skin_temperature,2m_temperature -y 2020 -m 6 --var-map variables.json
```

## CLI Arguments Reference

### Common Arguments (hourly/monthly)
- `-v, --variables`: Comma-separated variable names (required)
- `-y, --years`: Years as single value or range (e.g., `2020` or `2020-2022`)
- `-m, --months`: Months as single value or range (e.g., `6` or `6-8`) 
- `-o, --output-dir`: Output directory (default: `./ecmwf_data`)
- `--area`: Bounding box as `N,W,S,E` (default: global)
- `--grid`: Grid resolution (default: `0.5/0.5`)
- `--format`: Data format - `netcdf` or `grib` (default: `netcdf`)

### Hourly-specific Arguments
- `--dataset`: ECMWF dataset name (default: `reanalysis-era5-single-levels`)
- `--prefix`: File prefix (default: `ECMWF_HOURLY`)
- `--var-map`: JSON file with variable mappings

### Monthly-specific Arguments  
- `--product-type`: Product type (default: `monthly_averaged_reanalysis`)
- `--dataset`: ECMWF dataset name (default: `reanalysis-era5-single-levels-monthly-means`)
- `--prefix`: File prefix (default: `ECMWF_MONTHLY`)

### Predefined Configuration Arguments
- `-o, --output-dir`: Override default output directory
- `-y, --years`: Override default year range

## Configuration Options

### Area Definitions
- **Global**: `[-89.75, -179.75, 89.75, 179.75]`
- **CONUS**: `[60, -130, 20, -50]`
- **Custom**: `[North, West, South, East]` in degrees

### Grid Resolutions
- `["0.5/0.5"]` - 0.5° × 0.5°
- `["0.25/0.25"]` - 0.25° × 0.25°
- `["1.0/1.0"]` - 1.0° × 1.0°

### Variables
Common ERA5 variables:
- `"2m_temperature"`
- `"2m_dewpoint_temperature"`
- `"total_precipitation"`
- `"skin_temperature"`
- `"surface_solar_radiation_downwards"`
- `"snowfall"`

Full list: [ERA5 Variables](https://confluence.ecmwf.int/pages/viewpage.action?pageId=536218894)

### Product Types (Monthly Data)
- `"monthly_averaged_reanalysis"` - Standard monthly averages
- `"monthly_averaged_reanalysis_by_hour_of_day"` - Monthly averages by hour

## File Naming Convention

- **Hourly**: `{prefix}_{variable_abbr}_{MM}{YYYY}.nc`
- **Monthly**: `{prefix}_{variable}_{MM}{YYYY}.nc`

Example: `ECMWF_CARDAMOM_HOURLY_DRIVER_SKT_012020.nc`

## Features

- ✅ **Duplicate detection** - Skips existing files
- ✅ **Flexible parameters** - Single values or lists accepted
- ✅ **Variable mapping** - Custom abbreviations for filenames
- ✅ **Directory creation** - Auto-creates output directories
- ✅ **Progress logging** - Shows download progress
- ✅ **Error handling** - Graceful handling of API errors

## Troubleshooting

### Common Issues

1. **Authentication Error**
   - Check your `.cdsapirc` file is in the home directory
   - Verify your API key is correct

2. **Download Timeout**
   - Large requests may take time to process
   - Try smaller date ranges or fewer variables

3. **Invalid Variable Names**
   - Check the [ERA5 variable list](https://confluence.ecmwf.int/pages/viewpage.action?pageId=536218894)
   - Use exact variable names from ECMWF documentation

4. **Memory Issues**
   - Reduce spatial/temporal coverage
   - Download variables separately

### API Limits
- CDS API has usage limits and queuing system
- Large requests may be queued
- Consider breaking large downloads into smaller chunks

## Examples

### Download Temperature Data for Europe
```python
downloader = ECMWFDownloader(
    area=[75, -15, 35, 45],  # Europe
    output_dir="./europe_data"
)

downloader.download_monthly_data(
    variables="2m_temperature",
    years=2023,
    months=list(range(1, 13))
)
```

### Download Precipitation for Specific Months
```python
downloader.download_hourly_data(
    variables="total_precipitation",
    years=[2020, 2021],
    months=[6, 7, 8],  # Summer months
    file_prefix="SUMMER_PRECIP"
)
```

## References

- [CDS API Installation Guide](https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+macOS)
- [ERA5 Variable Documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=536218894)
- [Climate Data Store](https://cds.climate.copernicus.eu/)