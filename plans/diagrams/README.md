# CARDAMOM Preprocessor Data Flow Diagrams

This directory contains comprehensive data flow diagrams for the CARDAMOM STAC-based preprocessor system.

## Overview

These diagrams trace the complete data flow from CLI entry points through third-party API calls, data transformations, and final output generation. Each diagram documents:

- CLI command parameters
- Third-party API calls with specific parameters (CDS API, NOAA GML, ORNL DAAC)
- Data transformations and unit conversions
- Output data format, dimensions, and variable names

## Diagram Files

### ECMWF ERA5 Variables (8 diagrams)

ERA5 meteorological reanalysis data from ECMWF Climate Data Store:

1. **[ecmwf_t2m_min_flow.drawio](./ecmwf_t2m_min_flow.drawio)**
   - Monthly minimum 2-meter temperature
   - CDS API product: `monthly_averaged_reanalysis_by_hour_of_day`
   - Transformation: Extract min over 24 hourly values
   - Output: T2M_MIN [K], dimensions [time=1, lat=360, lon=720]

2. **[ecmwf_t2m_max_flow.drawio](./ecmwf_t2m_max_flow.drawio)**
   - Monthly maximum 2-meter temperature
   - CDS API product: `monthly_averaged_reanalysis_by_hour_of_day`
   - Transformation: Extract max over 24 hourly values
   - Output: T2M_MAX [K], dimensions [time=1, lat=360, lon=720]

3. **[ecmwf_vpd_flow.drawio](./ecmwf_vpd_flow.drawio)**
   - Vapor Pressure Deficit (requires 2 CDS API calls)
   - Variables: `2m_temperature` + `2m_dewpoint_temperature`
   - Transformation: Tetens equation calculation from max temperature and dewpoint
   - Output: VPD [hPa], dimensions [time=1, lat=360, lon=720]

4. **[ecmwf_total_prec_flow.drawio](./ecmwf_total_prec_flow.drawio)**
   - Total precipitation
   - CDS API product: `monthly_averaged_reanalysis`
   - Transformation: Convert m/s to mm/month
   - Output: TOTAL_PREC [mm], dimensions [time=1, lat=360, lon=720]

5. **[ecmwf_ssrd_flow.drawio](./ecmwf_ssrd_flow.drawio)**
   - Surface solar radiation downwards
   - CDS API product: `monthly_averaged_reanalysis`
   - Transformation: Convert J/m²/day to W/m² (divide by 86400)
   - Output: SSRD [W m-2], dimensions [time=1, lat=360, lon=720]

6. **[ecmwf_strd_flow.drawio](./ecmwf_strd_flow.drawio)**
   - Surface thermal radiation downwards
   - CDS API product: `monthly_averaged_reanalysis`
   - Transformation: Convert J/m²/day to W/m² (divide by 86400)
   - Output: STRD [W m-2], dimensions [time=1, lat=360, lon=720]

7. **[ecmwf_skt_flow.drawio](./ecmwf_skt_flow.drawio)**
   - Skin temperature
   - CDS API product: `monthly_averaged_reanalysis`
   - Transformation: Minimal (unit already in K)
   - Output: SKT [K], dimensions [time=1, lat=360, lon=720]

8. **[ecmwf_snowfall_flow.drawio](./ecmwf_snowfall_flow.drawio)**
   - Snowfall (water equivalent)
   - CDS API product: `monthly_averaged_reanalysis`
   - Transformation: Convert m to mm/month
   - Output: SNOWFALL [mm], dimensions [time=1, lat=360, lon=720]

### NOAA Data (1 diagram)

Global atmospheric CO2 concentration from NOAA Global Monitoring Laboratory:

9. **[noaa_co2_flow.drawio](./noaa_co2_flow.drawio)**
   - Atmospheric CO2 concentration
   - API: HTTPS download from `https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv`
   - Transformation: Parse CSV, create spatially-replicated grid (CO2 is uniform globally)
   - Output options:
     - Single month: CO2 [ppm], dimensions [time=1, lat=360, lon=720]
     - Full dataset: CO2 [ppm], dimensions [time=N, lat=360, lon=720] (N ≈ 546 months from 1979-2024)

### GFED Data (1 diagram)

Burned area from Global Fire Emissions Database:

10. **[gfed_burned_area_flow.drawio](./gfed_burned_area_flow.drawio)**
    - Monthly burned area fraction
    - API: HTTPS download from ORNL DAAC (`GFED4.1s_{year}.hdf5`)
    - Transformation:
      - Extract monthly data from HDF5 structure
      - Regrid from 0.25° native to 0.5° CARDAMOM (2×2 averaging)
    - Output: BURNED_AREA [fraction], dimensions [time=1, lat=360, lon=720]

### CBF Generator (1 diagram)

CARDAMOM Binary Format file generation from STAC data:

11. **[cbf_generator_flow.drawio](./cbf_generator_flow.drawio)**
    - Complete workflow from STAC data discovery to pixel-specific CBF files
    - STAC API queries for 10 required variables
    - Data validation (critical vs optional variables)
    - Meteorological data loading and assembly
    - Region definition (CONUS, global)
    - Pixel-level CBF file generation
    - Output: One CBF file per pixel (e.g., ~12,800 files for CONUS)
      - Dimensions per file: [time=N, lat=scalar, lon=scalar]
      - Variables: All 10 meteorological forcing variables
      - Format: CARDAMOM-compatible NetCDF with CF-1.8 conventions

## Common Output Specifications

All NetCDF outputs follow these standards:

- **Spatial Resolution**: 0.5° (CARDAMOM standard)
- **Latitude Grid**: -89.75° to 89.75° (360 points)
- **Longitude Grid**: -179.75° to 179.75° (720 points)
- **Conventions**: CF-1.8
- **Encoding**: float32, zlib compression level 4
- **Fill Value**: -9999.0
- **Time Encoding**: 'days since 2001-01-01'

## CDS API Authentication

ECMWF diagrams require CDS API credentials:
- Local development: `~/.cdsapirc` file with URL and Personal Access Token
- Get token from: https://cds.climate.copernicus.eu/profile

## Opening the Diagrams

These are [draw.io](https://app.diagrams.net/) (diagrams.net) files. Open them with:
- **Web**: Upload to https://app.diagrams.net/
- **Desktop**: Install draw.io desktop app
- **VS Code**: Install "Draw.io Integration" extension

## Diagram Color Scheme

- **Green**: CLI entry points and final outputs
- **Yellow**: Variable resolution and data parsing
- **Blue**: Third-party API calls
- **Purple**: Raw data files
- **Orange**: Data processing and transformations
- **Red/Pink**: Standard NetCDF creation
- **Gray**: Cleanup operations
- **Light Blue**: STAC operations (when included)

## Notes

- STAC metadata operations are excluded from these diagrams per specification
- Diagrams focus on data flow and transformations
- Variable names and dimensions are explicitly documented
- All third-party API parameters are detailed

## Related Documentation

- [STAC Implementation Summary](../../STAC_IMPLEMENTATION_SUMMARY.md)
- [Migration Notes](../../MIGRATION_NOTES.md)
- [Original MATLAB code](../../matlab-migration/)

---

*Generated: 2025-12-17*
*CARDAMOM Preprocessor v1.0*
