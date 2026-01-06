# CARDAMOM Preprocessor Data Flow Diagrams

This directory contains comprehensive data flow diagrams for the CARDAMOM STAC-based preprocessor system.

## Overview

These diagrams serve three purposes:

1. **System Architecture**: Document overall system design and component interactions
2. **Data Flow**: Trace complete data flow from CLI entry points through third-party API calls, transformations, and outputs
3. **Variable Documentation**: Detail specific transformations for each meteorological and fire emissions variable

The diagrams document:
- CLI command parameters and options
- Third-party API calls with specific parameters (CDS API, NOAA GML, ORNL DAAC)
- Data transformations and unit conversions
- STAC metadata generation and catalog organization
- Variable registry and type classification
- Time coordinate alignment and data assembly
- Output data format, dimensions, and variable names

## Diagram Files

### Architecture Diagrams (6 diagrams)

System-level diagrams documenting design, organization, and data flow:

1. **[system_architecture_overview.drawio](./system_architecture_overview.drawio)**
   - Complete system architecture with 5 layers
   - Data acquisition layer (ECMWF, NOAA, GFED)
   - STAC discovery layer (metadata-based queries)
   - Data loading layer (meteorology and observations)
   - Data assembly layer (regridding, alignment)
   - CBF generation layer (pixel processing)

2. **[variable_registry_system.drawio](./variable_registry_system.drawio)**
   - Central `CARDAMOM_VARIABLE_REGISTRY` as single source of truth
   - Variable type classification (meteorological, fire_emissions, observational)
   - Metadata fields per variable (source, units, interpolation, range)
   - Helper functions for variable lookup and filtering
   - Registry consumers (downloaders, STAC utils, CBF generator, met loader)

3. **[stac_catalog_structure.drawio](./stac_catalog_structure.drawio)**
   - STAC directory structure with type-based collections
   - `cardamom-meteorological-variables/`, `cardamom-fire-emissions-variables/`, etc.
   - STAC item properties (cardamom:variable, variable_type, time_steps)
   - Discovery process (recursive query, metadata filtering, data loading)
   - Incremental collection updates with merge policies

4. **[meteorology_discovery_flow.drawio](./meteorology_discovery_flow.drawio)**
   - 4-phase meteorology loading workflow
   - Discovery phase: Pure metadata filtering from STAC
   - Loading phase: Handle 3 temporal structures (monthly, yearly, full time-series)
   - Validation phase: FAIL if any variable/month missing (critical for science)
   - Assembly phase: Regrid, normalize coordinates, apply land-sea mask

5. **[time_coordinate_alignment.drawio](./time_coordinate_alignment.drawio)**
   - Architectural shift in time authority
   - OLD: Scaffold template as time source
   - NEW: Meteorology dataset as time authority
   - Observation alignment to meteorology time coordinate
   - Graceful degradation with NaN-fill for missing observations

6. **[observational_data_handling.drawio](./observational_data_handling.drawio)**
   - Graceful degradation strategy for optional observational data
   - Input files (all optional): main obs, SOM, FIR
   - Loading strategy: NaN-fill for missing files/variables
   - Pixel-level extraction with NaN fallback
   - Degradation scenarios (all missing, partial, complete, temporal mismatch)

### ECMWF ERA5 Variables (8 diagrams)

ERA5 meteorological reanalysis data from ECMWF Climate Data Store:

7. **[ecmwf_t2m_min_flow.drawio](./ecmwf_t2m_min_flow.drawio)**
   - Monthly minimum 2-meter temperature
   - Variable registry lookup from CARDAMOM_VARIABLE_REGISTRY
   - CDS API product: `monthly_averaged_reanalysis_by_hour_of_day`
   - Transformation: Extract min over 24 hourly values
   - STAC metadata generation: Includes cardamom:variable, variable_type, time_steps
   - Output: T2M_MIN [K], dimensions [time=1, lat=360, lon=720]

8. **[ecmwf_t2m_max_flow.drawio](./ecmwf_t2m_max_flow.drawio)**
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

### NOAA Data (1 diagram)

Global atmospheric CO2 concentration from NOAA Global Monitoring Laboratory:

15. **[noaa_co2_flow.drawio](./noaa_co2_flow.drawio)**
    - Atmospheric CO2 concentration
    - API: HTTPS download from `https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv`
    - Transformation: Parse CSV, create spatially-replicated grid (CO2 is uniform globally)
    - Output options:
      - Single month: CO2 [ppm], dimensions [time=1, lat=360, lon=720]
      - Full dataset: CO2 [ppm], dimensions [time=N, lat=360, lon=720] (N ≈ 546 months from 1979-2024)

### GFED Data (1 diagram)

Burned area from Global Fire Emissions Database:

16. **[gfed_burned_area_flow.drawio](./gfed_burned_area_flow.drawio)**
    - Monthly burned area fraction
    - API: HTTPS download from ORNL DAAC (`GFED4.1s_{year}.hdf5`)
    - Transformation:
      - Extract monthly data from HDF5 structure
      - Regrid from 0.25° native to 0.5° CARDAMOM (2×2 averaging)
    - Output: BURNED_AREA [fraction], dimensions [time=1, lat=360, lon=720]

### CBF Generator (1 diagram)

CARDAMOM Binary Format file generation from STAC data:

17. **[cbf_generator_flow.drawio](./cbf_generator_flow.drawio)** ⭐ **UPDATED**
    - Complete workflow from STAC data discovery to pixel-specific CBF files
    - Two parallel data paths:
      - **Meteorology Path** (REQUIRED): STAC discovery → Load → Validate (FAIL if incomplete) → Assemble
      - **Observation Path** (OPTIONAL): Load user files → NaN-fill → Align to meteorology time
    - Time coordinate authority: Meteorology (not scaffold)
    - Graceful degradation: Meteorology FAILS, observations NaN-fill
    - Land pixel identification and pixel-specific CBF generation
    - Output: One CBF file per pixel (e.g., ~12,800 files for CONUS)
      - Dimensions per file: [time=12, lat=scalar, lon=scalar]
      - Forcing variables: All 10 meteorological variables (from STAC meteorology)
      - Constraints: Observational data with NaN-fill for missing
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

## Recent Changes (December 2025)

### New Architecture Diagrams (6 files)
- **system_architecture_overview.drawio**: Complete 5-layer system architecture
- **variable_registry_system.drawio**: Central metadata registry documentation
- **stac_catalog_structure.drawio**: STAC collection organization and discovery
- **meteorology_discovery_flow.drawio**: 4-phase meteorology loading workflow
- **time_coordinate_alignment.drawio**: Time authority shift (scaffold → meteorology)
- **observational_data_handling.drawio**: Graceful degradation for optional data

### CBF Generator Diagram Updates
- Separated meteorology (required) and observation (optional) paths
- Added STAC discovery layer with validation checkpoints
- Documented time coordinate alignment
- Highlighted graceful degradation strategy
- Comprehensive pixel processing with constraint setting

### Variable Diagram Updates
- Added variable registry lookup step
- Added STAC metadata generation step (creates items with cardamom: properties)
- Batch processing capability noted for ECMWF diagrams
- Updated radiation units: W/m² → MJ/m²/day (for SSRD, STRD)

### Key Architectural Changes Documented
1. **Variable Type Classification** (Commit f5b9093)
   - Variables now classified as METEOROLOGICAL, OBSERVATIONAL, FIRE_EMISSIONS
   - Single source of truth in `CARDAMOM_VARIABLE_REGISTRY`
   - Type-based STAC collection organization

2. **STAC Integration** (Commits 7ad7d0a, 314486c, eeb9d08)
   - Pure metadata-based discovery (no catalog structure assumptions)
   - Handles 3 temporal file structures (monthly, yearly, full time-series)
   - Automatic regridding for resolution mismatches
   - Incremental catalog updates with merge policies

3. **Time Coordinate Authority Shift** (Commit b27635d)
   - OLD: Scaffold template defines time
   - NEW: Meteorology dataset defines time (CBF authority)
   - Observations align to meteorology time with NaN-fill

4. **Graceful Degradation Strategy**
   - Meteorology: FAIL if incomplete (scientific validity required)
   - Observations: NaN-fill if missing (optional for forward-only mode)

## Notes

- STAC metadata operations are now explicitly included in diagrams
- Diagrams document variable registry lookups and type classification
- All STAC properties and metadata fields are detailed
- Graceful degradation strategy clearly explained
- Time coordinate authority changes highlighted
- Variable names, units, and dimensions explicitly documented
- All third-party API parameters are detailed

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md): Project coding standards and architecture
- [STAC Implementation](../../src/stac_utils.py): STAC catalog management
- [Variable Registry](../../src/cardamom_variables.py): Central metadata
- [CBF Generation](../../src/cbf_main.py): Pixel-specific file generation
- [Migration Notes](../../matlab-migration/): MATLAB to Python migration

---

*Last Updated: 2025-12-17*
*Total Diagrams: 17 (6 architecture + 8 ECMWF + 1 NOAA + 1 GFED + 1 CBF)*
*CARDAMOM Preprocessor v1.0*
