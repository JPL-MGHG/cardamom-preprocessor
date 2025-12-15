# CARDAMOM Preprocessor - Comprehensive Onboarding Summary

## 1. Project Purpose & Context

### What is CARDAMOM?
**CARDAMOM** (Carbon Data-Model Framework) is a Bayesian ecosystem modeling framework that uses data assimilation to constrain carbon cycle parameters. It requires specific meteorological and observational inputs in a precise format called **CBF (CARDAMOM Binary Format)**.

### What Does This Repository Do?
The **cardamom-preprocessor** is a modular Python data orchestration system that:
- Downloads climate and carbon cycle data from multiple sources (ECMWF ERA5, NOAA, GFED4, MODIS)
- Processes raw data into standardized CARDAMOM-compatible formats
- Performs scientific calculations (VPD, radiation conversions, carbon flux downscaling)
- Generates CBF-compatible NetCDF files for ecosystem carbon cycle modeling

### Ultimate Goal
Create input files (`AllMet05x05_LFmasked.nc`, `AlltsObs05x05_LFmasked.nc`, etc.) that can be consumed by `matlab-migration/erens_cbf_code.py` to generate site-level CBF files for CARDAMOM analysis.

---

## 2. Repository Structure

```
cardamom-preprocessor/
├── src/                              # Main Python package (31 modules)
│   ├── Infrastructure & Orchestration
│   │   ├── cardamom_preprocessor.py   # Main CARDAMOMProcessor class
│   │   ├── config_manager.py          # Configuration management
│   │   ├── logging_utils.py           # Logging infrastructure
│   │   └── validation.py              # Quality assurance framework
│   │
│   ├── Data Downloaders
│   │   ├── base_downloader.py         # Abstract base class
│   │   ├── ecmwf_downloader.py        # ERA5 meteorology
│   │   ├── noaa_downloader.py         # CO2 concentrations
│   │   ├── gfed_downloader.py         # Fire emissions
│   │   ├── modis_downloader.py        # Land-sea masks
│   │   ├── downloader_factory.py      # Factory + retry logic
│   │   └── data_source_config.py      # Source configurations
│   │
│   ├── Data Processors
│   │   ├── cbf_met_processor.py       # Meteorological processing
│   │   ├── diurnal_processor.py       # Monthly→hourly downscaling
│   │   ├── gfed_processor.py          # Fire data processing
│   │   ├── met_driver_loader.py       # Met driver loading
│   │   ├── cms_flux_loader.py         # Carbon flux loading
│   │   └── gfed_diurnal_loader.py     # GFED diurnal patterns
│   │
│   ├── Scientific Calculations
│   │   ├── atmospheric_science.py     # VPD, humidity, radiation
│   │   ├── carbon_cycle.py            # NEE, GPP, respiration
│   │   ├── scientific_utils.py        # Generic scientific utils
│   │   ├── statistics_utils.py        # Aggregation, interpolation
│   │   ├── units_constants.py         # Physical constants
│   │   └── quality_control.py         # Data validation
│   │
│   ├── Infrastructure Utilities
│   │   ├── netcdf_infrastructure.py   # NetCDF file management
│   │   ├── coordinate_systems.py      # Geographic grids
│   │   ├── cardamom_variables.py      # Master variable registry
│   │   ├── time_utils.py              # Time standardization
│   │   └── cbf_cli.py                 # Command-line interface
│   │
├── matlab-migration/
│   └── erens_cbf_code.py             # CBF generation (end goal)
│
├── plans/                            # Workflow diagrams
├── environment.yml                   # Python environment
├── CLAUDE.md                         # Instructions for Claude Code
└── README.md                         # Project documentation
```

---

## 3. Architecture Overview

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     CARDAMOMProcessor                            │
│                (Main Orchestration Engine)                       │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ├─► CardamomConfig ──────► DataSourceConfig
           │   (YAML + env vars)       (Source definitions)
           │
           ├─► DownloaderFactory ──┬─► ECMWFDownloader
           │   (Factory pattern)    ├─► NOAADownloader
           │                        ├─► GFEDDownloader
           │                        └─► MODISDownloader
           │
           ├─► DiurnalProcessor ────┬─► DiurnalCalculator
           │   (Flux downscaling)   └─► DiurnalOutputWriters
           │
           ├─► CBFMetProcessor ──────► atmospheric_science.py
           │   (Met processing)       └─► carbon_cycle.py
           │
           ├─► GFEDProcessor ─────────► Fire emissions
           │
           ├─► CARDAMOMNetCDFWriter ──► NetCDF outputs
           │   (File generation)
           │
           ├─► CoordinateGrid ────────► Spatial grids
           │   (Geographic systems)
           │
           ├─► QualityAssurance ───────► Validation
           │   (QC framework)
           │
           └─► ProcessingLogger ───────► Structured logging
```

### Data Flow Pipeline

```
PHASE 1: DATA ACQUISITION
├─ ECMWFDownloader → ERA5 meteorology (hourly/monthly)
├─ NOAADownloader → Global CO2 concentrations
├─ GFEDDownloader → Burned area & fire emissions
└─ MODISDownloader → Land-sea masks

PHASE 2: SCIENTIFIC CALCULATIONS
├─ atmospheric_science.py
│  ├─ saturation_pressure_water_matlab()
│  ├─ calculate_vapor_pressure_deficit_matlab()
│  └─ calculate_humidity_index()
├─ carbon_cycle.py
│  ├─ calculate_net_ecosystem_exchange()
│  └─ validate_carbon_flux_mass_balance()
└─ scientific_utils.py
   ├─ convert_precipitation_units()
   └─ convert_radiation_units()

PHASE 3: DATA PROCESSING
├─ CBFMetProcessor: ERA5 → CBF meteorology
│  ├─ Unit conversions (K→C, m→mm, J/m²→W/m²)
│  ├─ Variable renaming (ERA5 names → CBF names)
│  ├─ Spatial regridding (0.25° → 0.5°)
│  └─ Quality control (range validation)
│
├─ DiurnalProcessor: Monthly → Hourly downscaling
│  ├─ GPP scaled by solar radiation
│  ├─ Respiration scaled by temperature
│  └─ Fire emissions with diurnal timing
│
└─ GFEDProcessor: GFED → CBF fire format
   ├─ Variable extraction (burned_area, DM)
   ├─ Unit conversion (fraction → gC/m²/day)
   └─ Gap-filling for missing data

PHASE 4: QUALITY ASSURANCE
├─ Physical plausibility checks
├─ Unit consistency validation
├─ Mass balance verification
└─ Data completeness reports

PHASE 5: OUTPUT GENERATION
├─ AllMet05x05_LFmasked.nc (meteorological drivers)
├─ AlltsObs05x05_LFmasked.nc (observational constraints)
├─ Diurnal flux files (hourly carbon fluxes)
└─ Processing logs & QA reports
```

---

## 4. Key Modules & Responsibilities

### 4.1 Core Orchestration

**`cardamom_preprocessor.py` - CARDAMOMProcessor**
- **Purpose**: Main orchestration engine coordinating all workflows
- **Key Methods**:
  - `process_global_monthly()` - Global monthly meteorology
  - `process_conus_diurnal()` - CONUS hourly flux downscaling
  - `process_batch()` - Batch processing with checkpointing
  - `validate_inputs()` - Pre-processing validation
- **Dependencies**: All other modules

**`config_manager.py` - CardamomConfig**
- **Purpose**: Centralized configuration management
- **Features**:
  - YAML-based configuration files
  - Environment variable overrides
  - Parameter validation
  - Default configurations
- **Key Methods**:
  - `get_workflow_config()`
  - `get_downloader_config()`
  - `get_quality_control_config()`

### 4.2 Data Downloaders (Factory Pattern)

**`base_downloader.py` - BaseDownloader (ABC)**
- **Purpose**: Abstract base class defining downloader interface
- **Required Methods**:
  - `download_data()` - Main download method
  - `check_existing_files()` - Skip completed downloads
  - `validate_downloaded_data()` - Post-download validation
  - `get_download_status()` - Progress tracking

**`ecmwf_downloader.py` - ECMWFDownloader**
- **Purpose**: Download ERA5 reanalysis data from ECMWF CDS
- **Data Variables**: 2m_temperature, 2m_dewpoint_temperature, total_precipitation, surface_solar_radiation_downwards, surface_thermal_radiation_downwards, skin_temperature, snowfall
- **Key Features**:
  - Hourly and monthly aggregation
  - Configurable spatial/temporal filtering
  - Job monitoring for long-running requests
  - Automatic retry on CDS queue failures
- **Authentication**: Requires `.cdsapirc` or `ECMWF_CDS_UID`/`ECMWF_CDS_KEY`

**`noaa_downloader.py` - NOAADownloader**
- **Purpose**: Download NOAA/ESRL global CO2 concentrations
- **Data**: Monthly mean CO2 (ppm) from Mauna Loa + global stations
- **Output Formats**:
  - Spatially replicated NetCDF (global constant)
  - Time series NetCDF
- **URL**: https://gml.noaa.gov/webdata/ccgg/trends/co2/

**`gfed_downloader.py` - GFEDDownloader**
- **Purpose**: Download GFED4.1s fire emissions data
- **Data**: Burned area, dry matter combustion, carbon emissions
- **Format**: HDF5 files (yearly)
- **Resolution**: 0.25° global
- **URL**: https://www.geo.vu.nl/~gwerf/GFED/GFED4/

**`modis_downloader.py` - MODISDownloader**
- **Purpose**: Generate land-sea fraction masks
- **Data Source**: MODIS-based land cover
- **Output**: Binary mask for filtering ocean pixels

**`downloader_factory.py` - DownloaderFactory + RetryManager**
- **Purpose**: Factory pattern + robust error handling
- **DownloaderFactory Methods**:
  - `create_downloader(source_name)` - Instantiate specific downloader
  - `create_all_downloaders()` - Create all configured downloaders
  - `check_downloader_dependencies()` - Validate prerequisites
- **RetryManager Methods**:
  - `download_with_retry()` - Exponential backoff with jitter
  - `categorize_error()` - Classify error types (network, auth, server)
  - `get_retry_statistics()` - Track retry performance

### 4.3 Data Processors

**`cbf_met_processor.py` - CBFMetProcessor**
- **Purpose**: Process ERA5 meteorology into CBF format
- **Transformations**:
  - Temperature: K → K (preserved), split into TMIN/TMAX
  - Precipitation: m → mm/day
  - Radiation: J/m² → W/m²
  - Snowfall: m water equiv → mm/day
  - VPD: Calculate from temperature + dewpoint
- **Key Methods**:
  - `process_single_file()` - Single variable processing
  - `process_batch()` - Batch processing
  - `regrid_to_target()` - Spatial regridding
  - `apply_unit_conversions()` - Unit standardization

**`diurnal_processor.py` - DiurnalProcessor**
- **Purpose**: Downscale monthly carbon fluxes to hourly resolution
- **Algorithm**:
  - GPP: Scaled by hourly solar radiation pattern
  - Respiration: Scaled by hourly temperature curve
  - Fire: Applied according to GFED diurnal timing
- **Inputs**:
  - Monthly GPP, REC, FIR (from CMS-Flux or models)
  - Hourly meteorology (SSRD, temperature)
  - GFED diurnal fire patterns
- **Output**: Hourly flux time series

**`gfed_processor.py` - GFEDProcessor**
- **Purpose**: Process GFED HDF5 files into CARDAMOM format
- **Key Methods**:
  - `process_gfed_monthly()` - Extract monthly data
  - `gap_fill_burned_area()` - Fill missing values
  - `to_netcdf_files()` - Create NetCDF outputs
  - `to_cardamom_format()` - Convert to CBF variable names

### 4.4 Scientific Calculation Modules

**`atmospheric_science.py`**
- **Purpose**: Atmospheric physics calculations
- **Functions**:
  - `saturation_pressure_water_matlab(temperature_kelvin)` - Magnus formula
  - `calculate_vapor_pressure_deficit_matlab(tmax_k, dewpoint_k)` - VPD from T and Td
  - `calculate_humidity_index(temperature, dewpoint)` - Humidity metrics
  - `calculate_photosynthetically_active_radiation(solar_rad)` - PAR from SSRD
- **Notes**: MATLAB-compatible implementations for consistency

**`carbon_cycle.py`**
- **Purpose**: Carbon cycle calculations
- **Functions**:
  - `calculate_net_ecosystem_exchange(gpp, respiration, fire)` - NEE = REC + FIR - GPP
  - `validate_carbon_flux_mass_balance(gpp, rec, nee)` - Mass balance checks
  - `calculate_carbon_use_efficiency(gpp, npp)` - CUE calculations
- **Sign Convention**: Positive = source to atmosphere, Negative = sink from atmosphere

**`scientific_utils.py`**
- **Purpose**: Generic scientific utilities
- **Functions**:
  - `convert_precipitation_units(precip_m_s, to_unit='mm_day')`
  - `convert_radiation_units(radiation_j_m2, to_unit='w_m2')`
  - `convert_temperature_units(temp_k, to_unit='celsius')`

**`statistics_utils.py`**
- **Purpose**: Statistical operations
- **Functions**:
  - `aggregate_hourly_to_monthly(hourly_data)`
  - `interpolate_spatial(data, target_grid)`
  - `calculate_temporal_statistics(data, stat='mean')`

**`units_constants.py`**
- **Purpose**: Physical constants and conversion factors
- **Constants**:
  - `SECONDS_PER_DAY = 86400`
  - `KELVIN_TO_CELSIUS = 273.15`
  - `MOLECULAR_WEIGHT_CO2 = 44.01 # g/mol`
  - `MOLECULAR_WEIGHT_C = 12.01 # g/mol`

**`quality_control.py`**
- **Purpose**: Data quality validation
- **Functions**:
  - `validate_temperature_range(temp_k)` - Check physical plausibility
  - `validate_vpd_calculation(vpd_hpa)` - Ensure VPD >= 0
  - `validate_carbon_flux_range(flux_gc_m2_day)` - Check flux ranges
  - `create_quality_report(dataset)` - Generate QA reports

### 4.5 Infrastructure Utilities

**`netcdf_infrastructure.py` - CARDAMOMNetCDFWriter**
- **Purpose**: NetCDF file creation and management
- **Key Methods**:
  - `write_2d_dataset()` - Spatial-only data (lat, lon)
  - `write_3d_dataset()` - Spatiotemporal data (time, lat, lon)
  - `write_cbf_file()` - CBF-specific format
  - `add_metadata()` - CF-compliant metadata
- **Features**:
  - CF-1.8 compliant
  - Compression and chunking
  - Custom fill values

**`coordinate_systems.py` - CoordinateGrid + StandardGrids**
- **Purpose**: Geographic grid management
- **StandardGrids**:
  - `GLOBAL_05DEG`: Global 0.5° (360×720 grid)
  - `GLOBAL_025DEG`: Global 0.25° (720×1440 grid)
  - `CONUS`: CONUS-specific grid (60°N-20°N, 130°W-50°W)
  - `GEOSCHEM_4x5`: GeosChem 4°×5° grid
- **CoordinateGrid Methods**:
  - `get_indices_for_region(lat_range, lon_range)`
  - `get_regional_subset(data, region)`
  - `get_grid_info()` - Grid metadata

**`cardamom_variables.py` - CARDAMOM_VARIABLE_REGISTRY**
- **Purpose**: Single source of truth for all variables
- **Structure**:
```python
CARDAMOM_VARIABLE_REGISTRY = {
    '2m_temperature': {
        'source': 'era5',
        'alternative_names': ['t2m', 'T2M'],
        'cbf_names': ['TMIN', 'TMAX'],
        'units': {'source': 'K', 'cbf': 'K'},
        'interpolation_method': 'linear',
        'essential': True,
        'data_type': 'forcing',
        'temporal_resolution': 'hourly',
        'description': '2-meter air temperature',
        ...
    },
    # ... ~30+ variables
}
```

**`time_utils.py`**
- **Purpose**: Time coordinate standardization
- **Functions**:
  - `standardize_time_units(dataset)` - Convert to 'days since 2001-01-01'
  - `create_time_bounds(start_date, end_date, freq)`
  - `validate_time_coordinates(dataset)`

**`logging_utils.py` - ProcessingLogger**
- **Purpose**: Structured logging infrastructure
- **Features**:
  - Context managers for processing stages
  - Progress tracking
  - Error recovery logging
  - QA report generation
- **Usage**:
```python
with ProcessingLogger("Processing ERA5 data") as logger:
    logger.info("Starting download")
    # ... processing ...
    logger.success("Download complete")
```

**`validation.py` - QualityAssurance**
- **Purpose**: Comprehensive validation framework
- **Key Methods**:
  - `validate_file_structure(filepath)`
  - `validate_variable_units(dataset, variable)`
  - `validate_coordinate_consistency(dataset)`
  - `generate_quality_report(dataset)`

---

## 5. Data Sources & Variables

### 5.1 ECMWF ERA5 Variables

| Variable Name | CBF Name | Units | Frequency | Description |
|--------------|----------|-------|-----------|-------------|
| `2m_temperature` | `TMIN`, `TMAX` | K | Hourly/Monthly | 2-meter air temperature |
| `2m_dewpoint_temperature` | `DEWPOINT` | K | Hourly/Monthly | 2-meter dewpoint temperature |
| `total_precipitation` | `TOTAL_PREC` | mm/day | Hourly/Monthly | Total precipitation |
| `surface_solar_radiation_downwards` | `SSRD` | W/m² | Hourly/Monthly | Downward solar radiation |
| `surface_thermal_radiation_downwards` | `STRD` | W/m² | Hourly/Monthly | Downward thermal radiation |
| `skin_temperature` | `SKT` | K | Hourly | Surface skin temperature |
| `snowfall` | `SNOWFALL` | mm/day | Monthly | Snowfall water equivalent |

**Derived Variables**:
- `VPD` (Vapor Pressure Deficit) - Calculated from TMAX and dewpoint

### 5.2 NOAA CO2 Data

| Variable | Units | Frequency | Source |
|----------|-------|-----------|--------|
| `CO2` | ppm | Monthly | NOAA/ESRL Global Monitoring Laboratory |

### 5.3 GFED4.1s Fire Data

| Variable | CBF Name | Units | Resolution |
|----------|----------|-------|------------|
| `burned_area` | `BURNED_AREA` | fraction | 0.25° monthly |
| `DM` (Dry Matter) | `FIR` | gC/m²/day | 0.25° monthly |
| `C` (Carbon emissions) | - | gC/m²/month | 0.25° monthly |

### 5.4 MODIS Data

| Variable | Units | Resolution |
|----------|-------|------------|
| `land_sea_mask` | binary (0/1) | 0.5° |

---

## 6. Predefined Workflows

### Workflow 1: Global Monthly Processing
```python
from cardamom_preprocessor import CARDAMOMProcessor

processor = CARDAMOMProcessor(config_file='config.yaml')

# Download and process global monthly meteorology
processor.process_global_monthly(
    years=range(2001, 2021),
    months=range(1, 13),
    variables=['2m_temperature', 'total_precipitation',
               'surface_solar_radiation_downwards', 'snowfall']
)

# Output: AllMet05x05_LFmasked.nc
```

### Workflow 2: CONUS Diurnal Flux Downscaling
```python
# Downscale monthly fluxes to hourly resolution
processor.process_conus_diurnal(
    years=range(2015, 2021),
    months=range(1, 13),
    flux_variables=['GPP', 'REC', 'FIR']
)

# Output: Hourly flux files for carbon analysis
```

### Workflow 3: Fire Emissions Processing
```python
from gfed_processor import GFEDProcessor

gfed_proc = GFEDProcessor()
gfed_proc.process_gfed_monthly(
    years=range(2001, 2021),
    output_format='cbf'
)

# Output: CARDAMOM-format fire emissions
```

---

## 7. Code Conventions & Patterns

### 7.1 Scientist-Friendly Coding Standards

**From CLAUDE.md**: This project prioritizes code readability for scientists who may not be proficient in Python.

**Key Principles**:
1. **Scientific Clarity Over Python Cleverness**
   - Avoid complex comprehensions, lambdas, and Python idioms
   - Prefer explicit operations over implicit ones

2. **Self-Documenting Code**
   - Variable names reflect scientific meaning + units
   - Function names describe scientific operations
   - Code structure mirrors scientific workflow

3. **Variable Naming Example**:
```python
# GOOD
temperature_celsius = era5_data['2m_temperature'] - 273.15
vapor_pressure_deficit_hpa = calculate_vpd(temp_max_k, dewpoint_k)

# AVOID
t = data['temp'] - 273.15
vpd = calc_vpd(tmax, td)
```

4. **Function Documentation**:
```python
def calculate_vapor_pressure_deficit(temperature_max_kelvin, dewpoint_temperature_kelvin):
    """
    Calculate Vapor Pressure Deficit from temperature and dewpoint.

    Scientific Background:
    VPD represents the atmospheric moisture demand and is crucial for
    understanding plant water stress and photosynthesis rates.

    VPD = e_sat(T_max) - e_sat(T_dewpoint)
    where e_sat is saturation vapor pressure (Tetens equation).

    Args:
        temperature_max_kelvin (float): Daily maximum temperature in K
            Typical range: 250-320 K (-23 to 47°C)
        dewpoint_temperature_kelvin (float): Dewpoint temperature in K
            Typical range: 230-300 K (-43 to 27°C)

    Returns:
        float: Vapor pressure deficit in hectopascals (hPa)
            Typical range: 0-60 hPa

    References:
        Tetens, O. (1930). Über einige meteorologische Begriffe.
        Zeitschrift für Geophysik, 6, 297-309.
    """
    # Step 1: Calculate saturation pressure at T_max
    temp_max_celsius = temperature_max_kelvin - 273.15
    sat_pressure_tmax = 6.1078 * np.exp(
        17.27 * temp_max_celsius / (temp_max_celsius + 237.3)
    )

    # Step 2: Calculate saturation pressure at dewpoint
    dewpoint_celsius = dewpoint_temperature_kelvin - 273.15
    sat_pressure_dewpoint = 6.1078 * np.exp(
        17.27 * dewpoint_celsius / (dewpoint_celsius + 237.3)
    )

    # Step 3: VPD = difference
    vpd = sat_pressure_tmax - sat_pressure_dewpoint

    # Step 4: Validate
    if np.any(vpd < 0):
        raise ValueError("VPD cannot be negative")

    return vpd
```

### 7.2 Design Patterns Used

1. **Abstract Base Class Pattern** (`BaseDownloader`)
   - All downloaders inherit from `BaseDownloader`
   - Enforces consistent interface

2. **Factory Pattern** (`DownloaderFactory`)
   - Centralized downloader creation
   - Dependency injection

3. **Strategy Pattern** (Processing strategies)
   - Different processing strategies for different data types
   - Pluggable algorithms

4. **Context Manager Pattern** (Logging)
   - `with ProcessingLogger()` for structured logging
   - Automatic resource cleanup

### 7.3 Import Conventions

**IMPORTANT**: Do not use relative imports - this is a Python package
```python
# GOOD
from cardamom_preprocessor.atmospheric_science import calculate_vpd

# AVOID
from .atmospheric_science import calculate_vpd
```

### 7.4 Python Environment

```bash
# Use .venv for Python commands
.venv/bin/python script.py
.venv/bin/pip install package

# Or activate environment
conda activate cardamom-ecmwf-downloader
```

---

## 8. Integration with MATLAB Migration

### 8.1 MATLAB Compatibility

**Key Requirement**: Many scientific functions are MATLAB-compatible implementations to ensure consistency with the original CARDAMOM MATLAB codebase.

**MATLAB Source Location**:
- `/Users/shah/Desktop/Development/ghg/CARDAMOM/MATLAB/prototypes`

**Example MATLAB-Compatible Function**:
```python
def saturation_pressure_water_matlab(temperature_kelvin):
    """
    MATLAB-compatible saturation vapor pressure calculation.

    Replicates MATLAB implementation for consistency with CARDAMOM framework.
    """
    temperature_celsius = temperature_kelvin - 273.15

    # Magnus formula (same as MATLAB version)
    saturation_pressure_hpa = 6.1078 * np.exp(
        17.27 * temperature_celsius / (temperature_celsius + 237.3)
    )

    return saturation_pressure_hpa
```

### 8.2 CBF Generation Pipeline

**End Goal**: Create inputs for `matlab-migration/erens_cbf_code.py`

**Required Files**:
1. `AllMet05x05_LFmasked.nc` - Meteorological drivers
2. `AlltsObs05x05_LFmasked.nc` - Observational constraints
3. `CARDAMOM-MAPS_05deg_LAND_SEA_FRAC.nc` - Land-sea mask
4. `CARDAMOM-MAPS_05deg_HWSD_PEQ_iniSOM.nc` - Initial soil carbon
5. `CARDAMOM-MAPS_05deg_GFED4_Mean_FIR.nc` - Mean fire emissions

**CBF File Structure** (from `erens_cbf_code.py`):
```python
# Geographic bounds for CONUS+
LAT_RANGE = np.arange(229, 301)  # ~20°N to 60°N
LON_RANGE = np.arange(110, 230)  # ~130°W to 50°W

# Output filename format
filename = f"site{lat_deg}_{lat_dec}N{lon_deg}_{lon_dec}W_ID{exp_id}exp0.cbf.nc"
# Example: site35_25N105_75W_ID001exp0.cbf.nc
```

**Variable Mapping** (ERA5 → CBF):
```python
MET_RENAME_MAP = {
    'CO2_2': 'CO2',
    'PREC': 'TOTAL_PREC',
    'BURN_2': 'BURNED_AREA',
    'TMIN': 'T2M_MIN',
    'TMAX': 'T2M_MAX',
    'SNOW': 'SNOWFALL'
}
```

---

## 9. Important File Locations

### 9.1 Key Configuration Files
- `environment.yml` - Python environment definition
- `CLAUDE.md` - Instructions for Claude Code (coding standards, workflows)
- `config.yaml` (if exists) - Runtime configuration
- `.cdsapirc` - ECMWF API credentials (in home directory)

### 9.2 Critical Source Files
- `src/cardamom_preprocessor.py` - Main orchestrator
- `src/cardamom_variables.py` - Variable registry (single source of truth)
- `src/atmospheric_science.py` - Core scientific calculations
- `src/ecmwf_downloader.py` - Primary data source
- `src/config_manager.py` - Configuration system

### 9.3 Reference Files
- `matlab-migration/erens_cbf_code.py` - CBF generation reference
- `/Users/shah/Desktop/Development/ghg/CARDAMOM/MATLAB/prototypes/` - Original MATLAB code

### 9.4 Documentation
- `README.md` - Project overview
- `plans/diagrams/` - Workflow diagrams
- Memory files in Serena:
  - `project_overview`
  - `codebase_structure`
  - `tech_stack`
  - `code_style_conventions`
  - `phase8_implementation_complete`

---

## 10. Testing & Validation

### 10.1 Testing Philosophy

**From CLAUDE.md**: "keep testing simple not too extensive"

### 10.2 Test Files
- `test_batch_download.py` - Batch download testing
- `test_consolidation.py` - Data consolidation testing

### 10.3 Running Tests
```bash
# Run tests with pytest
.venv/bin/python -m pytest tests/ -v

# Install package in development mode
.venv/bin/pip install -e .
```

### 10.4 Validation Approach
- Physical plausibility checks (temperature ranges, flux bounds)
- Unit consistency validation
- Mass balance verification for carbon fluxes
- Data completeness assessment

---

## 11. Common Tasks for New Agents

### Task 1: Add a New Variable
1. Add to `CARDAMOM_VARIABLE_REGISTRY` in `cardamom_variables.py`
2. Update relevant downloader (e.g., `ecmwf_downloader.py`)
3. Add unit conversion if needed in `scientific_utils.py`
4. Add validation in `quality_control.py`

### Task 2: Add a New Data Source
1. Create downloader class inheriting from `BaseDownloader`
2. Implement required methods: `download_data()`, `validate_downloaded_data()`
3. Add to `DownloaderFactory.create_downloader()`
4. Add configuration in `data_source_config.py`
5. Update `CARDAMOM_VARIABLE_REGISTRY` with new variables

### Task 3: Modify Processing Pipeline
1. Understand current workflow in `cardamom_preprocessor.py`
2. Identify relevant processor (CBFMetProcessor, DiurnalProcessor, etc.)
3. Modify processing logic
4. Update validation in `quality_control.py`
5. Test with sample data

### Task 4: Fix Scientific Calculation
1. Find relevant function in `atmospheric_science.py` or `carbon_cycle.py`
2. Check MATLAB reference in `/Users/shah/Desktop/Development/ghg/CARDAMOM/MATLAB/prototypes/`
3. Ensure MATLAB compatibility if modifying existing function
4. Update docstring with scientific justification
5. Add validation checks

---

## 12. Quick Reference Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate cardamom-ecmwf-downloader

# Install additional packages
.venv/bin/pip install cdsapi maap-py pystac pystac-client boto3
```

### Running the Preprocessor
```bash
# Process global monthly data
.venv/bin/python src/cardamom_preprocessor.py --workflow global_monthly --years 2020 --months 1-12

# Process CONUS diurnal fluxes
.venv/bin/python src/cardamom_preprocessor.py --workflow conus_diurnal --years 2015-2020
```

### Testing
```bash
# Run specific test
.venv/bin/python test_batch_download.py

# Run all tests
.venv/bin/python -m pytest tests/ -v

# Install in development mode
.venv/bin/pip install -e .
```

### Useful Inspection Commands
```bash
# Check NetCDF structure
ncdump -h output_file.nc

# List variables
ncdump -c output_file.nc

# Examine specific variable
.venv/bin/python -c "import xarray as xr; ds = xr.open_dataset('file.nc'); print(ds['variable'])"
```

---

## 13. Development Workflow

### Standard Development Process
1. **Understand**: Read relevant memories, explore codebase with Serena tools
2. **Plan**: Use planning mode if task is non-trivial
3. **Implement**: Follow scientist-friendly coding standards
4. **Validate**: Add quality checks, test with sample data
5. **Document**: Update docstrings, add to memories if significant

### When Modifying Scientific Code
1. **Check MATLAB Reference**: Ensure consistency with original implementation
2. **Preserve Function Signatures**: Maintain backward compatibility
3. **Document Scientific Basis**: Add references to literature
4. **Validate Results**: Compare with expected physical ranges
5. **Update Variable Registry**: If adding/modifying variables

### Error Handling Best Practices
1. **Use RetryManager**: For network/API operations
2. **Validate Early**: Check inputs before expensive operations
3. **Log Context**: Use ProcessingLogger for structured logging
4. **Provide Clear Messages**: Scientist-friendly error messages with context

---

## 14. Key Architectural Decisions

### Why This Architecture?

1. **Modularity**: Each data source has its own downloader
2. **Extensibility**: Easy to add new data sources or processing steps
3. **Robustness**: Retry logic, validation, checkpointing
4. **Scientific Rigor**: MATLAB-compatible calculations, extensive validation
5. **Scientist-Friendly**: Readable code for domain scientists

### Design Tradeoffs

1. **Verbosity vs. Clarity**: Chose verbosity for scientist accessibility
2. **Performance vs. Readability**: Chose readability over optimization
3. **Flexibility vs. Simplicity**: Chose flexibility with factory/strategy patterns
4. **Completeness vs. Maintainability**: Comprehensive variable registry over scattered definitions

---

## 15. Next Steps for Onboarding

### For Implementation Tasks
1. Read this document completely
2. Read relevant memories: `project_overview`, `codebase_structure`, `code_style_conventions`
3. Explore specific modules with Serena tools (`get_symbols_overview`, `find_symbol`)
4. Check `CLAUDE.md` for coding standards
5. Review `cardamom_variables.py` for variable definitions
6. Understand data flow from downloader → processor → output

### For Bug Fixes
1. Identify affected module
2. Read module with `get_symbols_overview`
3. Find relevant function with `find_symbol`
4. Check MATLAB reference if scientific calculation
5. Validate fix against physical constraints
6. Update quality control if needed

### For New Features
1. Understand existing patterns (downloaders, processors)
2. Identify integration points in `cardamom_preprocessor.py`
3. Plan architecture (factory pattern, inheritance)
4. Implement following scientist-friendly standards
5. Add to variable registry if new variables
6. Update configuration system

---

## 16. Contact & Resources

### Key Resources
- **CARDAMOM MATLAB**: `/Users/shah/Desktop/Development/ghg/CARDAMOM/MATLAB/`
- **ERA5 Documentation**: https://confluence.ecmwf.int/display/CKB/ERA5
- **GFED Documentation**: https://www.globalfiredata.org/
- **NOAA CO2 Data**: https://gml.noaa.gov/ccgg/trends/

### Project Location
- **Repository**: `/Users/shah/Desktop/Development/ghg/cardamom-preprocessor/`
- **Python Package**: `src/`
- **MATLAB Migration**: `matlab-migration/`

---

## Summary

The **CARDAMOM Preprocessor** is a comprehensive, modular data orchestration system that:

✅ **Downloads** climate and carbon data from ECMWF, NOAA, GFED, MODIS
✅ **Processes** raw data with scientific rigor (VPD, radiation, carbon fluxes)
✅ **Downscales** monthly fluxes to diurnal (hourly) resolution
✅ **Generates** CBF-compatible NetCDF files for CARDAMOM ecosystem modeling
✅ **Validates** data quality with extensive QC framework
✅ **Maintains** MATLAB compatibility for scientific consistency

**Architecture**: 31 Python modules organized into downloaders, processors, scientific calculators, and infrastructure utilities, all orchestrated by `CARDAMOMProcessor`.

**Philosophy**: Scientist-friendly code that prioritizes clarity, scientific accuracy, and accessibility over Python cleverness.

**End Goal**: Create `AllMet05x05_LFmasked.nc` and related files for `erens_cbf_code.py` to generate site-level CBF files for CARDAMOM carbon cycle analysis.
