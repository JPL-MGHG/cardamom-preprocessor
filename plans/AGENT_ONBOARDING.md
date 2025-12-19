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
├── src/                              # Main Python package (STAC-based architecture)
│   ├── STAC-Based Workflow
│   │   ├── stac_cli.py               # CLI for downloaders and CBF generation
│   │   ├── stac_utils.py             # STAC catalog management
│   │   ├── stac_met_loader.py        # Load meteorology from STAC catalogs
│   │   ├── cbf_main.py               # CBF generation orchestration
│   │   ├── cbf_obs_handler.py        # Observational data with NaN-fill
│   │   │
│   │   └── downloaders/              # Modular downloader package
│   │       ├── __init__.py
│   │       ├── base.py               # Abstract base class
│   │       ├── ecmwf_downloader.py   # ERA5 meteorology with STAC metadata
│   │       ├── noaa_downloader.py    # NOAA CO2 with STAC metadata
│   │       └── gfed_downloader.py    # GFED fire with STAC metadata
│   │
│   ├── Scientific Calculations
│   │   ├── atmospheric_science.py    # VPD, humidity, radiation
│   │   ├── carbon_cycle.py           # NEE, GPP, respiration
│   │   ├── scientific_utils.py       # Generic scientific utils
│   │   ├── statistics_utils.py       # Aggregation, interpolation
│   │   ├── units_constants.py        # Physical constants
│   │   └── quality_control.py        # Data validation
│   │
│   ├── Infrastructure Utilities
│   │   ├── netcdf_infrastructure.py  # NetCDF file management
│   │   ├── cardamom_variables.py     # Master variable registry
│   │   ├── time_utils.py             # Time standardization
│   │   ├── validation.py             # Quality assurance framework
│   │   └── data_source_config.py     # Source configurations
│   │
├── matlab-migration/
│   └── erens_cbf_code.py             # CBF generation (end goal)
│
├── plans/                            # Implementation plans and diagrams
├── environment.yml                   # Python environment
├── CLAUDE.md                         # Instructions for Claude Code
└── README.md                         # Project documentation
```

---

## 3. Architecture Overview

### System Architecture Diagram

```
┌──────────────────────────────────────────────────┐
│              STAC-Based CBF Workflow             │
└──────────┬───────────────────────────────────────┘
           │
           ├─► stac_cli.py ──┬─► ECMWFDownloader (src/downloaders/)
           │                 ├─► NOAADownloader
           │                 └─► GFEDDownloader
           │                      ↓
           │                 STAC Catalog (pystac)
           │                 [catalog.json, collections, items]
           │                      ↓
           ├─► stac_met_loader.py
           │   ├─ Discover all meteorological variables
           │   ├─ Validate completeness (FAIL if missing)
           │   └─ Load as unified xarray Dataset
           │                      ↓
           ├─► cbf_obs_handler.py
           │   ├─ Load user-provided obs data
           │   └─ NaN-fill for missing values
           │                      ↓
           └─► cbf_main.py
               ├─ Extract pixel-specific data
               ├─ Set forcing variables (from STAC met)
               ├─ Set observational constraints (from user data)
               └─ Generate CBF files
                    ↓
               Pixel-specific CBF files
               [site{lat}_{lon}_ID{exp}exp0.cbf.nc]
```

### Data Flow Pipeline

```
PHASE 1: DATA ACQUISITION
├─ ECMWFDownloader → ERA5 meteorology → STAC catalog
│  ├─ Monthly variables: T2M_MIN, T2M_MAX, VPD, TOTAL_PREC,
│  │                     SSRD, STRD, SKT, SNOWFALL
│  ├─ Derives VPD from temperature and dewpoint
│  └─ Creates STAC items with spatiotemporal metadata
│
├─ NOAADownloader → Global CO₂ → STAC catalog
│  ├─ Monthly CO2 concentrations (entire time series)
│  └─ Creates STAC item with temporal metadata
│
└─ GFEDDownloader → Burned area & fire → STAC catalog
   ├─ Monthly burned area fraction
   ├─ Fire emissions (if available)
   └─ Creates STAC items with spatiotemporal metadata

PHASE 2: METEOROLOGY DISCOVERY
└─ stac_met_loader.py
   ├─ Query STAC catalog for date range
   ├─ Discover all available meteorological variables
   ├─ Validate completeness: FAIL if any month missing
   ├─ Load NetCDF files into unified xarray Dataset
   ├─ Standardize time coordinates for CBF compatibility
   └─ Return meteorology ready for CBF generation

PHASE 3: OBSERVATIONAL DATA
└─ cbf_obs_handler.py
   ├─ Load user-provided observational NetCDF files:
   │  ├─ AlltsObs05x05.nc (LAI, GPP, ABGB, EWT, SCF)
   │  ├─ CARDAMOM-MAPS_05deg_HWSD_PEQ_iniSOM.nc
   │  └─ CARDAMOM-MAPS_05deg_GFED4_Mean_FIR.nc
   ├─ NaN-fill missing observations (graceful degradation)
   └─ Allow forward-only mode if no observations available

PHASE 4: CBF GENERATION
└─ cbf_main.py
   ├─ Load land-sea fraction mask
   ├─ Identify valid land pixels (>0.5 land fraction)
   ├─ For each pixel:
   │  ├─ Extract meteorology from STAC-loaded data
   │  ├─ Extract observations from user-provided data
   │  ├─ Set forcing variables (VPD, PREC, TMIN, TMAX, etc.)
   │  ├─ Set observation constraints (LAI, GPP, ABGB, etc.)
   │  ├─ Set single-value constraints (SOM, CUE, Mean_FIR)
   │  └─ Set MCMC configuration attributes
   └─ Save pixel-specific CBF NetCDF file

PHASE 5: QUALITY ASSURANCE (Built into each phase)
├─ STAC metadata validation during downloads
├─ Completeness checks in stac_met_loader
├─ Physical range validation (positive values where required)
└─ Unit consistency enforced by cardamom_variables.py
```

---

## 4. Key Modules & Responsibilities

### 4.1 STAC-Based Workflow Modules

**`stac_cli.py`** - Command-Line Interface
- **Purpose**: Main CLI for downloaders and CBF generation
- **Subcommands**:
  - `ecmwf` - Download ERA5 meteorological data
  - `noaa` - Download NOAA CO2 concentrations
  - `gfed` - Download GFED fire data
  - `cbf-generate` - Generate CBF files from STAC catalog
- **Usage**:
```bash
.venv/bin/python -m src.stac_cli ecmwf --variables t2m_min,t2m_max --year 2020 --month 1
.venv/bin/python -m src.stac_cli cbf-generate --stac-api file://./catalog.json --start 2020-01 --end 2020-12
```

**`stac_utils.py`** - STAC Catalog Management
- **Purpose**: Create and manage STAC catalogs for downloaded data
- **Key Functions**:
  - `create_root_catalog()` - Create root STAC catalog
  - `create_stac_collection()` - Create data collection
  - `create_stac_item()` - Add data item with metadata
  - `update_catalog_incremental()` - Incremental updates

**`stac_met_loader.py`** - Meteorology Discovery and Loading
- **Purpose**: Load meteorological data from STAC catalogs with validation
- **Process**:
  1. Query STAC catalog for specified date range
  2. Discover all available meteorological variables
  3. **Validate completeness**: FAIL if any required month is missing
  4. Load NetCDF files into unified xarray Dataset
  5. Return meteorology ready for CBF generation

**`cbf_main.py`** - CBF Generation Orchestration
- **Purpose**: Generate CBF files from STAC meteorology + user observations
- **Process**:
  1. Load meteorology from STAC catalog
  2. Load user-provided observational data
  3. Identify valid land pixels
  4. For each pixel: extract data, generate CBF file
- **Output**: Pixel-specific CBF NetCDF files

**`cbf_obs_handler.py`** - Observational Data Handler
- **Purpose**: Load observational data with graceful NaN-filling
- **Features**:
  - NaN-fills missing observations (graceful degradation)
  - Allows forward-only mode if no observations available
  - Loads: LAI, GPP, ABGB, EWT, SCF, SOM, Mean_FIR

### 4.2 Modular Downloaders (`src/downloaders/`)

**`downloaders/base.py`** - BaseDownloader (Abstract Base Class)
- **Purpose**: Abstract base class defining downloader interface
- **Required Methods**:
  - `download_data(year, month, **kwargs)` - Main download method
  - `_create_stac_item(file_path, year, month)` - Create STAC metadata for output
- **Features**:
  - Consistent interface across all downloaders
  - STAC metadata generation built into base class
  - Automatic catalog creation/update

**`downloaders/ecmwf_downloader.py`** - ECMWFDownloader
- **Purpose**: Download ERA5 meteorological data with STAC metadata
- **Data Variables**: T2M_MIN, T2M_MAX, VPD, TOTAL_PREC, SSRD, STRD, SKT, SNOWFALL
- **Key Features**:
  - Downloads from ECMWF Climate Data Store (CDS)
  - Calculates VPD from temperature and dewpoint
  - Creates STAC items for each output file
  - Supports incremental catalog updates
- **Authentication**: Requires `.cdsapirc` or `ECMWF_CDS_UID`/`ECMWF_CDS_KEY` environment variables
- **Usage**:
```python
downloader = ECMWFDownloader(output_directory='./era5_output')
downloader.download_and_process(
    variables=['t2m_min', 't2m_max', 'vpd'],
    year=2020,
    month=1,
    incremental=True
)
```

**`downloaders/noaa_downloader.py`** - NOAADownloader
- **Purpose**: Download NOAA CO2 concentrations with STAC metadata
- **Data**: Monthly mean CO2 (ppm) from global monitoring network
- **Key Features**:
  - Downloads entire time series (all available data)
  - Converts from NOAA HTTPS format to NetCDF
  - Creates STAC item with temporal metadata
- **URL**: https://gml.noaa.gov/webdata/ccgg/trends/co2/
- **Output**: Spatially-replicated NetCDF (global constant CO2 per month)

**`downloaders/gfed_downloader.py`** - GFEDDownloader
- **Purpose**: Download GFED fire data with STAC metadata
- **Data**: Burned area fraction, dry matter, fire emissions
- **Format**: Converts HDF5 → NetCDF with STAC metadata
- **Resolution**: 0.25° global
- **URL**: https://www.geo.vu.nl/~gwerf/GFED/GFED4/
- **Key Features**:
  - Automatic HDF5 to NetCDF conversion
  - Extracts burned area and fire carbon
  - Creates STAC items for each month

### 4.3 Scientific Calculation Modules

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

### 4.4 Infrastructure Utilities

**`netcdf_infrastructure.py`** - NetCDF File Management
- **Purpose**: NetCDF file creation and utilities
- **Features**:
  - CF-1.8 compliant file generation
  - Compression and chunking
  - Custom fill values
  - Metadata management

**`cardamom_variables.py`** - CARDAMOM_VARIABLE_REGISTRY
- **Purpose**: Single source of truth for all variables
- **Key Functions**:
  - `get_variable_config(variable_name)` - Get complete metadata
  - `get_interpolation_method(variable_name)` - Get spatial interpolation method
  - `get_cbf_name(variable_name)` - Get CBF naming convention
  - `get_variables_by_product_type(product_type)` - Group variables by ERA5 product
- **Features**:
  - Centralized variable definitions (name, units, interpolation, ranges)
  - Variable-specific interpolation methods based on spatial characteristics
  - Automatic unit conversion specifications

**`time_utils.py`**
- **Purpose**: Time coordinate standardization
- **Functions**:
  - `standardize_time_units(dataset)` - Convert to 'days since 2001-01-01'
  - `create_time_bounds(start_date, end_date, freq)` - Generate time bounds
  - `validate_time_coordinates(dataset)` - Validate time dimension

**`validation.py`** - Quality Assurance
- **Purpose**: Data validation framework
- **Key Functions**:
  - `validate_file_structure(filepath)` - Check file structure
  - `validate_variable_units(dataset, variable)` - Unit consistency
  - `validate_coordinate_consistency(dataset)` - Spatial/temporal checks
  - `generate_quality_report(dataset)` - QA report generation

**`data_source_config.py`**
- **Purpose**: Data source configurations
- **Contains**: URLs, authentication details, file patterns for data sources

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

### Workflow 1: Download Meteorology and Generate CBF Files
```bash
# Step 1: Download ERA5 meteorology for multiple months
.venv/bin/python -m src.stac_cli ecmwf \
    --variables t2m_min,t2m_max,vpd,ssrd,strd,total_prec,skt,snowfall \
    --year 2020 --month 1 \
    --output ./era5_output

# Repeat for additional months/years as needed

# Step 2: Generate CBF files from STAC catalog
.venv/bin/python -m src.stac_cli cbf-generate \
    --stac-api file://./era5_output/catalog.json \
    --start 2020-01 --end 2020-12 \
    --output ./cbf_output

# Output: Pixel-specific CBF files (site{lat}_{lon}_ID{exp}exp0.cbf.nc)
```

### Workflow 2: Download All Data Sources
```bash
# Download ERA5 meteorology
.venv/bin/python -m src.stac_cli ecmwf \
    --variables t2m_min,t2m_max,vpd,total_prec,ssrd,strd \
    --year 2020 --month 1-12 \
    --output ./era5_output

# Download NOAA CO2 (entire time series)
.venv/bin/python -m src.stac_cli noaa \
    --output ./noaa_output

# Download GFED fire data
.venv/bin/python -m src.stac_cli gfed \
    --year 2020 --month 1-12 \
    --output ./gfed_output

# All downloaders create STAC catalogs automatically
```

### Workflow 3: Programmatic CBF Generation
```python
from src.cbf_main import generate_cbf_files

# Generate CBF files from STAC catalog + user observations
result = generate_cbf_files(
    stac_source='./era5_output/catalog.json',
    start_date='2020-01',
    end_date='2020-12',
    output_directory='./cbf_output',
    obs_driver_file='input/AlltsObs05x05.nc',
    land_frac_file='input/CARDAMOM-MAPS_05deg_LAND_SEA_FRAC.nc'
)

print(f"Generated {result['successful_pixels']} CBF files")
# Output: Pixel-specific CBF files ready for CARDAMOM modeling
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
from src.atmospheric_science import calculate_vpd

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
- `src/stac_cli.py` - Main CLI interface
- `src/cbf_main.py` - CBF generation orchestration
- `src/cardamom_variables.py` - Variable registry (single source of truth)
- `src/atmospheric_science.py` - Core scientific calculations
- `src/downloaders/ecmwf_downloader.py` - ERA5 meteorology downloader
- `src/downloaders/noaa_downloader.py` - NOAA CO2 downloader
- `src/downloaders/gfed_downloader.py` - GFED fire downloader

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
1. Create downloader class in `src/downloaders/` inheriting from `downloaders.base.BaseDownloader`
2. Implement required methods: `download_data()`, `_create_stac_item()`
3. Add STAC metadata generation for outputs
4. Register new CLI subcommand in `src/stac_cli.py`
5. Update `CARDAMOM_VARIABLE_REGISTRY` in `src/cardamom_variables.py` with new variables

### Task 3: Modify CBF Generation
1. Understand current STAC-based workflow in `src/cbf_main.py`
2. Identify which component needs modification (downloader, loader, or generator)
3. Update relevant module (`stac_met_loader.py`, `cbf_obs_handler.py`, or `cbf_main.py`)
4. Update validation in `quality_control.py` if needed
5. Test with sample data and STAC catalog

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
# Download ERA5 meteorology
.venv/bin/python -m src.stac_cli ecmwf --variables t2m_min,t2m_max,vpd --year 2020 --month 1 --output ./era5_output

# Download NOAA CO2
.venv/bin/python -m src.stac_cli noaa --output ./co2_output

# Download GFED fire data
.venv/bin/python -m src.stac_cli gfed --year 2020 --month 1 --output ./gfed_output

# Generate CBF files from STAC catalog
.venv/bin/python -m src.stac_cli cbf-generate --stac-api file://./era5_output/catalog.json --start 2020-01 --end 2020-12 --output ./cbf_output
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
1. **Validate Early**: Check inputs before expensive operations
2. **Use Standard Logging**: Python logging module for structured logging
3. **STAC Validation**: Leverage STAC metadata for data completeness checks
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
1. Understand existing patterns (STAC downloaders, CBF generation)
2. Identify integration points in STAC workflow (`stac_cli.py`, `cbf_main.py`)
3. Plan architecture (BaseDownloader pattern, STAC metadata)
4. Implement following scientist-friendly standards
5. Add to variable registry if new variables
6. Update STAC catalog structure if needed

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

The **CARDAMOM Preprocessor** is a comprehensive, modular data preprocessing system that:

✅ **Downloads** climate and carbon data from ECMWF (ERA5), NOAA (CO2), GFED (fire)
✅ **Catalogs** data using STAC metadata for discoverability and validation
✅ **Processes** meteorological data with scientific rigor (VPD, radiation calculations)
✅ **Generates** pixel-specific CBF files for CARDAMOM ecosystem modeling
✅ **Validates** data completeness with STAC-based checks
✅ **Maintains** MATLAB compatibility for scientific consistency

**Architecture**: STAC-based workflow with modular downloaders, meteorology discovery, observational data handling, and CBF generation.

**Philosophy**: Scientist-friendly code that prioritizes clarity, scientific accuracy, and accessibility over Python cleverness. Focus on monthly-only workflow (no diurnal processing).

**End Goal**: Generate pixel-specific CBF files (`site{lat}_{lon}_ID{exp}exp0.cbf.nc`) for CARDAMOM carbon cycle analysis using STAC-discovered meteorology and user-provided observational constraints.
