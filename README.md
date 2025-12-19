# CARDAMOM Preprocessor

A comprehensive Python preprocessing system for the CARDAMOM carbon cycle modeling framework. This repository provides modular tools for downloading, processing, and preparing atmospheric and terrestrial data required for CARDAMOM carbon cycle analysis.

## 🌍 Overview

The CARDAMOM Preprocessor transforms raw Earth observation data into CARDAMOM-ready input files through a scientifically rigorous, modular Python pipeline. Originally developed as a MATLAB system, this Python implementation provides enhanced capabilities while maintaining complete scientific equivalence.

### Key Features

- **📡 Multi-source data integration**: ERA5 meteorology, NOAA CO₂, GFED fire emissions, MODIS land cover
- **🔬 Scientific accuracy**: MATLAB-equivalent calculations with complete function references
- **🛠️ Modular design**: Independent components that can be used together or separately
- **☁️ Cloud-ready**: NASA MAAP platform integration for scalable processing
- **📊 Quality control**: Comprehensive data validation and quality reporting
- **🧪 Scientist-friendly**: Clear documentation with physical units and typical value ranges

## 🏗️ Architecture

The system is organized into modular components across 8 development phases:

### **Phase 1: STAC-Based Workflow** ✅
- STAC catalog management for data discovery and metadata
- Modular downloader package (`src/downloaders/`) with pluggable architecture
- Meteorology loading from STAC catalogs with completeness validation
- CBF generation orchestration using STAC + user-provided data
- Graceful degradation for missing observational data (NaN-fill)
- Time coordinate standardization for CBF compatibility

### **Phase 2: Data Downloaders Package** ✅
- **Modular downloaders** (`src/downloaders/`)
  - `ECMWFDownloader`: ERA5 meteorological data via CDS API
  - `NOAADownloader`: CO₂ concentrations (migrated to HTTPS from deprecated FTP)
  - `GFEDDownloader`: Fire emissions and burned area data
  - `BaseDownloader`: Abstract base class for consistent interface
- **STAC Integration**: Each downloader generates STAC metadata for outputs
- **Incremental Updates**: Add new data without rebuilding catalogs

### **Phase 3: CBF File Generation** ✅
- STAC-based meteorology discovery and loading
- User-provided observational constraints with NaN-filling
- Pixel-specific CBF file generation
- Compatible with `matlab-migration/erens_cbf_code.py`
- Variable-aware processing using centralized configuration
- Automated unit conversions and spatial interpolation

### **Phase 4: Scientific Functions Library** ✅
- Atmospheric science calculations (VPD, humidity, radiation)
- Statistical utilities (temporal aggregation, spatial interpolation)
- Physical constants and unit conversions
- Carbon cycle modeling functions
- Enhanced data quality control
- **Centralized variable configuration system** (`cardamom_variables.py`)
  - Single source of truth for all variable metadata
  - Variable-specific interpolation methods based on spatial characteristics
  - Automated unit conversions and physical range validation

## 🚀 Quick Start

### Prerequisites

```bash
# Create conda environment
conda env create -f environment.yml
conda activate cardamom-ecmwf-downloader

# Install package in development mode
pip install -e .

# Or using virtual environment directly
.venv/bin/pip install -e .
```

### Basic Usage

```python
# Import STAC-based workflow components
from src.stac_cli import main as stac_cli_main
from src.cbf_main import generate_cbf_files
from src.stac_met_loader import load_met_data_from_stac
from src.cbf_obs_handler import load_observational_data_with_nan_fill

# Import modular downloaders
from src.downloaders.ecmwf_downloader import ECMWFDownloader
from src.downloaders.noaa_downloader import NOAADownloader
from src.downloaders.gfed_downloader import GFEDDownloader

# Import scientific utilities
from src.atmospheric_science import (
    saturation_pressure_water_matlab,
    calculate_vapor_pressure_deficit_matlab
)
from src.units_constants import PhysicalConstants
from src.cardamom_variables import (
    get_variable_config,
    get_interpolation_method,
    get_cbf_name,
    CARDAMOM_VARIABLE_REGISTRY
)

# Example 1: Download meteorology and create STAC catalog
downloader = ECMWFDownloader(output_directory='./era5_output')
downloader.download_data(
    variables=['t2m_min', 't2m_max', 'vpd', 'ssrd'],
    year=2020,
    month=1
)

# Example 2: Generate CBF files from STAC catalog + user obs data
result = generate_cbf_files(
    stac_source='./era5_output/catalog.json',  # STAC catalog from downloader
    start_date='2020-01',
    end_date='2020-12',
    output_directory='./cbf_output',
    obs_driver_file='input/AlltsObs05x05.nc',  # User-provided observations
    land_frac_file='input/CARDAMOM-MAPS_05deg_LAND_SEA_FRAC.nc'
)
print(f"Generated {result['successful_pixels']} CBF files")

# Example 3: Load meteorology from STAC catalog
met_data = load_met_data_from_stac(
    stac_source='./era5_output/catalog.json',
    start_date='2020-01',
    end_date='2020-12'
)
print(f"Loaded variables: {list(met_data.data_vars)}")
```

### Command Line Interface

```bash
# STAC-based CLI for downloaders and CBF generation
.venv/bin/python -m src.stac_cli --help

# Download ERA5 meteorology
.venv/bin/python -m src.stac_cli ecmwf \
    --variables t2m_min,t2m_max,vpd,ssrd,strd \
    --year 2020 --month 1 \
    --output ./era5_output

# Download NOAA CO2
.venv/bin/python -m src.stac_cli noaa \
    --year 2020 --month 1 \
    --output ./co2_output

# Download GFED fire data
.venv/bin/python -m src.stac_cli gfed \
    --year 2020 --month 1 \
    --output ./gfed_output

# Generate CBF files from STAC catalog
.venv/bin/python -m src.stac_cli cbf-generate \
    --stac-api file://./era5_output/catalog.json \
    --start 2020-01 --end 2020-12 \
    --output ./cbf_output

# Run tests
.venv/bin/python -m pytest tests/ -v
```

## 📚 Documentation

### Component Documentation
- **[STAC Implementation Summary](plans/STAC_IMPLEMENTATION_SUMMARY.md)** - STAC-based architecture overview
- **[CBF Implementation Summary](plans/CBF_IMPLEMENTATION_SUMMARY.md)** - CBF generation workflow
- **[Agent Onboarding Guide](plans/AGENT_ONBOARDING.md)** - Comprehensive onboarding for new team members

### Development Guidelines
- **[CLAUDE.md](CLAUDE.md)** - Scientist-friendly coding standards and best practices
- **[Migration Plans](plans/README.md)** - Complete 8-phase implementation roadmap
- **[Migration Notes](MIGRATION_NOTES.md)** - Recent changes and refactoring documentation

## 🔬 Scientific Validation

All components maintain scientific equivalence with original MATLAB implementations:

- **MATLAB Function References**: Complete line-by-line mapping in docstrings
- **Numerical Accuracy**: <0.1% difference from MATLAB reference results
- **Physical Validation**: Range checking and unit validation for all variables
- **Quality Control**: Comprehensive data validation with scientific guidance

### Example: MATLAB Equivalence

```python
# Python implementation with exact MATLAB reference
from src.atmospheric_science import saturation_pressure_water_matlab

# MATLAB: VPSAT=6.11*10.^(7.5*T./(237.3+T))./10 (SCIFUN_H2O_SATURATION_PRESSURE.m, line 19)
vpsat_kpa = saturation_pressure_water_matlab(25.0)  # 25°C
print(f"Saturation pressure: {vpsat_kpa:.3f} kPa")  # 3.169 kPa (matches MATLAB)
```

### Variable Configuration System

The preprocessor now uses a centralized variable registry (`src/cardamom_variables.py`) that eliminates scattered configuration:

```python
from src.cardamom_variables import (
    get_variable_config,
    get_interpolation_method,
    get_variables_by_product_type
)

# Get complete metadata for any variable
temp_config = get_variable_config('2m_temperature')
print(temp_config['interpolation_method'])  # 'linear'
print(temp_config['spatial_nature'])        # 'continuous'

# Variables are automatically assigned appropriate methods based on spatial characteristics
# - Continuous fields (temperature, CO2): linear interpolation
# - Patchy data (fire, snow): nearest neighbor
# - Radiation: linear with unit conversions

# Group variables by ERA5 product type for efficient downloads
hourly_vars = get_variables_by_product_type('monthly_averaged_reanalysis_by_hour_of_day')
# ['2m_temperature', '2m_dewpoint_temperature']

monthly_vars = get_variables_by_product_type('monthly_averaged_reanalysis')
# ['total_precipitation', 'skin_temperature', 'surface_solar_radiation_downwards', ...]
```

## 🌐 NASA MAAP Integration

Designed for deployment on the NASA MAAP (Multi-Mission Algorithm and Analysis Platform):

```yaml
# Algorithm configuration
algorithm_id: cardamom-preprocessor
queue: maap-dps-worker-8gb
container: Custom scientific Python environment
disk_space: 100GB (configurable)

# Key parameters
download_mode: [cardamom-monthly, cardamom-hourly, custom]
years: Range or list (e.g., "2020-2022")
months: Range or list (e.g., "6-8")
output_format: [netcdf, geoschem, cardamom]
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository>
cd cardamom-preprocessor
conda env create -f environment.yml
conda activate cardamom-ecmwf-downloader

# Install in development mode
.venv/bin/pip install -e .

# Run tests
.venv/bin/python -m pytest tests/ -v
```

### Adding New Components

1. **Follow scientist-friendly coding standards** in [CLAUDE.md](CLAUDE.md)
2. **Include MATLAB references** for all scientific functions
3. **Add comprehensive documentation** with units and typical ranges
4. **Validate against MATLAB** where applicable
5. **Add simple tests** focusing on functionality over edge cases

### Phase Implementation Status

- ✅ **Phase 1**: STAC-Based Workflow *(STAC catalog management, meteorology discovery)*
- ✅ **Phase 2**: Data Downloaders *(ECMWF ERA5, NOAA CO₂ via HTTPS, GFED)*
- ✅ **Phase 3**: CBF Input Generation *(STAC-based workflow, variable-aware processing)*
- ✅ **Phase 4**: Scientific Functions Library *(Atmospheric utilities, centralized variable registry)*
- 🚧 **Phase 5**: Enhanced CLI *(planned)*

### Recent Improvements (December 2025)

- **Codebase Cleanup**: Removed 17 obsolete files (diurnal processing, monolithic workflow, old downloaders)
- **STAC-Based Architecture**: Streamlined codebase to focus on STAC catalog workflow for data discovery
- **Centralized Variable Configuration**: New `cardamom_variables.py` module eliminates scattered variable definitions
- **NOAA HTTPS Migration**: Updated from deprecated FTP to modern HTTPS protocol with improved error handling
- **Simplified Processing**: Reduced ECMWF downloader complexity by 1,400+ lines through configuration centralization
- **Enhanced Time Handling**: Standardized time coordinates across all data sources for CBF compatibility
- **Variable-Specific Interpolation**: Automatic method selection based on spatial characteristics (continuous vs. patchy)

## 📄 License

[Include your license information here]

## 🙏 Acknowledgments

- **CARDAMOM Framework**: Bloom, A. A., et al. (2016). *Nature Geoscience*, 9(10), 796-800
- **ERA5 Reanalysis**: Copernicus Climate Change Service (C3S)
- **GFED Database**: van der Werf et al. (2017), Global Fire Emissions Database
- **NASA MAAP Platform**: Multi-Mission Algorithm and Analysis Platform

## 📞 Support

For questions about specific phases or components, refer to the individual README files or the detailed specifications within each phase plan in the [plans/](plans/) directory.

---

**CARDAMOM Preprocessor** - Transforming Earth observation data for carbon cycle science 🌱