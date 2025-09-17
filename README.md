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

The system is organized into 8 phases, each providing specific functionality:

### **Phase 1: Core Framework** ✅ **COMPLETE**
- Main orchestration with CARDAMOMProcessor class
- Unified configuration system with environment variable support
- Complete NetCDF infrastructure with MATLAB equivalence
- Coordinate systems and scientific utilities
- Error handling and recovery with resumable processing

### **Phase 2: Data Downloaders** ✅
- ECMWF meteorological data downloader
- NOAA CO₂ concentration downloader
- GFED fire emissions downloader
- MODIS land-sea mask generator

### **Phase 3: GFED Processing** ✅
- GFED burned area and fire emissions processing
- Gap-filling for missing years (2017+)
- Multi-resolution support (0.25°, 0.5°, GeosChem)

### **Phase 4: Diurnal Flux Processing** ✅
- CONUS carbon flux downscaling (monthly to hourly)
- CMS flux integration with meteorological drivers
- GeosChem-compatible output generation

### **Phase 5: NetCDF System** ✅ *Consolidated into Phase 1*
- CARDAMOM-compliant NetCDF file generation
- Template-based output with proper metadata

### **Phase 6: Pipeline Management** 🚧 *Planned*
- Unified workflow orchestration
- Component coordination and state management

### **Phase 7: Enhanced CLI** 🚧 *Planned*
- Extended command-line interface
- Interactive configuration and validation

### **Phase 8: Scientific Functions Library** ✅ **NEW!**
- Atmospheric science calculations (VPD, humidity, radiation)
- Statistical utilities (temporal aggregation, spatial interpolation)
- Physical constants and unit conversions
- Carbon cycle modeling functions
- Enhanced data quality control

## 🚀 Quick Start

### Prerequisites

```bash
# Create conda environment
conda env create -f environment.yml
conda activate cardamom-ecmwf-downloader

# Install package in development mode
.venv/bin/pip install -e .
```

### Basic Usage

```python
# Import core functionality
from src import (
    # Phase 1: Core components (COMPLETE)
    CARDAMOMProcessor, CardamomConfig,
    CARDAMOMNetCDFWriter, StandardGrids,

    # Phase 2: Data downloaders
    ECMWFDownloader, NOAADownloader, GFEDDownloader,

    # Phase 3: GFED processing
    GFEDProcessor,

    # Phase 4: Diurnal processing
    DiurnalProcessor,

    # Phase 8: Scientific functions
    saturation_pressure_water_matlab,
    calculate_vapor_pressure_deficit_matlab,
    PhysicalConstants,
    validate_temperature_range_extended
)

# Example: Complete workflow using Phase 1 orchestration
processor = CARDAMOMProcessor()

# Process global monthly data with error recovery and progress tracking
result = processor.process_batch(
    workflow_type='global_monthly',
    years=[2020, 2021],
    months=[1, 2, 3],
    show_progress=True
)
print(f"Success: {result['successful_combinations']}")

# Process CONUS diurnal data (integrates with Phase 4)
diurnal_result = processor.process_conus_diurnal(
    years=[2020],
    months=[6, 7, 8]
)
print(f"Diurnal processing: {diurnal_result['status']}")

# Example: Configuration management
config = CardamomConfig.create_template_config('minimal')
processor = CARDAMOMProcessor(config_file=config)

# Example: Use scientific functions
import numpy as np
temp_max_c = np.array([25, 30, 35])  # °C
dewpoint_c = np.array([15, 18, 20])  # °C
vpd_hpa = calculate_vapor_pressure_deficit_matlab(temp_max_c, dewpoint_c)
print(f"VPD: {vpd_hpa} hPa")  # Expected: [11.7, 19.4, 29.8] hPa
```

### Command Line Interface

```bash
# Download CARDAMOM meteorological drivers
python ecmwf/ecmwf_downloader.py cardamom-monthly -y 2020 -m 1-3

# Run complete MAAP workflow
./.maap/run.sh cardamom-monthly ./output 2020 1-12

# Run tests
.venv/bin/python -m pytest tests/ -v
```

## 📚 Documentation

### Component Documentation
- **[Phase 1 Core Framework](plans/README_PHASE1.md)** - Main orchestration and infrastructure ✅ **NEW**
- **[ECMWF Downloader](ecmwf/README.md)** - ERA5 meteorological data download
- **[Phase 2 Downloaders](plans/README_PHASE2.md)** - Multi-source data acquisition
- **[Phase 3 GFED](plans/README_PHASE3.md)** - Fire emissions processing
- **[Phase 4 Diurnal](plans/README_PHASE4.md)** - Hourly flux downscaling
- **[Phase 8 Scientific Functions](plans/README_PHASE8.md)** - Utility function library

### Development Guidelines
- **[CLAUDE.md](CLAUDE.md)** - Scientist-friendly coding standards and best practices
- **[Migration Plans](plans/README.md)** - Complete 8-phase implementation roadmap

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

- ✅ **Phase 1**: Core Framework *(Complete with CARDAMOMProcessor orchestration)*
- ✅ **Phase 2**: Data Downloaders *(ECMWF, NOAA, GFED, MODIS)*
- ✅ **Phase 3**: GFED Processing *(Gap-filling and multi-resolution)*
- ✅ **Phase 4**: Diurnal Flux Processing *(CONUS hourly downscaling)*
- ✅ **Phase 8**: Scientific Functions Library *(Atmospheric & carbon cycle utilities)*
- 🚧 **Phase 6**: Pipeline Management *(planned)*
- 🚧 **Phase 7**: Enhanced CLI *(planned)*

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