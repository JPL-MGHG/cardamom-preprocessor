# Phase 6: CBF Input Generation Pipeline ✅ **IMPLEMENTED**

## Overview

Phase 6 implements a **separated download/processing workflow** for generating CARDAMOM Binary Format (CBF) compatible input files. This phase addresses the critical need to create meteorological driver files that can be consumed by `erens_cbf_code.py` for carbon cycle modeling.

**Status: ✅ COMPLETE** - Implemented with 80% coverage from ERA5 data and separated workflow design.

## Key Achievements

### ✅ **Separated Workflow Architecture**
- **Independent download and processing phases** for resilience to data provider issues
- **Resumable processing** that works with previously downloaded files
- **Clean separation of concerns** between data acquisition and scientific processing

### ✅ **High Coverage CBF Generation**
- **80% coverage (8/10 variables)** directly from ERA5 reanalysis data
- **Automated derived variable calculation** (VPD from temperature + dewpoint)
- **Smart fallback values** for missing external data (CO2, fire emissions)

### ✅ **Production-Ready Implementation**
- **CBFMetProcessor class** for processing pre-downloaded files
- **Enhanced ECMWFDownloader** with separated download methods
- **Command-line interface** for easy operation
- **Comprehensive testing** with validation suite

## Implementation Status

### Current CBF Coverage Analysis:
✅ **ERA5 Variables (8/10 - 80% coverage)**:
- VPD *(calculated from temperature + dewpoint)*
- TOTAL_PREC, T2M_MIN, T2M_MAX *(from 2m_temperature, total_precipitation)*
- STRD, SSRD *(from surface radiation variables)*
- SNOWFALL, SKT *(direct mapping)*

✅ **External Data Integration**:
- CO2 *(NOAA data with 415 ppm fallback)*
- BURNED_AREA *(GFED data with zeros fallback)*

✅ **Processing Pipeline**:
- Unit conversions (J/m² → W/m², m → mm/month)
- Land fraction masking
- Spatial/temporal alignment
- NetCDF assembly with CBF compatibility

## Implemented Components

### ✅ 6.1 CBFMetProcessor (`src/cbf_met_processor.py`)
**Primary processor for converting downloaded ERA5 files to CBF format**

```python
from src.cbf_met_processor import CBFMetProcessor

processor = CBFMetProcessor(output_dir="./cbf_output")
cbf_file = processor.process_downloaded_files_to_cbf_met(
    input_dir="./era5_downloads",
    output_filename="AllMet05x05_LFmasked.nc",
    land_fraction_file="land_fraction.nc"  # Optional
)
```

**Key Features:**
- Processes 8/10 CBF variables from ERA5 data
- Handles VPD calculation from temperature + dewpoint
- Applies unit conversions (J/m² → W/m², m → mm, etc.)
- Integrates external data sources (CO2, fire emissions)
- Applies land masking for terrestrial focus

### ✅ 6.2 Enhanced ECMWFDownloader (`src/ecmwf_downloader.py`)
**Separated download workflow with CBF-specific methods**

```python
from src.ecmwf_downloader import ECMWFDownloader

downloader = ECMWFDownloader()

# Download only (resilient to processing failures)
result = downloader.download_cbf_met_variables(
    variables=['2m_temperature', 'total_precipitation', 'surface_solar_radiation_downwards'],
    years=[2020], months=[1, 2, 3],
    download_dir="./era5_downloads"
)
```

**New Methods:**
- `download_cbf_met_variables()` - Download-only workflow
- Automatic inclusion of dewpoint temperature for VPD calculation
- CBF-specific variable validation and metadata

### ✅ 6.3 CLI Interface (`src/cbf_cli.py`)
**Command-line interface for easy CBF processing**

```bash
# List supported variables and requirements
cd src && python cbf_cli.py list-variables

# Process downloaded files to CBF format
cd src && python cbf_cli.py process-met ./downloads/ --output AllMet05x05_LFmasked.nc

# With external data integration
cd src && python cbf_cli.py process-met ./downloads/ \
  --co2-data ./noaa_co2/ \
  --fire-data ./gfed_fire/ \
  --land-fraction land_frac.nc
```

### ✅ 6.4 Test Suite (`test_cbf_met.py`)
**Comprehensive validation of CBF processing capabilities**

```bash
.venv/bin/python test_cbf_met.py
```

**Test Results:**
- ✅ Variable coverage: 80% (8/10 from ERA5)
- ✅ Separated workflow validation
- ✅ CBF file structure compliance

## Integration with Existing Components

### Leverages Current Infrastructure:
- **Phase 1**: CardamomConfig for configuration management
- **Phase 2**: All downloaders (ECMWF, NOAA, GFED, MODIS) for raw data
- **Phase 3**: GFEDProcessor for fire data processing
- **Phase 4**: DiurnalProcessor for temporal processing if needed
- **Phase 5**: NetCDF infrastructure for file I/O
- **Phase 8**: Scientific utilities for calculations (VPD, unit conversions, etc.)

### New Capabilities Added:
- Observational data integration (SCF, LAI, GPP, ABGB, EWT, SOM)
- Multi-source data assembly and masking
- CBF-specific file format generation
- Spatial/temporal alignment tools

## Expected Outputs

### Generated CBF Input Files:
1. `CARDAMOM-MAPS_05deg_LAND_SEA_FRAC.nc` - Land fraction mask
2. `AllMet05x05_LFmasked.nc` - Complete meteorological drivers
3. `AlltsObs05x05newbiomass_LFmasked.nc` - Complete observational constraints
4. `CARDAMOM-MAPS_05deg_HWSD_PEQ_iniSOM.nc` - Soil organic matter
5. `CARDAMOM-MAPS_05deg_GFED4_Mean_FIR.nc` - Fire emissions

### Pipeline Features:
- Config-driven data source selection
- Incremental processing capability
- Quality control and validation
- Progress monitoring and logging
- Error handling and recovery

## Success Criteria
- Generate all input files required by `erens_cbf_code.py`
- Maintain data quality and scientific accuracy
- Support flexible spatial/temporal coverage
- Enable efficient incremental processing
- Provide clear documentation and error reporting

This approach transforms the existing downloaders and processors into a CBF input generation system without requiring MAAP-specific orchestration, focusing on the core scientific data processing needs.

## Key Design Principles

### Config-Driven Operation
- All component parameters driven by configuration files
- Support for environment-specific config overrides
- Validation of all configuration inputs
- Clear mapping from data sources to CBF variables

### Independent Component Design
- Each data generation step designed as standalone operation
- No complex job dependencies or coordination
- Simple input/output interfaces for each component
- Platform-agnostic execution

### Existing Infrastructure Integration
- Leverage all implemented downloaders and processors
- Use established CardamomConfig system
- Integrate with Phase 8 scientific utilities
- Maintain compatibility with existing CLI structure

### Scientist-Friendly Interface
- Clear component operation descriptions
- Scientific parameter documentation
- Error messages with scientific context
- Configuration templates for common use cases

## Implementation Notes

### Data Source Priorities
- Primary sources: ECMWF (meteorology), MODIS (land/vegetation), GFED (fire)
- Secondary sources: NOAA (CO2), FluxSat (GPP), GRACE (water storage)
- Fallback mechanisms for missing or corrupted data
- Quality flags and data provenance tracking

### Spatial/Temporal Considerations
- Standard 0.5° global grid for all outputs
- Monthly temporal resolution as baseline
- Land fraction masking applied consistently
- Coordinate system standardization (lat/lon WGS84)

### Quality Control Framework
- Range checking for all variables
- Temporal consistency validation
- Spatial coherence assessment
- Missing data handling strategies

### Performance Considerations
- Chunked processing for large datasets
- Memory-efficient operations using xarray/dask
- Parallel processing where beneficial
- Progress monitoring and resumable operations