# CARDAMOM MATLAB to Python Migration Plans

## Overview

This directory contains detailed implementation plans for converting the MATLAB CARDAMOM preprocessing system to Python. The migration is organized into 8 phases, each with specific deliverables and success criteria.

## Migration Scope

### Source MATLAB Files
1. **CARDAMOM_MAPS_05deg_DATASETS_JUL24.m** - Main data processing and NetCDF creation
2. **CARDAMOM_MAPS_READ_GFED_NOV24.m** - GFED burned area data reader with gap-filling
3. **PROJSCRIPT_DIURNAL_CMS_C_FLUXES_AUG25.m** - Diurnal flux processing for CONUS

### Target Python System
A comprehensive, modular preprocessing pipeline that:
- Maintains exact compatibility with MATLAB outputs
- Integrates with existing ECMWF downloader infrastructure
- Supports NASA MAAP platform deployment
- Provides enhanced CLI and pipeline management capabilities

## Phase Structure

### Phase 1: Core Data Processing Framework
**File**: `phase1_core_framework.md`

**Deliverables**:
- `CARDAMOMProcessor` main orchestration class
- Coordinate system management for multiple grids (0.25°, 0.5°, GeosChem)
- NetCDF infrastructure with CARDAMOM-compliant templates
- Scientific utility functions (VPD calculation, unit conversions)
- Configuration management and error handling

**Key Components**:
- `cardamom_preprocessor.py` - Main orchestration
- `coordinate_systems.py` - Grid management
- `netcdf_infrastructure.py` - NetCDF creation
- `scientific_utils.py` - Scientific calculations
- `config_manager.py` - Configuration handling

### Phase 2: Data Source-Specific Downloaders
**File**: `phase2_downloaders.md`

**Deliverables**:
- Enhanced ECMWF downloader with additional variables
- NOAA CO2 downloader with FTP access
- GFED downloader with HDF5 support and authentication
- MODIS land-sea mask downloader
- Unified downloader interface with retry logic and caching

**Key Components**:
- `ecmwf_downloader.py` - Enhanced ERA5 downloads
- `noaa_downloader.py` - NOAA CO2 data
- `gfed_downloader.py` - GFED fire data
- `modis_downloader.py` - Land-sea masks
- `base_downloader.py` - Common interface

### Phase 3: GFED Processing Module
**File**: `phase3_gfed_processor.md`

**Deliverables**:
- Comprehensive GFED data processor with gap-filling
- Multi-resolution support (0.25°, 0.5°, GeosChem 4×5)
- Climatology-based extrapolation for missing years (2017+)
- Emission factor calculations and species conversions
- Data quality validation and reporting

**Key Components**:
- `gfed_processor.py` - Main processing class
- `gap_filling.py` - Temporal gap-filling logic
- `resolution_converter.py` - Spatial aggregation
- `emission_factors.py` - Fire emission calculations
- `gfed_validation.py` - Quality control

### Phase 4: Diurnal Flux Processing
**File**: `phase4_diurnal_processor.md`

**Deliverables**:
- CONUS diurnal flux processing system
- CMS monthly flux loader with spatial interpolation
- ERA5 meteorological driver integration
- GFED diurnal fire pattern application
- Flux downscaling algorithms (GPP, REC, FIR, NEE, NBE)

**Key Components**:
- `diurnal_processor.py` - Main processing workflow
- `cms_flux_loader.py` - CMS data handling
- `met_driver_loader.py` - ERA5 diurnal data
- `diurnal_calculator.py` - Downscaling algorithms
- `diurnal_output_writers.py` - GeosChem format outputs

### Phase 5: NetCDF Template and Writing System
**File**: `phase5_netcdf_system.md`

**Deliverables**:
- CARDAMOM-compliant NetCDF writer system
- 2D and 3D template generators matching MATLAB structure
- Comprehensive metadata management
- Data variable and coordinate handling
- CF convention compliance and validation

**Key Components**:
- `netcdf_writer.py` - Main writing infrastructure
- `coordinate_manager.py` - Dimension handling
- `data_variable_manager.py` - Variable creation
- `metadata_manager.py` - Attribute management
- `template_generators.py` - Template creation

### Phase 6: Unified Processing Pipeline
**File**: `phase6_pipeline_manager.md`

**Deliverables**:
- Comprehensive pipeline orchestration system
- Component managers for downloaders, processors, and outputs
- Pipeline state management and resumability
- Error handling and recovery mechanisms
- Quality assurance and reporting system

**Key Components**:
- `pipeline_manager.py` - Main orchestrator
- `component_managers.py` - Module coordination
- `pipeline_state.py` - State tracking
- `error_handling.py` - Recovery strategies
- `qa_reporting.py` - Quality control

### Phase 7: Enhanced CLI Integration
**File**: `phase7_cli_integration.md`

**Deliverables**:
- Extended CLI maintaining backward compatibility
- Pipeline management commands
- Data source-specific commands
- Utility commands for validation and reporting
- Interactive configuration wizards

**Key Components**:
- `cardamom_cli.py` - Enhanced main CLI
- `pipeline_commands.py` - Pipeline interface
- `data_source_commands.py` - Individual source management
- `utility_commands.py` - Validation and utilities
- `interactive_cli.py` - Configuration wizards

### Phase 8: Scientific Functions Library
**File**: `phase8_scientific_utils.md`

**Deliverables**:
- Comprehensive atmospheric science calculations
- Carbon cycle and biogeochemistry functions
- Statistical and interpolation utilities
- Unit conversion and physical constants
- Data quality control and validation

**Key Components**:
- `atmospheric_science.py` - Water vapor, VPD, radiation
- `carbon_cycle.py` - Photosynthesis, respiration, fire emissions
- `statistics_utils.py` - Interpolation and time series analysis
- `units_constants.py` - Conversions and constants
- `quality_control.py` - Data validation

## Implementation Strategy

### Development Order
1. **Phases 1 & 8** can be developed in parallel as they provide foundational infrastructure
2. **Phase 2** follows Phase 1, extending existing downloader capabilities
3. **Phases 3 & 4** can be developed in parallel, both using Phase 1 infrastructure
4. **Phase 5** integrates with Phases 3 & 4 for output generation
5. **Phase 6** orchestrates all previous phases
6. **Phase 7** provides user interface to complete system

### Key Integration Points
- **Phase 1** provides the foundation used by all other phases
- **Phase 2** downloaders feed into **Phase 6** pipeline management
- **Phases 3 & 4** processors use **Phase 5** for output generation
- **Phase 6** coordinates **Phases 2-5** into unified workflows
- **Phase 7** provides user access to **Phase 6** capabilities
- **Phase 8** functions are used throughout **Phases 1-6**

### Testing Strategy
- Each phase includes comprehensive unit and integration tests
- Validation against MATLAB reference outputs
- Performance benchmarking for large datasets
- End-to-end pipeline testing with real data

## Success Criteria

### Functional Requirements
- [ ] Exact reproduction of MATLAB output files (bit-for-bit when possible)
- [ ] Support for all MATLAB workflows (global monthly, CONUS diurnal)
- [ ] Enhanced capabilities beyond original MATLAB system
- [ ] Seamless integration with existing ECMWF infrastructure

### Performance Requirements
- [ ] Process multi-year global datasets efficiently
- [ ] Support parallel processing where applicable
- [ ] Memory-efficient handling of large arrays
- [ ] Reasonable processing times for operational use

### Quality Requirements
- [ ] Comprehensive error handling and logging
- [ ] Extensive test coverage (>90% for each phase)
- [ ] Clear documentation and user guides
- [ ] Robust data validation and quality control

### Integration Requirements
- [ ] Compatible with NASA MAAP platform
- [ ] Backward compatible CLI interface
- [ ] Configurable for different deployment scenarios
- [ ] Support for different data source combinations

## Dependencies and Environment

### Core Dependencies
```yaml
# From existing environment.yml, enhanced
dependencies:
  - python>=3.9
  - numpy>=1.20
  - scipy>=1.7
  - xarray>=0.20
  - netcdf4>=1.5
  - h5py>=3.0
  - pandas>=1.3
  - pyyaml>=5.4
  - requests>=2.25
  - cdsapi>=0.5
  - click>=8.0  # For enhanced CLI
  - tqdm>=4.60  # For progress bars
```

### Development Dependencies
```yaml
development:
  - pytest>=6.0
  - pytest-cov>=2.10
  - black>=21.0
  - flake8>=3.8
  - sphinx>=4.0
  - jupyter>=1.0
```

## Getting Started

1. **Review Plans**: Read through each phase plan to understand the scope and approach
2. **Set Up Environment**: Install dependencies and development tools
3. **Choose Starting Phase**: Begin with Phase 1 or Phase 8 depending on team expertise
4. **Iterative Development**: Implement, test, and validate each component
5. **Integration Testing**: Combine phases and test against MATLAB outputs
6. **Documentation**: Maintain comprehensive documentation throughout development

## Support and Collaboration

- Each phase plan includes detailed implementation specifications
- Success criteria provide clear validation targets
- Test frameworks ensure quality and compatibility
- Modular design supports parallel development by multiple developers

For questions about specific phases, refer to the individual plan files or the detailed specifications within each phase document.