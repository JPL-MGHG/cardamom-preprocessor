# CARDAMOM Preprocessor - Phase 1 Core Framework

## Implementation Summary

Phase 1 of the CARDAMOM preprocessor has been successfully implemented, providing the foundational infrastructure for all CARDAMOM data processing workflows. This implementation creates the core modules that subsequent phases will build upon.

## Directory Structure

```
cardamom-preprocessor/
├── src/                          # Core modules (NEW)
│   ├── __init__.py              # Package initialization
│   ├── cardamom_preprocessor.py # Main orchestration module
│   ├── config_manager.py        # Unified configuration system
│   ├── coordinate_systems.py    # Grid management
│   ├── netcdf_infrastructure.py # NetCDF file creation
│   ├── scientific_utils.py      # Scientific calculations
│   ├── validation.py            # Data validation and QC
│   └── logging_utils.py         # Error handling and logging
├── tests/                       # Basic testing framework (NEW)
│   ├── __init__.py
│   ├── test_config_manager.py
│   ├── test_coordinate_systems.py
│   ├── test_netcdf_infrastructure.py
│   ├── test_scientific_utils.py
│   └── test_validation.py
├── requirements.txt             # Updated dependencies (NEW)
├── run_tests.py                 # Test runner script (NEW)
├── ecmwf/                       # Existing downloader (unchanged)
├── plans/                       # Documentation (unchanged)
└── ...
```

## Core Modules Implemented

### 1. Unified Configuration System (`config_manager.py`)
- **CardamomConfig class** with 4-tier hierarchy (defaults → file → environment → CLI)
- Support for YAML/JSON configuration files
- Environment variable mapping for credentials
- Dot notation access for nested configuration values
- Built-in validation and directory creation

### 2. Coordinate System Management (`coordinate_systems.py`)
- **CoordinateGrid class** for standard geographic grids
- Support for 0.5°, 0.25°, and GeosChem grids
- **StandardGrids factory** for common grid types
- Regional subsetting and regridding capabilities
- Grid creation matching MATLAB `loadworldmesh` function

### 3. NetCDF Infrastructure (`netcdf_infrastructure.py`)
- **CARDAMOMNetCDFWriter** main class replicating MATLAB templates
- **DimensionManager** for coordinate variable creation
- **DataVariableManager** for data handling with compression
- **MetadataManager** for global and variable attributes
- **TemplateGenerator** for 2D/3D template files
- Support for both single and multi-variable NetCDF files

### 4. Scientific Utility Functions (`scientific_utils.py`)
- Vapor pressure deficit calculation using Tetens equation
- Unit conversion functions (precipitation, radiation, carbon fluxes)
- Temperature and precipitation validation functions
- Growing degree days calculation
- All functions include scientific documentation and expected ranges

### 5. Data Validation and Quality Control (`validation.py`)
- Spatial coverage validation
- Temporal continuity checks
- Physical range validation by variable type
- Data consistency validation across variables
- **QualityAssurance class** for comprehensive QA workflows
- JSON report generation

### 6. Error Handling and Logging (`logging_utils.py`)
- Standardized logging setup for CARDAMOM processing
- **ProcessingLogger class** for workflow progress tracking
- Custom exception classes (CARDAMOMError, DataDownloadError, etc.)
- Error context management with detailed logging
- Processing session management

### 7. Main Orchestration Module (`cardamom_preprocessor.py`)
- **CARDAMOMProcessor class** coordinating all workflows
- Global monthly and CONUS diurnal processing pipelines
- Integration with existing ECMWFDownloader
- Progress tracking and batch processing support
- Quality assurance integration

## Key Features

### Scientist-Friendly Design
- Clear variable names with units and scientific meaning
- Comprehensive documentation with scientific references
- Self-validating functions with physical range checks
- Error messages with scientific context

### Configuration Management
- Centralized configuration with clear precedence
- Support for different deployment scenarios (dev, production, testing)
- Environment variable integration for credentials
- Validation and automatic directory creation

### Integration Ready
- Seamless integration with existing `ecmwf/ecmwf_downloader.py`
- Designed for future phase integration
- Modular architecture for easy extension
- Common interfaces across all modules

### Quality Assurance
- Comprehensive validation at multiple levels
- Automated QA report generation
- Physical and scientific reasonableness checks
- Structured logging for debugging

## Installation and Setup

### 1. Install Dependencies
```bash
# Using the project's virtual environment
pip install -r requirements.txt

# Or if using conda environment
conda install --file requirements.txt
```

### 2. Run Basic Tests
```bash
# Run the test suite
python run_tests.py

# Or use pytest directly (if installed)
python -m pytest tests/ -v
```

### 3. Basic Usage Example
```python
from src import CARDAMOMProcessor, CardamomConfig

# Initialize with default configuration
processor = CARDAMOMProcessor()

# Or with custom configuration
config = CardamomConfig(config_file="my_config.yaml")
processor = CARDAMOMProcessor(config_file="my_config.yaml")

# Run global monthly processing
results = processor.process_global_monthly(
    years=[2020, 2021],
    months=[1, 2, 3]
)

# Run CONUS diurnal processing
results = processor.process_conus_diurnal(
    years=[2020],
    months=[6, 7, 8]
)
```

## Testing Framework

The implementation includes a basic testing framework with:

- **Unit tests** for all core modules
- **Functionality verification** (not scientific validation)
- **Test runner script** (`run_tests.py`)
- **Import and initialization tests**
- **Basic validation tests**

Tests focus on ensuring modules load correctly, functions execute without errors, and basic validation works as expected.

## Integration with Existing Code

Phase 1 maintains full backward compatibility with existing code:

- **ECMWFDownloader** remains unchanged and functional
- **Environment setup** uses existing conda/venv structure
- **MAAP integration** preserved in `.maap/` directory
- **Project documentation** maintained in `plans/`

## Next Steps for Future Phases

Phase 1 provides the foundation for:

- **Phase 2**: Data downloaders will use coordinate systems and NetCDF infrastructure
- **Phase 3-4**: Processors will use scientific utilities and validation
- **Phase 6**: Pipeline manager will use configuration and logging
- **Phase 7**: CLI will use unified configuration system

## Success Criteria Met

✅ All core modules import successfully
✅ Basic functionality tests pass
✅ Configuration system loads properly
✅ NetCDF infrastructure creates valid structures
✅ Integration with existing ECMWF downloader works
✅ All modules follow scientist-friendly coding patterns
✅ Comprehensive error handling and logging
✅ Quality assurance framework operational

The Phase 1 implementation successfully establishes the core infrastructure needed for the complete CARDAMOM preprocessing system while maintaining the project's commitment to scientist-friendly, readable code.