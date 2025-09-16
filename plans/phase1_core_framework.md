# Phase 1: Core Data Processing Framework

## Overview
Establish the foundational infrastructure for CARDAMOM data processing, including the main orchestration module, NetCDF processing capabilities, and core scientific functions.

## 1.1 Main Orchestration Module (`cardamom_preprocessor.py`)

### CARDAMOMProcessor Class
```python
class CARDAMOMProcessor:
    """
    Main class for coordinating CARDAMOM data preprocessing workflows.
    Manages data flow between downloaders, processors, and writers.
    """

    def __init__(self, config_file=None, output_dir="./DATA/CARDAMOM-MAPS_05deg_MET/"):
        self.config = self._load_config(config_file)
        self.output_dir = output_dir
        self.coordinate_systems = self._init_coordinate_systems()

    def _load_config(self, config_file):
        """Load processing configuration from YAML/JSON file"""

    def _init_coordinate_systems(self):
        """Initialize standard coordinate grids (0.25°, 0.5°, GeosChem)"""

    def process_global_monthly(self, years, months=None):
        """Execute complete global monthly preprocessing pipeline"""

    def process_conus_diurnal(self, years, months=None):
        """Execute CONUS diurnal flux processing pipeline"""
```

### Key Methods
- `setup_data_directories()`: Create standardized output directory structure
- `validate_inputs()`: Check year/month ranges and data availability
- `process_batch()`: Handle multi-year/month processing with progress tracking
- `generate_summary_report()`: Create processing summary with statistics

## 1.2 Coordinate System Management (`coordinate_systems.py`)

### Grid Definitions
```python
class CoordinateGrid:
    """Represents a standard geographic grid used in CARDAMOM processing"""

    def __init__(self, resolution, bounds=None):
        self.resolution = resolution  # e.g., 0.5, 0.25
        self.bounds = bounds or [-89.75, -179.75, 89.75, 179.75]  # Global default
        self.x, self.y = self._create_grid()

    def _create_grid(self):
        """Create longitude/latitude arrays matching MATLAB loadworldmesh"""

    def get_indices_for_region(self, region_bounds):
        """Get array indices for a specific geographic region"""

    def regrid_data(self, data, target_grid):
        """Regrid data from this grid to target grid"""
```

### Standard Grids
- **Global 0.5°**: Primary CARDAMOM grid (720×360)
- **Global 0.25°**: High-resolution grid for GFED data (1440×720)
- **CONUS 0.5°**: Regional grid for diurnal processing
- **GeosChem 4×5°**: Model-specific grid (72×46)

## 1.3 NetCDF Infrastructure (`netcdf_infrastructure.py`)

### Base NetCDF Writer
```python
class CARDAMOMNetCDFWriter:
    """
    Base class for creating CARDAMOM-compliant NetCDF files.
    Ensures consistent metadata, dimensions, and variable structure.
    """

    def __init__(self, template_type="3D"):  # "2D" or "3D"
        self.template_type = template_type
        self.required_dimensions = self._get_required_dimensions()

    def create_file(self, filename, data_dict, grid, time_info=None):
        """Create NetCDF file with CARDAMOM-standard structure"""

    def add_cardamom_metadata(self, dataset, creation_info):
        """Add standard CARDAMOM global attributes"""

    def validate_data_structure(self, data_dict):
        """Ensure data conforms to CARDAMOM requirements"""
```

### Template Functions (from MATLAB)
```python
def create_2D_template(grid, data_shape):
    """Equivalent to CARDAMOM_MAPS_WRITE_2D_TEMPLATE"""

def create_3D_template(grid, data_shape, time_coords):
    """Equivalent to CARDAMOM_MAPS_WRITE_3D_TEMPLATE"""
```

## 1.4 Scientific Utility Functions (`scientific_utils.py`)

### Water Vapor Functions
```python
def saturation_pressure_water(temperature_kelvin):
    """
    Calculate saturation pressure of water vapor.
    Equivalent to MATLAB SCIFUN_H2O_SATURATION_PRESSURE

    Args:
        temperature_kelvin: Temperature in Kelvin

    Returns:
        Saturation pressure in hPa
    """
    # Implementation of Tetens formula or Magnus formula

def calculate_vpd(temp_max, dewpoint_temp):
    """
    Calculate Vapor Pressure Deficit.
    Replicates VPD calculation from MATLAB script line 202
    """
    sat_pressure_tmax = saturation_pressure_water(temp_max)
    sat_pressure_dewpoint = saturation_pressure_water(dewpoint_temp)
    return (sat_pressure_tmax - sat_pressure_dewpoint) * 10  # Convert to hPa
```

### Unit Conversion Functions
```python
def convert_precipitation_units(precip_meters_per_second, scale_factor=1e3):
    """Convert precipitation from m/s to mm/day"""
    return precip_meters_per_second * scale_factor

def convert_radiation_units(radiation_joules_per_m2):
    """Convert radiation units as needed for CARDAMOM"""

def convert_carbon_flux_units(flux_gc_m2_day):
    """Convert carbon fluxes to standard CARDAMOM units (Kg C/Km^2/sec)"""
    return flux_gc_m2_day * 1e3 / 24 / 3600
```

## 1.5 Configuration Management (`config_manager.py`)

### Configuration Schema
```yaml
# config/cardamom_processing.yaml
processing:
  output_directory: "./DATA/CARDAMOM-MAPS_05deg_MET/"
  temp_directory: "./temp/"

global_monthly:
  resolution: 0.5
  variables:
    era5: ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation",
           "skin_temperature", "surface_solar_radiation_downwards", "snowfall",
           "surface_thermal_radiation_downwards"]
    noaa: ["co2"]
    gfed: ["burned_area", "fire_carbon"]
    modis: ["land_sea_mask", "land_sea_fraction"]

conus_diurnal:
  resolution: 0.5
  region: [60, -130, 20, -50]  # N, W, S, E
  variables:
    cms: ["GPP", "NEE", "REC", "FIR", "NBE"]
    era5: ["skin_temperature", "surface_solar_radiation_downwards"]
    gfed: ["diurnal_fire_pattern"]
```

### Config Loader
```python
class ConfigManager:
    """Manage configuration loading and validation"""

    def load_config(self, config_path):
        """Load and validate configuration file"""

    def get_processing_config(self, workflow_type):
        """Get configuration for specific workflow (global_monthly, conus_diurnal)"""

    def validate_config(self, config):
        """Ensure configuration has all required fields"""
```

## 1.6 Data Validation and Quality Control

### Quality Control Functions
```python
def validate_spatial_coverage(data, expected_grid):
    """Ensure data covers expected spatial domain"""

def check_temporal_continuity(data, expected_time_range):
    """Verify temporal coverage and identify gaps"""

def validate_physical_ranges(data, variable_type):
    """Check if data values are within physically reasonable ranges"""

def generate_qa_report(processed_data, output_path):
    """Generate quality assurance report for processed data"""
```

## 1.7 Error Handling and Logging

### Logging Infrastructure
```python
import logging
from datetime import datetime

def setup_cardamom_logging(log_level="INFO", log_file=None):
    """Setup standardized logging for CARDAMOM processing"""

class ProcessingLogger:
    """Specialized logger for tracking processing progress"""

    def log_processing_start(self, workflow_type, parameters):
        """Log start of processing workflow"""

    def log_data_download(self, source, files_downloaded):
        """Log successful data downloads"""

    def log_processing_error(self, error_type, error_details):
        """Log processing errors with context"""

    def log_processing_complete(self, summary_stats):
        """Log completion with summary statistics"""
```

## 1.8 Testing Framework

### Unit Tests Structure
```
tests/
├── test_coordinate_systems.py
├── test_netcdf_infrastructure.py
├── test_scientific_utils.py
├── test_config_manager.py
└── fixtures/
    ├── sample_era5_data.nc
    ├── sample_config.yaml
    └── expected_outputs/
```

### Test Data Fixtures
- Small sample datasets for each data source
- Expected output files for validation
- Configuration files for different test scenarios

## 1.9 Dependencies and Integration

### Required Python Packages
```yaml
# Additional dependencies for environment.yml
- xarray>=0.20.0
- netcdf4>=1.5.0
- numpy>=1.20.0
- scipy>=1.7.0
- pyyaml>=5.4.0
- h5py>=3.0.0  # For GFED HDF5 files
- pandas>=1.3.0
- requests>=2.25.0
```

### Integration Points
- **Phase 2**: Downloaders will use coordinate systems and NetCDF infrastructure
- **Phase 3-4**: Processors will use scientific utilities and validation functions
- **Phase 6**: Pipeline manager will use configuration management and logging
- **Existing Code**: Seamless integration with current `ecmwf_downloader.py`

## 1.10 Success Criteria

### Functional Requirements
- [ ] Successfully replicate MATLAB coordinate system definitions
- [ ] Create NetCDF files with identical structure to MATLAB output
- [ ] Accurate VPD calculations matching MATLAB results
- [ ] Robust error handling and recovery mechanisms

### Performance Requirements
- [ ] Process global 0.5° monthly data within reasonable memory limits
- [ ] Support resumable processing for large datasets
- [ ] Efficient regridding between different coordinate systems

### Quality Requirements
- [ ] Comprehensive unit test coverage (>90%)
- [ ] Detailed documentation for all public APIs
- [ ] Clear error messages and debugging information