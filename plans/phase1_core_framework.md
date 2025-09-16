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

## 1.3 Comprehensive NetCDF Infrastructure (`netcdf_infrastructure.py`)

**Note**: This section consolidates the complete NetCDF system from the original Phase 5 plan.

### Main NetCDF Writer Class
```python
class CARDAMOMNetCDFWriter:
    """
    Comprehensive class for creating CARDAMOM-compliant NetCDF files.
    Reproduces exact structure and metadata from MATLAB templates.
    Consolidates all NetCDF functionality into Phase 1.
    """

    def __init__(self, template_type="3D", compression=True):
        self.template_type = template_type
        self.compression = compression
        self.global_attributes = self._setup_global_attributes()
        self.dimension_registry = self._setup_dimension_registry()

        # Initialize component managers
        self.dimension_manager = DimensionManager()
        self.data_variable_manager = DataVariableManager(compression)
        self.metadata_manager = MetadataManager()

    def write_2d_dataset(self, data_dict):
        """
        Write 2D dataset (spatial only).
        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_2D_DATASET.
        """
        self._validate_2d_input(data_dict)

        with netCDF4.Dataset(data_dict['filename'], 'w') as nc:
            self.dimension_manager.create_2d_dimensions(nc, data_dict)
            self.dimension_manager.create_coordinate_variables(nc, data_dict, spatial_only=True)
            self.data_variable_manager.create_2d_data_variables(nc, data_dict)
            self.metadata_manager.add_global_metadata(nc, data_dict)

        print(f"Done with {data_dict['filename']}")

    def write_3d_dataset(self, data_dict):
        """
        Write 3D dataset (spatial + temporal).
        Equivalent to MATLAB CARDAMOM_MAPS_WRITE_3D_DATASET.
        """
        self._validate_3d_input(data_dict)

        with netCDF4.Dataset(data_dict['filename'], 'w') as nc:
            self.dimension_manager.create_3d_dimensions(nc, data_dict)
            self.dimension_manager.create_coordinate_variables(nc, data_dict, spatial_only=False)
            self.data_variable_manager.create_3d_data_variables(nc, data_dict)
            self.metadata_manager.add_global_metadata(nc, data_dict)

        print(f"Done with {data_dict['filename']}")

    def write_template_2d(self, grid_info, filename):
        """Create 2D template file. Equivalent to MATLAB CARDAMOM_MAPS_WRITE_2D_TEMPLATE."""

    def write_template_3d(self, grid_info, filename):
        """Create 3D template file. Equivalent to MATLAB CARDAMOM_MAPS_WRITE_3D_TEMPLATE."""

    def _validate_2d_input(self, data_dict):
        """Validate input for 2D dataset creation"""
        required_fields = ['filename', 'x', 'y', 'data', 'info']
        for field in required_fields:
            if field not in data_dict:
                raise ValueError(f"Required field '{field}' missing from data_dict")

        # Validate data dimensions
        expected_shape = (len(data_dict['y']), len(data_dict['x']))
        if data_dict['data'].shape[:2] != expected_shape:
            raise ValueError(f"Data shape {data_dict['data'].shape} does not match grid {expected_shape}")

    def _validate_3d_input(self, data_dict):
        """Validate input for 3D dataset creation"""
        self._validate_2d_input(data_dict)

        if 't' not in data_dict:
            raise ValueError("Time coordinate 't' required for 3D datasets")

        expected_shape = (len(data_dict['y']), len(data_dict['x']), len(data_dict['t']))
        if data_dict['data'].shape != expected_shape:
            raise ValueError(f"Data shape {data_dict['data'].shape} does not match expected {expected_shape}")
```

### Dimension and Coordinate Management
```python
class DimensionManager:
    """Manage NetCDF dimensions and coordinate variables"""

    def __init__(self):
        self.standard_dimensions = {
            'longitude': {'var_name': 'longitude', 'units': 'degrees_east'},
            'latitude': {'var_name': 'latitude', 'units': 'degrees_north'},
            'time': {'var_name': 'time', 'units': 'months since Dec of previous year'}
        }

    def create_2d_dimensions(self, nc_dataset, data_dict):
        """Create dimensions for 2D datasets. Matches MATLAB dimension creation logic."""
        data_shape = data_dict['data'].shape
        nc_dataset.createDimension('latitude', data_shape[0])
        nc_dataset.createDimension('longitude', data_shape[1])

    def create_3d_dimensions(self, nc_dataset, data_dict):
        """Create dimensions for 3D datasets. Matches MATLAB dimension creation logic."""
        data_shape = data_dict['data'].shape
        nc_dataset.createDimension('latitude', data_shape[0])
        nc_dataset.createDimension('longitude', data_shape[1])
        nc_dataset.createDimension('time', data_shape[2])

    def create_coordinate_variables(self, nc_dataset, data_dict, spatial_only=True):
        """Create coordinate variables with proper attributes. Replicates MATLAB coordinate variable creation."""
        # Create longitude variable
        lon_var = nc_dataset.createVariable('longitude', 'f4', ('longitude',))
        lon_var[:] = data_dict['x']
        lon_var.units = 'degrees'

        # Create latitude variable
        lat_var = nc_dataset.createVariable('latitude', 'f4', ('latitude',))
        lat_var[:] = data_dict['y']
        lat_var.units = 'degrees'

        # Create time variable if 3D
        if not spatial_only:
            time_var = nc_dataset.createVariable('time', 'f4', ('time',))
            time_var[:] = data_dict['t']
            time_units = data_dict.get('timeunits', 'months since Dec of previous year')
            time_var.units = time_units
```

### Data Variable Management
```python
class DataVariableManager:
    """Manage creation of data variables with proper attributes"""

    def __init__(self, compression=True):
        self.compression = compression
        self.fill_value = -9999.0

    def create_2d_data_variables(self, nc_dataset, data_dict):
        """Create 2D data variables. Handles both single and multiple variables per file."""
        data_array = data_dict['data']
        info_list = data_dict['info']

        if data_array.ndim == 2:
            self._create_single_2d_variable(nc_dataset, data_array, info_list)
        elif data_array.ndim == 3:
            for var_idx in range(data_array.shape[2]):
                var_data = data_array[:, :, var_idx]
                var_info = info_list[var_idx] if isinstance(info_list, list) else info_list
                self._create_single_2d_variable(nc_dataset, var_data, var_info, var_idx)

    def create_3d_data_variables(self, nc_dataset, data_dict):
        """Create 3D data variables. Handles temporal dimension properly."""
        data_array = data_dict['data']
        info_list = data_dict['info']

        if data_array.ndim == 3:
            self._create_single_3d_variable(nc_dataset, data_array, info_list)
        elif data_array.ndim == 4:
            for var_idx in range(data_array.shape[3]):
                var_data = data_array[:, :, :, var_idx]
                var_info = info_list[var_idx] if isinstance(info_list, list) else info_list
                self._create_single_3d_variable(nc_dataset, var_data, var_info, var_idx)

    def _create_single_2d_variable(self, nc_dataset, data_array, var_info, var_idx=None):
        """Create a single 2D data variable"""
        var_name = self._get_variable_name(var_info, var_idx)

        if self.compression:
            var = nc_dataset.createVariable(
                var_name, 'f4', ('latitude', 'longitude'),
                zlib=True, complevel=6, fill_value=self.fill_value
            )
        else:
            var = nc_dataset.createVariable(
                var_name, 'f4', ('latitude', 'longitude'),
                fill_value=self.fill_value
            )

        var[:] = data_array
        self._add_variable_attributes(var, var_info)

    def _create_single_3d_variable(self, nc_dataset, data_array, var_info, var_idx=None):
        """Create a single 3D data variable"""
        var_name = self._get_variable_name(var_info, var_idx)

        if self.compression:
            var = nc_dataset.createVariable(
                var_name, 'f4', ('latitude', 'longitude', 'time'),
                zlib=True, complevel=6, fill_value=self.fill_value
            )
        else:
            var = nc_dataset.createVariable(
                var_name, 'f4', ('latitude', 'longitude', 'time'),
                fill_value=self.fill_value
            )

        var[:] = data_array
        self._add_variable_attributes(var, var_info)
```

### Metadata Management
```python
class MetadataManager:
    """Manage global and variable metadata for CARDAMOM NetCDF files"""

    def __init__(self):
        self.default_global_attrs = self._setup_default_global_attributes()
        self.cardamom_version = "CARDAMOM preprocessor v1.0"

    def add_global_metadata(self, nc_dataset, data_dict):
        """Add global attributes to NetCDF dataset. Replicates MATLAB global attribute structure."""
        nc_dataset.description = self._get_description(data_dict)
        nc_dataset.details = self._get_details(data_dict)
        nc_dataset.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nc_dataset.version = self.cardamom_version
        nc_dataset.contact = "CARDAMOM Development Team"

        if 'Attributes' in data_dict:
            self._add_custom_attributes(nc_dataset, data_dict['Attributes'])

    def _setup_default_global_attributes(self):
        """Setup default global attributes for CARDAMOM files"""
        return {
            'title': 'CARDAMOM Preprocessed Dataset',
            'institution': 'NASA Jet Propulsion Laboratory',
            'source': 'CARDAMOM Data Assimilation System',
            'conventions': 'CF-1.6',
            'history': f'Created by CARDAMOM preprocessor on {datetime.now().isoformat()}'
        }
```

### Template Generation
```python
class TemplateGenerator:
    """Generate template NetCDF files matching MATLAB templates"""

    def __init__(self):
        self.writer = CARDAMOMNetCDFWriter()

    def create_2d_template(self, output_path, grid_resolution=0.5):
        """Create 2D template file. Equivalent to MATLAB CARDAMOM_MAPS_WRITE_2D_TEMPLATE."""
        if grid_resolution == 0.5:
            lon_coords = np.arange(-179.75, 180, 0.5)  # 720 points
            lat_coords = np.arange(89.75, -90, -0.5)   # 360 points
        else:
            raise ValueError(f"Grid resolution {grid_resolution} not supported")

        template_data = np.full((len(lat_coords), len(lon_coords)), np.nan)

        data_dict = {
            'filename': output_path,
            'x': lon_coords,
            'y': lat_coords,
            'data': template_data,
            'info': {'name': 'data', 'units': 'unitless'},
            'Attributes': {'variable_info': 'TEMPLATE DATASET'}
        }

        self.writer.write_2d_dataset(data_dict)

    def create_3d_template(self, output_path, grid_resolution=0.5):
        """Create 3D template file. Equivalent to MATLAB CARDAMOM_MAPS_WRITE_3D_TEMPLATE."""
        if grid_resolution == 0.5:
            lon_coords = np.arange(-179.75, 180, 0.5)  # 720 points
            lat_coords = np.arange(89.75, -90, -0.5)   # 360 points
        else:
            raise ValueError(f"Grid resolution {grid_resolution} not supported")

        time_coords = np.arange(1, 13)  # 12 months
        template_data = np.full((len(lat_coords), len(lon_coords), len(time_coords)), np.nan)

        data_dict = {
            'filename': output_path,
            'x': lon_coords,
            'y': lat_coords,
            't': time_coords,
            'timeunits': 'months since Dec of previous year',
            'data': template_data,
            'info': {'name': 'data', 'units': 'unitless'},
            'Attributes': {'variable_info': 'TEMPLATE DATASET'}
        }

        self.writer.write_3d_dataset(data_dict)
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

## 1.5 Unified Configuration System (`config_manager.py`)

**Note**: This section provides the unified configuration system that consolidates configuration management across all phases, addressing the configuration overlap identified in the architectural review.

### Unified Configuration Architecture
```python
class CardamomConfig:
    """
    Unified configuration system for CARDAMOM preprocessing.

    Provides centralized configuration management with clear hierarchy:
    1. Built-in defaults
    2. Configuration files (YAML/JSON)
    3. Environment variables
    4. Command-line arguments (highest priority)

    This single class replaces all configuration management across phases.
    """

    def __init__(self, config_file=None, cli_args=None):
        self.config_file = config_file
        self.cli_args = cli_args or {}
        self._config = {}
        self._load_configuration()

    def _load_configuration(self):
        """Load configuration with proper precedence order"""
        # 1. Start with built-in defaults
        self._config = self._get_default_config()

        # 2. Override with config file
        if self.config_file:
            file_config = self._load_config_file(self.config_file)
            self._merge_config(self._config, file_config)

        # 3. Override with environment variables
        env_config = self._load_environment_config()
        self._merge_config(self._config, env_config)

        # 4. Override with CLI arguments (highest priority)
        if self.cli_args:
            self._merge_config(self._config, self.cli_args)

        # 5. Validate final configuration
        self._validate_configuration()

    def _get_default_config(self):
        """Built-in default configuration"""
        return {
            'processing': {
                'output_directory': "./DATA/CARDAMOM-MAPS_05deg_MET/",
                'temp_directory': "./temp/",
                'max_workers': 4,
                'retry_attempts': 3,
                'compression': True,
                'log_level': 'INFO'
            },
            'pipeline': {
                'resume_on_failure': True,
                'cleanup_temp_files': True,
                'generate_qa_reports': True,
                'parallel_downloads': True,
                'state_file': 'pipeline_state.json'
            },
            'global_monthly': {
                'resolution': 0.5,
                'grid_bounds': [-89.75, -179.75, 89.75, 179.75],  # S, W, N, E
                'variables': {
                    'era5': ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation",
                            "skin_temperature", "surface_solar_radiation_downwards", "snowfall",
                            "surface_thermal_radiation_downwards"],
                    'noaa': ["co2"],
                    'gfed': ["burned_area", "fire_carbon"],
                    'modis': ["land_sea_mask", "land_sea_fraction"]
                },
                'netcdf_template': '3D',
                'time_aggregation': 'monthly'
            },
            'conus_diurnal': {
                'resolution': 0.5,
                'region': [60, -130, 20, -50],  # N, W, S, E
                'variables': {
                    'cms': ["GPP", "NEE", "REC", "FIR", "NBE"],
                    'era5': ["skin_temperature", "surface_solar_radiation_downwards"],
                    'gfed': ["diurnal_fire_pattern"]
                },
                'netcdf_template': '3D',
                'time_aggregation': 'hourly',
                'diurnal_hours': list(range(24))
            },
            'downloaders': {
                'era5': {
                    'api_timeout': 3600,
                    'max_concurrent': 3,
                    'chunk_size': 'auto'
                },
                'noaa': {
                    'ftp_timeout': 300,
                    'retry_delay': 5
                },
                'gfed': {
                    'username': None,  # Set via environment
                    'password': None,  # Set via environment
                    'hdf5_chunks': True
                },
                'modis': {
                    'earthdata_username': None,  # Set via environment
                    'earthdata_password': None   # Set via environment
                }
            },
            'quality_control': {
                'enable_validation': True,
                'physical_range_checks': True,
                'spatial_continuity_checks': True,
                'temporal_continuity_checks': True,
                'missing_data_tolerance': 0.05  # 5% missing data allowed
            }
        }

    def _load_config_file(self, config_path):
        """Load configuration from YAML or JSON file"""
        import yaml
        import json
        from pathlib import Path

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def _load_environment_config(self):
        """Load configuration from environment variables"""
        import os

        env_config = {}

        # Map environment variables to config paths
        env_mappings = {
            'CARDAMOM_OUTPUT_DIR': 'processing.output_directory',
            'CARDAMOM_TEMP_DIR': 'processing.temp_directory',
            'CARDAMOM_LOG_LEVEL': 'processing.log_level',
            'CARDAMOM_MAX_WORKERS': 'processing.max_workers',
            'ECMWF_CDS_UID': 'downloaders.era5.username',
            'ECMWF_CDS_KEY': 'downloaders.era5.api_key',
            'NOAA_FTP_USER': 'downloaders.noaa.username',
            'NOAA_FTP_PASS': 'downloaders.noaa.password',
            'GFED_USERNAME': 'downloaders.gfed.username',
            'GFED_PASSWORD': 'downloaders.gfed.password',
            'EARTHDATA_USERNAME': 'downloaders.modis.earthdata_username',
            'EARTHDATA_PASSWORD': 'downloaders.modis.earthdata_password'
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_config(env_config, config_path, value)

        return env_config

    def _set_nested_config(self, config_dict, path, value):
        """Set nested configuration value using dot notation path"""
        keys = path.split('.')
        current = config_dict

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Convert string values to appropriate types
        if isinstance(value, str):
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)

        current[keys[-1]] = value

    def _merge_config(self, base_config, override_config):
        """Deep merge configuration dictionaries"""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value

    def _validate_configuration(self):
        """Validate final configuration"""
        required_sections = ['processing', 'pipeline', 'global_monthly', 'conus_diurnal', 'downloaders']

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Required configuration section missing: {section}")

        # Validate output directory
        output_dir = Path(self._config['processing']['output_directory'])
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Validate temp directory
        temp_dir = Path(self._config['processing']['temp_directory'])
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)

    # Public interface methods
    def get(self, path, default=None):
        """Get configuration value using dot notation path"""
        keys = path.split('.')
        current = self._config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def get_processing_config(self):
        """Get processing-specific configuration"""
        return self._config['processing']

    def get_pipeline_config(self):
        """Get pipeline-specific configuration"""
        return self._config['pipeline']

    def get_workflow_config(self, workflow_type):
        """Get configuration for specific workflow"""
        if workflow_type not in self._config:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        return self._config[workflow_type]

    def get_downloader_config(self, downloader_name):
        """Get configuration for specific downloader"""
        return self._config['downloaders'].get(downloader_name, {})

    def get_quality_control_config(self):
        """Get quality control configuration"""
        return self._config['quality_control']

    def to_dict(self):
        """Return complete configuration as dictionary"""
        return self._config.copy()

    def save_config(self, output_path):
        """Save current configuration to file"""
        import yaml
        from pathlib import Path

        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
```

### Configuration File Schema
```yaml
# Example: config/cardamom_processing.yaml
# All sections are optional - defaults will be used if not specified

processing:
  output_directory: "./DATA/CARDAMOM-MAPS_05deg_MET/"
  temp_directory: "./temp/"
  max_workers: 8
  compression: true
  log_level: "DEBUG"

pipeline:
  resume_on_failure: true
  cleanup_temp_files: false  # Keep temp files for debugging
  parallel_downloads: true

global_monthly:
  resolution: 0.25  # Override default 0.5
  variables:
    era5:
      - "2m_temperature"
      - "2m_dewpoint_temperature"
      - "total_precipitation"
    gfed:
      - "burned_area"

conus_diurnal:
  region: [50, -125, 25, -65]  # Custom CONUS bounds

downloaders:
  era5:
    max_concurrent: 5
  gfed:
    hdf5_chunks: false

quality_control:
  missing_data_tolerance: 0.10  # Allow 10% missing data
```

### Integration with Other Components
```python
# Example usage in main processor
class CARDAMOMProcessor:
    def __init__(self, config_file=None, cli_args=None):
        # Use unified configuration system
        self.config = CardamomConfig(config_file, cli_args)

        # Initialize components with config
        self.output_dir = self.config.get('processing.output_directory')
        self.netcdf_writer = CARDAMOMNetCDFWriter(
            compression=self.config.get('processing.compression', True)
        )

        # Pass relevant config sections to other components
        self.pipeline_manager = PipelineManager(self.config.get_pipeline_config())
        self.qa_system = QualityAssurance(self.config.get_quality_control_config())
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
- [ ] Successfully implement standard coordinate system definitions for CARDAMOM
- [ ] Create NetCDF files with CARDAMOM-compliant structure and metadata
- [ ] Accurate VPD calculations using validated scientific formulas
- [ ] Robust error handling and recovery mechanisms

### Performance Requirements
- [ ] Process global 0.5° monthly data within reasonable memory limits
- [ ] Support resumable processing for large datasets
- [ ] Efficient regridding between different coordinate systems

### Quality Requirements
- [ ] Comprehensive unit test coverage (>90%)
- [ ] Detailed documentation for all public APIs
- [ ] Clear error messages and debugging information