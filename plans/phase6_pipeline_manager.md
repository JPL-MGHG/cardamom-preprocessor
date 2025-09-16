# Phase 6: Component Access Library

## Overview
Provide simple access to individual CARDAMOM components without complex orchestration. Each component operates independently as a single-purpose operation, designed to run as separate MAAP jobs. Removes internal parallelism and coordination since the MAAP platform handles job scheduling and parallelism externally.

## 6.1 Component Access Manager (`component_access.py`)

### Simple Component Access
```python
class CARDAMOMComponents:
    """
    Provides access to individual CARDAMOM components without orchestration.
    Each component operates independently as single-purpose operations.
    Designed for MAAP platform where each operation runs as a separate job.
    """

    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()

        # Initialize individual components (no managers)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize individual component instances"""
        from ..downloaders import ECMWFDownloader, NOAADownloader, GFEDDownloader, MODISDownloader
        from ..processors import GFEDProcessor, DiurnalProcessor
        from ..core import CARDAMOMProcessor

        self.downloaders = {
            'ecmwf': ECMWFDownloader(
                output_dir=self.config['paths']['ecmwf_data']
            ),
            'noaa': NOAADownloader(
                output_dir=self.config['paths']['noaa_data']
            ),
            'gfed': GFEDDownloader(
                output_dir=self.config['paths']['gfed_data']
            ),
            'modis': MODISDownloader(
                output_dir=self.config['paths']['modis_data']
            )
        }

        self.processors = {
            'gfed': GFEDProcessor(
                data_dir=self.config['paths']['gfed_data'],
                output_dir=self.config['paths']['processed_gfed']
            ),
            'diurnal': DiurnalProcessor(
                config_file=self.config.get('diurnal_config')
            ),
            'cardamom': CARDAMOMProcessor(
                output_dir=self.config['paths']['cardamom_output']
            )
        }

    def get_downloader(self, source):
        """Get downloader instance for specified source"""
        if source not in self.downloaders:
            raise ValueError(f"Unknown data source: {source}")
        return self.downloaders[source]

    def get_processor(self, processor_type):
        """Get processor instance for specified type"""
        if processor_type not in self.processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
        return self.processors[processor_type]

    # Single-task operations (no orchestration)
    def download_era5_data(self, variables, years, months):
        """Download ERA5 data - single operation"""
        downloader = self.get_downloader('ecmwf')
        return downloader.download_data(variables, years, months)

    def download_gfed_data(self, year):
        """Download GFED data for single year - single operation"""
        downloader = self.get_downloader('gfed')
        return downloader.download_yearly_file(year)

    def process_gfed_year(self, year):
        """Process GFED data for single year - single operation"""
        processor = self.get_processor('gfed')
        return processor.process_year_data(year)

    def generate_monthly_netcdf(self, variable_group, year, months):
        """Generate monthly NetCDF files - single operation"""
        processor = self.get_processor('cardamom')
        return processor.create_monthly_files(variable_group, year, months)
```

### Individual Component Operations
```python
def download_noaa_co2(self, years):
    """Download NOAA CO2 data - single operation"""
    downloader = self.get_downloader('noaa')
    self.logger.info(f"Downloading NOAA CO2 data for {years}")

    try:
        result = downloader.download_raw_data(force_update=True)
        co2_data = downloader.parse_co2_data()
        return downloader.create_cardamom_co2_files(years, self.config['grid'])
    except Exception as e:
        self.logger.error(f"NOAA CO2 download failed: {e}")
        raise

def download_modis_land_sea_mask(self, resolution="0.5deg"):
    """Download MODIS land-sea mask - single operation"""
    downloader = self.get_downloader('modis')
    self.logger.info(f"Downloading MODIS land-sea mask at {resolution}")

    try:
        return downloader.download_land_sea_mask(resolution)
    except Exception as e:
        self.logger.error(f"MODIS land-sea mask download failed: {e}")
        raise

def process_diurnal_conus(self, year, months, experiment_numbers):
    """Process CONUS diurnal fluxes - single operation"""
    processor = self.get_processor('diurnal')
    self.logger.info(f"Processing CONUS diurnal fluxes for {year}")

    try:
        return processor.process_conus_diurnal(year, months, experiment_numbers)
    except Exception as e:
        self.logger.error(f"Diurnal processing failed: {e}")
        raise

def calculate_vpd_from_era5(self, year, months):
    """Calculate VPD from ERA5 temperature and dewpoint - single operation"""
    processor = self.get_processor('cardamom')
    self.logger.info(f"Calculating VPD for {year}")

    try:
        return processor.calculate_vpd(year, months)
    except Exception as e:
        self.logger.error(f"VPD calculation failed: {e}")
        raise
```

## 6.2 Configuration and Utilities (`config_utilities.py`)

### Configuration Loading
```python
class ComponentConfig:
    """Simple configuration management for individual components"""

    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)

    def _load_config(self, config_file):
        """Load configuration from file or use defaults"""
        # Import unified configuration from Phase 1
        from config_manager import CardamomConfig
        return CardamomConfig(config_file)

    def get_downloader_config(self, source):
        """Get configuration for specific downloader"""
        return self.config.get(f'downloaders.{source}', {})

    def get_processor_config(self, processor_type):
        """Get configuration for specific processor"""
        return self.config.get(f'processors.{processor_type}', {})

    def get_output_config(self):
        """Get output configuration"""
        return self.config.get('output', {})
```

### Error Handling Utilities
```python
class ComponentErrorHandler:
    """Simple error handling for individual component operations"""

    def __init__(self, logger=None):
        self.logger = logger or self._setup_default_logger()

    def _setup_default_logger(self):
        """Setup default logging for component operations"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def handle_component_error(self, component, operation, error):
        """Handle errors from individual component operations"""
        error_msg = f"{component} {operation} failed: {error}"
        self.logger.error(error_msg)

        # Simple error classification
        if isinstance(error, (ConnectionError, TimeoutError)):
            self.logger.info("Network-related error - operation may be retried")
        elif isinstance(error, FileNotFoundError):
            self.logger.info("File not found - check input data availability")
        elif isinstance(error, MemoryError):
            self.logger.info("Memory error - consider processing smaller time ranges")

        raise error  # Re-raise for caller to handle

    def log_operation_start(self, component, operation, params):
        """Log the start of a component operation"""
        self.logger.info(f"Starting {component} {operation} with params: {params}")

    def log_operation_success(self, component, operation, result_summary):
        """Log successful completion of component operation"""
        self.logger.info(f"Completed {component} {operation}: {result_summary}")
```

## 6.3 Usage Examples

### Individual Component Operations
```python
# Example usage of individual components for MAAP jobs

def main():
    """Example main function for a single MAAP job"""
    components = CARDAMOMComponents()

    # Example 1: Download ERA5 data for specific variables and time range
    era5_result = components.download_era5_data(
        variables=['2m_temperature', '2m_dewpoint_temperature'],
        years=[2020],
        months=[1, 2, 3]
    )

    # Example 2: Process GFED data for a single year
    gfed_result = components.process_gfed_year(2020)

    # Example 3: Calculate VPD from ERA5 data
    vpd_result = components.calculate_vpd_from_era5(2020, [1, 2, 3])

    # Example 4: Generate NetCDF files for specific variable group
    netcdf_result = components.generate_monthly_netcdf(
        'temperature', 2020, [1, 2, 3]
    )

if __name__ == "__main__":
    main()
```

### CLI Integration Examples
```python
# Example CLI commands that call individual components

@click.command()
@click.option('--year', type=int, required=True)
@click.option('--months', type=str, default='1-12')
def download_era5_temperature(year, months):
    """Download ERA5 temperature data for specific year/months"""
    components = CARDAMOMComponents()
    month_list = parse_month_range(months)

    result = components.download_era5_data(
        variables=['2m_temperature'],
        years=[year],
        months=month_list
    )

    click.echo(f"Downloaded ERA5 temperature data for {year}: {result}")

@click.command()
@click.option('--year', type=int, required=True)
def process_gfed_single_year(year):
    """Process GFED data for a single year"""
    components = CARDAMOMComponents()

    result = components.process_gfed_year(year)

    click.echo(f"Processed GFED data for {year}: {result}")
```

## 6.4 Testing Framework

### Component Testing
```python
def test_individual_components():
    """Test individual component operations"""

def test_era5_download():
    """Test ERA5 downloader component"""
    components = CARDAMOMComponents()
    result = components.download_era5_data(
        variables=['2m_temperature'],
        years=[2020],
        months=[1]
    )
    assert result is not None

def test_gfed_processing():
    """Test GFED processor component"""
    components = CARDAMOMComponents()
    result = components.process_gfed_year(2020)
    assert result is not None

def test_vpd_calculation():
    """Test VPD calculation component"""
    components = CARDAMOMComponents()
    result = components.calculate_vpd_from_era5(2020, [1])
    assert result is not None
```

## 6.5 Configuration Integration

**Note**: Phase 6 uses the unified `CardamomConfig` system from Phase 1 for all configuration management.

### Simple Configuration Usage
```python
# Import unified configuration system from Phase 1
from config_manager import CardamomConfig

class CARDAMOMComponents:
    """Uses unified configuration system from Phase 1"""

    def __init__(self, config_file=None, cli_args=None):
        # Use unified configuration system
        self.config = CardamomConfig(config_file, cli_args)
        self.logger = self._setup_logging()

        # Initialize individual components (no complex managers)
        self._initialize_components()

    def _setup_logging(self):
        """Setup logging using unified configuration"""
        log_level = self.config.get('processing.log_level', 'INFO')
        return setup_cardamom_logging(log_level)

    def get_component_config(self, component_type, component_name):
        """Get configuration for specific component"""
        return self.config.get(f'{component_type}.{component_name}', {})
```

### Simplified Configuration Access
- **No pipeline state management** - platform handles job state
- **No complex orchestration config** - each job is independent
- **Direct component configuration** - components get their own config sections
- **Unified system** - same CardamomConfig used throughout all phases

## 6.6 Success Criteria

### Functional Requirements
- [ ] Provide simple access to all CARDAMOM components
- [ ] Support individual component operations without orchestration
- [ ] Enable single-purpose MAAP jobs for each operation
- [ ] Maintain integration with existing CARDAMOM infrastructure

### Simplicity Requirements
- [ ] Remove complex pipeline orchestration and state management
- [ ] Eliminate internal parallelism (leverage MAAP platform parallelism)
- [ ] Provide clear, single-task operations for each component
- [ ] Use unified configuration system from Phase 1

### Platform Integration Requirements
- [ ] Design components to run as independent MAAP jobs
- [ ] Remove job coordination and dependency management
- [ ] Support platform-handled parallelism and scaling
- [ ] Enable external job submission and management

### Quality Requirements
- [ ] Simple error handling and logging for individual operations
- [ ] Clear documentation and usage examples for each component
- [ ] Basic data validation within individual components
- [ ] Integration testing for individual component operations

## 6.7 Testing Framework

### Individual Component Tests
```
tests/components/
├── test_component_access.py
├── test_individual_downloaders.py
├── test_individual_processors.py
├── test_configuration_integration.py
└── fixtures/
    ├── sample_configs/
    ├── test_datasets/
    └── expected_outputs/
```

### Component Operation Tests
```python
def test_era5_download_operation():
    """Test ERA5 download as individual operation"""

def test_gfed_processing_operation():
    """Test GFED processing as individual operation"""

def test_vpd_calculation_operation():
    """Test VPD calculation as individual operation"""

def test_netcdf_generation_operation():
    """Test NetCDF generation as individual operation"""
```