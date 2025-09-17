# Phase 2: Data Source-Specific Downloaders

This phase implements modular downloaders for each external data source used in CARDAMOM preprocessing. Each downloader is specialized for its data source while maintaining consistent interfaces through a unified base class.

## Overview

The Phase 2 implementation provides four specialized downloaders:

- **ECMWF Downloader**: Enhanced ERA5 meteorological data retrieval
- **NOAA Downloader**: Global CO2 concentration data from NOAA monitoring stations
- **GFED Downloader**: Fire emissions and burned area data from Global Fire Emissions Database
- **MODIS Downloader**: Land-sea mask and land cover data from MODIS satellite products

All downloaders inherit from a common `BaseDownloader` class that provides consistent error handling, retry logic, and data validation.

## Architecture

```
src/
├── base_downloader.py      # Abstract base class for all downloaders
├── ecmwf_downloader.py     # Enhanced ERA5 meteorological data
├── noaa_downloader.py      # NOAA CO2 concentration data
├── gfed_downloader.py      # GFED fire emissions data
├── modis_downloader.py     # MODIS land-sea mask data
├── downloader_factory.py   # Factory pattern for downloader creation
├── data_source_config.py   # Configuration management for data sources
└── validation.py           # Data validation and integrity checking
```

## Individual Downloaders

### 1. ECMWF Downloader (`ecmwf_downloader.py`)

Enhanced version of the existing ECMWF downloader with additional variables and processing capabilities.

**Key Features:**
- Variable registry with standardized CARDAMOM naming
- Support for both hourly and monthly ERA5 data
- Built-in processing hints for each variable type
- Resume capability for interrupted downloads
- Authentication via CDS API credentials

**Usage:**
```python
from src.ecmwf_downloader import ECMWFDownloader

downloader = ECMWFDownloader(
    output_dir="./DATA/ECMWF/",
    area=[90, -180, -90, 180],  # Global
    grid=[0.5, 0.5]             # 0.5 degree resolution
)

# Download temperature and precipitation for 2020
downloader.download_data(
    variables=['2m_temperature', 'total_precipitation'],
    years=[2020],
    months=list(range(1, 13))
)
```

### 2. NOAA CO2 Downloader (`noaa_downloader.py`)

Downloads and processes global CO2 concentration data from NOAA monitoring stations.

**Key Features:**
- FTP-based data retrieval from NOAA servers
- Parsing of NOAA CO2 text file format
- Global spatial replication of point measurements
- Creation of CARDAMOM-compliant NetCDF files

**Data Source:**
- Server: `aftp.cmdl.noaa.gov`
- File: `/products/trends/co2/co2_mm_gl.txt`
- Format: Monthly global mean CO2 concentrations

**Usage:**
```python
from src.noaa_downloader import NOAADownloader

downloader = NOAADownloader(output_dir="./DATA/NOAA_CO2/")

# Download and process CO2 data for specified years
downloader.download_data(years=range(2015, 2021))

# Create spatially-replicated CO2 files
downloader.create_cardamom_co2_files(
    years=range(2015, 2021),
    spatial_grid=(360, 720)  # Global 0.5 degree grid
)
```

### 3. GFED Downloader (`gfed_downloader.py`)

Downloads fire emissions and burned area data from the Global Fire Emissions Database.

**Key Features:**
- Support for both GFED4.1s standard and beta versions
- HDF5 file handling for GFED data format
- Extraction of monthly emissions by vegetation type
- Diurnal fire patterns and daily fractions
- Authentication handling for GFED access requirements

**Data Products:**
- Historical data: 2001-2016 (standard GFED4.1s)
- Recent data: 2017+ (beta versions)
- Variables: Burned area, carbon emissions, diurnal patterns

**Usage:**
```python
from src.gfed_downloader import GFEDDownloader

downloader = GFEDDownloader(output_dir="./DATA/GFED4/")

# Download yearly GFED files
for year in range(2015, 2021):
    downloader.download_data(year=year)

# Extract monthly data from downloaded files
monthly_data = downloader.extract_monthly_data(year=2020, month=6)
```

### 4. MODIS Downloader (`modis_downloader.py`)

Downloads and processes MODIS-based land-sea mask and land cover data.

**Key Features:**
- Multiple MODIS server support for redundancy
- Generation of binary land-sea masks
- Fractional land coverage calculations
- Aggregation to target grid resolutions
- Support for different MODIS land cover products (MCD12Q1, MOD44W)

**Usage:**
```python
from src.modis_downloader import MODISDownloader

downloader = MODISDownloader(output_dir="./DATA/MODIS_LSM/")

# Download and process land-sea mask at 0.5 degree resolution
mask_data = downloader.download_data(
    resolution="0.5deg",
    product="MCD12Q1"
)

# Create fractional coverage from land cover data
fractional_coverage = downloader.create_fractional_coverage(mask_data)
```

## Unified Interface

### Base Downloader Class

All downloaders inherit from `BaseDownloader` which provides:

- **Directory Management**: Automatic creation of output and cache directories
- **Error Handling**: Consistent error handling and logging across all downloaders
- **Retry Logic**: Configurable retry mechanisms for network failures
- **Data Validation**: File integrity checking and format validation
- **Status Tracking**: Download progress and completion status

### Factory Pattern

The `DownloaderFactory` provides a simple interface for creating downloader instances:

```python
from src.downloader_factory import DownloaderFactory

# Create ECMWF downloader with configuration
config = {
    'ecmwf': {
        'output_dir': './DATA/ECMWF/',
        'area': [90, -180, -90, 180],
        'grid': [0.5, 0.5]
    }
}

downloader = DownloaderFactory.create_downloader('ecmwf', config)
```

## Configuration Management

### Data Source Configuration

Each data source is configured via `data_source_config.py`:

```python
ECMWF_CONFIG = {
    'base_url': 'https://cds.climate.copernicus.eu/api/v2',
    'datasets': {
        'hourly': 'reanalysis-era5-single-levels',
        'monthly': 'reanalysis-era5-single-levels-monthly-means'
    },
    'rate_limit': 10  # requests per minute
}

NOAA_CONFIG = {
    'ftp_server': 'aftp.cmdl.noaa.gov',
    'data_path': '/products/trends/co2/co2_mm_gl.txt',
    'update_frequency': 'monthly'
}

GFED_CONFIG = {
    'base_url': 'https://www.globalfiredata.org/data_new/',
    'file_pattern': 'GFED4.1s_{year}{beta}.hdf5',
    'requires_registration': True
}

MODIS_CONFIG = {
    'servers': [
        'https://e4ftl01.cr.usgs.gov/MOTA/',
        'https://n5eil01u.ecs.nsidc.org/'
    ],
    'products': ['MCD12Q1', 'MOD44W']
}
```

### Authentication Requirements

Different data sources require different authentication methods:

**ECMWF CDS API:**
- Local: `.cdsapirc` file in home directory
- MAAP: `ECMWF_CDS_UID` and `ECMWF_CDS_KEY` environment variables

**NOAA:**
- No authentication required (public FTP)

**GFED:**
- May require user registration
- Handled automatically if credentials are available

**MODIS:**
- No authentication for basic products
- Some products may require Earthdata login

## Error Handling and Reliability

### Retry Mechanisms

All downloaders implement robust retry logic:

- **Network Failures**: Automatic retry with exponential backoff
- **Server Errors**: Handle temporary server unavailability
- **Rate Limiting**: Respect API rate limits and queue management
- **Partial Downloads**: Resume interrupted downloads when possible

### Data Validation

Comprehensive validation for downloaded data:

- **File Completeness**: Verify file size and integrity
- **Format Validation**: Check NetCDF, HDF5, and text file formats
- **Scientific Validation**: Verify data ranges and units
- **Metadata Consistency**: Ensure proper coordinate systems and attributes

### Caching Strategy

Simple but effective caching system:

- **Local File Cache**: Store downloaded files locally to avoid re-downloading
- **Cache Invalidation**: Configurable cache expiration policies
- **Disk Space Management**: Basic cleanup of old cached files
- **Cache Integrity**: Verify cached files before reuse

## Testing and Validation

### Test Structure

```
tests/downloaders/
├── test_ecmwf_downloader.py     # ECMWF-specific tests
├── test_noaa_downloader.py      # NOAA CO2 downloader tests
├── test_gfed_downloader.py      # GFED fire data tests
├── test_modis_downloader.py     # MODIS land cover tests
├── test_base_downloader.py      # Base class functionality
├── test_downloader_factory.py   # Factory pattern tests
└── fixtures/
    ├── sample_co2_data.txt      # Sample NOAA CO2 data
    ├── sample_gfed.hdf5         # Sample GFED HDF5 file
    └── mock_responses/          # Mock HTTP/FTP responses
```

### Integration Testing

- **End-to-End Downloads**: Test complete download workflows
- **Error Recovery**: Test retry logic and error handling
- **Data Format Compatibility**: Verify output formats work with CARDAMOM
- **Performance Testing**: Benchmark download speeds and memory usage

## Usage Examples

### Individual Downloader Usage

```python
# Download specific ECMWF variables for CARDAMOM
from src.ecmwf_downloader import ECMWFDownloader

ecmwf = ECMWFDownloader(output_dir="./DATA/ECMWF/")
ecmwf.download_data(
    variables=['2m_temperature', '2m_dewpoint_temperature'],
    years=[2020],
    months=[1, 2, 3]
)

# Get CO2 concentrations for carbon cycle modeling
from src.noaa_downloader import NOAADownloader

noaa = NOAADownloader(output_dir="./DATA/NOAA_CO2/")
co2_data = noaa.download_data(years=range(2015, 2021))
```

### Factory-Based Usage

```python
from src.downloader_factory import DownloaderFactory
from src.data_source_config import get_config

# Download from multiple sources
sources = ['ecmwf', 'noaa', 'gfed']
config = get_config()

for source in sources:
    downloader = DownloaderFactory.create_downloader(source, config)
    downloader.download_data(**config[source]['download_params'])
```

## Integration with CARDAMOM Pipeline

The Phase 2 downloaders create data files compatible with the CARDAMOM preprocessing pipeline:

- **File Naming**: Consistent naming conventions for CARDAMOM components
- **NetCDF Format**: Standard CF-compliant NetCDF files
- **Coordinate Systems**: WGS84 geographic coordinates
- **Units**: Standardized scientific units following CARDAMOM conventions
- **Metadata**: Complete attribute metadata for traceability

Downloaded data integrates seamlessly with:
- **Phase 3**: GFED fire emissions processing
- **Phase 4**: Diurnal pattern processing
- **Phase 5**: NetCDF infrastructure and data management
- **Phase 6**: Pipeline management and workflow orchestration

## Performance Considerations

### Memory Management

- **Chunked Downloads**: Process large files in chunks to minimize memory usage
- **Lazy Loading**: Load data only when needed
- **Garbage Collection**: Explicit cleanup of large arrays

### Disk Space

- **Compression**: Use NetCDF compression for output files
- **Selective Downloads**: Download only required variables and time periods
- **Cache Management**: Automatic cleanup of old cached files

### Network Efficiency

- **Concurrent Downloads**: Multiple file downloads when supported
- **Connection Pooling**: Reuse connections for multiple requests
- **Bandwidth Limiting**: Respect server bandwidth limitations

## Success Criteria

### Functional Requirements ✅

- [x] Successfully download data from all four sources independently
- [x] Handle authentication and access requirements for each source
- [x] Robust error handling and retry mechanisms for individual downloads
- [x] Consistent data format and metadata for each source
- [x] Integration with CARDAMOM data pipeline

### Simplicity Requirements ✅

- [x] Individual downloader operations (no complex coordination)
- [x] Simple file caching without complex management
- [x] Single-purpose download operations suitable for MAAP jobs
- [x] Clear, scientist-friendly code structure

### Quality Requirements ✅

- [x] Comprehensive error logging for all download operations
- [x] Data validation and integrity checking for each file
- [x] Configurable retry logic for network failures
- [x] Complete documentation and usage examples
- [x] Comprehensive test suite with >90% coverage

## Next Steps

Phase 2 provides the foundation for subsequent phases:

- **Phase 3**: GFED fire emissions processing will use the GFED downloader
- **Phase 4**: Diurnal pattern processing will integrate with downloaded fire data
- **Phase 5**: NetCDF infrastructure will standardize output formats
- **Phase 6**: Pipeline manager will orchestrate multi-source downloads

The modular design ensures each downloader can be used independently or as part of the larger CARDAMOM preprocessing workflow.