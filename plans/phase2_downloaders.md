# Phase 2: Data Source-Specific Downloaders

## Overview
Create modular downloaders for each external data source used in CARDAMOM preprocessing. Each downloader is specialized for its data source while maintaining consistent interfaces.

## 2.1 Enhanced ECMWF Downloader (`ecmwf_downloader.py`)

### Enhancements to Existing Code
```python
class ECMWFDownloader:
    """Enhanced version of existing ECMWFDownloader with additional variables"""

    def __init__(self, area=None, grid=None, data_format="netcdf",
                 download_format="unarchived", output_dir="."):
        # Existing initialization code
        self.variable_registry = self._setup_variable_registry()

    def _setup_variable_registry(self):
        """Registry of ERA5 variables with metadata"""
        return {
            "2m_temperature": {
                "cardamom_name": "T2M",
                "units": "K",
                "processing": "min_max_monthly"
            },
            "2m_dewpoint_temperature": {
                "cardamom_name": "D2M",
                "units": "K",
                "processing": "hourly_averaged"
            },
            "surface_thermal_radiation_downwards": {
                "cardamom_name": "STRD",
                "units": "J m-2",
                "processing": "monthly_mean"
            },
            # Add missing variables from MATLAB script
        }

    def download_with_processing(self, variables, years, months, processing_type):
        """Download and apply basic processing during download"""
```

### New Features
- **Variable Registry**: Standardized mapping between ERA5 and CARDAMOM variable names
- **Processing Hints**: Metadata about how each variable should be processed
- **Validation**: Check variable availability before attempting downloads
- **Resume Capability**: Enhanced file checking and partial download recovery

## 2.2 NOAA CO2 Downloader (`noaa_downloader.py`)

### Core Functionality
```python
class NOAADownloader:
    """
    Download and process NOAA global CO2 concentration data.
    Source: ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_gl.txt
    """

    def __init__(self, output_dir="./DATA/NOAA_CO2/"):
        self.output_dir = output_dir
        self.ftp_server = "aftp.cmdl.noaa.gov"
        self.data_path = "/products/trends/co2/co2_mm_gl.txt"
        self.cache_file = os.path.join(output_dir, "co2_mm_gl.txt")

    def download_raw_data(self, force_update=False):
        """Download raw CO2 text file from NOAA FTP server"""

    def parse_co2_data(self):
        """
        Parse NOAA CO2 text file format.
        Returns structured data with year, month, CO2 concentration.
        """

    def create_cardamom_co2_files(self, years, spatial_grid):
        """
        Create CARDAMOM-compliant NetCDF files with spatially-replicated CO2.
        Matches MATLAB logic from lines 164-176.
        """

    def get_co2_for_period(self, start_year, end_year):
        """Get CO2 concentrations for specified time period"""
```

### Data Processing Details
```python
def parse_noaa_co2_text(self, filepath):
    """
    Parse NOAA CO2 text file with format:
    # year  month  decimal_date  average  interpolated  trend  #days
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            data.append({
                'year': int(parts[0]),
                'month': int(parts[1]),
                'decimal_date': float(parts[2]),
                'co2_ppm': float(parts[3]) if parts[3] != '-99.99' else None,
                'interpolated': float(parts[4]) if parts[4] != '-99.99' else None
            })
    return data

def replicate_globally(self, co2_timeseries, spatial_shape):
    """
    Replicate point CO2 measurements across global grid.
    Matches MATLAB: repmat(permute(NOAACO2.data), spatial_dims)
    """
```

## 2.3 GFED Downloader (`gfed_downloader.py`)

### Main Class
```python
class GFEDDownloader:
    """
    Download GFED4.1s burned area data from Global Fire Emissions Database.
    Handles both historical (2001-2016) and beta versions (2017+).
    """

    def __init__(self, output_dir="./DATA/GFED4/", cache_dir="./cache/"):
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.base_url = "https://www.globalfiredata.org/data_new/"
        self.available_years = self._check_available_years()

    def download_yearly_file(self, year):
        """
        Download GFED4.1s HDF5 file for specific year.
        Format: GFED4.1s_YYYY.hdf5 or GFED4.1s_YYYY_beta.hdf5
        """

    def get_file_url(self, year):
        """Construct download URL based on year (beta vs standard)"""

    def verify_file_integrity(self, filepath):
        """Verify downloaded HDF5 file is complete and readable"""
```

### HDF5 Data Extraction
```python
class GFEDReader:
    """Read and extract data from GFED HDF5 files"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.h5file = h5py.File(filepath, 'r')

    def extract_monthly_data(self, year, month):
        """
        Extract burned area and emissions data for specific month.
        Reads HDF5 structure: /emissions/MM/partitioning/DM_TYPE
        """

    def get_diurnal_patterns(self, year, month):
        """
        Extract diurnal fire patterns for month.
        Reads: /emissions/MM/diurnal_cycle/UTC_H-Hh
        """

    def get_daily_fractions(self, year, month):
        """
        Extract daily fire fractions for month.
        Reads: /emissions/MM/daily_fraction/day_D
        """

    def get_vegetation_types(self):
        """Get vegetation type classifications (SAVA, BORF, TEMF, etc.)"""
```

### Authentication and Access
```python
def setup_gfed_access(self):
    """
    Setup access to GFED data. May require user registration.
    Handle authentication if required by GFED servers.
    """

def check_data_availability(self, years):
    """Check which years of GFED data are available for download"""
```

## 2.4 MODIS Land-Sea Mask Downloader (`modis_downloader.py`)

### Core Functionality
```python
class MODISDownloader:
    """
    Download MODIS-based land-sea mask data.
    Creates both binary mask and fractional coverage datasets.
    """

    def __init__(self, output_dir="./DATA/MODIS_LSM/"):
        self.output_dir = output_dir
        self.modis_servers = self._setup_server_list()

    def download_land_sea_mask(self, resolution="0.5deg"):
        """
        Download or generate MODIS land-sea mask at specified resolution.
        Creates both binary mask (land=1, sea=0) and fractional coverage.
        """

    def generate_mask_from_modis(self, modis_product="MCD12Q1"):
        """Generate land-sea mask from MODIS land cover product"""

    def create_fractional_coverage(self, land_cover_data):
        """Create fractional land coverage from land cover classifications"""
```

### Data Processing
```python
def process_modis_land_cover(self, raw_data, target_resolution):
    """
    Process raw MODIS land cover to create land-sea classifications.
    Handle different MODIS land cover schemes (IGBP, UMD, etc.)
    """

def aggregate_to_resolution(self, high_res_data, target_resolution):
    """Aggregate high-resolution MODIS data to target grid resolution"""

def create_mask_and_fraction(self, land_cover_data):
    """
    Create both binary mask and fractional coverage.
    Matches MATLAB loadlandseamask() functionality.
    """
```

## 2.5 Unified Downloader Interface (`base_downloader.py`)

### Abstract Base Class
```python
from abc import ABC, abstractmethod

class BaseDownloader(ABC):
    """Abstract base class for all data downloaders"""

    def __init__(self, output_dir, cache_dir=None):
        self.output_dir = output_dir
        self.cache_dir = cache_dir or os.path.join(output_dir, "cache")
        self.setup_directories()

    @abstractmethod
    def download_data(self, **kwargs):
        """Download data - implemented by each downloader"""

    def setup_directories(self):
        """Create necessary output directories"""

    def check_existing_files(self, file_pattern):
        """Check for existing files to avoid re-downloading"""

    def validate_downloaded_data(self, filepath):
        """Validate downloaded data integrity"""

    def get_download_status(self):
        """Return status of download operations"""
```

### Individual Downloader Access
```python
class DownloaderFactory:
    """Simple factory to create individual downloader instances"""

    @staticmethod
    def create_downloader(source, config):
        """Create downloader instance for specific source"""
        if source == 'ecmwf':
            return ECMWFDownloader(**config.get('ecmwf', {}))
        elif source == 'noaa':
            return NOAADownloader(**config.get('noaa', {}))
        elif source == 'gfed':
            return GFEDDownloader(**config.get('gfed', {}))
        elif source == 'modis':
            return MODISDownloader(**config.get('modis', {}))
        else:
            raise ValueError(f"Unknown downloader source: {source}")

    @staticmethod
    def check_downloader_dependencies(source):
        """Check dependencies for specific downloader"""
        # Simple dependency checks for individual downloaders
        pass
```

## 2.6 Error Handling and Retry Logic

### Robust Download Mechanisms
```python
class RetryManager:
    """Handle download failures and retry logic"""

    def __init__(self, max_retries=3, backoff_factor=2):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def download_with_retry(self, download_func, *args, **kwargs):
        """Execute download function with retry logic"""

    def handle_network_errors(self, error):
        """Handle specific network-related errors"""

    def handle_server_errors(self, error):
        """Handle server-side errors (500, 503, etc.)"""
```

### Data Validation
```python
def validate_file_completeness(filepath, expected_size=None):
    """Validate that downloaded file is complete"""

def validate_file_format(filepath, expected_format):
    """Validate file format (NetCDF, HDF5, text)"""

def quarantine_corrupted_files(filepath, quarantine_dir):
    """Move corrupted files to quarantine directory"""
```

## 2.7 Simple Caching

### Basic File Caching
```python
class SimpleDataCache:
    """Simple local file caching without complex management"""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def is_cached(self, cache_key):
        """Check if file exists in cache"""
        cache_path = os.path.join(self.cache_dir, cache_key)
        return os.path.exists(cache_path)

    def get_cache_path(self, cache_key):
        """Get path to cached file"""
        return os.path.join(self.cache_dir, cache_key)

    def cache_file(self, source_path, cache_key):
        """Copy file to cache"""
        cache_path = self.get_cache_path(cache_key)
        shutil.copy2(source_path, cache_path)
        return cache_path
```

## 2.8 Configuration and Credentials

### Credentials Management
```python
class CredentialsManager:
    """Manage authentication credentials for different data sources"""

    def __init__(self):
        self.credentials = self._load_credentials()

    def get_ecmwf_credentials(self):
        """Get ECMWF CDS API credentials"""

    def get_gfed_credentials(self):
        """Get GFED access credentials if required"""

    def validate_credentials(self, source):
        """Validate credentials for specific data source"""
```

### Source Configuration
```yaml
# config/data_sources.yaml
ecmwf:
  base_url: "https://cds.climate.copernicus.eu/api/v2"
  datasets:
    hourly: "reanalysis-era5-single-levels"
    monthly: "reanalysis-era5-single-levels-monthly-means"
  rate_limit: 10  # requests per minute

noaa:
  ftp_server: "aftp.cmdl.noaa.gov"
  data_path: "/products/trends/co2/co2_mm_gl.txt"
  update_frequency: "monthly"

gfed:
  base_url: "https://www.globalfiredata.org/data_new/"
  file_pattern: "GFED4.1s_{year}{beta}.hdf5"
  requires_registration: true

modis:
  servers:
    - "https://e4ftl01.cr.usgs.gov/MOTA/"
    - "https://n5eil01u.ecs.nsidc.org/"
  products:
    - "MCD12Q1"  # Land cover
    - "MOD44W"   # Water mask
```

## 2.9 Testing and Validation

### Unit Tests
```
tests/downloaders/
├── test_ecmwf_downloader.py
├── test_noaa_downloader.py
├── test_gfed_downloader.py
├── test_modis_downloader.py
├── test_base_downloader.py
└── fixtures/
    ├── sample_co2_data.txt
    ├── sample_gfed.hdf5
    └── mock_responses/
```

### Integration Tests
- Test complete download workflows
- Validate data format compatibility
- Test error handling and recovery
- Performance testing with large datasets

## 2.10 Success Criteria

### Functional Requirements
- [ ] Successfully download data from all four sources independently
- [ ] Handle authentication and access requirements for each source
- [ ] Simple error handling and retry mechanisms for individual downloads
- [ ] Consistent data format and metadata for each source

### Simplicity Requirements
- [ ] Individual downloader operations (no coordination between sources)
- [ ] Remove complex parallel download management
- [ ] Simple file caching without intelligent management
- [ ] Single-purpose download operations suitable for MAAP jobs

### Quality Requirements
- [ ] Basic error logging for individual download operations
- [ ] Data validation and integrity checking for each file
- [ ] Simple retry logic for network failures
- [ ] Clear documentation and usage examples for individual downloaders