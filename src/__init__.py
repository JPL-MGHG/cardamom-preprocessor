"""
CARDAMOM Preprocessor - Core and Data Downloader Modules

This package contains the core infrastructure for CARDAMOM data preprocessing,
including data orchestration, coordinate systems, NetCDF handling, scientific
utilities, configuration management, and data source-specific downloaders.

Phase 1: Core Framework
- Data orchestration and coordinate systems
- NetCDF infrastructure and scientific utilities
- Configuration and validation management

Phase 2: Data Source-Specific Downloaders
- ECMWF meteorological data downloader
- NOAA CO2 concentration downloader
- GFED fire emissions downloader
- MODIS land-sea mask generator
- Factory pattern and retry management
- Simple caching and configuration
"""

__version__ = "2.0.0"
__author__ = "CARDAMOM Development Team"

# Phase 1: Core module imports
from .cardamom_preprocessor import CARDAMOMProcessor
from .config_manager import CardamomConfig
from .coordinate_systems import CoordinateGrid
from .netcdf_infrastructure import CARDAMOMNetCDFWriter
from .scientific_utils import calculate_vapor_pressure_deficit, convert_precipitation_units

# Phase 2: Downloader module imports (with graceful fallback for missing dependencies)
try:
    from .base_downloader import BaseDownloader
    from .ecmwf_downloader import ECMWFDownloader
    from .noaa_downloader import NOAADownloader
    from .gfed_downloader import GFEDDownloader, GFEDReader
    from .modis_downloader import MODISDownloader
    from .downloader_factory import DownloaderFactory, RetryManager
    from .data_source_config import DataSourceConfig

    # All components available
    __all__ = [
        # Phase 1 components
        'CARDAMOMProcessor',
        'CardamomConfig',
        'CoordinateGrid',
        'CARDAMOMNetCDFWriter',
        'calculate_vapor_pressure_deficit',
        'convert_precipitation_units',
        # Phase 2 components
        'BaseDownloader',
        'ECMWFDownloader',
        'NOAADownloader',
        'GFEDDownloader',
        'GFEDReader',
        'MODISDownloader',
        'DownloaderFactory',
        'RetryManager',
        'DataSourceConfig'
    ]

except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Phase 2 downloaders unavailable due to missing dependencies: {e}")

    # Fall back to Phase 1 components only
    __all__ = [
        'CARDAMOMProcessor',
        'CardamomConfig',
        'CoordinateGrid',
        'CARDAMOMNetCDFWriter',
        'calculate_vapor_pressure_deficit',
        'convert_precipitation_units'
    ]