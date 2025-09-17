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

Phase 8: Scientific Functions Library
- Atmospheric science calculations (VPD, humidity, radiation)
- Statistical utilities (temporal aggregation, spatial interpolation)
- Physical constants and unit conversions
- Carbon cycle modeling functions
- Enhanced data quality control
"""

__version__ = "2.0.0"
__author__ = "CARDAMOM Development Team"

# Phase 1: Core module imports
from cardamom_preprocessor import CARDAMOMProcessor
from config_manager import CardamomConfig
from coordinate_systems import CoordinateGrid
from netcdf_infrastructure import CARDAMOMNetCDFWriter
from scientific_utils import calculate_vapor_pressure_deficit, convert_precipitation_units

# Phase 8: Scientific Functions Library imports
from atmospheric_science import saturation_pressure_water_matlab, calculate_vapor_pressure_deficit_matlab
from statistics_utils import nan_to_zero, monthly_to_annual, find_closest_grid_points
from units_constants import PhysicalConstants, temperature_celsius_to_kelvin
from carbon_cycle import calculate_net_ecosystem_exchange, validate_carbon_flux_mass_balance
from quality_control import validate_temperature_range_extended, DataQualityReport

# Phase 2: Downloader module imports (with graceful fallback for missing dependencies)
try:
    from base_downloader import BaseDownloader
    from ecmwf_downloader import ECMWFDownloader
    from noaa_downloader import NOAADownloader
    from gfed_downloader import GFEDDownloader, GFEDReader
    from modis_downloader import MODISDownloader
    from downloader_factory import DownloaderFactory, RetryManager
    from data_source_config import DataSourceConfig

    # All components available
    __all__ = [
        # Phase 1 components
        'CARDAMOMProcessor',
        'CardamomConfig',
        'CoordinateGrid',
        'CARDAMOMNetCDFWriter',
        'calculate_vapor_pressure_deficit',
        'convert_precipitation_units',
        # Phase 8 components
        'saturation_pressure_water_matlab',
        'calculate_vapor_pressure_deficit_matlab',
        'nan_to_zero',
        'monthly_to_annual',
        'find_closest_grid_points',
        'PhysicalConstants',
        'temperature_celsius_to_kelvin',
        'calculate_net_ecosystem_exchange',
        'validate_carbon_flux_mass_balance',
        'validate_temperature_range_extended',
        'DataQualityReport',
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
        'convert_precipitation_units',
        # Phase 8 components
        'saturation_pressure_water_matlab',
        'calculate_vapor_pressure_deficit_matlab',
        'nan_to_zero',
        'monthly_to_annual',
        'find_closest_grid_points',
        'PhysicalConstants',
        'temperature_celsius_to_kelvin',
        'calculate_net_ecosystem_exchange',
        'validate_carbon_flux_mass_balance',
        'validate_temperature_range_extended',
        'DataQualityReport'
    ]