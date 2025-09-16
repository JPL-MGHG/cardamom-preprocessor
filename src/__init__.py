"""
CARDAMOM Preprocessor Core Modules

This package contains the core infrastructure for CARDAMOM data preprocessing,
including data orchestration, coordinate systems, NetCDF handling, scientific
utilities, and configuration management.
"""

__version__ = "1.0.0"
__author__ = "CARDAMOM Development Team"

# Core module imports for convenient access
from .cardamom_preprocessor import CARDAMOMProcessor
from .config_manager import CardamomConfig
from .coordinate_systems import CoordinateGrid
from .netcdf_infrastructure import CARDAMOMNetCDFWriter
from .scientific_utils import calculate_vapor_pressure_deficit, convert_precipitation_units

__all__ = [
    'CARDAMOMProcessor',
    'CardamomConfig',
    'CoordinateGrid',
    'CARDAMOMNetCDFWriter',
    'calculate_vapor_pressure_deficit',
    'convert_precipitation_units'
]