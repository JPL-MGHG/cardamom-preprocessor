"""
CARDAMOM Preprocessor - STAC-Based Data Pipeline

This package contains the STAC-based data preprocessing pipeline for CARDAMOM,
including modular downloaders, STAC catalog management, meteorology loading,
and CBF file generation.

STAC-Based Workflow:
- STAC catalog management and data discovery
- Modular downloaders (ECMWF, NOAA, GFED)
- Meteorology loading from STAC catalogs
- CBF file generation with observational data handling

Scientific Utilities:
- Atmospheric science calculations (VPD, humidity, radiation)
- Statistical utilities (temporal aggregation, spatial interpolation)
- Physical constants and unit conversions
- Carbon cycle modeling functions
- Data quality control
"""

__version__ = "3.0.0"  # Updated for STAC-based architecture
__author__ = "CARDAMOM Development Team"
