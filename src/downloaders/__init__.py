"""
CARDAMOM Downloaders Package

This package contains independent downloaders for each meteorological data source
(ERA5, NOAA, GFED). Each downloader is a standalone CLI tool that produces:
1. Analysis-ready NetCDF files in standard format
2. STAC Item metadata describing the output files

Scientific Context:
The decoupled architecture allows each downloader to run independently on fresh
compute instances without shared filesystem requirements, suitable for distributed
processing systems like NASA MAAP.
"""

from downloaders.base import BaseDownloader

__all__ = ['BaseDownloader']
