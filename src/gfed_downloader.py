#!/usr/bin/env python3
"""
GFED Downloader for CARDAMOM

Downloads GFED4.1s burned area data from Global Fire Emissions Database.
Handles both historical (2001-2016) and beta versions (2017+) with
HDF5 data extraction capabilities.
"""

import os
import requests
import h5py
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from .base_downloader import BaseDownloader


class GFEDDownloader(BaseDownloader):
    """
    Download GFED4.1s burned area data from Global Fire Emissions Database.

    Scientific Context:
    GFED provides global fire emissions and burned area data essential for
    CARDAMOM fire disturbance modeling. Includes vegetation-specific fire
    emissions, diurnal fire patterns, and daily fire fractions needed for
    realistic fire impact on ecosystem carbon cycling.

    Handles both historical (2001-2016) and beta versions (2017+).
    """

    def __init__(self, output_dir: str = "./DATA/GFED4/"):
        """
        Initialize GFED downloader.

        Args:
            output_dir: Directory for GFED data files
        """
        super().__init__(output_dir)

        # GFED server configuration
        self.base_url = "https://www.globalfiredata.org/data_new/"
        self.available_years = self._determine_available_years()

        # Standard GFED vegetation types
        self.vegetation_types = [
            'SAVA',  # Savanna
            'BORF',  # Boreal Forest
            'TEMF',  # Temperate Forest
            'DEFO',  # Deforestation
            'PEAT',  # Peat
            'AGRI'   # Agriculture
        ]

    def _determine_available_years(self) -> Dict[str, List[int]]:
        """
        Determine which years are available for download.

        Returns:
            dict: Available years categorized by version type
        """
        # Historical GFED4.1s: 2001-2016
        # Beta versions: 2017+ (may not be complete)
        return {
            'historical': list(range(2001, 2017)),  # 2001-2016
            'beta': list(range(2017, 2025))         # 2017+ (estimated)
        }

    def download_data(self, years: List[int], **kwargs) -> Dict[str, Any]:
        """
        Download GFED data for specified years.

        Args:
            years: List of years to download
            **kwargs: Additional parameters

        Returns:
            dict: Download results with file information
        """
        return self.download_yearly_files(years)

    def download_yearly_files(self, years: List[int]) -> Dict[str, Any]:
        """
        Download GFED4.1s HDF5 files for specified years.

        Args:
            years: List of years to download

        Returns:
            dict: Download results with file paths and status
        """
        downloaded_files = []
        failed_downloads = []

        for year in years:
            try:
                file_url = self.get_file_url(year)
                filename = self.get_filename(year)
                filepath = os.path.join(self.output_dir, filename)

                # Skip if file already exists and is valid
                if os.path.exists(filepath) and self.verify_file_integrity(filepath):
                    self.logger.info(f"Valid GFED file already exists: {filename}")
                    downloaded_files.append(filepath)
                    self._record_download_attempt(filename, "skipped")
                    continue

                # Download directly to output directory
                self.logger.info(f"Downloading GFED data for {year} from {file_url}")

                success = self._download_file(file_url, filepath)

                if success and self.verify_file_integrity(filepath):
                    downloaded_files.append(filepath)
                    self._record_download_attempt(filename, "success")
                    self.logger.info(f"Successfully downloaded GFED file: {filename}")
                else:
                    failed_downloads.append({"year": year, "error": "Download or validation failed"})
                    self._record_download_attempt(filename, "failed", "Download or validation failed")

            except Exception as e:
                error_msg = f"Failed to download GFED data for year {year}: {e}"
                self.logger.error(error_msg)
                failed_downloads.append({"year": year, "error": str(e)})
                self._record_download_attempt(self.get_filename(year), "failed", str(e))

        return {
            "status": "completed",
            "downloaded_files": downloaded_files,
            "failed_downloads": failed_downloads,
            "total_successful": len(downloaded_files),
            "total_failed": len(failed_downloads)
        }

    def get_file_url(self, year: int) -> str:
        """
        Construct download URL based on year (beta vs standard).

        Args:
            year: Year for which to construct URL

        Returns:
            str: Complete download URL
        """
        filename = self.get_filename(year)
        return f"{self.base_url}{filename}"

    def get_filename(self, year: int) -> str:
        """
        Get GFED filename for specific year.

        Args:
            year: Year for filename

        Returns:
            str: GFED filename
        """
        if year in self.available_years['historical']:
            return f"GFED4.1s_{year}.hdf5"
        else:
            return f"GFED4.1s_{year}_beta.hdf5"

    def _download_file(self, url: str, filepath: str) -> bool:
        """
        Download file from URL with progress tracking.

        Args:
            url: URL to download from
            filepath: Local path to save file

        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Log progress for large files
                        if total_size > 0 and downloaded_size % (10 * 1024 * 1024) == 0:  # Every 10MB
                            progress = (downloaded_size / total_size) * 100
                            self.logger.info(f"Download progress: {progress:.1f}%")

            self.logger.info(f"Download completed: {filepath} ({downloaded_size} bytes)")
            return True

        except Exception as e:
            self.logger.error(f"Download failed for {url}: {e}")
            # Clean up partial download
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    def verify_file_integrity(self, filepath: str) -> bool:
        """
        Verify downloaded HDF5 file is complete and readable.

        Args:
            filepath: Path to HDF5 file to verify

        Returns:
            bool: True if file is valid, False otherwise
        """
        # Basic file validation from base class
        if not self.validate_downloaded_data(filepath):
            return False

        # HDF5-specific validation
        try:
            with h5py.File(filepath, 'r') as f:
                # Check for essential GFED structure
                required_groups = ['emissions', 'ancill']

                for group in required_groups:
                    if group not in f:
                        self.logger.error(f"Missing required group '{group}' in GFED file: {filepath}")
                        return False

                # Check for at least one month of data
                emissions_group = f['emissions']
                month_groups = [key for key in emissions_group.keys() if len(key) == 2 and key.isdigit()]

                if not month_groups:
                    self.logger.error(f"No monthly emission data found in GFED file: {filepath}")
                    return False

                # Check for spatial dimensions in ancillary data
                if 'grid' in f['ancill']:
                    grid_group = f['ancill']['grid']
                    required_coords = ['latitude', 'longitude']

                    for coord in required_coords:
                        if coord not in grid_group:
                            self.logger.warning(f"Missing coordinate '{coord}' in GFED file: {filepath}")

                self.logger.info(f"GFED file validation passed: {filepath}")
                return True

        except Exception as e:
            self.logger.error(f"HDF5 validation failed for {filepath}: {e}")
            return False


class GFEDReader:
    """
    Read and extract data from GFED HDF5 files.

    Provides methods to extract monthly emissions, diurnal patterns,
    daily fractions, and vegetation type classifications from GFED
    HDF5 data structure.
    """

    def __init__(self, filepath: str):
        """
        Initialize GFED reader for specific HDF5 file.

        Args:
            filepath: Path to GFED HDF5 file
        """
        self.filepath = filepath
        self.logger = logging.getLogger(self.__class__.__name__)

        # Verify file exists and is readable
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GFED file not found: {filepath}")

        try:
            # Test file opening
            with h5py.File(filepath, 'r') as f:
                self.logger.info(f"Opened GFED file: {filepath}")
        except Exception as e:
            raise ValueError(f"Cannot read GFED HDF5 file {filepath}: {e}")

    def extract_monthly_data(self, year: int, month: int) -> Dict[str, np.ndarray]:
        """
        Extract burned area and emissions data for specific month.

        HDF5 structure: /emissions/MM/partitioning/DM_TYPE

        Args:
            year: Year of data
            month: Month (1-12)

        Returns:
            dict: Monthly fire data by vegetation type
        """
        month_str = f"{month:02d}"

        try:
            with h5py.File(self.filepath, 'r') as f:
                if 'emissions' not in f:
                    raise ValueError("No emissions group found in GFED file")

                emissions_group = f['emissions']

                if month_str not in emissions_group:
                    raise ValueError(f"No data found for month {month:02d}")

                month_group = emissions_group[month_str]
                monthly_data = {}

                # Extract burned area if available
                if 'burned_area' in month_group:
                    monthly_data['burned_area'] = month_group['burned_area'][:]

                # Extract partitioning data by vegetation type
                if 'partitioning' in month_group:
                    partitioning_group = month_group['partitioning']
                    monthly_data['partitioning'] = {}

                    for veg_type in partitioning_group.keys():
                        monthly_data['partitioning'][veg_type] = partitioning_group[veg_type][:]

                # Extract total emissions if available
                if 'total_emission' in month_group:
                    monthly_data['total_emission'] = month_group['total_emission'][:]

                self.logger.info(f"Extracted GFED data for {year}-{month:02d}")
                return monthly_data

        except Exception as e:
            self.logger.error(f"Failed to extract monthly data for {year}-{month:02d}: {e}")
            raise

    def get_diurnal_patterns(self, year: int, month: int) -> Dict[str, np.ndarray]:
        """
        Extract diurnal fire patterns for month.

        HDF5 structure: /emissions/MM/diurnal_cycle/UTC_H-Hh

        Args:
            year: Year of data
            month: Month (1-12)

        Returns:
            dict: Diurnal fire patterns by hour
        """
        month_str = f"{month:02d}"

        try:
            with h5py.File(self.filepath, 'r') as f:
                emissions_group = f['emissions']

                if month_str not in emissions_group:
                    raise ValueError(f"No data found for month {month:02d}")

                month_group = emissions_group[month_str]

                if 'diurnal_cycle' not in month_group:
                    self.logger.warning(f"No diurnal cycle data for {year}-{month:02d}")
                    return {}

                diurnal_group = month_group['diurnal_cycle']
                diurnal_data = {}

                # Extract hourly patterns
                for hour_key in diurnal_group.keys():
                    if hour_key.startswith('UTC_'):
                        diurnal_data[hour_key] = diurnal_group[hour_key][:]

                self.logger.info(f"Extracted diurnal patterns for {year}-{month:02d}")
                return diurnal_data

        except Exception as e:
            self.logger.error(f"Failed to extract diurnal patterns for {year}-{month:02d}: {e}")
            raise

    def get_daily_fractions(self, year: int, month: int) -> Dict[str, np.ndarray]:
        """
        Extract daily fire fractions for month.

        HDF5 structure: /emissions/MM/daily_fraction/day_D

        Args:
            year: Year of data
            month: Month (1-12)

        Returns:
            dict: Daily fire fractions by day
        """
        month_str = f"{month:02d}"

        try:
            with h5py.File(self.filepath, 'r') as f:
                emissions_group = f['emissions']

                if month_str not in emissions_group:
                    raise ValueError(f"No data found for month {month:02d}")

                month_group = emissions_group[month_str]

                if 'daily_fraction' not in month_group:
                    self.logger.warning(f"No daily fraction data for {year}-{month:02d}")
                    return {}

                daily_group = month_group['daily_fraction']
                daily_data = {}

                # Extract daily fractions
                for day_key in daily_group.keys():
                    if day_key.startswith('day_'):
                        daily_data[day_key] = daily_group[day_key][:]

                self.logger.info(f"Extracted daily fractions for {year}-{month:02d}")
                return daily_data

        except Exception as e:
            self.logger.error(f"Failed to extract daily fractions for {year}-{month:02d}: {e}")
            raise

    def get_vegetation_types(self) -> List[str]:
        """
        Get vegetation type classifications available in the file.

        Returns:
            list: Available vegetation type classifications
        """
        try:
            with h5py.File(self.filepath, 'r') as f:
                vegetation_types = []

                # Look for vegetation types in first available month
                if 'emissions' in f:
                    emissions_group = f['emissions']
                    month_keys = [k for k in emissions_group.keys() if len(k) == 2 and k.isdigit()]

                    if month_keys:
                        first_month = emissions_group[month_keys[0]]
                        if 'partitioning' in first_month:
                            vegetation_types = list(first_month['partitioning'].keys())

                self.logger.info(f"Found vegetation types: {vegetation_types}")
                return vegetation_types

        except Exception as e:
            self.logger.error(f"Failed to get vegetation types: {e}")
            return []

    def get_spatial_coordinates(self) -> Dict[str, np.ndarray]:
        """
        Get spatial coordinate information from GFED file.

        Returns:
            dict: Latitude and longitude coordinate arrays
        """
        try:
            with h5py.File(self.filepath, 'r') as f:
                coords = {}

                if 'ancill' in f and 'grid' in f['ancill']:
                    grid_group = f['ancill']['grid']

                    if 'latitude' in grid_group:
                        coords['latitude'] = grid_group['latitude'][:]

                    if 'longitude' in grid_group:
                        coords['longitude'] = grid_group['longitude'][:]

                    self.logger.info(f"Extracted spatial coordinates: {list(coords.keys())}")

                return coords

        except Exception as e:
            self.logger.error(f"Failed to get spatial coordinates: {e}")
            return {}