"""
NOAA CO2 Data Downloader for CARDAMOM Preprocessing

This module downloads global CO2 concentration data from NOAA Global Monitoring
Laboratory and creates spatially-replicated NetCDF files for CARDAMOM carbon
cycle modeling.

Scientific Context:
Atmospheric CO2 concentration is a critical driver for photosynthesis calculations
in CARDAMOM. NOAA's Global Monitoring Laboratory provides monthly global CO2 data
that is spatially uniform but must be replicated to the CARDAMOM grid structure.

Key Features:
- Downloads entire CO2 time series by default (efficient for small CSV file)
- Supports single-month downloads for backwards compatibility
- Uses HTTPS for reliable data access
- Creates spatially-replicated grids at 0.5° resolution

Usage:
    # Download complete CO2 dataset (recommended)
    downloader = NOAADownloader(output_directory='./output')
    results = downloader.download_and_process()

    # Download specific month (backwards compatibility)
    results = downloader.download_and_process(year=2020, month=1)

References:
- NOAA GML CO2 Data: https://gml.noaa.gov/ccgg/trends/
- Data Source: Flask measurements from multiple observation sites
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import requests

from .base import BaseDownloader

logger = logging.getLogger(__name__)

# NOAA GML CO2 download URL
# Uses global monthly CO2 averages from NOAA's Global Monitoring Laboratory
# This dataset contains high-precision CO2 measurements from integrated global monitoring
NOAA_CO2_URL = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv'


class NOAADownloader(BaseDownloader):
    """
    Download and process NOAA global CO2 data for CARDAMOM.

    CO2 is atmospheric (spatially uniform), so this downloader:
    1. Fetches monthly global CO2 from NOAA
    2. Creates a spatially-replicated grid at CARDAMOM resolution
    3. Generates STAC metadata

    Attributes:
        co2_url (str): URL for NOAA CO2 data file
    """

    def __init__(
        self,
        output_directory: str,
        keep_raw_files: bool = False,
        verbose: bool = False,
        co2_url: Optional[str] = None,
    ):
        """
        Initialize NOAA CO2 downloader.

        Args:
            output_directory (str): Root output directory path
            keep_raw_files (bool): Retain raw data files. Default: False
            verbose (bool): Print debug messages. Default: False
            co2_url (Optional[str]): URL to fetch CO2 data from.
                Default: NOAA GML official URL
        """

        super().__init__(output_directory, keep_raw_files, verbose)

        self.co2_url = co2_url or NOAA_CO2_URL
        logger.info(f"NOAA downloader initialized with URL: {self.co2_url}")

    def _download_co2_data(self) -> Dict[int, Dict[int, float]]:
        """
        Download global CO2 data from NOAA.

        Parses the NOAA text file format to extract monthly CO2 concentrations.

        Returns:
            Dict[int, Dict[int, float]]: Nested dict {year: {month: co2_ppm}}

        Raises:
            RuntimeError: If download fails or file parsing fails
        """

        logger.info(f"Downloading NOAA CO2 data from {self.co2_url}")

        try:
            response = requests.get(self.co2_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download NOAA CO2 data: {e}") from e

        co2_data = {}

        # Parse NOAA CSV file format
        # Format: year,month,decimal_date,average_observed,std_dev,trend,trend_std
        # Header lines start with '#'
        # Example: 2020,1,2020.042,411.72,0.15,410.23,0.10
        for line in response.text.split('\n'):
            # Skip header and empty lines
            if not line or line.startswith('#'):
                continue

            # Split by comma for CSV format
            parts = line.split(',')
            if len(parts) < 4:
                continue

            try:
                year = int(parts[0].strip())
                month = int(parts[1].strip())
                co2_ppm = float(parts[3].strip())  # Use 'average_observed' column (index 3)

                if year not in co2_data:
                    co2_data[year] = {}

                co2_data[year][month] = co2_ppm

                logger.debug(f"Parsed CO2: {year}-{month:02d} = {co2_ppm:.2f} ppm")

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse line: {line}")
                continue

        if not co2_data:
            raise RuntimeError("No CO2 data found in NOAA file")

        logger.info(f"Parsed CO2 data for {len(co2_data)} years")

        return co2_data

    def _create_spatially_replicated_co2_grid(
        self,
        co2_value_ppm: float,
        latitude_extent: tuple = (-89.75, 89.75),
        longitude_extent: tuple = (-179.75, 179.75),
        resolution_degrees: float = 0.5,
    ) -> np.ndarray:
        """
        Create a 2D grid of CO2 concentrations replicated across the domain.

        Since CO2 is globally uniform, this creates an array with the same
        value everywhere, matching the CARDAMOM grid structure.

        Args:
            co2_value_ppm (float): Global CO2 concentration in ppm
            latitude_extent (tuple): (min_lat, max_lat)
            longitude_extent (tuple): (min_lon, max_lon)
            resolution_degrees (float): Grid resolution in degrees

        Returns:
            np.ndarray: 2D array [latitude, longitude] with replicated CO2 values
        """

        # Calculate grid dimensions
        num_lat = int((latitude_extent[1] - latitude_extent[0]) / resolution_degrees) + 1
        num_lon = int((longitude_extent[1] - longitude_extent[0]) / resolution_degrees) + 1

        # Create uniform grid
        co2_grid = np.full((num_lat, num_lon), co2_value_ppm, dtype=np.float32)

        logger.debug(
            f"Created spatially-replicated CO2 grid: shape={co2_grid.shape}, "
            f"value={co2_value_ppm:.2f} ppm"
        )

        return co2_grid

    def download_and_process(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Download and process NOAA CO2 data.

        Since the NOAA CO2 data is a small CSV file containing the entire historical
        record, this method downloads all available data by default and creates a
        single NetCDF file with complete time series.

        If year and month are specified, only that specific month's data is returned
        (for backwards compatibility).

        Workflow:
        1. Download global NOAA CO2 data (entire time series)
        2. If year/month specified: Extract specific month
           Otherwise: Process entire time series
        3. Create spatially-replicated grid(s)
        4. Write to NetCDF with STAC metadata

        Args:
            year (Optional[int]): Year to process. If None, processes all available data.
            month (Optional[int]): Month to process (1-12). If None, processes all available data.
            **kwargs: Additional arguments (unused, for compatibility)

        Returns:
            Dict[str, Any]: Results dictionary with keys:
                - 'output_files': List of generated NetCDF paths
                - 'stac_items': List of STAC Item objects
                - 'collection_id': STAC Collection ID
                - 'success': bool

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If download fails

        Scientific Note:
            Atmospheric CO2 is spatially uniform globally, so NOAA's global monthly
            mean values are replicated across all grid cells. Downloading the entire
            time series is more efficient than monthly downloads since the source
            file is small (~100KB) and contains all historical data.
        """

        # Validate parameters if provided
        if year is not None and month is not None:
            self.validate_temporal_parameters(year, month)
            logger.info(f"Starting NOAA CO2 download for {year}-{month:02d}")
            return self._download_single_month(year, month)
        elif year is not None or month is not None:
            raise ValueError(
                "Both year and month must be specified together, or both omitted. "
                f"Got: year={year}, month={month}"
            )
        else:
            logger.info("Starting NOAA CO2 download for entire available dataset")
            return self._download_all_data()

    def _download_single_month(
        self,
        year: int,
        month: int,
    ) -> Dict[str, Any]:
        """
        Download and process NOAA CO2 data for a specific month.

        This method maintains backwards compatibility with the original API.

        Args:
            year (int): Year to process
            month (int): Month to process (1-12)

        Returns:
            Dict[str, Any]: Results dictionary with single month's data
        """

        # Step 1: Download global CO2 data
        co2_data = self._download_co2_data()

        # Step 2: Extract CO2 for requested year/month
        if year not in co2_data or month not in co2_data[year]:
            raise ValueError(
                f"CO2 data not available for {year}-{month:02d}. "
                f"Available years: {sorted(co2_data.keys())}"
            )

        co2_value_ppm = co2_data[year][month]
        logger.info(f"CO2 for {year}-{month:02d}: {co2_value_ppm:.2f} ppm")

        # Step 3: Create spatially-replicated CO2 grid
        co2_grid = self._create_spatially_replicated_co2_grid(co2_value_ppm)

        # Step 4: Create standard NetCDF dataset
        dataset = self.create_standard_netcdf_dataset(
            {'CO2': co2_grid},
            year=year,
            month=month,
        )

        # Step 5: Write to NetCDF
        output_filename = f"co2_{year}_{month:02d}.nc"
        output_file = self.write_netcdf_file(
            dataset,
            output_filename,
            variable_units={'CO2': 'ppm'},
        )

        # Step 6: Generate STAC metadata
        stac_result = self.create_and_write_stac_metadata(
            collection_id='cardamom-co2',
            collection_description=(
                'Global atmospheric CO2 concentration from NOAA Global Monitoring Laboratory'
            ),
            collection_keywords=['co2', 'noaa', 'atmospheric', 'greenhouse-gas'],
            items_data=[
                {
                    'variable_name': 'CO2',
                    'year': year,
                    'month': month,
                    'data_file_path': f'data/{output_filename}',
                    'properties': {
                        'cardamom:units': 'ppm',
                        'cardamom:source': 'noaa-gml',
                        'noaa:co2_value': co2_value_ppm,
                    },
                }
            ],
            temporal_start=datetime(year, month, 1),
        )

        logger.info(f"Successfully created CO2 NetCDF: {output_file}")

        return {
            'output_files': [output_file],
            'stac_items': stac_result['items'],
            'collection_id': 'cardamom-co2',
            'success': True,
        }

    def _download_all_data(self) -> Dict[str, Any]:
        """
        Download and process all available NOAA CO2 data.

        This creates a single NetCDF file containing the complete CO2 time series
        with spatially-replicated values for each time step.

        Returns:
            Dict[str, Any]: Results dictionary with complete dataset
        """

        # Step 1: Download global CO2 data
        co2_data = self._download_co2_data()

        # Step 2: Extract all years and months
        all_years = sorted(co2_data.keys())
        start_year = all_years[0]
        end_year = all_years[-1]

        logger.info(
            f"Processing CO2 data from {start_year} to {end_year} "
            f"({len(all_years)} years)"
        )

        # Step 3: Build time series arrays
        time_steps = []
        co2_values = []

        for year in all_years:
            for month in sorted(co2_data[year].keys()):
                time_steps.append(datetime(year, month, 1))
                co2_values.append(co2_data[year][month])

        logger.info(f"Total time steps: {len(time_steps)}")

        # Step 4: Create spatially-replicated 3D CO2 array [time, lat, lon]
        # Get grid dimensions
        num_lat = int((89.75 - (-89.75)) / 0.5) + 1
        num_lon = int((179.75 - (-179.75)) / 0.5) + 1
        num_time = len(time_steps)

        # Create 3D array with replicated CO2 values
        co2_grid_3d = np.zeros((num_time, num_lat, num_lon), dtype=np.float32)
        for t_idx, co2_val in enumerate(co2_values):
            co2_grid_3d[t_idx, :, :] = co2_val

        logger.debug(
            f"Created 3D CO2 grid: shape={co2_grid_3d.shape}, "
            f"values range: {co2_grid_3d.min():.2f}-{co2_grid_3d.max():.2f} ppm"
        )

        # Step 5: Create xarray Dataset with complete time series
        lats = np.arange(-89.75, 90, 0.5)
        lons = np.arange(-179.75, 180, 0.5)

        dataset = xr.Dataset(
            {
                'CO2': (
                    ['time', 'latitude', 'longitude'],
                    co2_grid_3d,
                    {
                        'long_name': 'Atmospheric CO2 concentration',
                        'units': 'ppm',
                        'standard_name': 'mole_fraction_of_carbon_dioxide_in_air',
                        'description': (
                            'Global monthly mean atmospheric CO2 concentration '
                            'replicated spatially'
                        ),
                    },
                )
            },
            coords={
                'time': time_steps,
                'latitude': (['latitude'], lats, {'units': 'degrees_north'}),
                'longitude': (['longitude'], lons, {'units': 'degrees_east'}),
            },
            attrs={
                'title': 'NOAA Global CO2 Concentrations',
                'source': 'NOAA Global Monitoring Laboratory',
                'institution': 'NOAA GML',
                'references': 'https://gml.noaa.gov/ccgg/trends/',
                'Conventions': 'CF-1.8',
                'history': f'Created {datetime.now().isoformat()}',
            },
        )

        # Step 6: Write to NetCDF
        output_filename = f"co2_{start_year}_{end_year}.nc"
        output_file = self.write_netcdf_file(
            dataset,
            output_filename,
            variable_units={'CO2': 'ppm'},
        )

        # Step 7: Generate STAC metadata for the complete dataset
        stac_result = self.create_and_write_stac_metadata(
            collection_id='cardamom-co2',
            collection_description=(
                'Global atmospheric CO2 concentration from NOAA Global Monitoring Laboratory'
            ),
            collection_keywords=['co2', 'noaa', 'atmospheric', 'greenhouse-gas'],
            items_data=[
                {
                    'variable_name': 'CO2',
                    'year': start_year,
                    'month': 1,
                    'data_file_path': f'data/{output_filename}',
                    'properties': {
                        'cardamom:units': 'ppm',
                        'cardamom:source': 'noaa-gml',
                        'noaa:start_year': start_year,
                        'noaa:end_year': end_year,
                        'noaa:time_steps': len(time_steps),
                        'noaa:co2_min': float(co2_grid_3d.min()),
                        'noaa:co2_max': float(co2_grid_3d.max()),
                        'noaa:co2_mean': float(co2_grid_3d.mean()),
                    },
                }
            ],
            temporal_start=time_steps[0],
        )

        logger.info(f"Successfully created complete CO2 NetCDF: {output_file}")
        logger.info(
            f"Time range: {start_year}-{end_year}, "
            f"CO2 range: {co2_grid_3d.min():.2f}-{co2_grid_3d.max():.2f} ppm"
        )

        return {
            'output_files': [output_file],
            'stac_items': stac_result['items'],
            'collection_id': 'cardamom-co2',
            'success': True,
            'time_range': (start_year, end_year),
            'num_time_steps': len(time_steps),
        }
