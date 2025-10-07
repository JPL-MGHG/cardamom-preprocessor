#!/usr/bin/env python3
"""
NOAA CO2 Downloader for CARDAMOM

Downloads and processes NOAA global CO2 concentration data from ESRL/GMD.
Creates CARDAMOM-compliant NetCDF files with spatially-replicated CO2 data.
"""

import os
import requests
import numpy as np
import xarray as xr
from typing import Dict, List, Any, Tuple, Optional
import logging
from base_downloader import BaseDownloader
from time_utils import standardize_time_coordinate


class NOAADownloader(BaseDownloader):
    """
    Download and process NOAA global CO2 concentration data.

    Scientific Context:
    NOAA provides globally averaged atmospheric CO2 concentrations from the
    Mauna Loa observatory and global network. These measurements represent
    the background atmospheric CO2 that constrains carbon cycle models in
    CARDAMOM by providing the atmospheric boundary condition.

    Data Source: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.txt
    """

    def __init__(self, output_dir: str = "./DATA/NOAA_CO2/"):
        """
        Initialize NOAA CO2 downloader.

        Args:
            output_dir: Directory for NOAA CO2 data files
        """
        super().__init__(output_dir)

        # NOAA HTTPS server configuration
        self.base_url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/"
        self.data_file = "co2_mm_gl.txt"
        self.output_file = os.path.join(self.output_dir, "co2_mm_gl.txt")

        # Default spatial grid for CARDAMOM (matches ECMWF 0.5 degree grid)
        self.default_spatial_grid = {
            'longitude': np.arange(-179.75, 180, 0.5),
            'latitude': np.arange(-89.75, 90, 0.5)
        }

    def download_data(self, force_update: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Download raw CO2 data from NOAA HTTPS server.

        Args:
            force_update: If True, re-download even if file exists
            **kwargs: Additional parameters (unused for NOAA downloader)

        Returns:
            dict: Download results with file path and status
        """
        return self.download_raw_data(force_update)

    def download_raw_data(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Download raw CO2 text file from NOAA HTTPS server.

        Args:
            force_update: If True, re-download even if file exists

        Returns:
            dict: Download status and file information
        """
        # Check if file exists and is recent (unless forcing update)
        if os.path.exists(self.output_file) and not force_update:
            self.logger.info(f"Using existing NOAA CO2 file: {self.output_file}")
            return {
                "status": "success",
                "source": "existing",
                "file_path": self.output_file,
                "message": "Used existing file"
            }

        try:
            download_url = f"{self.base_url}{self.data_file}"
            self.logger.info(f"Downloading CO2 data from: {download_url}")

            # Download the CO2 data file using HTTPS
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Save the downloaded content
            with open(self.output_file, 'w', encoding='utf-8') as local_file:
                local_file.write(response.text)

            self.logger.info(f"Downloaded NOAA CO2 data to {self.output_file}")

            # Validate downloaded file
            if self.validate_downloaded_data(self.output_file):
                self._record_download_attempt("co2_mm_gl.txt", "success")
                return {
                    "status": "success",
                    "source": "downloaded",
                    "file_path": self.output_file,
                    "message": "Successfully downloaded NOAA CO2 data"
                }
            else:
                self._record_download_attempt("co2_mm_gl.txt", "failed", "File validation failed")
                return {
                    "status": "failed",
                    "error": "Downloaded file failed validation"
                }

        except Exception as e:
            error_msg = f"Failed to download NOAA CO2 data: {e}"
            self.logger.error(error_msg)
            self._record_download_attempt("co2_mm_gl.txt", "failed", str(e))
            return {
                "status": "failed",
                "error": error_msg
            }

    def parse_co2_data(self, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse NOAA CO2 text file format.

        NOAA CO2 file format (current):
        # Comment lines start with #
        # year   month   decimal   average   average_unc   trend   trend_unc
        # Missing values are represented as -99.99

        Args:
            filepath: Path to CO2 text file (uses output file if None)

        Returns:
            list: Structured CO2 data with year, month, concentration
        """
        if filepath is None:
            filepath = self.output_file

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NOAA CO2 file not found: {filepath}")

        co2_data = []

        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip comment lines and empty lines
                    if line.startswith('#') or not line.strip():
                        continue

                    try:
                        parts = line.strip().split()
                        if len(parts) < 4:
                            self.logger.warning(f"Line {line_num}: Insufficient data columns")
                            continue

                        # Parse data fields (new format: year month decimal average average_unc trend trend_unc)
                        year = int(parts[0])
                        month = int(parts[1])
                        decimal_date = float(parts[2])

                        # Handle missing values (-99.99)
                        co2_average = float(parts[3]) if parts[3] != '-99.99' else None

                        # New format has average_unc in column 4, trend in column 5
                        co2_average_unc = float(parts[4]) if len(parts) > 4 and parts[4] != '-99.99' else None
                        co2_trend = float(parts[5]) if len(parts) > 5 and parts[5] != '-99.99' else None
                        co2_trend_unc = float(parts[6]) if len(parts) > 6 and parts[6] != '-99.99' else None

                        co2_data.append({
                            'year': year,
                            'month': month,
                            'decimal_date': decimal_date,
                            'co2_ppm': co2_average,
                            'co2_average_unc': co2_average_unc,
                            'co2_trend_ppm': co2_trend,
                            'co2_trend_unc': co2_trend_unc
                        })

                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Line {line_num}: Error parsing data - {e}")
                        continue

            self.logger.info(f"Parsed {len(co2_data)} CO2 data records from {filepath}")

            # Validate parsed data
            if len(co2_data) == 0:
                raise ValueError("No valid CO2 data found in file")

            # Check for reasonable CO2 values
            valid_co2_values = [d['co2_ppm'] for d in co2_data if d['co2_ppm'] is not None]
            if valid_co2_values:
                min_co2 = min(valid_co2_values)
                max_co2 = max(valid_co2_values)

                # Reasonable atmospheric CO2 range (1958-2030): 300-450 ppm
                if min_co2 < 250 or max_co2 > 500:
                    self.logger.warning(f"CO2 values outside expected range: {min_co2:.1f} - {max_co2:.1f} ppm")

                self.logger.info(f"CO2 concentration range: {min_co2:.1f} - {max_co2:.1f} ppm")

            return co2_data

        except Exception as e:
            self.logger.error(f"Failed to parse NOAA CO2 data: {e}")
            raise

    def create_cardamom_co2_files(self,
                                 years: List[int],
                                 spatial_grid: Optional[Dict[str, np.ndarray]] = None,
                                 use_interpolated: bool = True) -> Dict[str, Any]:
        """
        Create CARDAMOM-compliant NetCDF files with spatially-replicated CO2.

        Matches MATLAB logic: creates global 2D+time datasets by replicating
        point CO2 measurements across the spatial domain.

        Args:
            years: List of years to process
            spatial_grid: Dictionary with 'longitude' and 'latitude' arrays
                         Uses default 0.5-degree grid if None
            use_interpolated: If True, use interpolated CO2 values when available

        Returns:
            dict: Processing results with created file paths
        """
        # Use default spatial grid if not provided
        if spatial_grid is None:
            spatial_grid = self.default_spatial_grid

        # Parse CO2 data
        try:
            co2_data = self.parse_co2_data()
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to parse CO2 data: {e}"
            }

        # Convert to xarray dataset for easier handling
        co2_records = []
        for record in co2_data:
            if record['year'] in years:
                # In new format, only average values are available (no separate interpolated values)
                co2_value = record['co2_ppm']
                if co2_value is not None:
                    co2_records.append({
                        'year': record['year'],
                        'month': record['month'],
                        'co2_ppm': co2_value
                    })

        if not co2_records:
            return {
                "status": "failed",
                "error": f"No valid CO2 data found for years {years}"
            }

        created_files = []

        # Create NetCDF file for each year
        for year in years:
            year_records = [r for r in co2_records if r['year'] == year]

            if not year_records:
                self.logger.warning(f"No CO2 data available for year {year}")
                continue

            try:
                # Create monthly time series for the year
                months = np.arange(1, 13)
                co2_monthly = np.full(12, np.nan)

                for record in year_records:
                    month_idx = record['month'] - 1  # Convert to 0-based index
                    if 0 <= month_idx < 12:
                        co2_monthly[month_idx] = record['co2_ppm']

                # Fill missing months with interpolation
                co2_monthly = self._interpolate_missing_months(co2_monthly)

                # Create spatial grid
                lon_grid, lat_grid = np.meshgrid(spatial_grid['longitude'], spatial_grid['latitude'])
                n_lat, n_lon = lon_grid.shape

                # Replicate CO2 across spatial domain (matches MATLAB repmat logic)
                co2_global = np.zeros((12, n_lat, n_lon))
                for month_idx in range(12):
                    co2_global[month_idx, :, :] = co2_monthly[month_idx]

                # Create time coordinate with proper datetime64 values
                time_values = np.array([f"{year}-{m:02d}-15" for m in range(1, 13)], dtype='datetime64[D]')

                # Create xarray dataset
                dataset = xr.Dataset(
                    {
                        'co2_mole_fraction': (
                            ['time', 'latitude', 'longitude'],
                            co2_global,
                            {
                                'long_name': 'Atmospheric CO2 mole fraction',
                                'units': 'ppm',
                                'standard_name': 'mole_fraction_of_carbon_dioxide_in_air',
                                'source': 'NOAA ESRL Global Monitoring Laboratory',
                                'description': 'Globally averaged atmospheric CO2 concentration replicated spatially'
                            }
                        )
                    },
                    coords={
                        'longitude': (
                            ['longitude'],
                            spatial_grid['longitude'],
                            {'units': 'degrees_east', 'long_name': 'Longitude'}
                        ),
                        'latitude': (
                            ['latitude'],
                            spatial_grid['latitude'],
                            {'units': 'degrees_north', 'long_name': 'Latitude'}
                        ),
                        'time': (
                            ['time'],
                            time_values,
                            {'long_name': 'Time (monthly mid-points)'}
                        )
                    },
                    attrs={
                        'title': f'NOAA Global CO2 Concentrations for CARDAMOM - {year}',
                        'source': 'NOAA ESRL Global Monitoring Laboratory',
                        'data_source_url': f'{self.base_url}{self.data_file}',
                        'spatial_resolution': '0.5 degrees',
                        'temporal_resolution': 'monthly',
                        'conventions': 'CF-1.8',
                        'creation_date': str(np.datetime64('now')),
                        'description': 'Atmospheric CO2 mole fraction from NOAA global mean, spatially replicated for CARDAMOM modeling'
                    }
                )

                # Standardize time coordinate to CARDAMOM convention
                dataset = standardize_time_coordinate(dataset)

                # Create output filename
                output_filename = f"NOAA_CO2_GLOBAL_{year}.nc"
                output_filepath = os.path.join(self.output_dir, output_filename)

                # Save to NetCDF
                dataset.to_netcdf(output_filepath)
                created_files.append(output_filepath)

                self.logger.info(f"Created CARDAMOM CO2 file: {output_filename}")
                self._record_download_attempt(output_filename, "success")

            except Exception as e:
                error_msg = f"Failed to create CO2 file for year {year}: {e}"
                self.logger.error(error_msg)
                self._record_download_attempt(f"NOAA_CO2_GLOBAL_{year}.nc", "failed", str(e))

        return {
            "status": "completed",
            "created_files": created_files,
            "total_files": len(created_files),
            "years_processed": years
        }

    def _interpolate_missing_months(self, co2_monthly: np.ndarray) -> np.ndarray:
        """
        Interpolate missing months in CO2 time series.

        Args:
            co2_monthly: Array of 12 monthly CO2 values (may contain NaN)

        Returns:
            np.ndarray: Interpolated monthly CO2 values
        """
        valid_mask = ~np.isnan(co2_monthly)

        if not np.any(valid_mask):
            self.logger.warning("No valid CO2 data for interpolation")
            return co2_monthly

        if np.all(valid_mask):
            return co2_monthly  # No interpolation needed

        # Simple linear interpolation for missing months
        months = np.arange(12)
        valid_months = months[valid_mask]
        valid_co2 = co2_monthly[valid_mask]

        # Interpolate missing values
        co2_interpolated = np.interp(months, valid_months, valid_co2)

        n_interpolated = np.sum(~valid_mask)
        self.logger.info(f"Interpolated {n_interpolated} missing months in CO2 data")

        return co2_interpolated

    def get_co2_for_period(self, start_year: int, end_year: int) -> Dict[str, Any]:
        """
        Get CO2 concentrations for specified time period.

        Args:
            start_year: First year to retrieve
            end_year: Last year to retrieve (inclusive)

        Returns:
            dict: CO2 data for the specified period
        """
        try:
            # Ensure we have the raw data
            download_result = self.download_raw_data()
            if download_result["status"] != "success":
                return download_result

            # Parse and filter data
            co2_data = self.parse_co2_data()
            period_data = [
                record for record in co2_data
                if start_year <= record['year'] <= end_year
            ]

            # Calculate statistics
            valid_co2_values = [
                record['co2_ppm'] for record in period_data
                if record['co2_ppm'] is not None
            ]

            if valid_co2_values:
                statistics = {
                    'mean_co2_ppm': np.mean(valid_co2_values),
                    'min_co2_ppm': np.min(valid_co2_values),
                    'max_co2_ppm': np.max(valid_co2_values),
                    'trend_ppm_per_year': (valid_co2_values[-1] - valid_co2_values[0]) / len(valid_co2_values) * 12
                }
            else:
                statistics = {}

            return {
                "status": "success",
                "data": period_data,
                "period": {"start_year": start_year, "end_year": end_year},
                "total_records": len(period_data),
                "statistics": statistics
            }

        except Exception as e:
            error_msg = f"Failed to get CO2 data for period {start_year}-{end_year}: {e}"
            self.logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg
            }