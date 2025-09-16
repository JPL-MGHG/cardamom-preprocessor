"""
GFED Diurnal Pattern Loader

Load GFED diurnal fire patterns for specific months.
Based on MATLAB load_gfed_diurnal_fields_05deg function.

This module extracts diurnal fire emission patterns from GFED4.1s HDF5 files
and provides them at the resolution needed for diurnal flux calculations.

Scientific Context:
GFED provides 3-hourly fire emission patterns that capture the typical
diurnal cycle of fire activity. These patterns are used to downscale
monthly fire emissions to hourly resolution while preserving realistic
fire timing.
"""

import os
import numpy as np
import h5py
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from gfed_downloader import GFEDDownloader


class GFEDDiurnalLoader:
    """
    Load GFED diurnal fire patterns for specific months.
    Based on MATLAB load_gfed_diurnal_fields_05deg function.

    MATLAB Reference: load_gfed_diurnal_fields_05deg.m
    """

    def __init__(self,
                 gfed_data_dir: str = "./DATA/GFED4/",
                 region_bounds: Optional[List[float]] = None):
        """
        Initialize GFED diurnal loader.

        Args:
            gfed_data_dir: Directory containing GFED HDF5 files
            region_bounds: Regional bounds [lon_min, lon_max, lat_min, lat_max]
                          Default: CONUS region
        """
        self.gfed_data_dir = gfed_data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

        # Default to CONUS region bounds - MATLAB: CONUS region definition
        self.region_bounds = region_bounds or [-124.75, -65.25, 24.75, 60.25]  # [W, E, S, N]

        # Initialize GFED downloader for file access
        self.downloader = GFEDDownloader(gfed_data_dir)

        # Setup emission factors - MATLAB: emission factor definitions
        self.emission_factors = self._setup_emission_factors()

    def _setup_emission_factors(self) -> Dict[str, Dict[str, float]]:
        """
        Setup emission factors for different vegetation types and species.

        MATLAB Reference: Emission factor matrix setup in MATLAB function

        Returns:
            dict: Emission factors by species and vegetation type
        """
        # GFED vegetation types
        vegetation_types = ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI']

        # Emission factors (g species / kg dry matter burned)
        # MATLAB Reference: Emission factor values from GFED documentation
        emission_factors = {
            'CO2': {
                'SAVA': 1686, 'BORF': 1489, 'TEMF': 1520,
                'DEFO': 1643, 'PEAT': 1703, 'AGRI': 1585
            },
            'CO': {
                'SAVA': 65, 'BORF': 127, 'TEMF': 88,
                'DEFO': 93, 'PEAT': 210, 'AGRI': 102
            },
            'CH4': {
                'SAVA': 1.9, 'BORF': 4.7, 'TEMF': 5.2,
                'DEFO': 5.7, 'PEAT': 21.0, 'AGRI': 2.3
            },
            'C': {  # Total carbon content
                'SAVA': 0.45, 'BORF': 0.45, 'TEMF': 0.45,
                'DEFO': 0.45, 'PEAT': 0.45, 'AGRI': 0.45
            }
        }

        return emission_factors

    def load_diurnal_fields(self, month: int, year: int, target_region: Optional[List[float]] = None) -> np.ndarray:
        """
        Load GFED diurnal fire patterns for specified month and year.

        MATLAB Reference: Main function body in load_gfed_diurnal_fields_05deg.m

        Args:
            month: Month to load (1-12)
            year: Year to load
            target_region: Optional regional bounds override

        Returns:
            ndarray: CO2 diurnal emissions array for the target region (3-hourly)
        """
        self.logger.info(f"Loading GFED diurnal fields for {year}-{month:02d}")

        if target_region is None:
            target_region = self.region_bounds

        # Determine file path (beta version for years >= 2017)
        beta_suffix = '_beta' if year >= 2017 else ''
        gfed_file = os.path.join(
            self.gfed_data_dir,
            f'GFED4.1s_{year}{beta_suffix}.hdf5'
        )

        if not os.path.exists(gfed_file):
            self.logger.warning(f"GFED file not found: {gfed_file}")
            # Try to download if not available
            self.downloader.download_data([year])

        # Load GFED data for the month
        gfed_data = self._load_monthly_gfed_data(gfed_file, month)

        # Extract regional CO2 diurnal patterns
        co2_diurnal = self._extract_regional_co2_diurnal(gfed_data, target_region)

        self.logger.info(f"Successfully loaded GFED diurnal fields for {year}-{month:02d}")
        self.logger.info(f"CO2 diurnal shape: {co2_diurnal.shape}")

        return co2_diurnal

    def _load_monthly_gfed_data(self, filepath: str, month: int) -> Dict[str, np.ndarray]:
        """
        Load monthly GFED data from HDF5 file.

        MATLAB Reference: HDF5 data loading section in MATLAB function

        Args:
            filepath: Path to GFED HDF5 file
            month: Month to extract (1-12)

        Returns:
            dict: Monthly GFED data components
        """
        month_str = f"{month:02d}"

        with h5py.File(filepath, 'r') as f:
            # Get number of days in month
            year = int(os.path.basename(filepath).split('_')[1][:4])
            days_in_month = self._get_days_in_month(year, month)

            # Load dry matter fractions by vegetation type
            # MATLAB Reference: Loading DM fractions for each vegetation type
            vegetation_types = ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI']
            dm_fractions = np.zeros((720, 1440, 6))  # 0.25° global grid

            for i, veg_type in enumerate(vegetation_types):
                field_path = f'/emissions/{month_str}/partitioning/DM_{veg_type}'
                if field_path in f:
                    # MATLAB Reference: flipud(transpose()) operations
                    dm_fractions[:, :, i] = np.flipud(f[field_path][:].T)

            # Load daily fractions - MATLAB: daily fire fraction loading
            daily_fractions = np.zeros((720, 1440, days_in_month))
            for day in range(1, days_in_month + 1):
                field_path = f'/emissions/{month_str}/daily_fraction/day_{day}'
                if field_path in f:
                    daily_fractions[:, :, day-1] = np.flipud(f[field_path][:].T)

            # Load diurnal fractions (8 × 3-hour periods) - MATLAB: diurnal cycle loading
            diurnal_fractions = np.zeros((720, 1440, 8))
            for hour_group in range(8):
                start_hour = hour_group * 3
                end_hour = (hour_group + 1) * 3
                field_path = f'/emissions/{month_str}/diurnal_cycle/UTC_{start_hour:02d}-{end_hour:02d}h'
                if field_path in f:
                    diurnal_fractions[:, :, hour_group] = np.flipud(f[field_path][:].T)

        gfed_data = {
            'dm_fractions': dm_fractions,
            'daily_fractions': daily_fractions,
            'diurnal_fractions': diurnal_fractions,
            'days_in_month': days_in_month
        }

        self.logger.debug(f"Loaded GFED data for month {month}: {days_in_month} days")
        return gfed_data

    def _calculate_co2_emissions(self, gfed_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate CO2 emissions from GFED dry matter and emission factors.
        Equivalent to complex nested loop in MATLAB (lines 598-609).

        MATLAB Reference: CO2 emission calculation loops in MATLAB function

        Args:
            gfed_data: GFED data components

        Returns:
            ndarray: CO2 emissions array (720, 1440, n_timesteps)
        """
        days = gfed_data['days_in_month']
        n_timesteps = days * 8  # 8 × 3-hour periods per day

        # Initialize emissions array
        co2_emissions = np.zeros((720, 1440, n_timesteps))

        # Get emission factor matrices
        co2_factors = self.emission_factors['CO2']
        carbon_factors = self.emission_factors['C']

        vegetation_types = ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI']

        # MATLAB Reference: Nested loops for emission calculation (lines 598-609)
        for day in range(days):
            for hour_group in range(8):  # 8 × 3-hour periods per day
                time_index = hour_group + day * 8

                # Calculate total CO2 emissions for this timestep
                co2_timestep = np.zeros((720, 1440))

                for veg_idx, veg_type in enumerate(vegetation_types):
                    # MATLAB Reference: Emission calculation per vegetation type
                    emission_rate = (
                        gfed_data['dm_fractions'][:, :, veg_idx] *
                        co2_factors[veg_type] *
                        gfed_data['daily_fractions'][:, :, day] *
                        gfed_data['diurnal_fractions'][:, :, hour_group] /
                        carbon_factors[veg_type]  # Normalize by carbon content
                    )

                    co2_timestep += emission_rate

                co2_emissions[:, :, time_index] = co2_timestep

        self.logger.debug(f"Calculated CO2 emissions for {n_timesteps} timesteps")
        return co2_emissions

    def _extract_regional_co2_diurnal(self, gfed_data: Dict[str, np.ndarray], target_region: List[float]) -> np.ndarray:
        """
        Extract CO2 diurnal patterns for target region.

        MATLAB Reference: Regional extraction and aggregation in MATLAB function

        Args:
            gfed_data: GFED data components
            target_region: Regional bounds [lon_min, lon_max, lat_min, lat_max]

        Returns:
            ndarray: Regional CO2 diurnal patterns
        """
        # Calculate full CO2 emissions at 0.25° resolution
        co2_global = self._calculate_co2_emissions(gfed_data)

        # Aggregate to 0.5° resolution - MATLAB: 2:2:end aggregation
        co2_05deg = self._aggregate_025_to_05_degree(co2_global)

        # Extract regional subset
        region_indices = self._get_region_indices(target_region)
        co2_regional = co2_05deg[region_indices]

        self.logger.debug(f"Extracted regional CO2 diurnal: {co2_regional.shape}")
        return co2_regional

    def _aggregate_025_to_05_degree(self, data_025deg: np.ndarray) -> np.ndarray:
        """
        Aggregate 0.25° data to 0.5° resolution.
        Equivalent to MATLAB 2:2:end indexing.

        MATLAB Reference: Resolution aggregation using 2:2:end indexing

        Args:
            data_025deg: Data at 0.25° resolution (720, 1440, time)

        Returns:
            ndarray: Data at 0.5° resolution (360, 720, time)
        """
        # MATLAB Reference: 2:2:end indexing for aggregation
        # Take every second grid point starting from index 1 (0-based: every second starting from 0)
        data_05deg = data_025deg[::2, ::2, :]

        self.logger.debug(f"Aggregated from 0.25° {data_025deg.shape} to 0.5° {data_05deg.shape}")
        return data_05deg

    def _get_region_indices(self, target_region: List[float]) -> Tuple[slice, slice, slice]:
        """
        Get array indices for target region extraction.

        Args:
            target_region: Regional bounds [lon_min, lon_max, lat_min, lat_max]

        Returns:
            tuple: (lat_slice, lon_slice, time_slice) for array indexing
        """
        lon_min, lon_max, lat_min, lat_max = target_region

        # 0.5° grid coordinates
        lons = np.arange(-179.75, 180, 0.5)
        lats = np.arange(-89.75, 90, 0.5)

        # Find indices
        lon_start = np.argmin(np.abs(lons - lon_min))
        lon_end = np.argmin(np.abs(lons - lon_max)) + 1
        lat_start = np.argmin(np.abs(lats - lat_min))
        lat_end = np.argmin(np.abs(lats - lat_max)) + 1

        lat_slice = slice(lat_start, lat_end)
        lon_slice = slice(lon_start, lon_end)
        time_slice = slice(None)  # All time steps

        self.logger.debug(f"Region indices - lat: {lat_slice}, lon: {lon_slice}")
        return lat_slice, lon_slice, time_slice

    def _get_days_in_month(self, year: int, month: int) -> int:
        """
        Get number of days in specified month.

        Args:
            year: Year
            month: Month (1-12)

        Returns:
            int: Number of days in month
        """
        # Days in month accounting for leap years
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        if month == 2 and self._is_leap_year(year):
            return 29
        else:
            return days_in_month[month - 1]

    def _is_leap_year(self, year: int) -> bool:
        """
        Check if year is a leap year.

        Args:
            year: Year to check

        Returns:
            bool: True if leap year
        """
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def get_diurnal_timing_statistics(self, co2_diurnal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate diurnal fire timing statistics.

        Args:
            co2_diurnal: Diurnal CO2 emission patterns

        Returns:
            dict: Timing statistics (peak hour, total emissions, etc.)
        """
        # Calculate daily total emissions
        n_days = co2_diurnal.shape[2] // 8
        daily_totals = np.zeros((*co2_diurnal.shape[:2], n_days))

        for day in range(n_days):
            start_idx = day * 8
            end_idx = (day + 1) * 8
            daily_totals[:, :, day] = np.sum(co2_diurnal[:, :, start_idx:end_idx], axis=2)

        # Find peak fire timing (3-hour period with maximum emissions)
        peak_periods = np.argmax(co2_diurnal, axis=2)

        # Convert to hours (each period represents 3 hours)
        peak_hours = peak_periods * 3

        statistics = {
            'daily_totals': daily_totals,
            'peak_hours': peak_hours,
            'diurnal_amplitude': np.max(co2_diurnal, axis=2) - np.min(co2_diurnal, axis=2)
        }

        return statistics

    def validate_diurnal_patterns(self, co2_diurnal: np.ndarray) -> bool:
        """
        Validate GFED diurnal patterns for reasonableness.

        Args:
            co2_diurnal: Diurnal CO2 emission patterns

        Returns:
            bool: True if validation passes
        """
        # Check for negative values
        if np.any(co2_diurnal < 0):
            self.logger.warning("Negative CO2 emissions detected in diurnal patterns")
            return False

        # Check for NaN values
        if np.any(np.isnan(co2_diurnal)):
            self.logger.warning("NaN values detected in diurnal patterns")

        # Check for reasonable peak timing (should peak during daylight hours)
        peak_hours = np.argmax(co2_diurnal, axis=2) * 3
        avg_peak_hour = np.nanmean(peak_hours)

        if avg_peak_hour < 6 or avg_peak_hour > 18:
            self.logger.warning(f"Unusual fire peak timing: {avg_peak_hour:.1f} hours UTC")

        self.logger.info("GFED diurnal pattern validation completed")
        return True