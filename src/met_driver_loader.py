"""
Meteorological Driver Loader

Load ERA5 diurnal meteorological fields for flux downscaling.
Based on MATLAB load_era5_diurnal_fields_new function.

This module handles loading and processing of hourly ERA5 meteorological
data used as drivers for diurnal flux downscaling, specifically skin
temperature and solar radiation.

Scientific Context:
ERA5 provides hourly meteorological reanalysis data that captures the
diurnal cycles needed to downscale monthly carbon fluxes to hourly
resolution. Solar radiation drives GPP patterns, while temperature
drives respiration patterns.
"""

import os
import numpy as np
import xarray as xr
import logging
from typing import Tuple, Optional

from validation import validate_temperature_data, validate_radiation_data


class ERA5DiurnalLoader:
    """
    Load ERA5 diurnal meteorological fields for specific months.
    Based on MATLAB load_era5_diurnal_fields_new function.

    MATLAB Reference: load_era5_diurnal_fields_new.m
    """

    def __init__(self, data_dir: str = "./DATA/ERA5_CUSTOM/CONUS_2015_2020_DIURNAL/"):
        """
        Initialize ERA5 diurnal loader.

        Args:
            data_dir: Directory containing ERA5 hourly NetCDF files
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

        # File naming pattern - MATLAB: ERA5 file naming convention
        self.file_pattern = "ECMWF_CARDAMOM_HOURLY_DRIVER_{var}_{month:02d}{year}.nc"

        # Variable mapping
        self.variable_mapping = {
            'SST': 'skt',   # Skin temperature
            'SSRD': 'ssrd'  # Surface solar radiation downwards
        }

    def load_diurnal_fields(self, month: int, year: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load skin temperature and solar radiation for specified month/year.

        MATLAB Reference: Main function body in load_era5_diurnal_fields_new.m

        Args:
            month: Month to load (1-12)
            year: Year to load

        Returns:
            tuple: (SSRD, SKT) arrays with hourly data
                SSRD: Solar radiation in J/m² (cumulative)
                SKT: Skin temperature in K
        """
        self.logger.info(f"Loading ERA5 diurnal fields for {year}-{month:02d}")

        # Load skin temperature data
        skt_file = os.path.join(
            self.data_dir,
            self.file_pattern.format(var='SKT', month=month, year=year)
        )
        skt = self._load_and_reorient(skt_file, 'skt')

        # Load solar radiation data
        ssrd_file = os.path.join(
            self.data_dir,
            self.file_pattern.format(var='SSRD', month=month, year=year)
        )
        ssrd = self._load_and_reorient(ssrd_file, 'ssrd')

        # Validate loaded data
        self._validate_met_data(ssrd, skt)

        self.logger.info(f"Successfully loaded ERA5 diurnal fields for {year}-{month:02d}")
        self.logger.info(f"SKT shape: {skt.shape}, range: [{np.nanmin(skt):.1f}, {np.nanmax(skt):.1f}] K")
        self.logger.info(f"SSRD shape: {ssrd.shape}, range: [{np.nanmin(ssrd):.0f}, {np.nanmax(ssrd):.0f}] J/m²")

        return ssrd, skt

    def _load_and_reorient(self, filepath: str, variable: str) -> np.ndarray:
        """
        Load NetCDF data and reorient to match MATLAB conventions.
        Equivalent to MATLAB: flipud(permute(data, [2,1,3]))

        MATLAB Reference: Data loading and reorientation in MATLAB function

        Args:
            filepath: Path to NetCDF file
            variable: Variable name to load

        Returns:
            ndarray: Reoriented data array
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ERA5 file not found: {filepath}")

        try:
            with xr.open_dataset(filepath) as ds:
                if variable not in ds:
                    available_vars = list(ds.data_vars.keys())
                    raise KeyError(f"Variable '{variable}' not found. Available: {available_vars}")

                data = ds[variable].values

                # MATLAB Reference: permute(data, [2,1,3]) - transpose spatial dimensions
                data_reoriented = np.transpose(data, (1, 0, 2))

                # MATLAB Reference: flipud() - flip vertically to match coordinate system
                data_reoriented = np.flipud(data_reoriented)

                self.logger.debug(f"Loaded and reoriented {variable} from {filepath}")
                self.logger.debug(f"Original shape: {data.shape}, reoriented shape: {data_reoriented.shape}")

                return data_reoriented

        except Exception as e:
            self.logger.error(f"Failed to load {variable} from {filepath}: {e}")
            raise

    def _validate_met_data(self, ssrd: np.ndarray, skt: np.ndarray) -> None:
        """
        Validate meteorological data ranges and consistency.

        Args:
            ssrd: Solar radiation array
            skt: Skin temperature array

        Raises:
            ValueError: If validation fails
        """
        self.logger.debug("Validating meteorological data")

        # Check temperature ranges (should be in Kelvin, reasonable values)
        if np.any(skt < 200) or np.any(skt > 350):
            temp_min, temp_max = np.nanmin(skt), np.nanmax(skt)
            raise ValueError(
                f"Skin temperature values outside reasonable range: [{temp_min:.1f}, {temp_max:.1f}] K. "
                f"Expected range: [200, 350] K"
            )

        # Check radiation values (should be non-negative)
        if np.any(ssrd < 0):
            negative_count = np.sum(ssrd < 0)
            raise ValueError(
                f"Solar radiation contains {negative_count} negative values. "
                f"SSRD should be non-negative (cumulative solar radiation)."
            )

        # Check for consistent spatial dimensions
        if ssrd.shape[:2] != skt.shape[:2]:
            raise ValueError(
                f"Spatial dimensions of SSRD {ssrd.shape[:2]} and SKT {skt.shape[:2]} do not match"
            )

        # Check temporal dimensions (should have same number of hours)
        if ssrd.shape[2] != skt.shape[2]:
            raise ValueError(
                f"Temporal dimensions of SSRD {ssrd.shape[2]} and SKT {skt.shape[2]} do not match"
            )

        # Additional scientific validation using existing validators
        try:
            # Convert temperature to Celsius for validation
            skt_celsius = skt - 273.15
            validate_temperature_data(skt_celsius.flatten())
            self.logger.debug("Temperature validation passed")
        except Exception as e:
            self.logger.warning(f"Temperature validation warning: {e}")

        try:
            # Validate radiation data
            validate_radiation_data(ssrd.flatten())
            self.logger.debug("Radiation validation passed")
        except Exception as e:
            self.logger.warning(f"Radiation validation warning: {e}")

        self.logger.debug("Meteorological data validation completed successfully")

    def convert_cumulative_to_instantaneous_radiation(self, ssrd_cumulative: np.ndarray) -> np.ndarray:
        """
        Convert cumulative solar radiation to instantaneous rates.

        MATLAB Reference: Processing cumulative SSRD to instantaneous values

        Args:
            ssrd_cumulative: Cumulative solar radiation in J/m²

        Returns:
            ndarray: Instantaneous solar radiation in W/m²
        """
        # Calculate time differences between consecutive hours
        # MATLAB Reference: diff() operation on cumulative radiation
        ssrd_instantaneous = np.zeros_like(ssrd_cumulative)

        # First hour uses cumulative value directly
        ssrd_instantaneous[:, :, 0] = ssrd_cumulative[:, :, 0] / 3600  # Convert J/m² to W/m²

        # Subsequent hours use differences
        for t in range(1, ssrd_cumulative.shape[2]):
            ssrd_diff = ssrd_cumulative[:, :, t] - ssrd_cumulative[:, :, t-1]
            ssrd_instantaneous[:, :, t] = ssrd_diff / 3600  # Convert J/m² to W/m²

        # Ensure non-negative values
        ssrd_instantaneous = np.maximum(ssrd_instantaneous, 0)

        self.logger.debug("Converted cumulative radiation to instantaneous rates")
        return ssrd_instantaneous

    def get_diurnal_temperature_cycle(self, skt: np.ndarray) -> np.ndarray:
        """
        Extract diurnal temperature cycle characteristics.

        Args:
            skt: Skin temperature array (K)

        Returns:
            ndarray: Temperature anomalies from daily mean
        """
        # Calculate daily mean temperature
        n_days = skt.shape[2] // 24
        daily_means = np.zeros((*skt.shape[:2], n_days))

        for day in range(n_days):
            start_hour = day * 24
            end_hour = (day + 1) * 24
            daily_means[:, :, day] = np.mean(skt[:, :, start_hour:end_hour], axis=2)

        # Calculate temperature anomalies
        skt_anomalies = np.zeros_like(skt)
        for day in range(n_days):
            start_hour = day * 24
            end_hour = (day + 1) * 24
            for hour in range(24):
                if start_hour + hour < skt.shape[2]:
                    skt_anomalies[:, :, start_hour + hour] = (
                        skt[:, :, start_hour + hour] - daily_means[:, :, day]
                    )

        return skt_anomalies

    def get_available_files(self, year: int) -> dict:
        """
        Get list of available ERA5 files for specified year.

        Args:
            year: Year to check

        Returns:
            dict: Available files by month and variable
        """
        available_files = {}

        for month in range(1, 13):
            available_files[month] = {}
            for var in ['SKT', 'SSRD']:
                filename = self.file_pattern.format(var=var, month=month, year=year)
                filepath = os.path.join(self.data_dir, filename)
                available_files[month][var] = os.path.exists(filepath)

        return available_files

    def preprocess_for_diurnal_calculation(self, ssrd: np.ndarray, skt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess meteorological data for diurnal flux calculation.

        MATLAB Reference: Data preprocessing before diurnal flux calculations

        Args:
            ssrd: Solar radiation array
            skt: Skin temperature array

        Returns:
            tuple: (processed_ssrd, processed_skt) ready for flux calculations
        """
        # Convert cumulative radiation to instantaneous if needed
        # Check if SSRD appears to be cumulative (monotonically increasing)
        if np.all(np.diff(ssrd[:, :, :24], axis=2) >= 0):  # Check first day
            self.logger.info("Converting cumulative SSRD to instantaneous values")
            ssrd_processed = self.convert_cumulative_to_instantaneous_radiation(ssrd)
        else:
            ssrd_processed = ssrd.copy()

        # Ensure temperature is in Kelvin
        skt_processed = skt.copy()
        if np.mean(skt_processed) < 100:  # Likely Celsius
            self.logger.info("Converting temperature from Celsius to Kelvin")
            skt_processed += 273.15

        # Apply smoothing to reduce noise if needed
        # MATLAB Reference: Optional smoothing of meteorological fields
        # (Implementation would go here if required)

        self.logger.info("Meteorological data preprocessing completed")
        return ssrd_processed, skt_processed