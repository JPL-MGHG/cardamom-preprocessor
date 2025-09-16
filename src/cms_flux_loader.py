"""
CMS Monthly Flux Loader

Load and process monthly CMS flux data from NetCDF files.
Based on MATLAB load_erens_cms_monthly_fluxes function.

This module handles loading monthly CMS flux data and provides spatial
interpolation for missing values using scattered interpolation equivalent
to MATLAB's scatteredInterpolant.

Scientific Context:
CMS (Carbon Monitoring System) provides monthly carbon flux estimates
that serve as the basis for diurnal downscaling. Includes GPP, respiration,
fire emissions, NEE, and NBE with uncertainties.
"""

import os
import numpy as np
import xarray as xr
import logging
from typing import Dict, Tuple, Optional
from scipy.interpolate import griddata

from coordinate_systems import CoordinateGrid


class CMSFluxLoader:
    """
    Load and process monthly CMS flux data from NetCDF files.
    Based on MATLAB load_erens_cms_monthly_fluxes function.

    MATLAB Reference: load_erens_cms_monthly_fluxes.m
    """

    def __init__(self, data_dir: str = "./DATA/DATA_FROM_EREN/CMS_CONUS_JUL25/"):
        """
        Initialize CMS flux loader.

        Args:
            data_dir: Directory containing CMS NetCDF files
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

        # CMS experiment file configuration - MATLAB: file definitions
        self.experiment_files = {
            1: {
                'mean': 'Outputmean_exp1redo5.nc',      # MATLAB: CMS experiment 1 mean file
                'std': 'Outputstd_exp1redo5.nc'        # MATLAB: CMS experiment 1 std file
            },
            2: {
                'mean': 'Outputmean_exp2redo5.nc',      # MATLAB: CMS experiment 2 mean file
                'std': 'Outputstd_exp2redo5.nc'        # MATLAB: CMS experiment 2 std file
            }
        }

    def load_monthly_fluxes(self, experiment_number: int) -> Dict[str, np.ndarray]:
        """
        Load monthly CMS fluxes for specified experiment.

        MATLAB Reference: Main function body in load_erens_cms_monthly_fluxes.m

        Args:
            experiment_number: CMS experiment number (1 or 2)

        Returns:
            dict: Dictionary with flux arrays and uncertainties
                Keys: 'GPP', 'REC', 'FIR', 'NEE', 'NBE', 'GPPunc', 'RECunc', etc.
        """
        if experiment_number not in self.experiment_files:
            raise ValueError(f"Invalid experiment number: {experiment_number}. Must be 1 or 2.")

        files = self.experiment_files[experiment_number]
        mean_file = os.path.join(self.data_dir, files['mean'])
        std_file = os.path.join(self.data_dir, files['std'])

        self.logger.info(f"Loading CMS fluxes for experiment {experiment_number}")
        self.logger.info(f"Mean file: {mean_file}")
        self.logger.info(f"Std file: {std_file}")

        # Verify files exist
        if not os.path.exists(mean_file):
            raise FileNotFoundError(f"CMS mean file not found: {mean_file}")
        if not os.path.exists(std_file):
            raise FileNotFoundError(f"CMS std file not found: {std_file}")

        # Load flux data - MATLAB: loading and permuting data arrays
        fluxes = {}

        # Load mean flux data with permutation to match MATLAB [2,1,3] order
        with xr.open_dataset(mean_file) as ds:
            # MATLAB Reference: permute(data, [2,1,3]) operation
            fluxes['GPP'] = ds['GPP'].values.transpose(1, 0, 2)          # Gross Primary Productivity
            fluxes['REC'] = ds['Resp_eco'].values.transpose(1, 0, 2)     # Ecosystem Respiration
            fluxes['FIR'] = ds['Fire'].values.transpose(1, 0, 2)         # Fire Emissions
            fluxes['NEE'] = ds['NEE'].values.transpose(1, 0, 2)          # Net Ecosystem Exchange
            fluxes['NBE'] = ds['NBE'].values.transpose(1, 0, 2)          # Net Biome Exchange

        # Load uncertainty data with same permutation
        with xr.open_dataset(std_file) as ds:
            # MATLAB Reference: permute(uncertainty_data, [2,1,3]) operation
            fluxes['GPPunc'] = ds['GPP'].values.transpose(1, 0, 2)
            fluxes['RECunc'] = ds['Resp_eco'].values.transpose(1, 0, 2)
            fluxes['FIRunc'] = ds['Fire'].values.transpose(1, 0, 2)
            fluxes['NEEunc'] = ds['NEE'].values.transpose(1, 0, 2)
            fluxes['NBEunc'] = ds['NBE'].values.transpose(1, 0, 2)

        # Apply spatial interpolation for missing values - MATLAB: scattered interpolation
        fluxes = self.patch_missing_values(fluxes)

        self.logger.info(f"Successfully loaded CMS fluxes for experiment {experiment_number}")
        self.logger.info(f"Flux data shape: {fluxes['GPP'].shape}")

        return fluxes

    def patch_missing_values(self, fluxes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Fill missing values using spatial interpolation.
        Equivalent to MATLAB scattered interpolation logic.

        MATLAB Reference: scatteredInterpolant usage in MATLAB function

        Args:
            fluxes: Dictionary with flux arrays containing NaN values

        Returns:
            dict: Flux arrays with missing values interpolated
        """
        self.logger.info("Applying spatial interpolation to fill missing values")

        # Load coordinate and land-sea mask data - MATLAB: coordinate system setup
        land_sea_mask = self._load_conus_land_sea_mask()
        x_coords, y_coords = self._get_coordinate_grids()

        # Identify finite and missing data points - MATLAB: finding valid/invalid points
        flux_sum = np.zeros_like(fluxes['GPP'][:, :, 0])
        for flux_name in ['GPP', 'REC', 'FIR', 'NEE', 'NBE']:
            flux_sum += np.nanmean(fluxes[flux_name], axis=2)

        # MATLAB Reference: Valid and missing point identification
        valid_points = np.isfinite(flux_sum) & (land_sea_mask > 0)
        missing_points = (~np.isfinite(flux_sum)) & (land_sea_mask > 0)

        n_valid = np.sum(valid_points)
        n_missing = np.sum(missing_points)

        self.logger.info(f"Found {n_valid} valid points and {n_missing} missing points")

        if n_missing == 0:
            self.logger.info("No missing values found, skipping interpolation")
            return fluxes

        # Apply interpolation to all flux variables
        for flux_name in fluxes.keys():
            self.logger.debug(f"Interpolating missing values for {flux_name}")
            fluxes[flux_name] = self._interpolate_flux_field(
                fluxes[flux_name], valid_points, missing_points, x_coords, y_coords
            )

        self.logger.info("Spatial interpolation completed successfully")
        return fluxes

    def _interpolate_flux_field(self,
                              flux_data: np.ndarray,
                              valid_points: np.ndarray,
                              missing_points: np.ndarray,
                              x_coords: np.ndarray,
                              y_coords: np.ndarray) -> np.ndarray:
        """
        Interpolate missing flux values using scattered interpolation.
        Equivalent to MATLAB scatteredInterpolant usage.

        MATLAB Reference: scatteredInterpolant interpolation in MATLAB function

        Args:
            flux_data: Flux array with shape (nx, ny, nt)
            valid_points: Boolean mask for valid data points
            missing_points: Boolean mask for missing data points
            x_coords: X coordinates grid
            y_coords: Y coordinates grid

        Returns:
            ndarray: Interpolated flux data
        """
        n_timesteps = flux_data.shape[2]
        interpolated_data = flux_data.copy()

        # Get coordinate arrays for valid and missing points
        valid_x = x_coords[valid_points]
        valid_y = y_coords[valid_points]
        missing_x = x_coords[missing_points]
        missing_y = y_coords[missing_points]

        # Interpolate each time step
        for t in range(n_timesteps):
            flux_slice = flux_data[:, :, t]

            # Get valid data values for this time step
            valid_values = flux_slice[valid_points]

            # Skip if no valid values or all NaN
            if len(valid_values) == 0 or np.all(np.isnan(valid_values)):
                continue

            # Remove NaN values from valid points
            finite_mask = np.isfinite(valid_values)
            if np.sum(finite_mask) < 3:  # Need at least 3 points for interpolation
                continue

            finite_x = valid_x[finite_mask]
            finite_y = valid_y[finite_mask]
            finite_values = valid_values[finite_mask]

            try:
                # MATLAB Reference: scatteredInterpolant linear interpolation
                interpolated_values = griddata(
                    (finite_x, finite_y), finite_values,
                    (missing_x, missing_y), method='linear', fill_value=np.nan
                )

                # Use nearest neighbor for remaining NaN values
                nan_mask = np.isnan(interpolated_values)
                if np.any(nan_mask):
                    nearest_values = griddata(
                        (finite_x, finite_y), finite_values,
                        (missing_x[nan_mask], missing_y[nan_mask]),
                        method='nearest', fill_value=0
                    )
                    interpolated_values[nan_mask] = nearest_values

                # Update interpolated data
                interpolated_data[missing_points, t] = interpolated_values

            except Exception as e:
                self.logger.warning(f"Interpolation failed for timestep {t}: {e}")
                continue

        return interpolated_data

    def _load_conus_land_sea_mask(self) -> np.ndarray:
        """
        Load or create CONUS land-sea mask.

        MATLAB Reference: Land-sea mask loading in MATLAB function

        Returns:
            ndarray: Land-sea mask (1=land, 0=ocean)
        """
        # Try to load existing mask, or create placeholder
        mask_file = os.path.join(self.data_dir, "conus_land_sea_mask.nc")

        if os.path.exists(mask_file):
            try:
                with xr.open_dataset(mask_file) as ds:
                    mask = ds['land_sea_mask'].values.transpose(1, 0)  # Match coordinate order
                    self.logger.info("Loaded existing CONUS land-sea mask")
                    return mask
            except Exception as e:
                self.logger.warning(f"Failed to load land-sea mask: {e}")

        # Create placeholder mask - assume CONUS grid dimensions
        # MATLAB Reference: Default mask creation for CONUS region
        self.logger.warning("Creating placeholder land-sea mask")
        mask = np.ones((120, 160))  # Typical CONUS 0.5° grid dimensions

        return mask

    def _get_coordinate_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get coordinate grids for interpolation.

        MATLAB Reference: Coordinate grid creation in MATLAB function

        Returns:
            tuple: (x_coords, y_coords) coordinate arrays
        """
        # Create CONUS coordinate grid - MATLAB: meshgrid equivalent
        lon_range = np.arange(-124.75, -65.25, 0.5)  # CONUS longitude range
        lat_range = np.arange(24.75, 60.25, 0.5)     # CONUS latitude range

        # Create coordinate grids
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

        self.logger.debug(f"Created coordinate grids: lon {lon_grid.shape}, lat {lat_grid.shape}")

        return lon_grid, lat_grid

    def validate_flux_data(self, fluxes: Dict[str, np.ndarray]) -> bool:
        """
        Validate loaded flux data for physical reasonableness.

        Args:
            fluxes: Dictionary with flux arrays

        Returns:
            bool: True if validation passes
        """
        validation_passed = True

        # Check for required flux types
        required_fluxes = ['GPP', 'REC', 'FIR', 'NEE', 'NBE']
        for flux_name in required_fluxes:
            if flux_name not in fluxes:
                self.logger.error(f"Missing required flux type: {flux_name}")
                validation_passed = False

        # Check data ranges
        for flux_name, flux_data in fluxes.items():
            if 'unc' in flux_name:
                continue  # Skip uncertainty validation

            # Check for reasonable carbon flux ranges (gC/m²/day)
            min_val = np.nanmin(flux_data)
            max_val = np.nanmax(flux_data)

            # Basic range checks based on typical carbon flux values
            if flux_name == 'GPP' and (min_val < -1 or max_val > 50):
                self.logger.warning(f"GPP values outside typical range: [{min_val:.2f}, {max_val:.2f}]")
            elif flux_name in ['REC', 'FIR'] and (min_val < -1 or max_val > 30):
                self.logger.warning(f"{flux_name} values outside typical range: [{min_val:.2f}, {max_val:.2f}]")
            elif flux_name in ['NEE', 'NBE'] and (min_val < -30 or max_val > 30):
                self.logger.warning(f"{flux_name} values outside typical range: [{min_val:.2f}, {max_val:.2f}]")

        self.logger.info("Flux data validation completed")
        return validation_passed