"""
Diurnal Flux Output Writers

Write diurnal flux data to NetCDF files in GeosChem-compatible format.
Based on MATLAB write functions.

This module handles the creation of NetCDF output files for both monthly
and hourly flux data, maintaining compatibility with GeosChem input
requirements and CARDAMOM metadata standards.

Scientific Context:
The output files provide carbon flux data at appropriate temporal
resolution for atmospheric transport modeling and carbon cycle analysis.
Monthly files include uncertainties, while hourly files provide the
diurnal variability needed for realistic transport modeling.
"""

import os
import numpy as np
import netCDF4
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from netcdf_infrastructure import CARDAMOMNetCDFWriter


class DiurnalFluxWriter:
    """
    Write diurnal flux data to NetCDF files in GeosChem-compatible format.
    Based on MATLAB write functions.

    MATLAB Reference: write_monthlyflux_to_geoschem_format.m and
                     write_hourly_flux_to_geoschem_format.m
    """

    def __init__(self, output_base_dir: str):
        """
        Initialize diurnal flux writer.

        Args:
            output_base_dir: Base directory for output files
        """
        self.output_base_dir = output_base_dir
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ensure output directory exists
        os.makedirs(output_base_dir, exist_ok=True)

        # Initialize NetCDF infrastructure
        self.netcdf_writer = CARDAMOMNetCDFWriter()

    def write_monthly_flux_to_geoschem_format(self,
                                            flux_data: np.ndarray,
                                            uncertainty_data: np.ndarray,
                                            year: int,
                                            month: int,
                                            aux_data: Dict[str, Any],
                                            flux_name: str,
                                            experiment: int) -> str:
        """
        Write monthly flux with uncertainty to NetCDF.
        Equivalent to MATLAB write_monthlyflux_to_geoschem_format.

        MATLAB Reference: write_monthlyflux_to_geoschem_format.m

        Args:
            flux_data: Monthly flux array (Kg C/Km²/sec)
            uncertainty_data: Uncertainty array (Kg C/Km²/sec)
            year: Year for filename
            month: Month for filename
            aux_data: Auxiliary data with coordinates
            flux_name: Type of flux (GPP, REC, FIR, NEE, NBE)
            experiment: CMS experiment number

        Returns:
            str: Path to created NetCDF file
        """
        # Setup directory structure - MATLAB: directory organization
        output_dir = self._setup_monthly_output_directory(flux_name, experiment)
        year_dir = os.path.join(output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        filename = os.path.join(year_dir, f"{month:02d}.nc")

        # Skip if file already exists - MATLAB: file existence check
        if os.path.exists(filename):
            self.logger.info(f"Monthly flux file already exists: {filename}")
            return filename

        self.logger.info(f"Writing monthly {flux_name} flux file: {filename}")

        # Prepare data arrays - MATLAB: data preparation
        uncertainty_factor = self._calculate_uncertainty_factor(flux_data, uncertainty_data)

        # Create NetCDF file - MATLAB: NetCDF file creation
        with netCDF4.Dataset(filename, 'w') as nc:
            self._create_monthly_netcdf_structure(nc, aux_data, flux_data.shape)

            # Write data - MATLAB: data writing
            nc.variables['CO2_Flux'][:] = self._nan_to_zero(flux_data)
            nc.variables['Uncertainty'][:] = uncertainty_factor

            # Add metadata - MATLAB: metadata addition
            self._add_monthly_metadata(nc, flux_name, year, month, experiment)

        self.logger.info(f"Successfully wrote monthly flux file: {filename}")
        return filename

    def write_hourly_flux_to_geoschem_format(self,
                                           flux_data: np.ndarray,
                                           year: int,
                                           month: int,
                                           aux_data: Dict[str, Any],
                                           flux_name: str,
                                           experiment: int) -> List[str]:
        """
        Write hourly flux data to daily NetCDF files.
        Equivalent to MATLAB write_hourly_flux_to_geoschem_format.

        MATLAB Reference: write_hourly_flux_to_geoschem_format.m

        Args:
            flux_data: Hourly flux array (Kg C/Km²/sec)
            year: Year for filename
            month: Month for filename
            aux_data: Auxiliary data with coordinates
            flux_name: Type of flux (GPP, REC, FIR, NEE, NBE)
            experiment: CMS experiment number

        Returns:
            list: Paths to created NetCDF files
        """
        # Setup directory structure - MATLAB: hourly directory organization
        output_dir = self._setup_diurnal_output_directory(flux_name, experiment)
        month_dir = os.path.join(output_dir, str(year), f"{month:02d}")
        os.makedirs(month_dir, exist_ok=True)

        # Get days in month - MATLAB: days calculation
        days_in_month = self._get_days_in_month(year, month)

        created_files = []

        # Write daily files - MATLAB: daily file loop
        for day in range(1, days_in_month + 1):
            day_filename = os.path.join(month_dir, f"{day:02d}.nc")

            # Skip if file already exists
            if os.path.exists(day_filename):
                self.logger.debug(f"Hourly flux file already exists: {day_filename}")
                created_files.append(day_filename)
                continue

            # Extract 24 hours for this day - MATLAB: daily data extraction
            hour_start = (day - 1) * 24
            hour_end = day * 24

            if hour_end <= flux_data.shape[2]:
                daily_flux = flux_data[:, :, hour_start:hour_end]

                # Create daily NetCDF file - MATLAB: daily NetCDF creation
                with netCDF4.Dataset(day_filename, 'w') as nc:
                    self._create_hourly_netcdf_structure(nc, aux_data, daily_flux.shape)
                    nc.variables['CO2_Flux'][:] = self._nan_to_zero(daily_flux)
                    self._add_hourly_metadata(nc, flux_name, year, month, day, experiment)

                created_files.append(day_filename)
                self.logger.debug(f"Created hourly flux file: {day_filename}")

        self.logger.info(f"Created {len(created_files)} hourly {flux_name} files for {year}-{month:02d}")
        return created_files

    def _setup_monthly_output_directory(self, flux_name: str, experiment: int) -> str:
        """
        Setup directory structure for monthly files.

        MATLAB Reference: Monthly directory structure in MATLAB functions

        Args:
            flux_name: Type of flux
            experiment: CMS experiment number

        Returns:
            str: Output directory path
        """
        exp_suffix = f"_EXP{experiment}"
        output_dir = os.path.join(
            self.output_base_dir,
            f"MONTHLY_{flux_name.upper()}{exp_suffix}"
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _setup_diurnal_output_directory(self, flux_name: str, experiment: int) -> str:
        """
        Setup directory structure for diurnal files.

        MATLAB Reference: Diurnal directory structure in MATLAB functions

        Args:
            flux_name: Type of flux
            experiment: CMS experiment number

        Returns:
            str: Output directory path
        """
        exp_suffix = f"_EXP{experiment}"
        output_dir = os.path.join(
            self.output_base_dir,
            f"DIURNAL_{flux_name.upper()}{exp_suffix}"
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _create_monthly_netcdf_structure(self,
                                       nc_dataset: netCDF4.Dataset,
                                       aux_data: Dict[str, Any],
                                       data_shape: tuple) -> None:
        """
        Create NetCDF structure for monthly files.

        MATLAB Reference: NetCDF structure creation in MATLAB functions

        Args:
            nc_dataset: NetCDF dataset object
            aux_data: Auxiliary data with coordinates
            data_shape: Shape of flux data
        """
        # Create dimensions - MATLAB: dimension creation
        nc_dataset.createDimension('longitude', data_shape[1])
        nc_dataset.createDimension('latitude', data_shape[0])
        nc_dataset.createDimension('time', 1)

        # Create coordinate variables - MATLAB: coordinate variable creation
        lon_var = nc_dataset.createVariable('longitude', 'f4', ('longitude',))
        lat_var = nc_dataset.createVariable('latitude', 'f4', ('latitude',))
        time_var = nc_dataset.createVariable('time', 'f4', ('time',))

        # Create data variables - MATLAB: data variable creation
        flux_var = nc_dataset.createVariable('CO2_Flux', 'f4',
                                           ('latitude', 'longitude', 'time'),
                                           zlib=True, complevel=4)
        unc_var = nc_dataset.createVariable('Uncertainty', 'f4',
                                          ('latitude', 'longitude', 'time'),
                                          zlib=True, complevel=4)

        # Set coordinate data - MATLAB: coordinate data assignment
        lon_var[:] = self._get_longitude_coordinates(data_shape[1])
        lat_var[:] = self._get_latitude_coordinates(data_shape[0])
        time_var[:] = 1

        # Add coordinate attributes - MATLAB: coordinate attributes
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'longitude'
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'latitude'
        time_var.units = 'N/A'
        time_var.long_name = 'time'

        # Add data variable attributes - MATLAB: data variable attributes
        flux_var.units = 'Kg C/Km^2/sec'
        flux_var.long_name = 'Monthly Carbon Flux'
        unc_var.units = 'factor'
        unc_var.long_name = 'Uncertainty Factor'

    def _create_hourly_netcdf_structure(self,
                                      nc_dataset: netCDF4.Dataset,
                                      aux_data: Dict[str, Any],
                                      data_shape: tuple) -> None:
        """
        Create NetCDF structure for hourly files.

        MATLAB Reference: Hourly NetCDF structure creation in MATLAB functions

        Args:
            nc_dataset: NetCDF dataset object
            aux_data: Auxiliary data with coordinates
            data_shape: Shape of flux data (lat, lon, 24)
        """
        # Create dimensions - MATLAB: hourly dimension creation
        nc_dataset.createDimension('longitude', data_shape[1])
        nc_dataset.createDimension('latitude', data_shape[0])
        nc_dataset.createDimension('time', 24)  # 24 hours

        # Create coordinate variables
        lon_var = nc_dataset.createVariable('longitude', 'f4', ('longitude',))
        lat_var = nc_dataset.createVariable('latitude', 'f4', ('latitude',))
        time_var = nc_dataset.createVariable('time', 'f4', ('time',))

        # Create data variable
        flux_var = nc_dataset.createVariable('CO2_Flux', 'f4',
                                           ('latitude', 'longitude', 'time'),
                                           zlib=True, complevel=4)

        # Set coordinate data
        lon_var[:] = self._get_longitude_coordinates(data_shape[1])
        lat_var[:] = self._get_latitude_coordinates(data_shape[0])
        time_var[:] = np.arange(0.5, 24, 1)  # Hour centers: 0.5, 1.5, ..., 23.5

        # Add attributes
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'longitude'
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'latitude'
        time_var.units = 'hour'
        time_var.long_name = 'hour of day'
        flux_var.units = 'Kg C/Km^2/sec'
        flux_var.long_name = 'Hourly Carbon Flux'

    def _add_monthly_metadata(self,
                            nc_dataset: netCDF4.Dataset,
                            flux_name: str,
                            year: int,
                            month: int,
                            experiment: int) -> None:
        """
        Add metadata to monthly NetCDF file.

        MATLAB Reference: Metadata addition in MATLAB functions

        Args:
            nc_dataset: NetCDF dataset object
            flux_name: Type of flux
            year: Year
            month: Month
            experiment: CMS experiment number
        """
        # Global attributes - MATLAB: global attribute setting
        nc_dataset.title = f"CARDAMOM Monthly {flux_name.upper()} Flux"
        nc_dataset.institution = "NASA Jet Propulsion Laboratory"
        nc_dataset.source = f"CARDAMOM Diurnal Processor - CMS Experiment {experiment}"
        nc_dataset.history = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        nc_dataset.references = "Bloom et al. (2016), Nature Geoscience"
        nc_dataset.conventions = "CF-1.6"
        nc_dataset.experiment_number = experiment
        nc_dataset.year = year
        nc_dataset.month = month

        # Flux-specific metadata
        if flux_name.upper() == 'GPP':
            nc_dataset.description = "Gross Primary Productivity from CMS inversion"
        elif flux_name.upper() == 'REC':
            nc_dataset.description = "Ecosystem Respiration from CMS inversion"
        elif flux_name.upper() == 'FIR':
            nc_dataset.description = "Fire Emissions from CMS inversion"
        elif flux_name.upper() == 'NEE':
            nc_dataset.description = "Net Ecosystem Exchange from CMS inversion"
        elif flux_name.upper() == 'NBE':
            nc_dataset.description = "Net Biome Exchange from CMS inversion"

    def _add_hourly_metadata(self,
                           nc_dataset: netCDF4.Dataset,
                           flux_name: str,
                           year: int,
                           month: int,
                           day: int,
                           experiment: int) -> None:
        """
        Add metadata to hourly NetCDF file.

        Args:
            nc_dataset: NetCDF dataset object
            flux_name: Type of flux
            year: Year
            month: Month
            day: Day
            experiment: CMS experiment number
        """
        # Global attributes
        nc_dataset.title = f"CARDAMOM Hourly {flux_name.upper()} Flux"
        nc_dataset.institution = "NASA Jet Propulsion Laboratory"
        nc_dataset.source = f"CARDAMOM Diurnal Processor - CMS Experiment {experiment}"
        nc_dataset.history = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        nc_dataset.references = "Bloom et al. (2016), Nature Geoscience"
        nc_dataset.conventions = "CF-1.6"
        nc_dataset.experiment_number = experiment
        nc_dataset.year = year
        nc_dataset.month = month
        nc_dataset.day = day

        # Add diurnal processing information
        nc_dataset.processing_method = "Monthly to hourly downscaling using meteorological drivers"
        if flux_name.upper() == 'GPP':
            nc_dataset.driver_variable = "Solar radiation (SSRD)"
        elif flux_name.upper() == 'REC':
            nc_dataset.driver_variable = "Skin temperature (SKT)"
        elif flux_name.upper() == 'FIR':
            nc_dataset.driver_variable = "GFED diurnal fire patterns"

    def _calculate_uncertainty_factor(self,
                                    flux_data: np.ndarray,
                                    uncertainty_data: np.ndarray) -> np.ndarray:
        """
        Calculate uncertainty factor from flux and uncertainty data.

        MATLAB Reference: Uncertainty factor calculation in MATLAB functions

        Args:
            flux_data: Flux values
            uncertainty_data: Uncertainty values

        Returns:
            ndarray: Uncertainty factors
        """
        # Calculate relative uncertainty factor
        uncertainty_factor = np.ones_like(flux_data)

        # Avoid division by zero
        nonzero_mask = np.abs(flux_data) > 1e-10

        uncertainty_factor[nonzero_mask] = (
            1.0 + np.abs(uncertainty_data[nonzero_mask] / flux_data[nonzero_mask])
        )

        # Cap uncertainty factors at reasonable values
        uncertainty_factor = np.clip(uncertainty_factor, 1.0, 10.0)

        return uncertainty_factor

    def _nan_to_zero(self, data: np.ndarray) -> np.ndarray:
        """
        Convert NaN values to zero for NetCDF output.

        Args:
            data: Input array

        Returns:
            ndarray: Array with NaN values replaced by zeros
        """
        result = data.copy()
        result[np.isnan(result)] = 0.0
        return result

    def _get_longitude_coordinates(self, n_lon: int) -> np.ndarray:
        """
        Get longitude coordinates for CONUS grid.

        Args:
            n_lon: Number of longitude points

        Returns:
            ndarray: Longitude coordinates
        """
        # CONUS longitude range with 0.5° resolution
        lon_start = -124.75
        lon_step = 0.5
        return np.arange(lon_start, lon_start + n_lon * lon_step, lon_step)

    def _get_latitude_coordinates(self, n_lat: int) -> np.ndarray:
        """
        Get latitude coordinates for CONUS grid.

        Args:
            n_lat: Number of latitude points

        Returns:
            ndarray: Latitude coordinates
        """
        # CONUS latitude range with 0.5° resolution
        lat_start = 24.75
        lat_step = 0.5
        return np.arange(lat_start, lat_start + n_lat * lat_step, lat_step)

    def _get_days_in_month(self, year: int, month: int) -> int:
        """
        Get number of days in specified month.

        Args:
            year: Year
            month: Month (1-12)

        Returns:
            int: Number of days in month
        """
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

    def create_file_manifest(self, created_files: List[str], output_dir: str) -> str:
        """
        Create manifest file listing all created NetCDF files.

        Args:
            created_files: List of created file paths
            output_dir: Output directory for manifest

        Returns:
            str: Path to manifest file
        """
        manifest_file = os.path.join(output_dir, "file_manifest.txt")

        with open(manifest_file, 'w') as f:
            f.write(f"# CARDAMOM Diurnal Flux File Manifest\n")
            f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write(f"# Total files: {len(created_files)}\n\n")

            for filepath in sorted(created_files):
                f.write(f"{filepath}\n")

        self.logger.info(f"Created file manifest: {manifest_file}")
        return manifest_file