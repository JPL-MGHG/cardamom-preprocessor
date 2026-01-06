"""
ECMWF ERA5 Data Downloader for CARDAMOM Preprocessing

This module downloads ERA5 reanalysis data from ECMWF Climate Data Store and
processes it into analysis-ready NetCDF files with STAC metadata.

Scientific Context:
ERA5 provides global, high-resolution meteorological reanalysis data essential
for CARDAMOM carbon cycle modeling, including temperature, precipitation, and
radiation fields required for ecosystem carbon flux calculations.

Features:
- Downloads raw ERA5 variables via ECMWF CDS API
- Calculates derived variables (VPD, temperature extrema)
- Applies unit conversions to CARDAMOM standards
- Generates analysis-ready NetCDF output
- Creates STAC Item metadata for each output file
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import xarray as xr
import cdsapi
import warnings
import zipfile

from downloaders.base import BaseDownloader
from atmospheric_science import (
    calculate_vapor_pressure_deficit_matlab,
    saturation_pressure_water_matlab,
)
from cardamom_variables import CARDAMOM_VARIABLE_REGISTRY
from units_constants import temperature_kelvin_to_celsius

logger = logging.getLogger(__name__)


class ECMWFDownloader(BaseDownloader):
    """
    Download and process ERA5 meteorological data for CARDAMOM.

    This downloader handles ERA5 variables with special logic for derived
    variables like VPD that require multiple raw inputs. Uses the cdsapi
    library for Climate Data Store (CDS) access.

    Attributes:
        cds_client: cdsapi.Client instance for CDS API access
        requested_variables: List of output variables to generate
        raw_variables_needed: List of raw ERA5 variables required
    """

    def __init__(
        self,
        output_directory: str,
        keep_raw_files: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize ECMWF downloader using cdsapi for Climate Data Store access.

        Credentials must be configured in ~/.cdsapirc file with:
            url: https://cds.climate.copernicus.eu/api
            key: <your-personal-access-token>

        Args:
            output_directory (str): Root output directory path
            keep_raw_files (bool): Retain intermediate raw files. Default: False
            verbose (bool): Print debug messages. Default: False

        Raises:
            Exception: If cdsapi client cannot be initialized or credentials are missing
        """

        super().__init__(output_directory, keep_raw_files, verbose)

        # Initialize cdsapi client for Climate Data Store access
        try:
            self.cds_client = cdsapi.Client()
            logger.info("Initialized CDS API client via cdsapi")
        except Exception as e:
            logger.error(f"Failed to initialize CDS API client: {e}")
            logger.error(
                "Ensure ~/.cdsapirc credentials are configured with URL and API key. "
                "Get your Personal Access Token from https://cds.climate.copernicus.eu/profile"
            )
            raise

        self.requested_variables = []
        self.raw_variables_needed = []

    def _get_time_dimension(self, dataset: xr.Dataset) -> str:
        """
        Identify the time dimension name in an xarray Dataset.

        ERA5 datasets may use 'time', 'valid_time', or other time dimension names.
        This method detects which one is present.

        Args:
            dataset (xr.Dataset): xarray Dataset to check

        Returns:
            str: Name of the time dimension

        Raises:
            ValueError: If no recognized time dimension is found
        """

        time_dim_candidates = ['time', 'valid_time']

        for candidate in time_dim_candidates:
            if candidate in dataset.dims:
                logger.debug(f"Detected time dimension: {candidate}")
                return candidate

        raise ValueError(
            f"No recognized time dimension found in dataset. "
            f"Available dimensions: {list(dataset.dims.keys())}"
        )
    def _download_variable_batch(
        self,
        raw_variables: List[str],
        product_type: str,
        year: int,
        month: int,
    ) -> Dict[str, Path]:
        """
        Download multiple ERA5 variables in a single API request.

        Batching variables sharing the same product type significantly reduces
        API overhead and download time compared to individual requests.

        Args:
            raw_variables (List[str]): Raw ERA5 variable names to download together
            product_type (str): Either 'monthly_averaged_reanalysis' or
                'monthly_averaged_reanalysis_by_hour_of_day'
            year (int): Year to download
            month (int): Month to download (1-12)

        Returns:
            Dict[str, Path]: Mapping of variable names to file paths

        Raises:
            RuntimeError: If download fails or files cannot be extracted
        """

        logger.info(
            f"Downloading batch of {len(raw_variables)} variables "
            f"({product_type}): {raw_variables}"
        )

        # Create output directory
        output_dir = self.output_directory / self.raw_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual variable file paths that will be extracted
        extracted_files = {}
        for raw_variable in raw_variables:
            extracted_files[raw_variable] = (
                output_dir / f"{raw_variable}_{year}_{month:02d}.nc"
            )

        # Check if individual files already exist
        all_exist = all(filepath.exists() for filepath in extracted_files.values())
        if all_exist:
            logger.info(
                f"All {len(raw_variables)} variables already exist, skipping batch download"
            )
            return extracted_files

        # Prepare time parameter based on product type
        if product_type == 'monthly_averaged_reanalysis_by_hour_of_day':
            time_param = [f'{h:02d}:00' for h in range(24)]
        else:
            time_param = '00:00'

        try:
            # Build CDS API request for multiple variables
            cds_request = {
                'product_type': product_type,
                'variable': raw_variables,  # Pass as list for batch download
                'year': str(year),
                'month': f'{month:02d}',
                'time': time_param,
                'data_format': 'netcdf',  # Request NetCDF format
                'download_format': 'zip',  # Download as ZIP archive
            }

            logger.debug(f"CDS request: {cds_request}")

            # Download combined file as ZIP
            zip_filepath = output_dir / f"batch_{product_type}_{year}_{month:02d}.zip"
            self.cds_client.retrieve(
                'reanalysis-era5-single-levels-monthly-means',
                cds_request,
                str(zip_filepath),
            )

            # Verify ZIP file was created
            if not zip_filepath.exists():
                raise RuntimeError(
                    f"Batch download verification failed: ZIP file not found at {zip_filepath}"
                )

            file_size_mb = zip_filepath.stat().st_size / (1024 * 1024)
            logger.info(
                f"Successfully downloaded batch ZIP file: {zip_filepath} ({file_size_mb:.2f} MB)"
            )

            # Unzip the archive and extract variables from the NC files inside
            logger.info(f"Extracting ZIP archive to get NetCDF files")
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                # Find all NetCDF files in the ZIP
                nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                if not nc_files:
                    raise RuntimeError(
                        f"No NetCDF files found in ZIP archive. Contents: {zip_ref.namelist()}"
                    )

                logger.info(f"Found {len(nc_files)} NetCDF files in ZIP: {nc_files}")

                # Extract each NC file to a temporary location and process
                temp_nc_files = []
                for nc_file in nc_files:
                    # Extract NC file from ZIP to temporary location
                    temp_nc_path = output_dir / f"temp_{Path(nc_file).name}"
                    with zip_ref.open(nc_file) as nc_source:
                        with open(temp_nc_path, 'wb') as nc_dest:
                            nc_dest.write(nc_source.read())
                    temp_nc_files.append(temp_nc_path)
                    logger.debug(f"Extracted {nc_file} from ZIP to {temp_nc_path}")

            # Clean up ZIP file
            zip_filepath.unlink()
            logger.debug(f"Cleaned up temporary ZIP file: {zip_filepath}")

            # Process the extracted NC files to find and extract requested variables
            logger.info(f"Processing {len(temp_nc_files)} NC files to extract requested variables")
            variables_found = {}

            for temp_nc_path in temp_nc_files:
                try:
                    nc_ds = xr.open_dataset(temp_nc_path, engine='netcdf4')
                    available_vars = list(nc_ds.data_vars)
                    logger.debug(f"File {temp_nc_path.name} contains variables: {available_vars}")

                    # Check which requested variables are in this file
                    for requested_var in raw_variables:
                        # Check if variable exists directly or as an alias
                        actual_var_name = None
                        if requested_var in nc_ds.data_vars:
                            actual_var_name = requested_var
                        else:
                            # Try aliases
                            name_mapping = self._get_variable_name_mapping()
                            if requested_var in name_mapping:
                                for alias in name_mapping[requested_var]:
                                    if alias in nc_ds.data_vars:
                                        actual_var_name = alias
                                        break

                        if actual_var_name:
                            variables_found[requested_var] = (temp_nc_path, actual_var_name)
                            logger.debug(
                                f"Found {requested_var} (as '{actual_var_name}') in {temp_nc_path.name}"
                            )

                    nc_ds.close()
                except Exception as e:
                    logger.error(f"Failed to read {temp_nc_path}: {e}")
                    raise

            # Verify all requested variables were found
            missing_vars = set(raw_variables) - set(variables_found.keys())
            if missing_vars:
                raise RuntimeError(
                    f"Could not find all requested variables. Missing: {missing_vars}. "
                    f"Found: {list(variables_found.keys())}"
                )

            # Now extract individual variables and save them
            logger.info(f"Extracting individual variables from source NC files")
            for requested_var, (source_file, actual_var_name) in variables_found.items():
                output_filepath = extracted_files[requested_var]

                if output_filepath.exists():
                    logger.debug(f"File already exists, skipping: {output_filepath}")
                    continue

                # Read variable from source file and save to individual file
                source_ds = xr.open_dataset(source_file, engine='netcdf4')
                var_data = source_ds[[actual_var_name]]
                var_data.to_netcdf(output_filepath)
                source_ds.close()

                logger.debug(f"Extracted {requested_var} (as '{actual_var_name}') to {output_filepath}")

            # Clean up temporary NC files
            for temp_nc_path in temp_nc_files:
                temp_nc_path.unlink()
                logger.debug(f"Cleaned up temporary NC file: {temp_nc_path}")

            # All variables have been extracted from the multiple NC files
            logger.info(f"Successfully extracted all {len(extracted_files)} requested variables from batch download")

            return extracted_files

        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            # Clean up temporary NC files if extraction was interrupted
            try:
                for temp_file in temp_nc_files:
                    if temp_file.exists():
                        temp_file.unlink()
                        logger.debug(f"Cleaned up temporary file: {temp_file}")
            except (NameError, UnboundLocalError):
                # temp_nc_files may not be defined if error occurred before extraction
                pass
            raise RuntimeError(f"Failed to download variable batch: {e}") from e

    def _get_variable_name_mapping(self) -> Dict[str, List[str]]:
        """
        Get mapping of ERA5 variable names to their possible aliases in NetCDF files.

        ERA5 CDS API returns variables with abbreviated names that may differ
        from the requested full names. This mapping handles common aliases.

        Returns:
            Dict[str, List[str]]: Maps requested variable names to possible aliases
        """

        return {
            '2m_temperature': ['2m_temperature', 't2m', '2t'],
            '2m_dewpoint_temperature': ['2m_dewpoint_temperature', 'd2m', '2d'],
            'total_precipitation': ['total_precipitation', 'tp'],
            'surface_solar_radiation_downwards': [
                'surface_solar_radiation_downwards',
                'ssrd',
            ],
            'surface_thermal_radiation_downwards': [
                'surface_thermal_radiation_downwards',
                'strd',
            ],
            'skin_temperature': ['skin_temperature', 'skt'],
            'snowfall': ['snowfall', 'sf'],
        }

    def _find_variable_in_dataset(
        self, batch_ds: xr.Dataset, requested_variable: str
    ) -> str:
        """
        Find actual variable name in dataset given a requested ERA5 variable name.

        Handles cases where CDS API returns abbreviated names instead of full names.

        Args:
            batch_ds (xr.Dataset): Opened batch download dataset
            requested_variable (str): Variable name that was requested

        Returns:
            str: Actual variable name found in the dataset

        Raises:
            ValueError: If variable not found under any known alias
        """

        # Try exact match first
        if requested_variable in batch_ds.data_vars:
            return requested_variable

        # Try known aliases for this variable
        name_mapping = self._get_variable_name_mapping()
        if requested_variable in name_mapping:
            for alias in name_mapping[requested_variable]:
                if alias in batch_ds.data_vars:
                    logger.debug(
                        f"Found variable {requested_variable} as '{alias}' in batch file"
                    )
                    return alias

        # Fallback: Try fuzzy matching
        available_vars = list(batch_ds.data_vars)
        raise ValueError(
            f"Variable '{requested_variable}' not found in batch file. "
            f"Checked aliases: {name_mapping.get(requested_variable, [])}. "
            f"Available variables: {available_vars}"
        )

    def _extract_variables_from_batch(
        self,
        batch_filepath: Path,
        extracted_files: Dict[str, Path],
    ) -> None:
        """
        Extract individual variables from a batch-downloaded file.

        Each variable is saved to its own file for compatibility with the
        existing processing pipeline. Handles variable name aliasing since
        CDS API may return abbreviated variable names.

        Args:
            batch_filepath (Path): Path to the combined batch download file
            extracted_files (Dict[str, Path]): Mapping of variable names to output paths

        Raises:
            RuntimeError: If extraction fails
        """

        try:
            # Open batch file (using netcdf4 engine for CDS downloads)
            # CDS API now returns unarchived NetCDF4 format due to download_format parameter
            batch_ds = xr.open_dataset(batch_filepath, engine='netcdf4')

            # Extract each variable to its own file
            for requested_variable, output_filepath in extracted_files.items():
                if output_filepath.exists():
                    logger.debug(
                        f"File already exists, skipping extraction: {output_filepath}"
                    )
                    continue

                # Find actual variable name in dataset (handles aliases)
                actual_var_name = self._find_variable_in_dataset(
                    batch_ds, requested_variable
                )

                # Extract variable
                var_data = batch_ds[[actual_var_name]]

                # Save variable to individual file
                var_data.to_netcdf(output_filepath)
                logger.debug(
                    f"Extracted {requested_variable} (as '{actual_var_name}') to {output_filepath}"
                )

            batch_ds.close()
            logger.info(f"Successfully extracted {len(extracted_files)} variables")

        except Exception as e:
            logger.error(f"Variable extraction failed: {e}")
            raise RuntimeError(f"Failed to extract variables from batch: {e}") from e

    def _resolve_variable_dependencies(
        self, variables: List[str]
    ) -> Dict[str, List[str]]:
        """
        Resolve raw ERA5 variables needed to produce requested output variables.

        Special handling: VPD requires both 2m_temperature and 2m_dewpoint_temperature.

        Args:
            variables (List[str]): List of CARDAMOM output variable names

        Returns:
            Dict[str, List[str]]: Mapping of raw ERA5 variable names to their
                CARDAMOM output variables

        Example:
            >>> deps = downloader._resolve_variable_dependencies(['t2m_min', 'vpd'])
            >>> print(deps)
            {'2m_temperature': ['t2m_min', 'vpd'],
             '2m_dewpoint_temperature': ['vpd']}
        """

        # Map output variables to raw ERA5 variables they depend on
        output_to_raw_mapping = {
            't2m_min': ['2m_temperature'],
            't2m_max': ['2m_temperature'],
            'vpd': ['2m_temperature', '2m_dewpoint_temperature'],
            'total_prec': ['total_precipitation'],
            'ssrd': ['surface_solar_radiation_downwards'],
            'strd': ['surface_thermal_radiation_downwards'],
            'skt': ['skin_temperature'],
            'snowfall': ['snowfall'],
        }

        # Collect all unique raw variables needed
        raw_variables_dict = {}

        for output_var in variables:
            if output_var not in output_to_raw_mapping:
                raise ValueError(
                    f"Unknown output variable: {output_var}. "
                    f"Available: {list(output_to_raw_mapping.keys())}"
                )

            for raw_var in output_to_raw_mapping[output_var]:
                if raw_var not in raw_variables_dict:
                    raw_variables_dict[raw_var] = []
                raw_variables_dict[raw_var].append(output_var)

        logger.debug(f"Variable dependency resolution: {raw_variables_dict}")
        return raw_variables_dict

    def _download_raw_era5(
        self,
        raw_variables: Dict[str, List[str]],
        year: int,
        month: int,
    ) -> Dict[str, Path]:
        """
        Download raw ERA5 data from Climate Data Store using batch API calls.

        Optimizes performance by grouping variables by product type and downloading
        them together in a single API request, rather than making separate requests
        for each variable.

        Product types:
        - 'monthly_averaged_reanalysis_by_hour_of_day': For temperature/dewpoint
          (provides 24 hourly values for extrema calculation)
        - 'monthly_averaged_reanalysis': For all other variables

        Args:
            raw_variables (Dict[str, List[str]]): Mapping of raw variable names to
                their dependent output variables
            year (int): Year to download
            month (int): Month to download (1-12)

        Returns:
            Dict[str, Path]: Dictionary mapping variable names to file paths

        Raises:
            RuntimeError: If download fails or CDS API returns errors
        """

        logger.info(f"Downloading ERA5 data for {year}-{month:02d}")

        # Group variables by product type for efficient batch downloading
        hourly_product_variables = []
        monthly_product_variables = []

        for raw_variable in raw_variables.keys():
            if raw_variable in ['2m_temperature', '2m_dewpoint_temperature']:
                hourly_product_variables.append(raw_variable)
            else:
                monthly_product_variables.append(raw_variable)

        downloaded_files = {}

        # Download hourly-type variables in batch
        if hourly_product_variables:
            logger.info(
                f"Downloading {len(hourly_product_variables)} hourly variables in batch"
            )
            try:
                hourly_files = self._download_variable_batch(
                    hourly_product_variables,
                    'monthly_averaged_reanalysis_by_hour_of_day',
                    year,
                    month,
                )
                downloaded_files.update(hourly_files)
            except Exception as e:
                logger.error(f"Failed to download hourly product variables: {e}")
                raise

        # Download monthly-type variables in batch
        if monthly_product_variables:
            logger.info(
                f"Downloading {len(monthly_product_variables)} monthly variables in batch"
            )
            try:
                monthly_files = self._download_variable_batch(
                    monthly_product_variables,
                    'monthly_averaged_reanalysis',
                    year,
                    month,
                )
                downloaded_files.update(monthly_files)
            except Exception as e:
                logger.error(f"Failed to download monthly product variables: {e}")
                raise

        logger.info(f"Successfully downloaded all {len(downloaded_files)} raw variables")
        return downloaded_files

    def _process_variable(
        self,
        output_variable: str,
        raw_files: Dict[str, Path],
        year: int,
        month: int,
    ) -> xr.Dataset:
        """
        Process raw ERA5 data to produce a CARDAMOM output variable.

        Handles variable-specific logic:
        - T2M_MIN/MAX: Extract monthly extrema from temperature
        - VPD: Calculate from temperature and dewpoint
        - Others: Unit conversion only

        Args:
            output_variable (str): CARDAMOM output variable name
            raw_files (Dict[str, Path]): Downloaded raw file paths
            year (int): Year being processed
            month (int): Month being processed

        Returns:
            xr.Dataset: Processed data ready for NetCDF output
        """

        logger.debug(f"Processing {output_variable} for {year}-{month:02d}")

        if output_variable == 't2m_min':
            return self._process_temperature_extrema(
                raw_files['2m_temperature'],
                extrema_type='min',
            )

        elif output_variable == 't2m_max':
            return self._process_temperature_extrema(
                raw_files['2m_temperature'],
                extrema_type='max',
            )

        elif output_variable == 'vpd':
            return self._process_vpd(
                raw_files['2m_temperature'],
                raw_files['2m_dewpoint_temperature'],
            )

        elif output_variable == 'total_prec':
            return self._process_precipitation(raw_files['total_precipitation'])

        elif output_variable in ['ssrd', 'strd']:
            return self._process_radiation(
                raw_files[
                    'surface_solar_radiation_downwards'
                    if output_variable == 'ssrd'
                    else 'surface_thermal_radiation_downwards'
                ],
                radiation_type=output_variable,
            )

        elif output_variable == 'skt':
            return self._process_skin_temperature(raw_files['skin_temperature'])

        elif output_variable == 'snowfall':
            return self._process_snowfall(raw_files['snowfall'])

        else:
            raise ValueError(f"Unknown output variable: {output_variable}")

    def _process_temperature_extrema(
        self,
        temperature_file: Path,
        extrema_type: str = 'min',
    ) -> xr.Dataset:
        """
        Extract monthly minimum or maximum temperature from hourly data.

        Uses 'monthly_averaged_reanalysis_by_hour_of_day' product which provides
        24 values (one for each hour UTC) from which monthly extrema are computed.

        Args:
            temperature_file (Path): NetCDF file with hourly temperature data
            extrema_type (str): 'min' or 'max'

        Returns:
            xr.Dataset: Dataset with extrema temperature in Celsius
        """

        ds = xr.open_dataset(temperature_file)
        temp_var = '2m_temperature' if '2m_temperature' in ds else list(
            ds.data_vars
        )[0]

        time_dim = self._get_time_dimension(ds)

        if extrema_type == 'min':
            extrema_values = ds[temp_var].min(dim=time_dim, skipna=True)
        else:
            extrema_values = ds[temp_var].max(dim=time_dim, skipna=True)

        # Convert from Kelvin to Celsius while preserving xarray structure
        extrema_data = extrema_values.copy()
        extrema_data.values = temperature_kelvin_to_celsius(extrema_values.values)

        # Update units metadata to reflect Celsius conversion
        extrema_data.attrs['units'] = 'deg C'
        if 'GRIB_units' in extrema_data.attrs:
            extrema_data.attrs['GRIB_units'] = 'deg C'

        # Update long_name for better CARDAMOM context
        extrema_data.attrs['long_name'] = f'Monthly {extrema_type}imum 2 metre temperature'

        ds.close()

        # Return as dataset with single time value (month)
        result = extrema_data.to_dataset(name=extrema_type.upper())

        return result

    def _process_vpd(
        self,
        temperature_file: Path,
        dewpoint_file: Path,
    ) -> xr.Dataset:
        """
        Calculate Vapor Pressure Deficit from temperature and dewpoint.

        Uses MATLAB-equivalent formula for consistency with original CARDAMOM workflows.
        MATLAB Reference: /MATLAB/prototypes/CARDAMOM_MAPS_05deg_DATASETS_JUL24.m line 202
        Formula: VPD=(SCIFUN_H2O_SATURATION_PRESSURE(ET2M.datamax) - SCIFUN_H2O_SATURATION_PRESSURE(ED2M.datamax))*10

        Args:
            temperature_file (Path): NetCDF with 2m_temperature
            dewpoint_file (Path): NetCDF with 2m_dewpoint_temperature

        Returns:
            xr.Dataset: Dataset with VPD variable

        Raises:
            FileNotFoundError: If required dewpoint temperature data is missing
            ValueError: If data files are invalid or incompatible
        """

        # Validate required files exist (following MATLAB requirements exactly)
        if not temperature_file.exists():
            raise FileNotFoundError(
                f"Temperature file required for VPD calculation: {temperature_file}"
            )

        if not dewpoint_file.exists():
            raise FileNotFoundError(
                f"Dewpoint temperature file required for VPD calculation: {dewpoint_file}. "
                f"VPD calculation follows MATLAB implementation which requires both "
                f"temperature (ET2M.datamax) and dewpoint (ED2M.datamax) data. "
                f"Cannot proceed without dewpoint temperature data."
            )

        # Load data
        ds_temp = xr.open_dataset(temperature_file)
        ds_dew = xr.open_dataset(dewpoint_file)

        temp_var = '2m_temperature' if '2m_temperature' in ds_temp else list(
            ds_temp.data_vars
        )[0]
        dew_var = '2m_dewpoint_temperature' if '2m_dewpoint_temperature' in ds_dew else list(
            ds_dew.data_vars
        )[0]

        time_dim_temp = self._get_time_dimension(ds_temp)
        time_dim_dew = self._get_time_dimension(ds_dew)

        # Get monthly maximum temperature (convert from Kelvin to Celsius)
        temperature_max_kelvin = ds_temp[temp_var].max(dim=time_dim_temp, skipna=True).values
        temperature_max_celsius = temperature_max_kelvin - 273.15

        # Get monthly maximum dewpoint temperature (for consistency with MATLAB formula)
        dewpoint_celsius = ds_dew[dew_var].max(dim=time_dim_dew, skipna=True).values - 273.15

        # Calculate VPD using MATLAB-equivalent formula
        vpd_hpa = calculate_vapor_pressure_deficit_matlab(
            temperature_max_celsius,
            dewpoint_celsius,  # Use actual dewpoint temperature for correct VPD calculation
        )

        ds_temp.close()
        ds_dew.close()

        # Create result dataset
        vpd_array = xr.DataArray(
            vpd_hpa,
            coords={
                'latitude': ds_temp['latitude'],
                'longitude': ds_temp['longitude'],
            },
            dims=['latitude', 'longitude'],
            name='VPD',
        )

        result = vpd_array.to_dataset()

        logger.debug(f"Calculated VPD: min={vpd_hpa.min():.2f}, max={vpd_hpa.max():.2f} hPa")

        return result

    def _process_precipitation(self, precip_file: Path) -> xr.Dataset:
        """
        Convert precipitation from m/s to mm/month.

        Args:
            precip_file (Path): NetCDF with total_precipitation

        Returns:
            xr.Dataset: Dataset with monthly precipitation in mm
        """

        ds = xr.open_dataset(precip_file)
        precip_var = 'total_precipitation' if 'total_precipitation' in ds else list(
            ds.data_vars
        )[0]

        time_dim = self._get_time_dimension(ds)

        # Sum over time dimension to get monthly total
        precip_monthly_m = ds[precip_var].sum(dim=time_dim, skipna=True).values

        # Convert m to mm
        precip_mm = precip_monthly_m * 1000

        ds.close()

        precip_array = xr.DataArray(
            precip_mm,
            coords={
                'latitude': ds['latitude'],
                'longitude': ds['longitude'],
            },
            dims=['latitude', 'longitude'],
            name='TOTAL_PREC',
        )

        result = precip_array.to_dataset()

        logger.debug(
            f"Processed precipitation: mean={precip_mm.mean():.2f} mm/month"
        )

        return result

    def _process_radiation(
        self,
        radiation_file: Path,
        radiation_type: str,
    ) -> xr.Dataset:
        """
        Convert radiation from J/m² to MJ/m²/day for CARDAMOM CBF format.

        ERA5 monthly_averaged_reanalysis provides mean daily accumulation in J/m².
        CARDAMOM requires daily accumulated radiation energy in MJ/m²/day for
        daily carbon cycle modeling.

        Scientific Background:
        - Input: Mean daily radiation accumulation (J/m² per day)
        - Output: Daily radiation accumulation (MJ/m²/day)
        - Conversion: J/m²/day ÷ 1,000,000 J/MJ = MJ/m²/day

        Physical Interpretation:
        This represents total radiation energy received per square meter per day.
        CARDAMOM uses a daily timestep and requires accumulated daily energy,
        not instantaneous power flux.

        Typical ranges:
        - SSRD (solar): 0-40 MJ/m²/day
        - STRD (thermal): 0-50 MJ/m²/day

        Args:
            radiation_file (Path): NetCDF with radiation data in J/m²
            radiation_type (str): 'ssrd' or 'strd'

        Returns:
            xr.Dataset: Dataset with radiation in MJ/m²/day

        References:
            ERA5 documentation: https://confluence.ecmwf.int/display/CKB/ERA5
            CARDAMOM CBF format specification
        """

        ds = xr.open_dataset(radiation_file)
        rad_var = list(ds.data_vars)[0]

        time_dim = self._get_time_dimension(ds)

        radiation_j_m2 = ds[rad_var].mean(dim=time_dim, skipna=True).values

        # Convert from J/m²/day to MJ/m²/day for CARDAMOM CBF format
        # 1 MJ = 1,000,000 J
        joules_per_megajoule = 1_000_000

        radiation_mj_m2_day = radiation_j_m2 / joules_per_megajoule

        ds.close()

        rad_array = xr.DataArray(
            radiation_mj_m2_day,
            coords={
                'latitude': ds['latitude'],
                'longitude': ds['longitude'],
            },
            dims=['latitude', 'longitude'],
            name=radiation_type.upper(),
        )

        result = rad_array.to_dataset()

        # Validate physical reasonableness
        mean_radiation = radiation_mj_m2_day.mean()
        max_radiation = radiation_mj_m2_day.max()

        # Expected ranges for MJ/m²/day
        expected_max = 50.0 if radiation_type == 'strd' else 40.0

        if max_radiation > expected_max * 1.5:  # Allow 50% margin for edge cases
            logger.warning(
                f"Unusually high {radiation_type} values detected: "
                f"max={max_radiation:.2f} MJ/m²/day (expected max ~{expected_max} MJ/m²/day). "
                f"Check data quality."
            )

        logger.debug(
            f"Processed {radiation_type}: mean={mean_radiation:.2f} MJ/m²/day, "
            f"max={max_radiation:.2f} MJ/m²/day"
        )

        return result

    def _process_skin_temperature(self, skt_file: Path) -> xr.Dataset:
        """
        Process skin temperature and convert to Celsius.

        Args:
            skt_file (Path): NetCDF with skin_temperature

        Returns:
            xr.Dataset: Dataset with SKT in Celsius
        """

        ds = xr.open_dataset(skt_file)
        skt_var = 'skin_temperature' if 'skin_temperature' in ds else list(
            ds.data_vars
        )[0]

        time_dim = self._get_time_dimension(ds)

        # Take monthly mean
        skt_mean = ds[skt_var].mean(dim=time_dim, skipna=True).values

        # Convert from Kelvin to Celsius following CARDAMOM standards
        skt_mean = temperature_kelvin_to_celsius(skt_mean)

        ds.close()

        skt_array = xr.DataArray(
            skt_mean,
            coords={
                'latitude': ds['latitude'],
                'longitude': ds['longitude'],
            },
            dims=['latitude', 'longitude'],
            name='SKT',
            attrs={
                'units': 'deg C',
                'long_name': 'Monthly mean skin temperature',
                'standard_name': 'surface_temperature',
            }
        )

        result = skt_array.to_dataset()

        return result

    def _process_snowfall(self, snowfall_file: Path) -> xr.Dataset:
        """
        Convert snowfall from m to mm/month.

        Args:
            snowfall_file (Path): NetCDF with snowfall data

        Returns:
            xr.Dataset: Dataset with monthly snowfall in mm
        """

        ds = xr.open_dataset(snowfall_file)
        snow_var = 'snowfall' if 'snowfall' in ds else list(
            ds.data_vars
        )[0]

        time_dim = self._get_time_dimension(ds)

        # Sum over time to get monthly total
        snow_monthly_m = ds[snow_var].sum(dim=time_dim, skipna=True).values

        # Convert m to mm
        snow_mm = snow_monthly_m * 1000

        ds.close()

        snow_array = xr.DataArray(
            snow_mm,
            coords={
                'latitude': ds['latitude'],
                'longitude': ds['longitude'],
            },
            dims=['latitude', 'longitude'],
            name='SNOWFALL',
        )

        result = snow_array.to_dataset()

        return result

    def download_and_process(
        self,
        variables: List[str],
        year: int,
        month: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Download and process ERA5 data for CARDAMOM.

        Main entry point that orchestrates the entire download-to-STAC workflow.

        Args:
            variables (List[str]): Output variables to generate
                (e.g., ['t2m_min', 't2m_max', 'vpd'])
            year (int): Year to download
            month (int): Month to download (1-12)
            **kwargs: Additional arguments (unused, for compatibility)

        Returns:
            Dict[str, Any]: Results dictionary with keys:
                - 'output_files': List of generated NetCDF paths
                - 'stac_items': List of STAC Item objects
                - 'collection_id': STAC Collection ID
                - 'success': bool

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If download or processing fails
        """

        self.validate_temporal_parameters(year, month)

        logger.info(f"Starting ECMWF download for {year}-{month:02d}")
        logger.info(f"Requested variables: {variables}")

        # Step 1: Resolve variable dependencies
        raw_variables = self._resolve_variable_dependencies(variables)

        # Step 2: Download raw ERA5 data
        raw_files = self._download_raw_era5(raw_variables, year, month)

        # Step 3: Process each output variable
        output_files = []
        stac_items_data = []

        for output_variable in variables:
            try:
                logger.info(f"Processing {output_variable}")

                # Process the variable
                processed_data = self._process_variable(
                    output_variable, raw_files, year, month
                )

                # Create standard NetCDF dataset
                var_units = self._get_variable_units(output_variable)

                # Determine output filename
                output_filename = f"{output_variable}_{year}_{month:02d}.nc"

                # Write to NetCDF
                output_file = self.write_netcdf_file(
                    processed_data,
                    output_filename,
                    variable_units=var_units,
                )

                output_files.append(output_file)

                # Prepare STAC item data
                stac_items_data.append(
                    {
                        'variable_name': output_variable.upper(),
                        'year': year,
                        'month': month,
                        'data_file_path': f"data/{output_filename}",
                        'properties': {
                            'cardamom:units': var_units.get(output_variable.upper(), ''),
                            'cardamom:source': 'era5',
                        },
                    }
                )

            except Exception as e:
                logger.error(f"Failed to process {output_variable}: {e}")
                raise

        # Step 4: Generate STAC metadata
        stac_result = self.create_and_write_stac_metadata(
            collection_id='cardamom-era5-variables',
            collection_description='ERA5 meteorological variables for CARDAMOM',
            collection_keywords=['era5', 'meteorology', 'cardamom'],
            items_data=stac_items_data,
            temporal_start=datetime(year, month, 1),
            incremental=kwargs.get('incremental', True),
            duplicate_policy=kwargs.get('duplicate_policy', 'update'),
        )

        # Step 5: Clean up raw files if requested
        self.cleanup_raw_files(list(raw_files.values()))

        logger.info(f"Successfully processed {len(output_files)} variables")

        return {
            'output_files': output_files,
            'stac_items': stac_result['items'],
            'collection_id': 'cardamom-era5-variables',
            'success': True,
        }

    @staticmethod
    def _get_variable_units(variable_name: str) -> Dict[str, str]:
        """
        Get standard CARDAMOM CBF units for a variable.

        Returns:
            Dict mapping variable name to its CBF-standard unit string

        Note:
            Radiation units are MJ m-2 day-1 (daily accumulation) per CARDAMOM requirements,
            NOT W m-2 (instantaneous flux)
        """

        units_map = {
            'T2M_MIN': 'deg C',
            'T2M_MAX': 'deg C',
            'VPD': 'hPa',
            'TOTAL_PREC': 'mm',
            'SSRD': 'MJ m-2 day-1',  # Daily accumulated solar radiation
            'STRD': 'MJ m-2 day-1',  # Daily accumulated thermal radiation
            'SKT': 'deg C',
            'SNOWFALL': 'mm',
        }

        return {variable_name.upper(): units_map.get(variable_name.upper(), 'unknown')}
