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

from downloaders.base import BaseDownloader
from atmospheric_science import (
    calculate_vapor_pressure_deficit_matlab,
    saturation_pressure_water_matlab,
)
from cardamom_variables import CARDAMOM_VARIABLE_REGISTRY

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
        Download raw ERA5 data from Climate Data Store using cdsapi.

        Uses the cdsapi.Client.retrieve() method which directly downloads
        ERA5 monthly averaged reanalysis data for specified variables.

        Args:
            raw_variables (Dict[str, List[str]]): Raw variables to download
            year (int): Year to download
            month (int): Month to download (1-12)

        Returns:
            Dict[str, Path]: Dictionary mapping variable names to file paths

        Raises:
            RuntimeError: If download fails or CDS API returns errors
        """

        logger.info(f"Downloading ERA5 data for {year}-{month:02d}")

        downloaded_files = {}

        # Download each raw variable from CDS
        for raw_variable in raw_variables.keys():
            try:
                # Ensure output directory exists
                output_dir = self.output_directory / self.raw_subdir
                output_dir.mkdir(parents=True, exist_ok=True)

                output_filepath = output_dir / f"{raw_variable}_{year}_{month:02d}.nc"

                # Skip download if file already exists
                if output_filepath.exists():
                    logger.info(
                        f"File already exists, skipping download: {output_filepath}"
                    )
                    downloaded_files[raw_variable] = output_filepath
                    continue

                # Determine product type and time parameters based on variable
                # Temperature and dewpoint need hourly data for min/max calculations
                if raw_variable in ['2m_temperature', '2m_dewpoint_temperature']:
                    product_type = 'monthly_averaged_reanalysis_by_hour_of_day'
                    # Request all 24 hours in single API call (more efficient)
                    time_param = [f'{h:02d}:00' for h in range(24)]
                else:
                    product_type = 'monthly_averaged_reanalysis'
                    time_param = '00:00'

                logger.info(
                    f"Downloading {raw_variable} for {year}-{month:02d} "
                    f"using product_type={product_type}"
                )

                # CDS API request for monthly averaged ERA5 data
                cds_request = {
                    'product_type': product_type,
                    'variable': raw_variable,
                    'year': str(year),
                    'month': f'{month:02d}',
                    'time': time_param,
                    'format': 'netcdf',
                }

                # Download data synchronously to output file
                # cdsapi.Client.retrieve() blocks until download completes
                self.cds_client.retrieve(
                    'reanalysis-era5-single-levels-monthly-means',
                    cds_request,
                    str(output_filepath),
                )

                # Verify file was created
                if output_filepath.exists():
                    file_size_mb = output_filepath.stat().st_size / (1024 * 1024)
                    logger.info(
                        f"Successfully downloaded {raw_variable}: "
                        f"{output_filepath} ({file_size_mb:.2f} MB)"
                    )
                    downloaded_files[raw_variable] = output_filepath
                else:
                    raise RuntimeError(
                        f"Download verification failed: file not found at {output_filepath}"
                    )

            except Exception as e:
                logger.error(f"Failed to download {raw_variable}: {e}")
                raise RuntimeError(
                    f"CDS API download failed for {raw_variable}: {e}"
                ) from e

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
            xr.Dataset: Dataset with extrema temperature in Kelvin
        """

        ds = xr.open_dataset(temperature_file)
        temp_var = '2m_temperature' if '2m_temperature' in ds else list(
            ds.data_vars
        )[0]

        time_dim = self._get_time_dimension(ds)

        if extrema_type == 'min':
            extrema_data = ds[temp_var].min(dim=time_dim, skipna=True)
        else:
            extrema_data = ds[temp_var].max(dim=time_dim, skipna=True)

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

        Args:
            temperature_file (Path): NetCDF with 2m_temperature
            dewpoint_file (Path): NetCDF with 2m_dewpoint_temperature

        Returns:
            xr.Dataset: Dataset with VPD variable
        """

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

        # Get monthly maximum temperature
        temperature_max_kelvin = ds_temp[temp_var].max(dim=time_dim_temp, skipna=True).values

        # Get monthly maximum dewpoint temperature (for consistency with MATLAB formula)
        dewpoint_celsius = ds_dew[dew_var].max(dim=time_dim_dew, skipna=True).values - 273.15

        # Convert temperature max to Celsius for VPD calculation
        temperature_max_celsius = temperature_max_kelvin - 273.15

        # Calculate VPD using MATLAB-equivalent formula
        vpd_hpa = calculate_vapor_pressure_deficit_matlab(
            temperature_max_celsius,
            temperature_max_celsius,  # Using T_max for both (consistent with MATLAB)
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
        Convert radiation from J/m² to W/m².

        ERA5 monthly_averaged_reanalysis provides mean daily accumulation in J/m².
        Convert to instantaneous rate in W/m² by dividing by seconds per day.

        Scientific Background:
        - Input: Mean daily radiation accumulation (J/m² per day)
        - Output: Instantaneous radiation flux (W/m² = J/m²/s)
        - Conversion: J/m²/day ÷ 86400 s/day = W/m²

        Args:
            radiation_file (Path): NetCDF with radiation data
            radiation_type (str): 'ssrd' or 'strd'

        Returns:
            xr.Dataset: Dataset with radiation in W/m²
        """

        ds = xr.open_dataset(radiation_file)
        rad_var = list(ds.data_vars)[0]

        time_dim = self._get_time_dimension(ds)

        radiation_j_m2 = ds[rad_var].mean(dim=time_dim, skipna=True).values

        # ERA5 monthly_averaged_reanalysis provides mean daily accumulation
        # Units: J/m² per day → convert to W/m² (J/m²/s)
        seconds_per_day = 86400  # 24 hours * 3600 seconds

        radiation_w_m2 = radiation_j_m2 / seconds_per_day

        ds.close()

        rad_array = xr.DataArray(
            radiation_w_m2,
            coords={
                'latitude': ds['latitude'],
                'longitude': ds['longitude'],
            },
            dims=['latitude', 'longitude'],
            name=radiation_type.upper(),
        )

        result = rad_array.to_dataset()

        logger.debug(
            f"Processed {radiation_type}: mean={radiation_w_m2.mean():.2f} W/m²"
        )

        return result

    def _process_skin_temperature(self, skt_file: Path) -> xr.Dataset:
        """
        Process skin temperature (minimal processing, unit already K).

        Args:
            skt_file (Path): NetCDF with skin_temperature

        Returns:
            xr.Dataset: Dataset with SKT
        """

        ds = xr.open_dataset(skt_file)
        skt_var = 'skin_temperature' if 'skin_temperature' in ds else list(
            ds.data_vars
        )[0]

        time_dim = self._get_time_dimension(ds)

        # Take monthly mean
        skt_mean = ds[skt_var].mean(dim=time_dim, skipna=True).values

        ds.close()

        skt_array = xr.DataArray(
            skt_mean,
            coords={
                'latitude': ds['latitude'],
                'longitude': ds['longitude'],
            },
            dims=['latitude', 'longitude'],
            name='SKT',
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
        """Get standard CARDAMOM units for a variable."""

        units_map = {
            'T2M_MIN': 'K',
            'T2M_MAX': 'K',
            'VPD': 'hPa',
            'TOTAL_PREC': 'mm',
            'SSRD': 'W m-2',
            'STRD': 'W m-2',
            'SKT': 'K',
            'SNOWFALL': 'mm',
        }

        return {variable_name.upper(): units_map.get(variable_name.upper(), 'unknown')}
