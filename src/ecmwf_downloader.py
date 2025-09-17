#!/usr/bin/env python3
"""
Enhanced ECMWF Downloader for CARDAMOM with Variable Registry

Enhanced version of existing ECMWFDownloader with additional variables,
processing hints, and validation capabilities for CARDAMOM preprocessing.
"""

import cdsapi
import os
import time
import glob
from typing import List, Dict, Union, Any
import logging
import numpy as np
import xarray as xr
from base_downloader import BaseDownloader
from atmospheric_science import calculate_vapor_pressure_deficit_matlab
# Unit conversion now handled by CBFMetProcessor


class ECMWFDownloader(BaseDownloader):
    """
    Enhanced ECMWF data downloader with variable registry and processing hints.

    Provides specialized interface for downloading ERA5 reanalysis data with
    CARDAMOM-specific variable mappings, validation, and processing metadata.

    Scientific Context:
    ERA5 provides essential meteorological drivers for CARDAMOM carbon cycle
    modeling including temperature, precipitation, radiation, and humidity
    variables at global scale with high temporal resolution.
    """

    def __init__(self,
                 area: List[float] = None,
                 grid: List[str] = None,
                 data_format: str = "netcdf",
                 download_format: str = "unarchived",
                 output_dir: str = ".",
):
        """
        Initialize enhanced ECMWF downloader with variable registry.

        Args:
            area: [North, West, South, East] bounding box in decimal degrees
                 Default: Global coverage [-89.75, -179.75, 89.75, 179.75]
            grid: Grid resolution as list (default: ["0.5/0.5"])
            data_format: Output format - "netcdf" or "grib" (default: "netcdf")
            download_format: Download format (default: "unarchived")
            output_dir: Directory for downloaded files
        """
        # Initialize base downloader
        super().__init__(output_dir)

        # ECMWF-specific configuration
        self.area = area or [-89.75, -179.75, 89.75, 179.75]
        self.grid = grid or ["0.5/0.5"]
        self.data_format = data_format
        self.download_format = download_format

        # Initialize ECMWF CDS API client
        try:
            self.client = cdsapi.Client()
            self.logger.info("ECMWF CDS API client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ECMWF CDS API client: {e}")
            raise

        # Setup variable registry with CARDAMOM-specific metadata
        self.variable_registry = self._setup_variable_registry()

    def _setup_variable_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Registry of ERA5 variables with CARDAMOM-specific metadata.

        Returns:
            dict: Variable registry with processing hints and metadata
        """
        return {
            "2m_temperature": {
                "cardamom_name": "T2M",
                "cbf_names": ["T2M_MIN", "T2M_MAX"],  # CBF requires min/max temperatures
                "units": "K",
                "cbf_units": "K",
                "processing": "min_max_monthly",
                "description": "Air temperature at 2m height for photosynthesis calculations",
                "typical_range": [233, 323],  # -40°C to 50°C
                "required_for": ["photosynthesis", "respiration", "vpd_calculation"],
                "cbf_processing": "derive_min_max"
            },
            "2m_dewpoint_temperature": {
                "cardamom_name": "D2M",
                "cbf_names": ["D2M"],
                "units": "K",
                "cbf_units": "K",
                "processing": "hourly_averaged",
                "description": "Dewpoint temperature for vapor pressure deficit calculations",
                "typical_range": [193, 303],  # -80°C to 30°C
                "required_for": ["vpd_calculation", "humidity_stress"],
                "cbf_processing": "monthly_average"
            },
            "surface_solar_radiation_downwards": {
                "cardamom_name": "SSRD",
                "cbf_names": ["SSRD"],
                "units": "J m-2",
                "cbf_units": "W m-2",  # CBF expects W/m²
                "processing": "monthly_mean",
                "description": "Downward solar radiation for photosynthesis light limitation",
                "typical_range": [0, 3.6e7],  # 0 to ~36 MJ/m²/day
                "required_for": ["photosynthesis", "par_calculation"],
                "cbf_processing": "convert_to_watts"
            },
            "surface_thermal_radiation_downwards": {
                "cardamom_name": "STRD",
                "cbf_names": ["STRD"],
                "units": "J m-2",
                "cbf_units": "W m-2",  # CBF expects W/m²
                "processing": "monthly_mean",
                "description": "Downward thermal radiation for energy balance",
                "typical_range": [1e7, 5e7],  # ~10-50 MJ/m²/day
                "required_for": ["energy_balance"],
                "cbf_processing": "convert_to_watts"
            },
            "total_precipitation": {
                "cardamom_name": "TP",
                "cbf_names": ["TOTAL_PREC"],  # CBF requires this exact name
                "units": "m",
                "cbf_units": "mm/month",  # CBF expects mm/month
                "processing": "monthly_sum",
                "description": "Total precipitation for soil moisture and plant water availability",
                "typical_range": [0, 1.0],  # 0 to 1000 mm/month
                "required_for": ["soil_moisture", "water_stress"],
                "cbf_processing": "convert_to_mm"
            },
            "skin_temperature": {
                "cardamom_name": "SKT",
                "cbf_names": ["SKT"],
                "units": "K",
                "cbf_units": "K",
                "processing": "monthly_mean",
                "description": "Surface skin temperature for soil respiration",
                "typical_range": [223, 333],  # -50°C to 60°C
                "required_for": ["soil_respiration", "surface_fluxes"],
                "cbf_processing": "monthly_average"
            },
            "snowfall": {
                "cardamom_name": "SF",
                "cbf_names": ["SNOWFALL"],  # CBF requires this exact name
                "units": "m of water equivalent",
                "cbf_units": "mm/month",  # CBF expects mm/month
                "processing": "monthly_sum",
                "description": "Snowfall for seasonal carbon cycle dynamics",
                "typical_range": [0, 0.5],  # 0 to 500 mm water equivalent
                "required_for": ["seasonal_dynamics", "snow_cover"],
                "cbf_processing": "convert_to_mm"
            }
        }

    def validate_variables(self, variables: List[str]) -> Dict[str, bool]:
        """
        Validate that requested variables are available in the registry.

        Args:
            variables: List of ERA5 variable names to validate

        Returns:
            dict: Validation results for each variable
        """
        validation_results = {}

        for variable in variables:
            if variable in self.variable_registry:
                validation_results[variable] = True
                self.logger.info(f"Variable '{variable}' validated successfully")
            else:
                validation_results[variable] = False
                self.logger.warning(f"Unknown variable '{variable}' not in registry")

        available_vars = list(self.variable_registry.keys())
        self.logger.info(f"Available variables: {available_vars}")

        return validation_results

    def get_variable_metadata(self, variable: str) -> Union[Dict[str, Any], None]:
        """
        Get metadata for a specific variable from the registry.

        Args:
            variable: ERA5 variable name

        Returns:
            dict: Variable metadata or None if not found
        """
        return self.variable_registry.get(variable)

    def download_data(self,
                     variables: Union[str, List[str]],
                     years: Union[int, List[int]],
                     months: Union[int, List[int]],
                     processing_type: str = "monthly",
                     **kwargs) -> Dict[str, Any]:
        """
        Download ERA5 data with validation and processing hints.

        Args:
            variables: Variable name(s) to download
            years: Year(s) to download
            months: Month(s) to download
            processing_type: "hourly" or "monthly" processing
            **kwargs: Additional parameters for specific download methods

        Returns:
            dict: Download results with status and file information
        """
        # Ensure variables is a list
        if isinstance(variables, str):
            variables = [variables]

        # Validate variables before attempting download
        validation_results = self.validate_variables(variables)
        invalid_variables = [v for v, valid in validation_results.items() if not valid]

        if invalid_variables:
            error_msg = f"Invalid variables: {invalid_variables}"
            self.logger.error(error_msg)
            return {"status": "failed", "error": error_msg}

        # Route to appropriate download method based on processing type
        try:
            if processing_type == "hourly":
                return self._download_hourly_data(variables, years, months, **kwargs)
            elif processing_type == "monthly":
                return self._download_monthly_data(variables, years, months, **kwargs)
            else:
                error_msg = f"Unknown processing type: {processing_type}"
                self.logger.error(error_msg)
                return {"status": "failed", "error": error_msg}

        except Exception as e:
            error_msg = f"Download failed: {e}"
            self.logger.error(error_msg)
            return {"status": "failed", "error": error_msg}

    def _download_hourly_data(self,
                             variables: List[str],
                             years: Union[int, List[int]],
                             months: Union[int, List[int]],
                             days: List[str] = None,
                             times: List[str] = None,
                             dataset: str = "reanalysis-era5-single-levels",
                             file_prefix: str = "ECMWF_HOURLY") -> Dict[str, Any]:
        """Download hourly ERA5 data with CARDAMOM variable mapping."""

        # Ensure lists
        if isinstance(years, int):
            years = [years]
        if isinstance(months, int):
            months = [months]

        # Default days and times
        if days is None:
            days = [f"{i:02d}" for i in range(1, 32)]
        if times is None:
            times = [f"{i:02d}:00" for i in range(24)]

        downloaded_files = []

        for year in years:
            for month in months:
                for variable in variables:
                    # Use CARDAMOM name from registry if available
                    var_metadata = self.get_variable_metadata(variable)
                    var_abbr = var_metadata.get("cardamom_name", variable) if var_metadata else variable

                    filename = f"{file_prefix}_{var_abbr}_{month:02d}{year}.nc"
                    filepath = os.path.join(self.output_dir, filename)

                    # Skip if file already exists
                    if os.path.exists(filepath):
                        self.logger.info(f"File '{filename}' already exists. Skipping download.")
                        self._record_download_attempt(filename, "skipped")
                        continue

                    try:
                        # Prepare download request
                        request = {
                            "product_type": ["reanalysis"],
                            "variable": variable,
                            "year": str(year),
                            "month": f"{month:02d}",
                            "day": days,
                            "time": times,
                            "data_format": self.data_format,
                            "grid": self.grid,
                            "download_format": self.download_format,
                            "area": self.area
                        }

                        self.logger.info(f"Downloading {variable} for {month:02d}/{year}...")

                        # Download directly to output directory
                        self.client.retrieve(dataset, request).download(filepath)

                        # Validate downloaded file
                        if self.validate_downloaded_data(filepath):
                            downloaded_files.append(filepath)
                            self._record_download_attempt(filename, "success")
                            self.logger.info(f"Successfully downloaded {filename}")
                        else:
                            self._record_download_attempt(filename, "failed", "File validation failed")

                    except Exception as e:
                        error_msg = f"Download failed for {variable} {month:02d}/{year}: {e}"
                        self.logger.error(error_msg)
                        self._record_download_attempt(filename, "failed", str(e))

        return {
            "status": "completed",
            "downloaded_files": downloaded_files,
            "total_files": len(downloaded_files),
            "processing_type": "hourly"
        }

    def _download_monthly_data(self,
                              variables: List[str],
                              years: Union[int, List[int]],
                              months: Union[int, List[int]],
                              product_type: str = "monthly_averaged_reanalysis",
                              times: List[str] = None,
                              dataset: str = "reanalysis-era5-single-levels-monthly-means",
                              file_prefix: str = "ECMWF_MONTHLY") -> Dict[str, Any]:
        """Download monthly ERA5 data with CARDAMOM variable mapping."""

        # Ensure lists
        if isinstance(years, int):
            years = [years]
        if isinstance(months, int):
            months = [months]

        # Default times for monthly data
        if times is None:
            if "by_hour" in product_type:
                times = [f"{i:02d}:00" for i in range(24)]
            else:
                times = ["00:00"]

        downloaded_files = []

        for year in years:
            for month in months:
                for variable in variables:
                    filename = f"{file_prefix}_{variable}_{month:02d}{year}.nc"
                    filepath = os.path.join(self.output_dir, filename)

                    # Skip if file already exists
                    if os.path.exists(filepath):
                        self.logger.info(f"File '{filename}' already exists. Skipping download.")
                        self._record_download_attempt(filename, "skipped")
                        continue

                    try:
                        # Prepare download request
                        request = {
                            "product_type": [product_type],
                            "variable": variable,
                            "year": str(year),
                            "month": f"{month:02d}",
                            "time": times,
                            "data_format": self.data_format,
                            "download_format": self.download_format,
                            "area": self.area
                        }

                        self.logger.info(f"Downloading {variable} for {month:02d}/{year}...")

                        # Download directly to output directory
                        self.client.retrieve(dataset, request).download(filepath)

                        # Validate downloaded file
                        if self.validate_downloaded_data(filepath):
                            downloaded_files.append(filepath)
                            self._record_download_attempt(filename, "success")
                            self.logger.info(f"Successfully downloaded {filename}")
                        else:
                            self._record_download_attempt(filename, "failed", "File validation failed")

                    except Exception as e:
                        error_msg = f"Download failed for {variable} {month:02d}/{year}: {e}"
                        self.logger.error(error_msg)
                        self._record_download_attempt(filename, "failed", str(e))

        return {
            "status": "completed",
            "downloaded_files": downloaded_files,
            "total_files": len(downloaded_files),
            "processing_type": "monthly"
        }

    def download_with_processing(self,
                                variables: List[str],
                                years: List[int],
                                months: List[int],
                                processing_type: str) -> Dict[str, Any]:
        """
        Download and apply basic processing during download based on variable metadata.

        Args:
            variables: List of ERA5 variable names
            years: List of years to download
            months: List of months to download
            processing_type: Processing type ("hourly" or "monthly")

        Returns:
            dict: Download results with processing information
        """
        # Get processing hints from variable registry
        processing_info = {}
        for variable in variables:
            metadata = self.get_variable_metadata(variable)
            if metadata:
                processing_info[variable] = metadata.get("processing", "standard")

        self.logger.info(f"Processing hints: {processing_info}")

        # Perform standard download
        results = self.download_data(
            variables=variables,
            years=years,
            months=months,
            processing_type=processing_type
        )

        # Add processing information to results
        results["processing_hints"] = processing_info
        results["variable_metadata"] = {v: self.get_variable_metadata(v) for v in variables}

        return results

    def calculate_derived_variables(self,
                                  temperature_file: str,
                                  dewpoint_file: str,
                                  output_dir: str = None) -> Dict[str, str]:
        """
        Calculate derived variables required for CBF input generation.

        Calculates VPD from temperature and dewpoint data, and derives
        temperature min/max from hourly or daily temperature data.

        Args:
            temperature_file: Path to ERA5 temperature NetCDF file
            dewpoint_file: Path to ERA5 dewpoint temperature NetCDF file
            output_dir: Output directory for derived variable files

        Returns:
            dict: Paths to generated derived variable files
        """
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.logger.info("Calculating derived variables for CBF input")

        derived_files = {}

        try:
            # Load temperature and dewpoint data
            self.logger.info(f"Loading temperature data from {temperature_file}")
            temp_ds = xr.open_dataset(temperature_file)

            self.logger.info(f"Loading dewpoint data from {dewpoint_file}")
            dewpoint_ds = xr.open_dataset(dewpoint_file)

            # Get variable names (ERA5 uses standard names)
            temp_var = '2m_temperature' if '2m_temperature' in temp_ds.data_vars else 't2m'
            dewpoint_var = '2m_dewpoint_temperature' if '2m_dewpoint_temperature' in dewpoint_ds.data_vars else 'd2m'

            # Extract temperature and dewpoint arrays
            temperature_kelvin = temp_ds[temp_var]
            dewpoint_kelvin = dewpoint_ds[dewpoint_var]

            # Convert to Celsius for VPD calculation
            temperature_celsius = temperature_kelvin - 273.15
            dewpoint_celsius = dewpoint_kelvin - 273.15

            # Calculate VPD using Phase 8 atmospheric science function
            self.logger.info("Calculating Vapor Pressure Deficit (VPD)")

            # For monthly data, use the temperature directly
            # For hourly/daily data, we need temperature maximum
            if 'time' in temperature_celsius.dims:
                if len(temperature_celsius.time) > 31:  # Likely hourly or daily data
                    # Calculate monthly maximum temperature for VPD
                    temp_max_celsius = temperature_celsius.groupby('time.month').max(dim='time')
                    dewpoint_avg_celsius = dewpoint_celsius.groupby('time.month').mean(dim='time')
                else:
                    # Already monthly data
                    temp_max_celsius = temperature_celsius
                    dewpoint_avg_celsius = dewpoint_celsius
            else:
                temp_max_celsius = temperature_celsius
                dewpoint_avg_celsius = dewpoint_celsius

            # Calculate VPD using MATLAB-equivalent function
            vpd_hpa = calculate_vapor_pressure_deficit_matlab(
                temp_max_celsius.values,
                dewpoint_avg_celsius.values
            )

            # Create VPD DataArray with proper coordinates
            vpd_da = xr.DataArray(
                vpd_hpa,
                coords=temp_max_celsius.coords,
                dims=temp_max_celsius.dims,
                attrs={
                    'units': 'hPa',
                    'long_name': 'Vapor Pressure Deficit',
                    'description': 'Atmospheric moisture demand calculated from ERA5 temperature and dewpoint',
                    'calculation_method': 'MATLAB SCIFUN equivalent',
                    'source': 'ERA5 2m_temperature and 2m_dewpoint_temperature'
                }
            )

            # Save VPD file
            vpd_filename = os.path.basename(temperature_file).replace(temp_var, 'VPD')
            vpd_filepath = os.path.join(output_dir, vpd_filename)

            vpd_ds = xr.Dataset({'VPD': vpd_da})
            vpd_ds.to_netcdf(vpd_filepath)
            derived_files['VPD'] = vpd_filepath
            self.logger.info(f"VPD calculation completed: {vpd_filepath}")

            # Calculate temperature min/max if we have sub-monthly data
            if 'time' in temperature_celsius.dims and len(temperature_celsius.time) > 31:
                self.logger.info("Calculating monthly temperature min/max")

                # Calculate monthly statistics
                temp_min_monthly = temperature_kelvin.groupby('time.month').min(dim='time')
                temp_max_monthly = temperature_kelvin.groupby('time.month').max(dim='time')

                # Create min temperature DataArray
                temp_min_da = xr.DataArray(
                    temp_min_monthly.values,
                    coords=temp_min_monthly.coords,
                    dims=temp_min_monthly.dims,
                    attrs={
                        'units': 'K',
                        'long_name': 'Monthly Minimum 2m Temperature',
                        'description': 'Monthly minimum air temperature at 2m height',
                        'source': 'ERA5 2m_temperature'
                    }
                )

                # Create max temperature DataArray
                temp_max_da = xr.DataArray(
                    temp_max_monthly.values,
                    coords=temp_max_monthly.coords,
                    dims=temp_max_monthly.dims,
                    attrs={
                        'units': 'K',
                        'long_name': 'Monthly Maximum 2m Temperature',
                        'description': 'Monthly maximum air temperature at 2m height',
                        'source': 'ERA5 2m_temperature'
                    }
                )

                # Save min temperature file
                tmin_filename = os.path.basename(temperature_file).replace(temp_var, 'T2M_MIN')
                tmin_filepath = os.path.join(output_dir, tmin_filename)
                tmin_ds = xr.Dataset({'T2M_MIN': temp_min_da})
                tmin_ds.to_netcdf(tmin_filepath)
                derived_files['T2M_MIN'] = tmin_filepath

                # Save max temperature file
                tmax_filename = os.path.basename(temperature_file).replace(temp_var, 'T2M_MAX')
                tmax_filepath = os.path.join(output_dir, tmax_filename)
                tmax_ds = xr.Dataset({'T2M_MAX': temp_max_da})
                tmax_ds.to_netcdf(tmax_filepath)
                derived_files['T2M_MAX'] = tmax_filepath

                self.logger.info(f"Temperature min/max calculation completed")

            # Close datasets
            temp_ds.close()
            dewpoint_ds.close()

            return derived_files

        except Exception as e:
            error_msg = f"Failed to calculate derived variables: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def convert_units_for_cbf(self,
                            input_file: str,
                            variable_name: str,
                            output_dir: str = None) -> str:
        """
        Convert variable units to CBF-compatible format.

        Args:
            input_file: Path to input NetCDF file
            variable_name: ERA5 variable name to convert
            output_dir: Output directory for converted files

        Returns:
            str: Path to converted file
        """
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Get variable metadata
        var_metadata = self.get_variable_metadata(variable_name)
        if not var_metadata:
            raise ValueError(f"Unknown variable: {variable_name}")

        cbf_processing = var_metadata.get('cbf_processing', 'no_conversion')
        cbf_names = var_metadata.get('cbf_names', [variable_name])
        cbf_units = var_metadata.get('cbf_units', var_metadata.get('units'))

        self.logger.info(f"Converting {variable_name} units for CBF compatibility")

        try:
            # Load dataset
            ds = xr.open_dataset(input_file)

            # Get the variable (ERA5 uses standard names)
            if variable_name in ds.data_vars:
                data_var = ds[variable_name]
            else:
                # Try common ERA5 abbreviations
                var_mappings = {
                    'total_precipitation': 'tp',
                    '2m_temperature': 't2m',
                    'surface_solar_radiation_downwards': 'ssrd',
                    'surface_thermal_radiation_downwards': 'strd'
                }
                alt_name = var_mappings.get(variable_name, variable_name)
                if alt_name in ds.data_vars:
                    data_var = ds[alt_name]
                else:
                    raise ValueError(f"Variable {variable_name} not found in {input_file}")

            # Apply unit conversions based on CBF requirements
            if cbf_processing == 'convert_to_mm':
                # Convert precipitation from meters to mm
                converted_data = data_var * 1000  # m to mm
                converted_data.attrs['units'] = 'mm'

            elif cbf_processing == 'convert_to_watts':
                # Convert radiation from J/m² to W/m²
                # Assuming monthly data: J/m²/month to W/m²
                seconds_per_month = 30.44 * 24 * 3600  # Average seconds per month
                converted_data = data_var / seconds_per_month
                converted_data.attrs['units'] = 'W m-2'

            else:
                # No conversion needed
                converted_data = data_var

            # Update variable name for CBF compatibility
            cbf_var_name = cbf_names[0]
            converted_data.name = cbf_var_name

            # Create output dataset
            output_ds = xr.Dataset({cbf_var_name: converted_data})

            # Generate output filename
            base_name = os.path.basename(input_file)
            output_filename = base_name.replace(variable_name, cbf_var_name)
            output_filepath = os.path.join(output_dir, output_filename)

            # Save converted file
            output_ds.to_netcdf(output_filepath)

            # Close datasets
            ds.close()
            output_ds.close()

            self.logger.info(f"Unit conversion completed: {output_filepath}")
            return output_filepath

        except Exception as e:
            error_msg = f"Failed to convert units for {variable_name}: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def apply_land_masking(self,
                          input_files: Union[str, List[str]],
                          land_fraction_file: str,
                          land_threshold: float = 0.5,
                          output_dir: str = None) -> Union[str, List[str]]:
        """
        Apply land fraction masking to ECMWF data files.

        Applies MODIS land fraction masks to meteorological variables,
        setting ocean/water pixels to NaN based on land fraction threshold.

        Args:
            input_files: Path(s) to ECMWF data files to mask
            land_fraction_file: Path to MODIS land fraction NetCDF file
            land_threshold: Land fraction threshold (0.0-1.0, default: 0.5)
            output_dir: Output directory for masked files

        Returns:
            Path(s) to masked files
        """
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Handle single file or list of files
        if isinstance(input_files, str):
            input_files = [input_files]
            return_single = True
        else:
            return_single = False

        self.logger.info(f"Applying land masking to {len(input_files)} files")
        self.logger.info(f"Land fraction threshold: {land_threshold}")

        masked_files = []

        try:
            # Load land fraction data
            self.logger.info(f"Loading land fraction data from {land_fraction_file}")
            land_ds = xr.open_dataset(land_fraction_file)

            # Get land fraction variable (common names in MODIS data)
            land_var_names = ['data', 'land_fraction', 'land_sea_frac', 'fraction']
            land_var = None
            for var_name in land_var_names:
                if var_name in land_ds.data_vars:
                    land_var = land_ds[var_name]
                    break

            if land_var is None:
                available_vars = list(land_ds.data_vars.keys())
                raise ValueError(f"Land fraction variable not found. Available variables: {available_vars}")

            self.logger.info(f"Using land fraction variable: {land_var.name}")

            # Create land mask (True where land fraction > threshold)
            land_mask = land_var > land_threshold

            # Process each input file
            for input_file in input_files:
                self.logger.info(f"Processing file: {os.path.basename(input_file)}")

                # Load input data
                input_ds = xr.open_dataset(input_file)

                # Create output dataset
                masked_ds = input_ds.copy()

                # Apply land masking to all data variables
                for var_name in input_ds.data_vars:
                    var_data = input_ds[var_name]

                    # Ensure coordinate alignment between land mask and variable data
                    try:
                        # Interpolate land mask to match variable grid if necessary
                        if not (np.array_equal(land_mask.latitude.values, var_data.latitude.values) and
                                np.array_equal(land_mask.longitude.values, var_data.longitude.values)):

                            self.logger.info(f"Interpolating land mask to match {var_name} grid")
                            land_mask_interp = land_mask.interp_like(var_data, method='nearest')
                        else:
                            land_mask_interp = land_mask

                        # Apply mask: set ocean/water pixels to NaN
                        if 'time' in var_data.dims:
                            # For time-varying data, broadcast mask across time
                            mask_broadcast = land_mask_interp.broadcast_like(var_data)
                            masked_var = var_data.where(mask_broadcast)
                        else:
                            # For time-invariant data
                            masked_var = var_data.where(land_mask_interp)

                        # Update attributes to reflect masking
                        masked_var.attrs.update(var_data.attrs)
                        masked_var.attrs['land_masking'] = f'Applied with threshold {land_threshold}'
                        masked_var.attrs['land_fraction_source'] = land_fraction_file
                        masked_var.attrs['masked_pixels'] = 'Ocean/water pixels set to NaN'

                        masked_ds[var_name] = masked_var

                    except Exception as e:
                        self.logger.warning(f"Failed to apply land mask to {var_name}: {e}")
                        # Keep original data if masking fails
                        masked_ds[var_name] = var_data

                # Generate output filename
                base_name = os.path.basename(input_file)
                if 'LFmasked' not in base_name:
                    name_parts = base_name.split('.')
                    name_parts[0] += '_LFmasked'
                    masked_filename = '.'.join(name_parts)
                else:
                    masked_filename = base_name

                masked_filepath = os.path.join(output_dir, masked_filename)

                # Save masked file
                masked_ds.to_netcdf(masked_filepath)
                masked_files.append(masked_filepath)

                # Close datasets
                input_ds.close()
                masked_ds.close()

                self.logger.info(f"Land masking completed: {masked_filename}")

            # Close land fraction dataset
            land_ds.close()

            self.logger.info(f"Land masking completed for all {len(masked_files)} files")

            return masked_files[0] if return_single else masked_files

        except Exception as e:
            error_msg = f"Failed to apply land masking: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def integrate_external_data(self,
                               ecmwf_files: List[str],
                               co2_data_dir: str = None,
                               fire_data_dir: str = None,
                               output_dir: str = None) -> List[str]:
        """
        Integrate external data (NOAA CO2, GFED fire) with ECMWF meteorological data.

        Combines ECMWF meteorological variables with CO2 concentration data
        from NOAA and fire emissions/burned area from GFED to create
        comprehensive meteorological driver files.

        Args:
            ecmwf_files: List of ECMWF NetCDF files to integrate with
            co2_data_dir: Directory containing NOAA CO2 data files
            fire_data_dir: Directory containing GFED fire data files
            output_dir: Output directory for integrated files

        Returns:
            List of paths to integrated files
        """
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"Integrating external data with {len(ecmwf_files)} ECMWF files")

        integrated_files = []

        try:
            # Initialize external data loaders
            co2_data = None
            fire_data = None

            # Load CO2 data if directory provided
            if co2_data_dir and os.path.exists(co2_data_dir):
                self.logger.info(f"Loading NOAA CO2 data from {co2_data_dir}")
                co2_data = self._load_noaa_co2_data(co2_data_dir)

            # Load fire data if directory provided
            if fire_data_dir and os.path.exists(fire_data_dir):
                self.logger.info(f"Loading GFED fire data from {fire_data_dir}")
                fire_data = self._load_gfed_fire_data(fire_data_dir)

            # Process each ECMWF file
            for ecmwf_file in ecmwf_files:
                self.logger.info(f"Integrating data for {os.path.basename(ecmwf_file)}")

                # Load ECMWF data
                ecmwf_ds = xr.open_dataset(ecmwf_file)

                # Create integrated dataset starting with ECMWF data
                integrated_ds = ecmwf_ds.copy()

                # Integrate CO2 data
                if co2_data is not None:
                    try:
                        co2_var = self._integrate_co2_data(integrated_ds, co2_data)
                        integrated_ds['CO2'] = co2_var
                        self.logger.info("CO2 data integration completed")
                    except Exception as e:
                        self.logger.warning(f"Failed to integrate CO2 data: {e}")

                # Integrate fire data
                if fire_data is not None:
                    try:
                        fire_vars = self._integrate_fire_data(integrated_ds, fire_data)
                        for var_name, var_data in fire_vars.items():
                            integrated_ds[var_name] = var_data
                        self.logger.info("Fire data integration completed")
                    except Exception as e:
                        self.logger.warning(f"Failed to integrate fire data: {e}")

                # Generate output filename
                base_name = os.path.basename(ecmwf_file)
                if 'integrated' not in base_name:
                    name_parts = base_name.split('.')
                    name_parts[0] += '_integrated'
                    integrated_filename = '.'.join(name_parts)
                else:
                    integrated_filename = base_name

                integrated_filepath = os.path.join(output_dir, integrated_filename)

                # Save integrated file
                integrated_ds.to_netcdf(integrated_filepath)
                integrated_files.append(integrated_filepath)

                # Close datasets
                ecmwf_ds.close()
                integrated_ds.close()

                self.logger.info(f"Integration completed: {integrated_filename}")

            return integrated_files

        except Exception as e:
            error_msg = f"Failed to integrate external data: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _load_noaa_co2_data(self, co2_data_dir: str) -> xr.Dataset:
        """Load NOAA CO2 data from directory."""
        # Look for NOAA CO2 files (common patterns)
        co2_files = []
        for pattern in ['*co2*.nc', '*CO2*.nc', '*noaa*.nc', '*NOAA*.nc']:
            co2_files.extend(glob.glob(os.path.join(co2_data_dir, pattern)))

        if not co2_files:
            raise FileNotFoundError(f"No CO2 data files found in {co2_data_dir}")

        # Load and combine CO2 files if multiple
        if len(co2_files) == 1:
            return xr.open_dataset(co2_files[0])
        else:
            # Combine multiple files
            datasets = [xr.open_dataset(f) for f in co2_files]
            combined = xr.concat(datasets, dim='time')
            return combined

    def _load_gfed_fire_data(self, fire_data_dir: str) -> xr.Dataset:
        """Load GFED fire data from directory."""
        # Look for GFED fire files (common patterns)
        fire_files = []
        for pattern in ['*gfed*.nc', '*GFED*.nc', '*fire*.nc', '*FIRE*.nc', '*burned*.nc']:
            fire_files.extend(glob.glob(os.path.join(fire_data_dir, pattern)))

        if not fire_files:
            raise FileNotFoundError(f"No fire data files found in {fire_data_dir}")

        # Load and combine fire files if multiple
        if len(fire_files) == 1:
            return xr.open_dataset(fire_files[0])
        else:
            # Combine multiple files
            datasets = [xr.open_dataset(f) for f in fire_files]
            combined = xr.concat(datasets, dim='time')
            return combined

    def _integrate_co2_data(self, target_ds: xr.Dataset, co2_data: xr.Dataset) -> xr.DataArray:
        """Integrate CO2 data with target dataset."""
        # Get CO2 variable (common names)
        co2_var_names = ['co2', 'CO2', 'co2_concentration', 'mole_fraction']
        co2_var = None
        for var_name in co2_var_names:
            if var_name in co2_data.data_vars:
                co2_var = co2_data[var_name]
                break

        if co2_var is None:
            available_vars = list(co2_data.data_vars.keys())
            raise ValueError(f"CO2 variable not found. Available variables: {available_vars}")

        # Interpolate CO2 data to match target grid and time
        co2_interp = co2_var.interp_like(target_ds, method='linear')

        # Set attributes for CBF compatibility
        co2_interp.attrs.update({
            'units': 'ppm',
            'long_name': 'Atmospheric CO2 Concentration',
            'description': 'Atmospheric carbon dioxide concentration from NOAA measurements',
            'source': 'NOAA GML',
            'cbf_variable': 'CO2'
        })

        return co2_interp

    def _integrate_fire_data(self, target_ds: xr.Dataset, fire_data: xr.Dataset) -> Dict[str, xr.DataArray]:
        """Integrate fire data with target dataset."""
        fire_vars = {}

        # Map GFED variables to CBF names
        fire_var_mapping = {
            'burned_area': 'BURNED_AREA',
            'BURNED_AREA': 'BURNED_AREA',
            'burned_fraction': 'BURNED_AREA',
            'fire_emissions': 'Mean_FIR',
            'Mean_FIR': 'Mean_FIR',
            'co2_emissions': 'Mean_FIR'
        }

        for gfed_var, cbf_var in fire_var_mapping.items():
            if gfed_var in fire_data.data_vars:
                fire_var_data = fire_data[gfed_var]

                # Interpolate to match target grid and time
                fire_interp = fire_var_data.interp_like(target_ds, method='nearest')

                # Set attributes for CBF compatibility
                if cbf_var == 'BURNED_AREA':
                    fire_interp.attrs.update({
                        'units': 'fraction',
                        'long_name': 'Burned Area Fraction',
                        'description': 'Fraction of grid cell burned by fires',
                        'source': 'GFED4',
                        'cbf_variable': 'BURNED_AREA'
                    })
                elif cbf_var == 'Mean_FIR':
                    fire_interp.attrs.update({
                        'units': 'gC/m2/day',
                        'long_name': 'Fire CO2 Emissions',
                        'description': 'Carbon dioxide emissions from fires',
                        'source': 'GFED4',
                        'cbf_variable': 'Mean_FIR'
                    })

                fire_vars[cbf_var] = fire_interp
                break  # Use first available variable for each CBF variable

        return fire_vars

    def download_cbf_met_variables(self,
                                 variables: List[str],
                                 years: List[int],
                                 months: List[int],
                                 download_dir: str = None) -> Dict[str, Any]:
        """
        Download ERA5 variables required for CBF meteorological drivers.

        This method only downloads data and does not perform processing.
        Use CBFMetProcessor for processing downloaded files into CBF format.

        Args:
            variables: List of ECMWF variable names to download
            years: List of years to download
            months: List of months to download
            download_dir: Directory for downloaded files (default: self.output_dir)

        Returns:
            dict: Download results with file paths and status
        """
        if download_dir is None:
            download_dir = self.output_dir

        # Add dewpoint temperature for VPD calculation if temperature is requested
        download_variables = variables.copy()
        if '2m_temperature' in variables and '2m_dewpoint_temperature' not in download_variables:
            download_variables.append('2m_dewpoint_temperature')
            self.logger.info("Added 2m_dewpoint_temperature for VPD calculation")

        self.logger.info("Starting CBF meteorological data download")
        self.logger.info(f"Variables: {download_variables}")
        self.logger.info(f"Years: {years}")
        self.logger.info(f"Months: {months}")
        self.logger.info(f"Download directory: {download_dir}")

        # Download ERA5 data
        download_result = self.download_data(
            variables=download_variables,
            years=years,
            months=months,
            processing_type="monthly"
        )

        if download_result.get('status') != 'completed':
            error_msg = f"ERA5 download failed: {download_result.get('error')}"
            self.logger.error(error_msg)
            return {"status": "failed", "error": error_msg}

        downloaded_files = download_result.get('downloaded_files', [])
        self.logger.info(f"Downloaded {len(downloaded_files)} ERA5 files")

        return {
            "status": "completed",
            "downloaded_files": downloaded_files,
            "download_directory": download_dir,
            "variables": download_variables,
            "message": f"Downloaded {len(downloaded_files)} files for CBF processing"
        }

    def generate_cbf_met_drivers(self,
                               variables: List[str],
                               years: List[int],
                               months: List[int],
                               co2_data_dir: str = None,
                               fire_data_dir: str = None,
                               land_fraction_file: str = None,
                               land_threshold: float = 0.5,
                               output_filename: str = "AllMet05x05_LFmasked.nc",
                               output_dir: str = None) -> str:
        """
        Generate unified CBF meteorological driver files (download + process).

        DEPRECATED: Use download_cbf_met_variables() + CBFMetProcessor for better
        separation of concerns and resilience to download failures.

        Complete pipeline that downloads ECMWF data, calculates derived variables,
        integrates external data, applies land masking, and creates the unified
        meteorological driver file required by erens_cbf_code.py.

        Args:
            variables: List of ECMWF variable names to download
            years: List of years to process
            months: List of months to process
            co2_data_dir: Directory containing NOAA CO2 data (optional)
            fire_data_dir: Directory containing GFED fire data (optional)
            land_fraction_file: Path to MODIS land fraction file (optional)
            land_threshold: Land fraction threshold for masking (default: 0.5)
            output_filename: Name of output CBF file (default: AllMet05x05_LFmasked.nc)
            output_dir: Output directory for final CBF file

        Returns:
            str: Path to generated CBF meteorological driver file
        """
        self.logger.warning("generate_cbf_met_drivers() is deprecated. "
                           "Use download_cbf_met_variables() + CBFMetProcessor for better reliability.")

        # Import here to avoid circular imports
        from src.cbf_met_processor import CBFMetProcessor

        if output_dir is None:
            output_dir = self.output_dir

        # Create subdirectories for intermediate files
        intermediate_dir = os.path.join(output_dir, 'intermediate')
        os.makedirs(intermediate_dir, exist_ok=True)

        try:
            # Step 1: Download ECMWF data using new separated method
            self.logger.info("Step 1: Downloading ECMWF data")
            download_result = self.download_cbf_met_variables(
                variables=variables,
                years=years,
                months=months,
                download_dir=intermediate_dir
            )

            if download_result.get('status') != 'completed':
                raise ValueError(f"ECMWF download failed: {download_result.get('error')}")

            # Step 2: Process downloaded files using CBFMetProcessor
            self.logger.info("Step 2: Processing downloaded files with CBFMetProcessor")
            processor = CBFMetProcessor(output_dir=output_dir)

            cbf_filepath = processor.process_downloaded_files_to_cbf_met(
                input_dir=intermediate_dir,
                output_filename=output_filename,
                land_fraction_file=land_fraction_file,
                land_threshold=land_threshold,
                co2_data_dir=co2_data_dir,
                fire_data_dir=fire_data_dir
            )

            self.logger.info(f"CBF meteorological driver generation completed: {cbf_filepath}")
            return cbf_filepath

        except Exception as e:
            error_msg = f"Failed to generate CBF meteorological drivers: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _create_unified_cbf_file(self,
                               input_files: List[str],
                               output_filename: str,
                               output_dir: str) -> str:
        """Create unified CBF file from individual variable files."""
        output_filepath = os.path.join(output_dir, output_filename)

        self.logger.info(f"Creating unified CBF file: {output_filename}")
        self.logger.info(f"Combining {len(input_files)} input files")

        try:
            # Load all datasets
            datasets = []
            for file_path in input_files:
                try:
                    ds = xr.open_dataset(file_path)
                    datasets.append(ds)
                    self.logger.debug(f"Loaded {os.path.basename(file_path)}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")

            if not datasets:
                raise ValueError("No valid datasets to combine")

            # Merge all datasets
            self.logger.info("Merging datasets...")

            # Start with the first dataset as base
            combined_ds = datasets[0].copy()

            # Add variables from other datasets
            for ds in datasets[1:]:
                for var_name in ds.data_vars:
                    if var_name not in combined_ds.data_vars:
                        # Interpolate to match the combined dataset's coordinates
                        var_data = ds[var_name].interp_like(combined_ds, method='nearest')
                        combined_ds[var_name] = var_data

            # Ensure all required CBF variables are present
            required_cbf_vars = [
                'VPD', 'TOTAL_PREC', 'T2M_MIN', 'T2M_MAX',
                'STRD', 'SSRD', 'SNOWFALL', 'SKT'
            ]

            # Optional variables that may be present
            optional_cbf_vars = ['CO2', 'BURNED_AREA', 'Mean_FIR']

            missing_vars = []
            for var in required_cbf_vars:
                if var not in combined_ds.data_vars:
                    missing_vars.append(var)

            if missing_vars:
                self.logger.warning(f"Missing required CBF variables: {missing_vars}")

            # Set global attributes for CBF compatibility
            combined_ds.attrs.update({
                'title': 'CARDAMOM Meteorological Drivers',
                'description': 'Unified meteorological driver file for CBF input generation',
                'source': 'ERA5 reanalysis with NOAA CO2 and GFED fire data',
                'grid_resolution': '0.5 degrees',
                'land_masking': 'Applied with MODIS land fraction',
                'created_by': 'CARDAMOM ECMWF Downloader',
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'cbf_compatible': 'true'
            })

            # Set encoding for consistent NetCDF format
            encoding = {}
            for var_name in combined_ds.data_vars:
                encoding[var_name] = {
                    'zlib': True,
                    'complevel': 6,
                    'dtype': 'float32',
                    '_FillValue': -9999.0
                }

            # Save unified CBF file
            combined_ds.to_netcdf(output_filepath, encoding=encoding)

            # Close all datasets
            combined_ds.close()
            for ds in datasets:
                ds.close()

            self.logger.info(f"Unified CBF file created successfully: {output_filepath}")
            return output_filepath

        except Exception as e:
            error_msg = f"Failed to create unified CBF file: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _validate_cbf_file(self, cbf_filepath: str) -> bool:
        """Validate CBF file for compatibility with erens_cbf_code.py."""
        self.logger.info(f"Validating CBF file: {os.path.basename(cbf_filepath)}")

        try:
            # Load CBF file
            cbf_ds = xr.open_dataset(cbf_filepath)

            # Required variables from erens_cbf_code.py
            required_vars = [
                'VPD', 'TOTAL_PREC', 'T2M_MIN', 'T2M_MAX',
                'STRD', 'SSRD', 'SNOWFALL', 'SKT'
            ]

            # Check for required variables
            missing_vars = []
            present_vars = []
            for var in required_vars:
                if var in cbf_ds.data_vars:
                    present_vars.append(var)
                else:
                    missing_vars.append(var)

            # Check for optional variables
            optional_vars = ['CO2', 'BURNED_AREA', 'Mean_FIR']
            optional_present = []
            for var in optional_vars:
                if var in cbf_ds.data_vars:
                    optional_present.append(var)

            # Check spatial dimensions
            has_lat = 'latitude' in cbf_ds.dims or 'lat' in cbf_ds.dims
            has_lon = 'longitude' in cbf_ds.dims or 'lon' in cbf_ds.dims
            has_time = 'time' in cbf_ds.dims

            # Log validation results
            self.logger.info(f"Present required variables ({len(present_vars)}/{len(required_vars)}): {present_vars}")
            if missing_vars:
                self.logger.warning(f"Missing required variables: {missing_vars}")

            if optional_present:
                self.logger.info(f"Present optional variables: {optional_present}")

            self.logger.info(f"Spatial dimensions - Latitude: {has_lat}, Longitude: {has_lon}")
            self.logger.info(f"Temporal dimension - Time: {has_time}")

            # Check data ranges for key variables
            for var_name in present_vars[:3]:  # Check first few variables
                var_data = cbf_ds[var_name]
                data_min = float(var_data.min().values)
                data_max = float(var_data.max().values)
                data_mean = float(var_data.mean().values)

                self.logger.info(f"{var_name}: range [{data_min:.2f}, {data_max:.2f}], mean {data_mean:.2f}")

            cbf_ds.close()

            # Validation criteria
            validation_passed = (
                len(missing_vars) == 0 and
                has_lat and has_lon and has_time
            )

            if validation_passed:
                self.logger.info("CBF file validation PASSED")
            else:
                self.logger.warning("CBF file validation FAILED")

            return validation_passed

        except Exception as e:
            self.logger.error(f"CBF file validation error: {e}")
            return False