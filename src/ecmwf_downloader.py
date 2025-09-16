#!/usr/bin/env python3
"""
Enhanced ECMWF Downloader for CARDAMOM with Variable Registry

Enhanced version of existing ECMWFDownloader with additional variables,
processing hints, and validation capabilities for CARDAMOM preprocessing.
"""

import cdsapi
import os
import time
from typing import List, Dict, Union, Any
import logging
from .base_downloader import BaseDownloader


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
                "units": "K",
                "processing": "min_max_monthly",
                "description": "Air temperature at 2m height for photosynthesis calculations",
                "typical_range": [233, 323],  # -40°C to 50°C
                "required_for": ["photosynthesis", "respiration", "vpd_calculation"]
            },
            "2m_dewpoint_temperature": {
                "cardamom_name": "D2M",
                "units": "K",
                "processing": "hourly_averaged",
                "description": "Dewpoint temperature for vapor pressure deficit calculations",
                "typical_range": [193, 303],  # -80°C to 30°C
                "required_for": ["vpd_calculation", "humidity_stress"]
            },
            "surface_solar_radiation_downwards": {
                "cardamom_name": "SSRD",
                "units": "J m-2",
                "processing": "monthly_mean",
                "description": "Downward solar radiation for photosynthesis light limitation",
                "typical_range": [0, 3.6e7],  # 0 to ~36 MJ/m²/day
                "required_for": ["photosynthesis", "par_calculation"]
            },
            "surface_thermal_radiation_downwards": {
                "cardamom_name": "STRD",
                "units": "J m-2",
                "processing": "monthly_mean",
                "description": "Downward thermal radiation for energy balance",
                "typical_range": [1e7, 5e7],  # ~10-50 MJ/m²/day
                "required_for": ["energy_balance"]
            },
            "total_precipitation": {
                "cardamom_name": "TP",
                "units": "m",
                "processing": "monthly_sum",
                "description": "Total precipitation for soil moisture and plant water availability",
                "typical_range": [0, 1.0],  # 0 to 1000 mm/month
                "required_for": ["soil_moisture", "water_stress"]
            },
            "skin_temperature": {
                "cardamom_name": "SKT",
                "units": "K",
                "processing": "monthly_mean",
                "description": "Surface skin temperature for soil respiration",
                "typical_range": [223, 333],  # -50°C to 60°C
                "required_for": ["soil_respiration", "surface_fluxes"]
            },
            "snowfall": {
                "cardamom_name": "SF",
                "units": "m of water equivalent",
                "processing": "monthly_sum",
                "description": "Snowfall for seasonal carbon cycle dynamics",
                "typical_range": [0, 0.5],  # 0 to 500 mm water equivalent
                "required_for": ["seasonal_dynamics", "snow_cover"]
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
                times = ["00:00", "01:00"]

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