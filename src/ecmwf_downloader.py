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
import zipfile
from typing import List, Dict, Union, Any
import logging
import numpy as np
import shutil
import xarray as xr
from base_downloader import BaseDownloader
from atmospheric_science import calculate_vapor_pressure_deficit_matlab
from time_utils import standardize_time_coordinate, ensure_monotonic_time
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
                 download_format: str = "zip",
                 output_dir: str = ".",
):
        """
        Initialize enhanced ECMWF downloader with variable registry.

        Args:
            area: [North, West, South, East] bounding box in decimal degrees
                 Default: Global coverage [-89.75, -179.75, 89.75, 179.75]
            grid: Grid resolution as list (default: ["0.5/0.5"])
            data_format: Output format - "netcdf" or "grib" (default: "netcdf")
            download_format: Download format (default: "zip")
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

        # Setup variable and coordinate name mappings for ERA5 abbreviations
        self.era5_name_mapping = self._setup_era5_name_mapping()
        self.coordinate_name_mapping = self._setup_coordinate_name_mapping()

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
                "download_source": "hourly",  # Must download from hourly ERA5 dataset
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
                "download_source": "hourly",  # Must download from hourly ERA5 dataset
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
                "download_source": "monthly",  # Download from monthly ERA5 dataset
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
                "download_source": "monthly",  # Download from monthly ERA5 dataset
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
                "download_source": "monthly",  # Download from monthly ERA5 dataset
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
                "download_source": "monthly",  # Download from monthly ERA5 dataset
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
                "download_source": "monthly",  # Download from monthly ERA5 dataset
                "processing": "monthly_sum",
                "description": "Snowfall for seasonal carbon cycle dynamics",
                "typical_range": [0, 0.5],  # 0 to 500 mm water equivalent
                "required_for": ["seasonal_dynamics", "snow_cover"],
                "cbf_processing": "convert_to_mm"
            }
        }

    def _setup_era5_name_mapping(self) -> Dict[str, List[str]]:
        """
        Mapping of ERA5 variable names to possible alternative names in NetCDF files.

        ECMWF returns abbreviated variable names in downloaded files that differ
        from the full names used in API requests. This mapping handles the conversion.

        Returns:
            dict: Mapping from full variable names to list of alternative names
        """
        return {
            # Core CARDAMOM meteorological variables
            "2m_temperature": ["2m_temperature", "t2m", "T2M"],
            "2m_dewpoint_temperature": ["2m_dewpoint_temperature", "d2m", "D2M"],
            "surface_solar_radiation_downwards": ["surface_solar_radiation_downwards", "ssrd", "SSRD"],
            "surface_thermal_radiation_downwards": ["surface_thermal_radiation_downwards", "strd", "STRD"],
            "total_precipitation": ["total_precipitation", "tp", "TP"],
            "skin_temperature": ["skin_temperature", "skt", "SKT"],
            "snowfall": ["snowfall", "sf", "SF"],
            # Additional meteorological variables
            "10m_u_component_of_wind": ["10m_u_component_of_wind", "u10", "U10"],
            "10m_v_component_of_wind": ["10m_v_component_of_wind", "v10", "V10"],
            "mean_sea_level_pressure": ["mean_sea_level_pressure", "msl", "MSL"],
            "surface_pressure": ["surface_pressure", "sp", "SP"]
        }

    def _setup_coordinate_name_mapping(self) -> Dict[str, List[str]]:
        """
        Mapping of coordinate names to possible alternative names in NetCDF files.

        ECMWF files may use different coordinate names like 'valid_time' instead of 'time'.

        Returns:
            dict: Mapping from standard coordinate names to alternatives
        """
        return {
            "time": ["time", "valid_time"],
            "latitude": ["latitude", "lat"],
            "longitude": ["longitude", "lon"]
        }

    def _get_actual_variable_name(self, requested_var: str, dataset: 'xr.Dataset') -> str:
        """
        Find the actual variable name in the dataset for a requested variable.

        Checks the requested name and all known alternative names to find
        what's actually present in the downloaded NetCDF file.

        Args:
            requested_var: The variable name as requested in the API call
            dataset: The xarray Dataset to search

        Returns:
            str: The actual variable name found in the dataset, or None if not found
        """
        all_vars = set(dataset.data_vars.keys()) | set(dataset.coords.keys())

        # First try the requested name directly
        if requested_var in all_vars:
            return requested_var

        # Try alternative names from the variable mapping
        alternative_names = self.era5_name_mapping.get(requested_var, [])
        for alt_name in alternative_names:
            if alt_name in all_vars:
                self.logger.info(f"Found variable '{requested_var}' as '{alt_name}' in downloaded file")
                return alt_name

        # Variable not found under any known name
        available_vars = list(dataset.data_vars.keys())
        self.logger.warning(f"Variable '{requested_var}' not found in file. Available variables: {available_vars}")
        return None

    def _get_actual_coordinate_name(self, requested_coord: str, dataset: 'xr.Dataset') -> str:
        """
        Find the actual coordinate name in the dataset for a requested coordinate.

        Handles cases where ECMWF uses 'valid_time' instead of 'time', etc.

        Args:
            requested_coord: The coordinate name to find
            dataset: The xarray Dataset to search

        Returns:
            str: The actual coordinate name found in the dataset, or requested_coord if not found
        """
        all_coords = set(dataset.coords.keys()) | set(dataset.dims.keys())

        # First try the requested coordinate name directly
        if requested_coord in all_coords:
            return requested_coord

        # Try alternative coordinate names from the mapping
        alternative_names = self.coordinate_name_mapping.get(requested_coord, [])
        for alt_name in alternative_names:
            if alt_name in all_coords:
                self.logger.info(f"Found coordinate '{requested_coord}' as '{alt_name}' in downloaded file")
                return alt_name

        # Coordinate not found under any known name, return requested name
        self.logger.warning(f"Coordinate '{requested_coord}' not found in file. Available coordinates: {list(all_coords)}")
        return requested_coord

    def standardize_dataset_coordinates(self, dataset: 'xr.Dataset') -> 'xr.Dataset':
        """
        Standardize coordinate names in a dataset to use standard names.

        Renames coordinates like 'valid_time' to 'time' for consistency.

        Args:
            dataset: The xarray Dataset to standardize

        Returns:
            xr.Dataset: Dataset with standardized coordinate names
        """
        rename_dict = {}

        for standard_coord, alternatives in self.coordinate_name_mapping.items():
            for alt_coord in alternatives[1:]:  # Skip first (standard) name
                if alt_coord in dataset.coords or alt_coord in dataset.dims:
                    if standard_coord not in dataset.coords and standard_coord not in dataset.dims:
                        rename_dict[alt_coord] = standard_coord
                        self.logger.info(f"Standardizing coordinate '{alt_coord}' to '{standard_coord}'")
                        break

        if rename_dict:
            return dataset.rename(rename_dict)
        else:
            return dataset

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

    def _determine_required_datasets(self, variables: List[str]) -> Dict[str, List[str]]:
        """
        Analyze requested variables to determine which ERA5 datasets are needed.

        Args:
            variables: List of ERA5 variable names to analyze

        Returns:
            dict: Dataset requirements with keys:
                - 'hourly_vars': Variables that need hourly ERA5 dataset
                - 'monthly_vars': Variables that need monthly ERA5 dataset
                - 'missing_vars': Variables not found in registry

        Raises:
            ValueError: If no valid variables are found
        """
        hourly_vars = []
        monthly_vars = []
        missing_vars = []

        for variable in variables:
            var_metadata = self.get_variable_metadata(variable)
            if var_metadata is None:
                missing_vars.append(variable)
                continue

            download_source = var_metadata.get('download_source', 'monthly')  # Default to monthly

            if download_source == 'hourly':
                hourly_vars.append(variable)
                self.logger.info(f"Variable '{variable}' requires hourly ERA5 dataset")
            elif download_source == 'monthly':
                monthly_vars.append(variable)
                self.logger.info(f"Variable '{variable}' requires monthly ERA5 dataset")
            else:
                self.logger.warning(f"Unknown download_source '{download_source}' for variable '{variable}', defaulting to monthly")
                monthly_vars.append(variable)

        # Log summary
        if hourly_vars:
            self.logger.info(f"Hourly dataset required for {len(hourly_vars)} variables: {hourly_vars}")
        if monthly_vars:
            self.logger.info(f"Monthly dataset required for {len(monthly_vars)} variables: {monthly_vars}")
        if missing_vars:
            self.logger.warning(f"Unknown variables (will be skipped): {missing_vars}")

        # Validate that we have at least some valid variables
        if not hourly_vars and not monthly_vars:
            error_msg = f"No valid variables found in registry. Unknown variables: {missing_vars}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        return {
            'hourly_vars': hourly_vars,
            'monthly_vars': monthly_vars,
            'missing_vars': missing_vars
        }

    def _extract_zip_file(self, zip_filepath: str, extract_dir: str = None) -> List[str]:
        """
        Extract a ZIP file and return list of extracted NetCDF files.
        
        Args:
            zip_filepath: Path to the ZIP file to extract
            extract_dir: Directory to extract files to (default: temp directory next to ZIP)
            
        Returns:
            List of paths to extracted NetCDF files
            
        Raises:
            Exception: If ZIP extraction fails or no NetCDF files found
        """
        extracted_files = []
        
        # Create extraction directory if not provided
        if extract_dir is None:
            zip_dir = os.path.dirname(zip_filepath)
            zip_name = os.path.splitext(os.path.basename(zip_filepath))[0]
            extract_dir = os.path.join(zip_dir, f"temp_extract_{zip_name}")
        
        # Ensure extraction directory exists
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            self.logger.info(f"Extracting ZIP file: {zip_filepath} to {extract_dir}")
            
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(extract_dir)
                
                # Find all NetCDF files in the extracted content
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if file.lower().endswith(('.nc', '.netcdf')):
                            full_path = os.path.join(root, file)
                            extracted_files.append(full_path)
                            self.logger.info(f"Found NetCDF file: {file}")
            
            if not extracted_files:
                raise Exception(f"No NetCDF files found in ZIP archive: {zip_filepath}")
                
            self.logger.info(f"Successfully extracted {len(extracted_files)} NetCDF files from ZIP")
            return extracted_files
            
        except zipfile.BadZipFile:
            raise Exception(f"Invalid or corrupted ZIP file: {zip_filepath}")
        except Exception as e:
            # Cleanup extraction directory on failure
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise Exception(f"Failed to extract ZIP file {zip_filepath}: {e}")

    
    def _cleanup_temp_files(self, zip_filepath: str, extracted_files: List[str]):
        """
        Clean up temporary ZIP file and extracted files.
        
        Args:
            zip_filepath: Path to the ZIP file to remove
            extracted_files: List of extracted file paths to remove
        """
        try:
            # Remove the ZIP file
            if os.path.exists(zip_filepath):
                os.remove(zip_filepath)
                self.logger.info(f"Removed temporary ZIP file: {zip_filepath}")
            
            # Remove extracted files and their parent directories
            extraction_dirs = set()
            for extracted_file in extracted_files:
                if os.path.exists(extracted_file):
                    extraction_dirs.add(os.path.dirname(extracted_file))
                    os.remove(extracted_file)
            
            # Remove extraction directories if they're empty
            for extract_dir in extraction_dirs:
                try:
                    if os.path.exists(extract_dir) and not os.listdir(extract_dir):
                        os.rmdir(extract_dir)
                        self.logger.info(f"Removed empty extraction directory: {extract_dir}")
                except OSError:
                    # Directory not empty or other error, ignore
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Failed to clean up some temporary files: {e}")

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
        """Download hourly ERA5 data with optimized bulk requests."""

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

        # Optimize: Make one request per year for all variables and months
        for year in years:
            # Prepare month list as strings
            month_strings = [f"{month:02d}" for month in months]
            
            # Create temporary filename for multi-variable download (ZIP format)
            temp_filename = f"{file_prefix}_MULTI_{year}.zip"
            temp_filepath = os.path.join(self.output_dir, temp_filename)

            try:
                # Prepare bulk download request for all variables and months in this year
                request = {
                    "product_type": ["reanalysis"],
                    "variable": variables,  # Download all variables at once
                    "year": str(year),
                    "month": month_strings,  # Download all months at once
                    "day": days,
                    "time": times,
                    "data_format": self.data_format,
                    "grid": self.grid,
                    "download_format": self.download_format,  # This will be "zip"
                    "area": self.area
                }

                self.logger.info(f"Downloading {len(variables)} variables for {len(months)} months in {year}...")
                self.logger.info(f"Variables: {variables}")
                self.logger.info(f"Months: {month_strings}")

                # Download bulk ZIP file with all variables and months
                target = self.client.retrieve(dataset, request).download()
                shutil.move(target, temp_filepath)

                self.logger.info(f"Bulk ZIP download completed: {temp_filename}")

                # Extract ZIP file and get list of NetCDF files
                extracted_files = self._extract_zip_file(temp_filepath)

                # Consolidate extracted files into single file with year range
                consolidated_file = self._consolidate_extracted_files(
                    extracted_files=extracted_files,
                    file_prefix=file_prefix,
                    years=[year]
                )

                downloaded_files.append(consolidated_file)

                # Clean up temporary files
                self._cleanup_temp_files(temp_filepath, extracted_files)

                # Record successful download of consolidated file
                consolidated_filename = os.path.basename(consolidated_file)
                self._record_download_attempt(consolidated_filename, "success")

                self.logger.info(f"Successfully consolidated {len(variables)} variables for {len(months)} months in {year}")

            except Exception as e:
                error_msg = f"Bulk download failed for {year}: {e}"
                self.logger.error(error_msg)
                
                # Record failed download
                failed_filename = f"{file_prefix}_{year}.nc"
                self._record_download_attempt(failed_filename, "failed", str(e))

                # Clean up temp files if they exist but failed
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except:
                        pass

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
        """Download monthly ERA5 data with optimized bulk requests."""

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

        # Optimize: Make one request per year for all variables and months
        for year in years:
            # Prepare month list as strings
            month_strings = [f"{month:02d}" for month in months]
            
            # Create temporary filename for multi-variable download (ZIP format)
            temp_filename = f"{file_prefix}_MULTI_{year}.zip"
            temp_filepath = os.path.join(self.output_dir, temp_filename)

            try:
                # Prepare bulk download request for all variables and months in this year
                request = {
                    "product_type": [product_type],
                    "variable": variables,  # Download all variables at once
                    "year": str(year),
                    "month": month_strings,  # Download all months at once
                    "time": times,
                    "data_format": self.data_format,
                    "download_format": self.download_format,  # This will be "zip"
                    "area": self.area
                }

                self.logger.info(f"Downloading {len(variables)} variables for {len(months)} months in {year}...")
                self.logger.info(f"Variables: {variables}")
                self.logger.info(f"Months: {month_strings}")

                # Download bulk ZIP file with all variables and months
                target = self.client.retrieve(dataset, request).download()
                shutil.move(target, temp_filepath)

                self.logger.info(f"Bulk ZIP download completed: {temp_filename}")

                # Extract ZIP file and get list of NetCDF files
                extracted_files = self._extract_zip_file(temp_filepath)

                # Consolidate extracted files into single file with year range
                consolidated_file = self._consolidate_extracted_files(
                    extracted_files=extracted_files,
                    file_prefix=file_prefix,
                    years=[year]
                )

                downloaded_files.append(consolidated_file)

                # Clean up temporary files
                self._cleanup_temp_files(temp_filepath, extracted_files)

                # Record successful download of consolidated file
                consolidated_filename = os.path.basename(consolidated_file)
                self._record_download_attempt(consolidated_filename, "success")

                self.logger.info(f"Successfully consolidated {len(variables)} variables for {len(months)} months in {year}")

            except Exception as e:
                error_msg = f"Bulk download failed for {year}: {e}"
                self.logger.error(error_msg)
                
                # Record failed download
                failed_filename = f"{file_prefix}_{year}.nc"
                self._record_download_attempt(failed_filename, "failed", str(e))

                # Clean up temp file if it exists but failed
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except:
                        pass

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

            # Get actual variable names using mapping system
            temp_var = self._get_actual_variable_name('2m_temperature', temp_ds)
            dewpoint_var = self._get_actual_variable_name('2m_dewpoint_temperature', dewpoint_ds)

            if temp_var is None:
                raise ValueError("Temperature variable not found in temperature file")
            if dewpoint_var is None:
                raise ValueError("Dewpoint temperature variable not found in dewpoint file")

            # Extract temperature and dewpoint arrays
            temperature_kelvin = temp_ds[temp_var]
            dewpoint_kelvin = dewpoint_ds[dewpoint_var]

            # Convert to Celsius for VPD calculation
            temperature_celsius = temperature_kelvin - 273.15
            dewpoint_celsius = dewpoint_kelvin - 273.15

            # Calculate VPD using Phase 8 atmospheric science function
            self.logger.info("Calculating Vapor Pressure Deficit (VPD)")

            # Get actual time coordinate name
            time_coord = self._get_actual_coordinate_name('time', temp_ds)

            # For monthly data, use the temperature directly
            # For hourly/daily data, we need temperature maximum
            if time_coord in temperature_celsius.dims:
                if len(temperature_celsius[time_coord]) > 31:  # Likely hourly or daily data
                    # Calculate monthly maximum temperature for VPD
                    temp_max_celsius = temperature_celsius.groupby(f'{time_coord}.month').max(dim=time_coord)
                    dewpoint_avg_celsius = dewpoint_celsius.groupby(f'{time_coord}.month').mean(dim=time_coord)
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
            if time_coord in temperature_celsius.dims and len(temperature_celsius[time_coord]) > 31:
                self.logger.info("Calculating monthly temperature min/max")

                # Calculate monthly statistics
                temp_min_monthly = temperature_kelvin.groupby(f'{time_coord}.month').min(dim=time_coord)
                temp_max_monthly = temperature_kelvin.groupby(f'{time_coord}.month').max(dim=time_coord)

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

            # Get the variable using the centralized name mapping
            actual_var_name = self._get_actual_variable_name(variable_name, ds)
            if actual_var_name:
                data_var = ds[actual_var_name]
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
        co2_var_names = ['co2', 'CO2', 'co2_concentration', 'mole_fraction', 'co2_mole_fraction']
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

        # Analyze variables to determine required datasets
        try:
            dataset_requirements = self._determine_required_datasets(download_variables)
        except ValueError as e:
            return {"status": "failed", "error": str(e)}

        all_downloaded_files = []

        # Download from hourly dataset if needed
        if dataset_requirements.get('hourly_vars'):
            self.logger.info(f"Downloading hourly variables: {dataset_requirements['hourly_vars']}")
            hourly_result = self.download_data(
                variables=dataset_requirements['hourly_vars'],
                years=years,
                months=months,
                processing_type="hourly"  # Auto-determined based on variable requirements
            )

            if hourly_result.get('status') != 'completed':
                error_msg = f"Hourly ERA5 download failed: {hourly_result.get('error')}"
                self.logger.error(error_msg)
                return {"status": "failed", "error": error_msg}

            hourly_files = hourly_result.get('downloaded_files', [])
            all_downloaded_files.extend(hourly_files)
            self.logger.info(f"Downloaded {len(hourly_files)} hourly ERA5 files")

        # Download from monthly dataset if needed
        if dataset_requirements.get('monthly_vars'):
            self.logger.info(f"Downloading monthly variables: {dataset_requirements['monthly_vars']}")
            monthly_result = self.download_data(
                variables=dataset_requirements['monthly_vars'],
                years=years,
                months=months,
                processing_type="monthly"  # Auto-determined based on variable requirements
            )

            if monthly_result.get('status') != 'completed':
                error_msg = f"Monthly ERA5 download failed: {monthly_result.get('error')}"
                self.logger.error(error_msg)
                return {"status": "failed", "error": error_msg}

            monthly_files = monthly_result.get('downloaded_files', [])
            all_downloaded_files.extend(monthly_files)
            self.logger.info(f"Downloaded {len(monthly_files)} monthly ERA5 files")

        downloaded_files = all_downloaded_files
        self.logger.info(f"Total downloaded files: {len(downloaded_files)} ERA5 files")

        return {
            "status": "completed",
            "downloaded_files": downloaded_files,
            "download_directory": download_dir,
            "variables": download_variables,
            "dataset_requirements": dataset_requirements,
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

    def _split_multivariable_file(self,
                                 multivariable_filepath: str,
                                 variables: List[str],
                                 file_prefix: str,
                                 year: int,
                                 month: int) -> List[str]:
        """
        Split a multi-variable NetCDF file into individual variable files.
        
        Args:
            multivariable_filepath: Path to the downloaded multi-variable NetCDF file
            variables: List of variable names to extract
            file_prefix: Prefix for output files (e.g., "ECMWF_MONTHLY")
            year: Year for filename generation
            month: Month for filename generation
            
        Returns:
            List of paths to created individual variable files
        """
        split_files = []
        
        try:
            self.logger.info(f"Splitting multi-variable file: {multivariable_filepath}")
            
            # Open the multi-variable dataset
            ds = xr.open_dataset(multivariable_filepath)
            
            # Extract each variable into a separate file
            for variable in variables:
                # Find the actual variable name in the dataset (handles ERA5 abbreviations)
                actual_var_name = self._get_actual_variable_name(variable, ds)
                if actual_var_name:
                    # Create individual file path
                    filename = f"{file_prefix}_{variable}_{month:02d}{year}.nc"
                    individual_filepath = os.path.join(self.output_dir, filename)

                    # Extract single variable dataset using actual name
                    var_ds = ds[[actual_var_name]]

                    # Rename the variable back to the requested name for consistency
                    var_ds = var_ds.rename({actual_var_name: variable})

                    # Save individual variable file
                    var_ds.to_netcdf(individual_filepath)
                    split_files.append(individual_filepath)

                    self.logger.info(f"Extracted {variable} to: {filename}")

                    # Close variable dataset
                    var_ds.close()
            
            # Close the original dataset
            ds.close()
            
            # Remove the temporary multi-variable file
            os.remove(multivariable_filepath)
            self.logger.info(f"Removed temporary multi-variable file: {multivariable_filepath}")
            
        except Exception as e:
            error_msg = f"Failed to split multi-variable file {multivariable_filepath}: {e}"
            self.logger.error(error_msg)
            # Don't raise exception - return what we managed to create
        
        return split_files

    def _consolidate_extracted_files(self,
                                     extracted_files: List[str],
                                     file_prefix: str,
                                     years: List[int]) -> str:
        """
        Consolidate multiple NetCDF files into a single file with year range naming.

        Args:
            extracted_files: List of extracted NetCDF file paths to consolidate
            file_prefix: Prefix for output file (e.g., "ECMWF_MONTHLY")
            years: List of years to determine year range for filename

        Returns:
            Path to the consolidated file
        """
        try:
            self.logger.info(f"Consolidating {len(extracted_files)} extracted files")

            if not extracted_files:
                raise ValueError("No extracted files provided for consolidation")

            # Determine year range for filename
            min_year = min(years)
            max_year = max(years)

            # Create consolidated filename with year range
            if min_year == max_year:
                consolidated_filename = f"{file_prefix}_{min_year}.nc"
            else:
                consolidated_filename = f"{file_prefix}_{min_year}_{max_year}.nc"

            consolidated_filepath = os.path.join(self.output_dir, consolidated_filename)

            # Open all datasets
            datasets = []
            for filepath in extracted_files:
                try:
                    ds = xr.open_dataset(filepath, engine='netcdf4')
                    if "valid_time" in ds.dims or "valid_time" in ds.coords:
                        ds = ds.rename({"valid_time": "time"})
                    datasets.append(ds)
                    self.logger.info(f"Loaded dataset from: {os.path.basename(filepath)}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {filepath}: {e}")
                    continue

            if not datasets:
                raise ValueError("No valid datasets could be loaded from extracted files")

            # Consolidate datasets
            if len(datasets) == 1:
                # Single dataset, just copy it
                consolidated_ds = datasets[0]
            else:
                # Multiple datasets - concatenate along time dimension if it exists
                # or merge if different variables
                try:
                    # Check if all datasets have time dimension
                    has_time = all('time' in ds.dims for ds in datasets)

                    if has_time:
                        # Check if all datasets have the same time coords
                        same_time = all(ds['time'].equals(datasets[0]['time']) for ds in datasets)

                        if same_time:
                            # Merge variables (since time coords are identical)
                            consolidated_ds = xr.merge(datasets)
                            self.logger.info("Consolidated datasets by merging variables (same time coords)")
                        else:
                            # Concatenate (datasets have different time periods)
                            consolidated_ds = xr.concat(datasets, dim='time')
                            consolidated_ds = consolidated_ds.sortby('time')
                            self.logger.info("Consolidated datasets by concatenating along time dimension")
                    else:
                        # Merge datasets (no time dimension at all)
                        consolidated_ds = xr.merge(datasets)
                        self.logger.info("Consolidated datasets by merging variables (no time dim)")

                except Exception as e:
                    self.logger.warning(f"Failed to concatenate/merge datasets: {e}")
                    # Fallback: just use the first dataset
                    consolidated_ds = datasets[0]
                    self.logger.warning("Using first dataset as fallback")

            # Standardize time coordinate to CARDAMOM convention
            consolidated_ds = standardize_time_coordinate(consolidated_ds)
            consolidated_ds = ensure_monotonic_time(consolidated_ds)
            self.logger.info("Standardized time coordinate to CARDAMOM convention (days since 2001-01-01)")

            # Save consolidated file
            consolidated_ds.to_netcdf(consolidated_filepath)
            self.logger.info(f"Saved consolidated file: {consolidated_filename}")

            # Close all datasets
            for ds in datasets:
                ds.close()

            return consolidated_filepath

        except Exception as e:
            error_msg = f"Failed to consolidate extracted files: {e}"
            self.logger.error(error_msg)
            raise e
