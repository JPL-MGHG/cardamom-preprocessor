"""
CBF Meteorological Data Processor

Processes pre-downloaded meteorological data files into CARDAMOM Binary Format (CBF)
compatible meteorological driver files. Separates processing from download to enable
resilient workflows that can handle data source provider issues.

This module creates the unified AllMet05x05_LFmasked.nc file required by erens_cbf_code.py
by processing ERA5 meteorological variables and integrating external data sources.
"""

import os
import glob
import time
from typing import List, Dict, Union, Any
import numpy as np
import xarray as xr
from pathlib import Path

from logging_utils import setup_cardamom_logging
from atmospheric_science import calculate_vapor_pressure_deficit_matlab
from time_utils import standardize_time_coordinate, ensure_monotonic_time, get_time_range_info


class CBFMetProcessor:
    """
    Processes downloaded meteorological data into CBF-compatible format.

    This class takes pre-downloaded ERA5 NetCDF files and processes them into
    the unified meteorological driver file format required by CARDAMOM CBF
    generation pipeline.

    Scientific Context:
    CBF (CARDAMOM Binary Format) requires specific meteorological variables
    in standardized units and naming conventions. This processor handles
    the transformation from raw ERA5 data to CBF requirements.
    """

    def __init__(self, output_dir: str = ".",
                 target_grid_resolution: float = 0.5,
                 target_lat_range: tuple = (-89.75, 89.75),
                 target_lon_range: tuple = (-179.75, 179.75)):
        """
        Initialize CBF meteorological processor.

        Args:
            output_dir: Directory for processed CBF files
            target_grid_resolution: Target grid resolution in degrees (default: 0.5)
            target_lat_range: Target latitude range as (min, max) in degrees (default: (-89.75, 89.75))
            target_lon_range: Target longitude range as (min, max) in degrees (default: (-179.75, 179.75))
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Target grid configuration
        self.target_grid_resolution = target_grid_resolution
        self.target_lat_range = target_lat_range
        self.target_lon_range = target_lon_range

        # Setup logging
        import logging
        self.logger = setup_cardamom_logging(log_level="DEBUG")

        # CBF variable requirements from erens_cbf_code.py (original names before internal renaming)
        self.cbf_met_variables = [
            'VPD', 'PREC', 'TMIN', 'TMAX',
            'STRD', 'SSRD', 'SNOW', 'CO2_2', 'BURN_2', 'SKT',
            'DISTURBANCE_FLUX', 'YIELD'  # Framework variables required by erens_cbf_code.py
        ]

        # ERA5 to CBF variable mapping (using original names that erens_cbf_code.py expects)
        self.era5_to_cbf_mapping = {
            'total_precipitation': 'PREC',
            '2m_temperature': ['TMIN', 'TMAX'],  # Derive min/max
            'surface_thermal_radiation_downwards': 'STRD',
            'surface_solar_radiation_downwards': 'SSRD',
            'snowfall': 'SNOW',
            'skin_temperature': 'SKT',
            # VPD calculated from temperature and dewpoint
            # CO2_2 and BURN_2 from external sources
        }

        # Unit conversion factors for CBF compatibility
        self.unit_conversions = {
            'PREC': {'from': 'm', 'to': 'mm/month', 'factor': 1000},
            'SNOW': {'from': 'm of water equivalent', 'to': 'mm/month', 'factor': 1000},
            'STRD': {'from': 'J m-2', 'to': 'W m-2', 'method': 'radiation_monthly'},
            'SSRD': {'from': 'J m-2', 'to': 'W m-2', 'method': 'radiation_monthly'},
        }

        # Setup variable and coordinate name mappings for ERA5 abbreviations
        self.era5_name_mapping = self._setup_era5_name_mapping()
        self.coordinate_name_mapping = self._setup_coordinate_name_mapping()

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

    def _create_target_grid_coordinates(self) -> Dict[str, np.ndarray]:
        """
        Create target grid coordinate arrays based on configuration.

        Generates latitude and longitude coordinate arrays at the specified
        target resolution and spatial bounds. This ensures all processed
        variables are regridded to a consistent, user-defined grid.

        Returns:
            dict: Dictionary with 'latitude' and 'longitude' coordinate arrays
        """
        # Generate latitude array (from min to max at target resolution)
        lat_min, lat_max = self.target_lat_range
        num_lat_points = int((lat_max - lat_min) / self.target_grid_resolution) + 1
        target_latitudes = np.linspace(lat_min, lat_max, num_lat_points)

        # Generate longitude array (from min to max at target resolution)
        lon_min, lon_max = self.target_lon_range
        num_lon_points = int((lon_max - lon_min) / self.target_grid_resolution) + 1
        target_longitudes = np.linspace(lon_min, lon_max, num_lon_points)

        self.logger.debug(
            f"Created target grid: {len(target_latitudes)}×{len(target_longitudes)} "
            f"at {self.target_grid_resolution}° resolution"
        )

        return {
            'latitude': target_latitudes,
            'longitude': target_longitudes
        }

    def _get_actual_variable_name(self, requested_var: str, dataset: 'xr.Dataset') -> str:
        """
        Find the actual variable name in the dataset for a requested variable.

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

    def process_downloaded_files_to_cbf_met(self,
                                          input_dir: str,
                                          output_filename: str = "AllMet05x05_LFmasked.nc",
                                          land_fraction_file: str = None,
                                          land_threshold: float = 0.5,
                                          co2_data_dir: str = None,
                                          fire_data_dir: str = None) -> str:
        """
        Process downloaded ERA5 files into unified CBF meteorological driver file.

        Args:
            input_dir: Directory containing downloaded ERA5 NetCDF files
            output_filename: Name of output CBF file
            land_fraction_file: Path to land fraction file for masking
            land_threshold: Land fraction threshold for masking
            co2_data_dir: Directory containing CO2 data (optional)
            fire_data_dir: Directory containing fire data (optional)

        Returns:
            str: Path to generated CBF meteorological driver file
        """
        input_path = Path(input_dir)
        output_filepath = self.output_dir / output_filename

        self.logger.info("Starting CBF meteorological processing")
        self.logger.info(f"Input directory: {input_path}")
        self.logger.info(f"Output file: {output_filepath}")

        try:
            # Step 1: Load all ERA5 files into unified dataset
            unified_era5_dataset = self._load_unified_era5_dataset(input_path)
            self.logger.info(f"Loaded unified ERA5 dataset with {len(unified_era5_dataset.data_vars)} variables")

            # Step 2: Process core ERA5 variables from unified dataset
            processed_variables = self._process_era5_variables(unified_era5_dataset)

            # Step 3: Calculate derived variables (VPD from temperature/dewpoint)
            derived_variables = self._calculate_derived_variables(unified_era5_dataset)
            processed_variables.update(derived_variables)

            # Step 4: Integrate external data (CO2, fire) if available
            external_variables = self._integrate_external_data(
                co2_data_dir, fire_data_dir, processed_variables
            )
            processed_variables.update(external_variables)

            # Step 5: Apply land masking if land fraction file provided
            if land_fraction_file and os.path.exists(land_fraction_file):
                masked_variables = self._apply_land_masking(
                    processed_variables, land_fraction_file, land_threshold
                )
            else:
                masked_variables = processed_variables
                self.logger.info("No land fraction file provided, skipping land masking")

            # Step 6: Create unified CBF file
            cbf_filepath = self._create_unified_cbf_file(
                masked_variables, output_filepath
            )

            # Step 7: Validate CBF file
            self._validate_cbf_file(cbf_filepath)

            # Step 8: Clean up unified dataset
            unified_era5_dataset.close()

            self.logger.info(f"CBF meteorological processing completed: {cbf_filepath}")
            return str(cbf_filepath)

        except Exception as e:
            error_msg = f"Failed to process downloaded files to CBF format: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _load_unified_era5_dataset(self, input_dir: Path) -> xr.Dataset:
        """
        Load all ERA5 NetCDF files into a unified dataset in memory.

        This replaces the previous approach of cataloging files and loading them
        multiple times per variable. Instead, we load all files once and process
        variables from the unified dataset.

        Args:
            input_dir: Directory containing ERA5 NetCDF files

        Returns:
            xr.Dataset: Unified dataset containing all available ERA5 variables
        """
        # Search for NetCDF files
        nc_files = list(input_dir.glob("*.nc"))
        self.logger.info(f"Found {len(nc_files)} NetCDF files in {input_dir}")

        if not nc_files:
            raise ValueError(f"No NetCDF files found in {input_dir}")

        # Track which variables we find
        found_variables = set()
        datasets_to_merge = []

        # Load each file and collect variables
        for nc_file in nc_files:
            try:
                ds = xr.open_dataset(nc_file)
                file_variables = list(ds.data_vars.keys())

                # Log what we found
                self.logger.debug(f"File {nc_file.name} contains variables: {file_variables}")
                found_variables.update(file_variables)

                # Add to merge list
                datasets_to_merge.append(ds)

            except Exception as e:
                self.logger.warning(f"Failed to read {nc_file}: {e}")

        if not datasets_to_merge:
            raise ValueError("No valid NetCDF files could be loaded")

        # Combine all datasets into unified dataset
        self.logger.info("Combining datasets into unified ERA5 dataset")

        try:
            if len(datasets_to_merge) == 1:
                unified_dataset = datasets_to_merge[0]
            else:
                # Try to merge along time dimension if possible, otherwise use merge
                try:
                    # Sort datasets by time if time dimension exists
                    datasets_with_time = []
                    datasets_without_time = []

                    for ds in datasets_to_merge:
                        # Rename valid_time to time
                        if "valid_time" in ds.dims or "valid_time" in ds.coords:
                            ds = ds.rename({"valid_time": "time"})


                        if 'time' in ds.dims and len(ds.time) > 0:
                            datasets_with_time.append(ds)
                        else:
                            datasets_without_time.append(ds)

                    if datasets_with_time:
                        # Sort by first time value
                        datasets_with_time.sort(key=lambda ds: ds.time.values[0])

                        # Concatenate along time dimension
                        time_combined = xr.concat(datasets_with_time, dim='time')

                        # Merge with any datasets without time dimension
                        if datasets_without_time:
                            unified_dataset = xr.merge([time_combined] + datasets_without_time)
                        else:
                            unified_dataset = time_combined
                    else:
                        # No time dimension, just merge
                        unified_dataset = xr.merge(datasets_to_merge)

                except Exception as e:
                    self.logger.warning(f"Failed to concatenate along time: {e}, using merge instead")
                    unified_dataset = xr.merge(datasets_to_merge)

        except Exception as e:
            self.logger.error(f"Failed to combine datasets: {e}")
            raise ValueError(f"Could not create unified dataset: {e}")

        # Report what we have in the unified dataset
        available_vars = list(unified_dataset.data_vars.keys())
        self.logger.info(f"Unified dataset contains {len(available_vars)} variables: {available_vars}")

        # Check for expected variables
        expected_vars = list(self.era5_to_cbf_mapping.keys()) + ['2m_dewpoint_temperature']
        missing_vars = []
        for var in expected_vars:
            actual_var_name = self._get_actual_variable_name(var, unified_dataset)
            if actual_var_name is None:
                missing_vars.append(var)

        if missing_vars:
            self.logger.warning(f"Expected variables not found: {missing_vars}")

        # Standardize time coordinate to CARDAMOM convention
        unified_dataset = standardize_time_coordinate(unified_dataset)
        unified_dataset = ensure_monotonic_time(unified_dataset)

        # Log time range information
        time_info = get_time_range_info(unified_dataset)
        if time_info.get('has_time'):
            self.logger.info(f"Time range: {time_info['start']} to {time_info['end']} ({time_info['count']} timesteps, {time_info['resolution']} resolution)")
            self.logger.info("Standardized time coordinate to CARDAMOM convention (days since 2001-01-01)")

        return unified_dataset

    def _process_era5_variables(self, unified_dataset: xr.Dataset) -> Dict[str, xr.Dataset]:
        """
        Process core ERA5 variables into CBF format from unified dataset.

        Args:
            unified_dataset: Unified dataset containing all ERA5 variables

        Returns:
            dict: Processed variables as xarray Datasets
        """
        processed_variables = {}

        for era5_var, cbf_var in self.era5_to_cbf_mapping.items():
            # Get actual variable name from mapping
            actual_var_name = self._get_actual_variable_name(era5_var, unified_dataset)
            if actual_var_name is None:
                self.logger.warning(f"Variable {era5_var} not found in unified dataset, skipping")
                continue

            self.logger.info(f"Processing {era5_var} (found as {actual_var_name}) -> {cbf_var}")

            try:
                # Extract variable from unified dataset
                variable_data = unified_dataset[[actual_var_name]]

                # Handle special cases
                if era5_var == '2m_temperature':
                    # Derive TMIN and TMAX from temperature data
                    temp_min, temp_max = self._derive_temperature_extremes(variable_data)
                    processed_variables['TMIN'] = temp_min
                    processed_variables['TMAX'] = temp_max
                else:
                    # Apply unit conversion if needed
                    converted_data = self._apply_unit_conversion(variable_data, cbf_var)
                    processed_variables[cbf_var] = converted_data

            except Exception as e:
                self.logger.error(f"Failed to process {era5_var}: {e}")
                continue

        return processed_variables

    def _calculate_derived_variables(self, unified_dataset: xr.Dataset) -> Dict[str, xr.Dataset]:
        """
        Calculate derived variables like VPD from temperature and dewpoint.

        Args:
            unified_dataset: Unified dataset containing all ERA5 variables

        Returns:
            dict: Derived variables as xarray Datasets
        """
        derived_variables = {}

        # Calculate VPD if both temperature and dewpoint are available
        temp_var = self._get_actual_variable_name('2m_temperature', unified_dataset)
        dewpoint_var = self._get_actual_variable_name('2m_dewpoint_temperature', unified_dataset)

        if (temp_var and dewpoint_var):

            self.logger.info("Calculating Vapor Pressure Deficit (VPD)")

            try:
                # Extract temperature and dewpoint data from unified dataset
                temp_data = unified_dataset[[temp_var]]
                dewpoint_data = unified_dataset[[dewpoint_var]]

                # Calculate VPD
                vpd_data = self._calculate_vpd_from_datasets(temp_data, dewpoint_data)
                derived_variables['VPD'] = vpd_data

                self.logger.info("VPD calculation completed")

            except Exception as e:
                self.logger.error(f"Failed to calculate VPD: {e}")
        else:
            self.logger.warning("Cannot calculate VPD: missing temperature or dewpoint data")

        return derived_variables


    def _derive_temperature_extremes(self, temp_dataset: xr.Dataset) -> tuple:
        """
        Derive monthly minimum and maximum temperatures.

        Args:
            temp_dataset: Temperature dataset

        Returns:
            tuple: (TMIN dataset, TMAX dataset)
        """
        temp_var_name = self._get_actual_variable_name('2m_temperature', temp_dataset)
        if temp_var_name is None:
            available_vars = list(temp_dataset.data_vars.keys())
            raise ValueError(f"Temperature variable not found. Available: {available_vars}")

        temp_var = temp_dataset[temp_var_name]
        time_coord = self._get_actual_coordinate_name('time', temp_dataset)

        # Ensure time is in datetime format
        temp_var[time_coord] = xr.conventions.decode_cf_datetime(temp_var[time_coord], units=temp_var[time_coord].units) \
            if not np.issubdtype(temp_var[time_coord].dtype, np.datetime64) else temp_var[time_coord]

        # If sub-monthly data, calculate monthly extremes
        if time_coord in temp_var.dims and len(temp_var[time_coord]) > 31:
            self.logger.info("Calculating monthly temperature extremes from sub-monthly data")

            # Use resample to get proper datetime index for TMIN/TMAX
            temp_min_grouped = temp_var.resample({time_coord: 'MS'}).min()  # 'MS' = month start
            temp_max_grouped = temp_var.resample({time_coord: 'MS'}).max()  # 'MS' = month start
        else:
            # Already monthly or less frequent data
            temp_min_grouped = temp_var
            temp_max_grouped = temp_var

        # Create datasets with proper attributes
        temp_min_ds = xr.Dataset({
            'TMIN': temp_min_grouped.assign_attrs({
                'units': 'K',
                'long_name': 'Monthly Minimum 2m Temperature',
                'description': 'Monthly minimum air temperature at 2m height',
                'source': 'ERA5 2m_temperature'
            })
        })

        temp_max_ds = xr.Dataset({
            'TMAX': temp_max_grouped.assign_attrs({
                'units': 'K',
                'long_name': 'Monthly Maximum 2m Temperature',
                'description': 'Monthly maximum air temperature at 2m height',
                'source': 'ERA5 2m_temperature'
            })
        })

        return temp_min_ds, temp_max_ds

    def _calculate_vpd_from_datasets(self, temp_dataset: xr.Dataset, dewpoint_dataset: xr.Dataset) -> xr.Dataset:
        """
        Calculate VPD from temperature and dewpoint datasets.

        Args:
            temp_dataset: Temperature dataset
            dewpoint_dataset: Dewpoint temperature dataset

        Returns:
            xr.Dataset: VPD dataset
        """
        # Get variable names using mapping system
        temp_var_name = self._get_actual_variable_name('2m_temperature', temp_dataset)
        dewpoint_var_name = self._get_actual_variable_name('2m_dewpoint_temperature', dewpoint_dataset)

        if temp_var_name is None or dewpoint_var_name is None:
            raise ValueError("Temperature or dewpoint variable not found")

        temp_var = temp_dataset[temp_var_name]
        dewpoint_var = dewpoint_dataset[dewpoint_var_name]

        # Align datasets spatially and temporally
        dewpoint_aligned = dewpoint_var.interp_like(temp_var, method='nearest')

        # Convert to Celsius for VPD calculation
        temp_celsius = temp_var - 273.15
        dewpoint_celsius = dewpoint_aligned - 273.15

        # For VPD, use maximum temperature if we have sub-monthly data
        if 'time' in temp_celsius.dims and len(temp_celsius.time) > 31:
            self.logger.info("Resampling temperature to monthly max and dewpoint to monthly mean for VPD calculation.")
            # Use resample to get a proper time series of monthly values
            temp_max_celsius = temp_celsius.resample(time='MS').max()
            dewpoint_avg_celsius = dewpoint_celsius.resample(time='MS').mean()
        else:
            temp_max_celsius = temp_celsius
            dewpoint_avg_celsius = dewpoint_celsius

        # Calculate VPD using MATLAB-equivalent function
        vpd_hpa = calculate_vapor_pressure_deficit_matlab(
            temp_max_celsius.values,
            dewpoint_avg_celsius.values
        )

        # Create VPD DataArray
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

        return xr.Dataset({'VPD': vpd_da})

    def _apply_unit_conversion(self, dataset: xr.Dataset, cbf_variable: str) -> xr.Dataset:
        """
        Apply unit conversions for CBF compatibility.

        Args:
            dataset: Input dataset
            cbf_variable: CBF variable name

        Returns:
            xr.Dataset: Dataset with converted units
        """
        if cbf_variable not in self.unit_conversions:
            return dataset

        conversion = self.unit_conversions[cbf_variable]

        # Get the main data variable
        data_vars = list(dataset.data_vars.keys())
        if not data_vars:
            return dataset

        main_var = dataset[data_vars[0]]

        if conversion.get('method') == 'radiation_monthly':
            # Convert radiation from J/m² to W/m² (monthly average)
            seconds_per_month = 30.44 * 24 * 3600  # Average seconds per month
            converted_data = main_var / seconds_per_month
            unit_str = conversion['to']
        else:
            # Simple multiplication factor
            factor = conversion['factor']
            converted_data = main_var * factor
            unit_str = conversion['to']

        # Update attributes
        converted_data.attrs.update(main_var.attrs)
        converted_data.attrs['units'] = unit_str
        converted_data.attrs['unit_conversion'] = f"Converted from {conversion['from']} to {conversion['to']}"

        # Rename variable to CBF name
        converted_data.name = cbf_variable

        return xr.Dataset({cbf_variable: converted_data})

    def _integrate_external_data(self, co2_data_dir: str, fire_data_dir: str,
                                processed_variables: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """
        Integrate external data sources (CO2, fire) with processed variables.

        Args:
            co2_data_dir: Directory containing CO2 data
            fire_data_dir: Directory containing fire data
            processed_variables: Already processed ERA5 variables

        Returns:
            dict: External variables as xarray Datasets
        """
        external_variables = {}

        # Get reference coordinates from processed variables
        reference_coords = None
        for var_name, dataset in processed_variables.items():
            if dataset and len(dataset.data_vars) > 0:
                reference_coords = dataset
                break

        # Handle CO2 data
        if co2_data_dir and os.path.exists(co2_data_dir):
            try:
                co2_dataset = self._load_co2_data(co2_data_dir, reference_coords)
                external_variables['CO2_2'] = co2_dataset
                self.logger.info("CO2 data integration completed")
            except Exception as e:
                self.logger.warning(f"Failed to integrate CO2 data: {e}")
                # Use constant CO2 value as fallback
                external_variables['CO2_2'] = self._create_constant_co2_dataset(reference_coords)
        else:
            self.logger.info("No CO2 data directory provided, using constant value")
            external_variables['CO2_2'] = self._create_constant_co2_dataset(reference_coords)

        # Handle fire data
        if fire_data_dir and os.path.exists(fire_data_dir):
            try:
                fire_dataset = self._load_fire_data(fire_data_dir, reference_coords)
                external_variables['BURN_2'] = fire_dataset
                self.logger.info("Fire data integration completed")
            except Exception as e:
                self.logger.warning(f"Failed to integrate fire data: {e}")
                # Use zeros as fallback
                external_variables['BURN_2'] = self._create_zero_burned_area_dataset(reference_coords)
        else:
            self.logger.info("No fire data directory provided, using zeros")
            external_variables['BURN_2'] = self._create_zero_burned_area_dataset(reference_coords)

        return external_variables

    def _create_constant_co2_dataset(self, reference_coords: xr.Dataset) -> xr.Dataset:
        """Create constant CO2 dataset based on reference coordinates."""
        if reference_coords is None:
            raise ValueError("No reference coordinates available for CO2 dataset")

        # Use reference variable to get coordinate structure
        ref_var = list(reference_coords.data_vars.values())[0]

        # Create constant CO2 value (415 ppm - typical recent value)
        co2_constant = 415.0

        co2_data = xr.full_like(ref_var, co2_constant)
        co2_data.attrs = {
            'units': 'ppm',
            'long_name': 'Atmospheric CO2 Concentration',
            'description': 'Constant atmospheric CO2 concentration',
            'source': 'Constant value (fallback)',
            'value': co2_constant
        }
        co2_data.name = 'CO2_2'

        return xr.Dataset({'CO2_2': co2_data})

    def _create_zero_burned_area_dataset(self, reference_coords: xr.Dataset) -> xr.Dataset:
        """Create zero burned area dataset based on reference coordinates."""
        if reference_coords is None:
            raise ValueError("No reference coordinates available for burned area dataset")

        # Use reference variable to get coordinate structure
        ref_var = list(reference_coords.data_vars.values())[0]

        # Create zero burned area
        burned_area_data = xr.zeros_like(ref_var)
        burned_area_data.attrs = {
            'units': 'fraction',
            'long_name': 'Burned Area Fraction',
            'description': 'Fraction of grid cell burned by fires (zero fallback)',
            'source': 'Zero values (fallback)'
        }
        burned_area_data.name = 'BURN_2'

        return xr.Dataset({'BURN_2': burned_area_data})

    def _load_co2_data(self, co2_data_dir: str, reference_coords: xr.Dataset) -> xr.Dataset:
        """Load CO2 data from directory and interpolate to reference grid."""
        co2_files = []
        for pattern in ['*co2*.nc', '*CO2*.nc', '*noaa*.nc', '*NOAA*.nc']:
            co2_files.extend(glob.glob(os.path.join(co2_data_dir, pattern)))

        if not co2_files:
            raise FileNotFoundError(f"No CO2 data files found in {co2_data_dir}")

        # Load CO2 data
        if len(co2_files) == 1:
            co2_ds = xr.open_dataset(co2_files[0])
        else:
            datasets = [xr.open_dataset(f) for f in co2_files]
            co2_ds = xr.concat(datasets, dim='time')

        # Find CO2 variable
        co2_var_names = ['co2', 'CO2', 'co2_concentration', 'mole_fraction', 'co2_mole_fraction']
        co2_var = None
        for var_name in co2_var_names:
            if var_name in co2_ds.data_vars:
                co2_var = co2_ds[var_name]
                break

        if co2_var is None:
            raise ValueError(f"CO2 variable not found in {co2_files}")

        # Interpolate to target grid
        target_coords = self._create_target_grid_coordinates()
        co2_interp = co2_var.interp(
            latitude=target_coords['latitude'],
            longitude=target_coords['longitude'],
            method='linear'
        )

        co2_interp.attrs.update({
            'units': 'ppm',
            'long_name': 'Atmospheric CO2 Concentration',
            'description': 'Atmospheric carbon dioxide concentration from NOAA measurements',
            'source': 'NOAA GML'
        })
        co2_interp.name = 'CO2_2'

        return xr.Dataset({'CO2_2': co2_interp})

    def _load_fire_data(self, fire_data_dir: str, reference_coords: xr.Dataset) -> xr.Dataset:
        """Load fire data from directory and interpolate to reference grid."""
        fire_files = []
        for pattern in ['*gfed*.nc', '*GFED*.nc', '*fire*.nc', '*FIRE*.nc', '*burned*.nc']:
            fire_files.extend(glob.glob(os.path.join(fire_data_dir, pattern)))

        if not fire_files:
            raise FileNotFoundError(f"No fire data files found in {fire_data_dir}")

        # Load fire data
        if len(fire_files) == 1:
            fire_ds = xr.open_dataset(fire_files[0])
        else:
            datasets = [xr.open_dataset(f) for f in fire_files]
            fire_ds = xr.concat(datasets, dim='time')

        # Find burned area variable
        fire_var_names = ['burned_area', 'BURNED_AREA', 'burned_fraction', 'ba']
        fire_var = None
        for var_name in fire_var_names:
            if var_name in fire_ds.data_vars:
                fire_var = fire_ds[var_name]
                break

        if fire_var is None:
            raise ValueError(f"Burned area variable not found in {fire_files}")

        # Interpolate to target grid
        target_coords = self._create_target_grid_coordinates()
        fire_interp = fire_var.interp(
            latitude=target_coords['latitude'],
            longitude=target_coords['longitude'],
            method='nearest'
        )

        fire_interp.attrs.update({
            'units': 'fraction',
            'long_name': 'Burned Area Fraction',
            'description': 'Fraction of grid cell burned by fires',
            'source': 'GFED4'
        })
        fire_interp.name = 'BURN_2'

        return xr.Dataset({'BURN_2': fire_interp})

    def _apply_land_masking(self, variables: Dict[str, xr.Dataset],
                          land_fraction_file: str, land_threshold: float) -> Dict[str, xr.Dataset]:
        """
        Apply land fraction masking to all variables.

        Args:
            variables: Dictionary of variable datasets
            land_fraction_file: Path to land fraction file
            land_threshold: Land fraction threshold

        Returns:
            dict: Masked variable datasets
        """
        self.logger.info(f"Applying land masking with threshold {land_threshold}")

        # Load land fraction data
        land_ds = xr.open_dataset(land_fraction_file)

        # Find land fraction variable
        land_var_names = ['data', 'land_fraction', 'land_sea_frac', 'fraction']
        land_var = None
        for var_name in land_var_names:
            if var_name in land_ds.data_vars:
                land_var = land_ds[var_name]
                break

        if land_var is None:
            raise ValueError(f"Land fraction variable not found in {land_fraction_file}")

        # Create land mask and interpolate to target grid
        land_mask = land_var > land_threshold
        target_coords = self._create_target_grid_coordinates()
        land_mask_on_target_grid = land_mask.interp(
            latitude=target_coords['latitude'],
            longitude=target_coords['longitude'],
            method='nearest'
        )

        # Apply mask to all variables
        masked_variables = {}
        for var_name, dataset in variables.items():
            try:
                masked_ds = dataset.copy()

                for data_var_name in dataset.data_vars:
                    data_var = dataset[data_var_name]

                    # Use pre-interpolated land mask on target grid
                    land_mask_interp = land_mask_on_target_grid

                    # Apply mask
                    if 'time' in data_var.dims:
                        mask_broadcast = land_mask_interp.broadcast_like(data_var)
                        masked_var = data_var.where(mask_broadcast)
                    else:
                        masked_var = data_var.where(land_mask_interp)

                    # Update attributes
                    masked_var.attrs.update(data_var.attrs)
                    masked_var.attrs['land_masking'] = f'Applied with threshold {land_threshold}'

                    masked_ds[data_var_name] = masked_var

                masked_variables[var_name] = masked_ds

            except Exception as e:
                self.logger.warning(f"Failed to apply land mask to {var_name}: {e}")
                masked_variables[var_name] = dataset

        land_ds.close()
        return masked_variables

    def _validate_output_grid(self, dataset: xr.Dataset) -> None:
        """
        Validate that output grid matches target specification.

        Checks that the output dataset has the expected spatial dimensions
        and grid resolution matching the target configuration.

        Args:
            dataset: Output dataset to validate

        Raises:
            Warning logs if grid doesn't match specification
        """
        # Get actual grid dimensions
        actual_lat_size = len(dataset.latitude)
        actual_lon_size = len(dataset.longitude)

        # Calculate expected grid dimensions
        target_coords = self._create_target_grid_coordinates()
        expected_lat_size = len(target_coords['latitude'])
        expected_lon_size = len(target_coords['longitude'])

        # Check dimensions
        if actual_lat_size != expected_lat_size or actual_lon_size != expected_lon_size:
            self.logger.error(
                f"Output grid dimensions mismatch! "
                f"Expected: {expected_lat_size}×{expected_lon_size}, "
                f"Actual: {actual_lat_size}×{actual_lon_size}"
            )
        else:
            self.logger.info(
                f"Output grid validation PASSED: {actual_lat_size}×{actual_lon_size} "
                f"at {self.target_grid_resolution}° resolution"
            )

        # Calculate actual grid resolution
        if len(dataset.latitude) > 1:
            actual_lat_res = abs(dataset.latitude.values[1] - dataset.latitude.values[0])
            actual_lon_res = abs(dataset.longitude.values[1] - dataset.longitude.values[0])

            # Check resolution (allow small floating point differences)
            lat_res_match = abs(actual_lat_res - self.target_grid_resolution) < 0.01
            lon_res_match = abs(actual_lon_res - self.target_grid_resolution) < 0.01

            if not (lat_res_match and lon_res_match):
                self.logger.warning(
                    f"Grid resolution mismatch! "
                    f"Expected: {self.target_grid_resolution}°, "
                    f"Actual: lat={actual_lat_res:.4f}°, lon={actual_lon_res:.4f}°"
                )

        # Check spatial bounds
        actual_lat_min = float(dataset.latitude.min())
        actual_lat_max = float(dataset.latitude.max())
        actual_lon_min = float(dataset.longitude.min())
        actual_lon_max = float(dataset.longitude.max())

        expected_lat_min, expected_lat_max = self.target_lat_range
        expected_lon_min, expected_lon_max = self.target_lon_range

        bounds_match = (
            abs(actual_lat_min - expected_lat_min) < 0.01 and
            abs(actual_lat_max - expected_lat_max) < 0.01 and
            abs(actual_lon_min - expected_lon_min) < 0.01 and
            abs(actual_lon_max - expected_lon_max) < 0.01
        )

        if not bounds_match:
            self.logger.warning(
                f"Spatial bounds mismatch! "
                f"Expected lat: [{expected_lat_min}, {expected_lat_max}], "
                f"lon: [{expected_lon_min}, {expected_lon_max}]. "
                f"Actual lat: [{actual_lat_min}, {actual_lat_max}], "
                f"lon: [{actual_lon_min}, {actual_lon_max}]"
            )

    def _create_unified_cbf_file(self, variables: Dict[str, xr.Dataset],
                               output_filepath: Path) -> str:
        """
        Create unified CBF meteorological driver file.

        Args:
            variables: Dictionary of processed variable datasets
            output_filepath: Path for output file

        Returns:
            str: Path to created CBF file
        """
        self.logger.info(f"Creating unified CBF file: {output_filepath.name}")

        # Get target grid coordinates
        target_coords = self._create_target_grid_coordinates()
        self.logger.info(
            f"Regridding all variables to target grid: "
            f"{len(target_coords['latitude'])}×{len(target_coords['longitude'])} "
            f"at {self.target_grid_resolution}° resolution"
        )

        # Start with the first dataset, regridded to target
        first_var = list(variables.keys())[0]
        first_dataset = variables[first_var]

        # Regrid first variable to target grid
        first_var_name = list(first_dataset.data_vars.keys())[0]
        first_data_var = first_dataset[first_var_name]
        first_regridded = first_data_var.interp(
            latitude=target_coords['latitude'],
            longitude=target_coords['longitude'],
            method='nearest'
        )
        combined_ds = xr.Dataset({first_var_name: first_regridded})

        # Add all other variables, regridded to target grid
        for var_name, dataset in variables.items():
            if var_name == first_var:
                continue

            for data_var_name in dataset.data_vars:
                data_var = dataset[data_var_name]

                # Interpolate to target grid coordinates
                data_interp = data_var.interp(
                    latitude=target_coords['latitude'],
                    longitude=target_coords['longitude'],
                    method='nearest'
                )
                combined_ds[data_var_name] = data_interp

        # Add required CBF framework variables (DISTURBANCE_FLUX and YIELD)
        self._add_cbf_framework_variables(combined_ds)

        # Check for required CBF variables
        present_vars = []
        missing_vars = []
        for var in self.cbf_met_variables:
            if var in combined_ds.data_vars:
                present_vars.append(var)
            else:
                missing_vars.append(var)

        self.logger.info(f"Present CBF variables ({len(present_vars)}/{len(self.cbf_met_variables)}): {present_vars}")
        if missing_vars:
            self.logger.warning(f"Missing CBF variables: {missing_vars}")

        # Calculate actual grid resolution from output coordinates
        actual_lat_res = abs(combined_ds.latitude.values[1] - combined_ds.latitude.values[0])
        actual_lon_res = abs(combined_ds.longitude.values[1] - combined_ds.longitude.values[0])
        grid_resolution_str = f"{actual_lat_res:.2f}° lat × {actual_lon_res:.2f}° lon"

        # Set global attributes for CBF compatibility
        combined_ds.attrs.update({
            'title': 'CARDAMOM Meteorological Drivers',
            'description': 'Unified meteorological driver file for CBF input generation',
            'source': 'ERA5 reanalysis with external CO2 and fire data',
            'target_grid_resolution': f'{self.target_grid_resolution} degrees',
            'actual_grid_resolution': grid_resolution_str,
            'latitude_range': f'{self.target_lat_range[0]} to {self.target_lat_range[1]}',
            'longitude_range': f'{self.target_lon_range[0]} to {self.target_lon_range[1]}',
            'created_by': 'CBF Meteorological Processor',
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cbf_compatible': 'true',
            'variables_included': ', '.join(present_vars)
        })

        # Set encoding for consistent NetCDF format matching CARDAMOM expectations
        encoding = {}
        for var_name in combined_ds.data_vars:
            encoding[var_name] = {
                'zlib': True,
                'complevel': 6,
                'dtype': 'float64',  # Match sample file (was float32)
                '_FillValue': np.nan  # Match sample file (was -9999.0)
            }

        # Add time coordinate encoding to match CARDAMOM convention
        if 'time' in combined_ds.coords:
            encoding['time'] = {
                'units': 'days since 2001-01-01',
                'calendar': 'proleptic_gregorian',
                'dtype': 'int64'
            }
            self.logger.info("Set time encoding: days since 2001-01-01")

        # Validate output grid before saving
        self._validate_output_grid(combined_ds)

        # Save unified CBF file
        combined_ds.to_netcdf(output_filepath, encoding=encoding)
        combined_ds.close()

        self.logger.info(f"Unified CBF file created: {output_filepath}")
        return str(output_filepath)

    def _add_cbf_framework_variables(self, combined_ds: xr.Dataset) -> None:
        """
        Add required CBF framework variables (DISTURBANCE_FLUX and YIELD) as zeros.

        These variables are required by erens_cbf_code.py but are typically zero
        for meteorological processing. They are added to ensure compatibility.

        Args:
            combined_ds: Combined dataset to add variables to
        """
        self.logger.info("Adding CBF framework variables: DISTURBANCE_FLUX, YIELD")

        # Get reference variable for coordinate structure
        ref_var = list(combined_ds.data_vars.values())[0]

        # Add DISTURBANCE_FLUX (zeros)
        disturbance_flux = xr.zeros_like(ref_var)
        disturbance_flux.attrs = {
            'units': 'gC/m2/day',
            'long_name': 'Disturbance Flux',
            'description': 'Carbon flux from disturbances (zeros for meteorological processing)',
            'source': 'Zero values (required by CBF framework)'
        }
        disturbance_flux.name = 'DISTURBANCE_FLUX'
        combined_ds['DISTURBANCE_FLUX'] = disturbance_flux

        # Add YIELD (zeros)
        yield_flux = xr.zeros_like(ref_var)
        yield_flux.attrs = {
            'units': 'gC/m2/day',
            'long_name': 'Yield Flux',
            'description': 'Carbon flux from harvesting/yields (zeros for meteorological processing)',
            'source': 'Zero values (required by CBF framework)'
        }
        yield_flux.name = 'YIELD'
        combined_ds['YIELD'] = yield_flux

        self.logger.info("CBF framework variables added successfully")

    def _validate_cbf_file(self, cbf_filepath: str) -> bool:
        """
        Validate CBF file for compatibility with erens_cbf_code.py.

        Args:
            cbf_filepath: Path to CBF file

        Returns:
            bool: True if validation passes
        """
        self.logger.info(f"Validating CBF file: {os.path.basename(cbf_filepath)}")

        try:
            cbf_ds = xr.open_dataset(cbf_filepath)

            # Check for required variables
            present_vars = []
            missing_vars = []
            for var in self.cbf_met_variables:
                if var in cbf_ds.data_vars:
                    present_vars.append(var)
                else:
                    missing_vars.append(var)

            # Check spatial dimensions
            has_lat = 'latitude' in cbf_ds.dims or 'lat' in cbf_ds.dims
            has_lon = 'longitude' in cbf_ds.dims or 'lon' in cbf_ds.dims
            has_time = 'time' in cbf_ds.dims

            # Log validation results
            self.logger.info(f"Present variables ({len(present_vars)}/{len(self.cbf_met_variables)}): {present_vars}")
            if missing_vars:
                self.logger.warning(f"Missing variables: {missing_vars}")

            self.logger.info(f"Spatial dims - Lat: {has_lat}, Lon: {has_lon}, Time: {has_time}")

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
                len(present_vars) >= 8 and  # At least 8/10 variables
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

    def get_supported_variables(self) -> List[str]:
        """
        Get list of CBF meteorological variables that can be processed.

        Returns:
            list: Supported CBF variable names
        """
        return self.cbf_met_variables.copy()

    def get_era5_requirements(self) -> List[str]:
        """
        Get list of required ERA5 variables for full CBF processing.

        Returns:
            list: Required ERA5 variable names
        """
        required_era5_vars = list(self.era5_to_cbf_mapping.keys())
        required_era5_vars.append('2m_dewpoint_temperature')  # For VPD calculation
        return required_era5_vars