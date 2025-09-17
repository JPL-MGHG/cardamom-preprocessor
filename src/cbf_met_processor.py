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

    def __init__(self, output_dir: str = "."):
        """
        Initialize CBF meteorological processor.

        Args:
            output_dir: Directory for processed CBF files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)

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
            # Step 1: Discover and categorize input files
            file_catalog = self._catalog_input_files(input_path)
            self.logger.info(f"Found {len(file_catalog)} ERA5 variable files")

            # Step 2: Process core ERA5 variables
            processed_variables = self._process_era5_variables(file_catalog)

            # Step 3: Calculate derived variables (VPD from temperature/dewpoint)
            derived_variables = self._calculate_derived_variables(file_catalog)
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

            self.logger.info(f"CBF meteorological processing completed: {cbf_filepath}")
            return str(cbf_filepath)

        except Exception as e:
            error_msg = f"Failed to process downloaded files to CBF format: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _catalog_input_files(self, input_dir: Path) -> Dict[str, List[str]]:
        """
        Catalog input NetCDF files by ERA5 variable type.

        Args:
            input_dir: Directory containing ERA5 NetCDF files

        Returns:
            dict: Mapping of ERA5 variable names to file paths
        """
        file_catalog = {}

        # Search for NetCDF files
        nc_files = list(input_dir.glob("*.nc"))
        self.logger.info(f"Found {len(nc_files)} NetCDF files in {input_dir}")

        # Categorize files by variable type
        for era5_var in self.era5_to_cbf_mapping.keys():
            matching_files = []

            # Match files containing the variable name
            for nc_file in nc_files:
                filename = nc_file.name.lower()
                var_name_normalized = era5_var.replace('_', '').replace('-', '')

                if var_name_normalized in filename.replace('_', '').replace('-', ''):
                    matching_files.append(str(nc_file))

            if matching_files:
                file_catalog[era5_var] = sorted(matching_files)
                self.logger.info(f"Found {len(matching_files)} files for {era5_var}")
            else:
                self.logger.warning(f"No files found for ERA5 variable: {era5_var}")

        # Also look for dewpoint temperature files for VPD calculation
        dewpoint_files = []
        for nc_file in nc_files:
            filename = nc_file.name.lower()
            if 'dewpoint' in filename or 'd2m' in filename:
                dewpoint_files.append(str(nc_file))

        if dewpoint_files:
            file_catalog['2m_dewpoint_temperature'] = sorted(dewpoint_files)
            self.logger.info(f"Found {len(dewpoint_files)} dewpoint files for VPD calculation")

        return file_catalog

    def _process_era5_variables(self, file_catalog: Dict[str, List[str]]) -> Dict[str, xr.Dataset]:
        """
        Process core ERA5 variables into CBF format.

        Args:
            file_catalog: Mapping of ERA5 variables to file paths

        Returns:
            dict: Processed variables as xarray Datasets
        """
        processed_variables = {}

        for era5_var, cbf_var in self.era5_to_cbf_mapping.items():
            if era5_var not in file_catalog:
                self.logger.warning(f"No files found for {era5_var}, skipping")
                continue

            self.logger.info(f"Processing {era5_var} -> {cbf_var}")

            try:
                # Load and combine files for this variable
                variable_files = file_catalog[era5_var]
                combined_data = self._load_and_combine_files(variable_files, era5_var)

                # Handle special cases
                if era5_var == '2m_temperature':
                    # Derive TMIN and TMAX from temperature data
                    temp_min, temp_max = self._derive_temperature_extremes(combined_data)
                    processed_variables['TMIN'] = temp_min
                    processed_variables['TMAX'] = temp_max
                else:
                    # Apply unit conversion if needed
                    converted_data = self._apply_unit_conversion(combined_data, cbf_var)
                    processed_variables[cbf_var] = converted_data

            except Exception as e:
                self.logger.error(f"Failed to process {era5_var}: {e}")
                continue

        return processed_variables

    def _calculate_derived_variables(self, file_catalog: Dict[str, List[str]]) -> Dict[str, xr.Dataset]:
        """
        Calculate derived variables like VPD from temperature and dewpoint.

        Args:
            file_catalog: Mapping of ERA5 variables to file paths

        Returns:
            dict: Derived variables as xarray Datasets
        """
        derived_variables = {}

        # Calculate VPD if both temperature and dewpoint are available
        if ('2m_temperature' in file_catalog and
            '2m_dewpoint_temperature' in file_catalog):

            self.logger.info("Calculating Vapor Pressure Deficit (VPD)")

            try:
                # Load temperature data
                temp_files = file_catalog['2m_temperature']
                temp_data = self._load_and_combine_files(temp_files, '2m_temperature')

                # Load dewpoint data
                dewpoint_files = file_catalog['2m_dewpoint_temperature']
                dewpoint_data = self._load_and_combine_files(dewpoint_files, '2m_dewpoint_temperature')

                # Calculate VPD
                vpd_data = self._calculate_vpd_from_datasets(temp_data, dewpoint_data)
                derived_variables['VPD'] = vpd_data

                self.logger.info("VPD calculation completed")

            except Exception as e:
                self.logger.error(f"Failed to calculate VPD: {e}")
        else:
            self.logger.warning("Cannot calculate VPD: missing temperature or dewpoint data")

        return derived_variables

    def _load_and_combine_files(self, file_paths: List[str], variable_name: str) -> xr.Dataset:
        """
        Load and combine multiple NetCDF files for a variable.

        Args:
            file_paths: List of NetCDF file paths
            variable_name: ERA5 variable name

        Returns:
            xr.Dataset: Combined dataset
        """
        datasets = []

        for file_path in file_paths:
            try:
                ds = xr.open_dataset(file_path)
                datasets.append(ds)
                self.logger.debug(f"Loaded {os.path.basename(file_path)}")
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")

        if not datasets:
            raise ValueError(f"No valid datasets loaded for {variable_name}")

        # Combine along time dimension if multiple files
        if len(datasets) == 1:
            combined = datasets[0]
        else:
            # Sort by time if time dimension exists
            try:
                datasets.sort(key=lambda ds: ds.time.values[0] if 'time' in ds.dims else 0)
                combined = xr.concat(datasets, dim='time')
            except Exception as e:
                self.logger.warning(f"Failed to concatenate along time: {e}, using merge instead")
                combined = xr.merge(datasets)

        return combined

    def _derive_temperature_extremes(self, temp_dataset: xr.Dataset) -> tuple:
        """
        Derive monthly minimum and maximum temperatures.

        Args:
            temp_dataset: Temperature dataset

        Returns:
            tuple: (TMIN dataset, TMAX dataset)
        """
        # Get temperature variable (ERA5 uses standard names)
        temp_var_names = ['2m_temperature', 't2m', 'temperature']
        temp_var = None

        for var_name in temp_var_names:
            if var_name in temp_dataset.data_vars:
                temp_var = temp_dataset[var_name]
                break

        if temp_var is None:
            available_vars = list(temp_dataset.data_vars.keys())
            raise ValueError(f"Temperature variable not found. Available: {available_vars}")

        # Calculate monthly extremes if we have sub-monthly data
        if 'time' in temp_var.dims and len(temp_var.time) > 31:
            self.logger.info("Calculating monthly temperature extremes from sub-monthly data")
            temp_min_monthly = temp_var.groupby('time.month').min(dim='time')
            temp_max_monthly = temp_var.groupby('time.month').max(dim='time')
        else:
            # Already monthly data - use as is for both min and max
            self.logger.info("Using monthly temperature data as-is")
            temp_min_monthly = temp_var
            temp_max_monthly = temp_var

        # Create datasets with proper attributes
        temp_min_ds = xr.Dataset({
            'TMIN': temp_min_monthly.assign_attrs({
                'units': 'K',
                'long_name': 'Monthly Minimum 2m Temperature',
                'description': 'Monthly minimum air temperature at 2m height',
                'source': 'ERA5 2m_temperature'
            })
        })

        temp_max_ds = xr.Dataset({
            'TMAX': temp_max_monthly.assign_attrs({
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
        # Get variable names
        temp_var_names = ['2m_temperature', 't2m', 'temperature']
        dewpoint_var_names = ['2m_dewpoint_temperature', 'd2m', 'dewpoint_temperature']

        temp_var = None
        for var_name in temp_var_names:
            if var_name in temp_dataset.data_vars:
                temp_var = temp_dataset[var_name]
                break

        dewpoint_var = None
        for var_name in dewpoint_var_names:
            if var_name in dewpoint_dataset.data_vars:
                dewpoint_var = dewpoint_dataset[var_name]
                break

        if temp_var is None or dewpoint_var is None:
            raise ValueError("Temperature or dewpoint variable not found")

        # Align datasets spatially and temporally
        dewpoint_aligned = dewpoint_var.interp_like(temp_var, method='nearest')

        # Convert to Celsius for VPD calculation
        temp_celsius = temp_var - 273.15
        dewpoint_celsius = dewpoint_aligned - 273.15

        # For VPD, use maximum temperature if we have sub-monthly data
        if 'time' in temp_celsius.dims and len(temp_celsius.time) > 31:
            temp_max_celsius = temp_celsius.groupby('time.month').max(dim='time')
            dewpoint_avg_celsius = dewpoint_celsius.groupby('time.month').mean(dim='time')
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
        co2_var_names = ['co2', 'CO2', 'co2_concentration', 'mole_fraction']
        co2_var = None
        for var_name in co2_var_names:
            if var_name in co2_ds.data_vars:
                co2_var = co2_ds[var_name]
                break

        if co2_var is None:
            raise ValueError(f"CO2 variable not found in {co2_files}")

        # Interpolate to reference grid
        ref_var = list(reference_coords.data_vars.values())[0]
        co2_interp = co2_var.interp_like(ref_var, method='linear')

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

        # Interpolate to reference grid
        ref_var = list(reference_coords.data_vars.values())[0]
        fire_interp = fire_var.interp_like(ref_var, method='nearest')

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

        # Create land mask
        land_mask = land_var > land_threshold

        # Apply mask to all variables
        masked_variables = {}
        for var_name, dataset in variables.items():
            try:
                masked_ds = dataset.copy()

                for data_var_name in dataset.data_vars:
                    data_var = dataset[data_var_name]

                    # Interpolate land mask to match variable grid
                    land_mask_interp = land_mask.interp_like(data_var, method='nearest')

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

        # Start with the first dataset as base
        first_var = list(variables.keys())[0]
        combined_ds = variables[first_var].copy()

        # Add all other variables
        for var_name, dataset in variables.items():
            if var_name == first_var:
                continue

            for data_var_name in dataset.data_vars:
                data_var = dataset[data_var_name]

                # Interpolate to match combined dataset coordinates
                data_interp = data_var.interp_like(combined_ds, method='nearest')
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

        # Set global attributes for CBF compatibility
        combined_ds.attrs.update({
            'title': 'CARDAMOM Meteorological Drivers',
            'description': 'Unified meteorological driver file for CBF input generation',
            'source': 'ERA5 reanalysis with external CO2 and fire data',
            'grid_resolution': '0.5 degrees',
            'created_by': 'CBF Meteorological Processor',
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cbf_compatible': 'true',
            'variables_included': ', '.join(present_vars)
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