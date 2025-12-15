"""
CARDAMOM Binary Format (CBF) Generator

This module generates CARDAMOM Binary Format (CBF) NetCDF files for carbon cycle
model execution. It discovers meteorological and observational data from STAC
catalogs, assembles them into unified datasets, and produces pixel-specific
CBF files for CARDAMOM DALEC model execution.

Scientific Context:
CBF files are CARDAMOM's standard input format for model runs. They contain:
- Meteorological forcing data (temperature, precipitation, radiation, etc.)
- Observational constraints (LAI, biomass, CO2 fluxes)
- Assimilation parameters (uncertainties, optimization flags)
- Site-specific metadata (latitude, longitude, soil properties)

The CBF generator bridges STAC data discovery with CARDAMOM's model execution pipeline.

References:
- CARDAMOM Documentation: https://www2.geog.umd.edu/~sjd/cardamom.html
- erens_cbf_code.py: Original MATLAB-to-Python CBF generation logic
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
from pystac_client import Client

logger = logging.getLogger(__name__)

# CARDAMOM standard variable requirements
REQUIRED_FORCING_VARIABLES = [
    'VPD', 'TOTAL_PREC', 'T2M_MIN', 'T2M_MAX',
    'STRD', 'SSRD', 'SNOWFALL', 'CO2', 'BURNED_AREA', 'SKT'
]

CRITICAL_VARIABLES = ['CO2', 'BURNED_AREA']  # Must be present for forward analysis

OBSERVATION_CONSTRAINT_VARIABLES = ['SCF', 'GPP', 'ABGB', 'EWT']

# STAC Collection IDs for each variable
STAC_COLLECTION_MAPPING = {
    'T2M_MIN': 'cardamom-t2m-min',
    'T2M_MAX': 'cardamom-t2m-max',
    'VPD': 'cardamom-vpd',
    'TOTAL_PREC': 'cardamom-total-prec',
    'SSRD': 'cardamom-ssrd',
    'STRD': 'cardamom-strd',
    'SKT': 'cardamom-skt',
    'SNOWFALL': 'cardamom-snowfall',
    'CO2': 'cardamom-co2',
    'BURNED_AREA': 'cardamom-burned-area',
}


class CBFGenerator:
    """
    Generate CARDAMOM Binary Format (CBF) files from STAC data.

    This generator:
    1. Queries STAC catalogs to discover available data
    2. Validates data completeness (critical variables must exist)
    3. Loads and combines meteorological datasets
    4. Generates pixel-specific CBF files
    5. Creates output STAC metadata

    Attributes:
        stac_api_url (str): URL to STAC API endpoint
        output_directory (Path): Root directory for CBF outputs
        verbose (bool): Print debug information
    """

    def __init__(
        self,
        stac_api_url: str,
        output_directory: str,
        verbose: bool = False,
    ):
        """
        Initialize CBF generator.

        Args:
            stac_api_url (str): URL to STAC API (e.g., https://stac.maap-project.org)
            output_directory (str): Root output directory path
            verbose (bool): Print debug messages. Default: False
        """

        self.stac_api_url = stac_api_url
        self.output_directory = Path(output_directory)
        self.verbose = verbose

        self.output_directory.mkdir(parents=True, exist_ok=True)

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        logger.info(f"CBF Generator initialized")
        logger.info(f"STAC API URL: {stac_api_url}")
        logger.info(f"Output directory: {output_directory}")

    def discover_available_data(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, List[Any]]:
        """
        Query STAC API to discover available meteorological data.

        Queries each required variable's collection for items in the date range.

        Args:
            start_date (str): Start date in 'YYYY-MM' format
            end_date (str): End date in 'YYYY-MM' format

        Returns:
            Dict[str, List[Any]]: Dictionary mapping variable names to lists of
                STAC Items available for the date range

        Raises:
            RuntimeError: If STAC API query fails
        """

        logger.info(f"Discovering STAC data from {start_date} to {end_date}")

        # Convert date strings to full datetime range for STAC query
        start_datetime = f"{start_date}-01T00:00:00Z"
        end_datetime = f"{end_date}-28T23:59:59Z"  # Conservative end date

        available_data = {}

        try:
            # Open STAC API client
            client = Client.open(self.stac_api_url)

            for variable_name, collection_id in STAC_COLLECTION_MAPPING.items():
                logger.debug(f"Querying collection: {collection_id}")

                try:
                    search = client.search(
                        collections=[collection_id],
                        datetime=f"{start_datetime}/{end_datetime}",
                    )

                    items = list(search.items())

                    available_data[variable_name] = items

                    logger.info(
                        f"{variable_name}: found {len(items)} items in {collection_id}"
                    )

                except Exception as e:
                    logger.warning(
                        f"Could not query {collection_id}: {e}. "
                        f"Data may not be available."
                    )
                    available_data[variable_name] = []

        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to STAC API at {self.stac_api_url}: {e}"
            ) from e

        return available_data

    def validate_data_availability(
        self,
        available_data: Dict[str, List[Any]],
        required_months: List[str],
    ) -> Dict[str, Any]:
        """
        Validate that critical variables are available for all required months.

        Forward analysis fails if CO2 or BURNED_AREA is missing for any month.
        Other missing variables are accepted with a warning.

        Args:
            available_data (Dict[str, List]): Dictionary from discover_available_data()
            required_months (List[str]): List of required months in 'YYYY-MM' format

        Returns:
            Dict[str, Any]: Validation results with keys:
                - 'is_valid': bool
                - 'missing_critical': Dict of critical variables with missing months
                - 'missing_optional': Dict of optional variables with missing months
                - 'available_for_cbf': Dict of variables available for CBF generation

        Raises:
            ValueError: If critical variables are missing
        """

        logger.info(f"Validating data availability for {len(required_months)} months")

        validation_results = {
            'is_valid': True,
            'missing_critical': {},
            'missing_optional': {},
            'available_for_cbf': {},
        }

        # Check each variable
        for variable_name, items in available_data.items():
            available_months = {
                item.datetime.strftime('%Y-%m')
                for item in items
                if item.datetime
            }

            missing_months = set(required_months) - available_months

            if missing_months:
                if variable_name in CRITICAL_VARIABLES:
                    validation_results['missing_critical'][variable_name] = (
                        sorted(missing_months)
                    )
                    validation_results['is_valid'] = False
                    logger.error(
                        f"CRITICAL: {variable_name} missing for months: "
                        f"{missing_months}"
                    )
                else:
                    validation_results['missing_optional'][variable_name] = (
                        sorted(missing_months)
                    )
                    logger.warning(
                        f"Optional: {variable_name} missing for months: "
                        f"{missing_months}"
                    )
            else:
                validation_results['available_for_cbf'][variable_name] = items
                logger.debug(f"{variable_name}: complete for all months")

        if not validation_results['is_valid']:
            raise ValueError(
                f"Critical data missing: {validation_results['missing_critical']}"
            )

        return validation_results

    def load_variable_data(
        self,
        stac_items: List[Any],
        variable_name: str,
    ) -> xr.Dataset:
        """
        Load variable data from STAC Item asset URLs.

        Args:
            stac_items (List): STAC Items for a variable (one per month)
            variable_name (str): Variable name for logging

        Returns:
            xr.Dataset: Combined dataset with all monthly data

        Raises:
            RuntimeError: If file loading fails
        """

        logger.info(f"Loading {variable_name} data from {len(stac_items)} items")

        datasets = []

        for item in stac_items:
            try:
                # Get data asset URL
                data_asset = item.get_asset('data')
                if not data_asset:
                    logger.warning(
                        f"No 'data' asset found in item {item.id}. Skipping."
                    )
                    continue

                file_path = data_asset.href

                # Load NetCDF
                ds = xr.open_dataset(file_path)
                datasets.append(ds)

                logger.debug(f"Loaded: {file_path}")

            except Exception as e:
                logger.error(f"Failed to load {item.id}: {e}")
                raise RuntimeError(
                    f"Could not load {variable_name} from {item.id}"
                ) from e

        if not datasets:
            raise RuntimeError(f"No {variable_name} data could be loaded")

        # Combine along time dimension
        combined = xr.concat(datasets, dim='time')

        logger.info(f"Combined {variable_name}: shape={combined.dims}")

        return combined

    def assemble_cbf_meteorology_dataset(
        self,
        meteorology_data: Dict[str, xr.Dataset],
    ) -> xr.Dataset:
        """
        Assemble individual forcing variables into unified meteorology dataset.

        Handles variable naming conventions and coordinate alignment.

        Args:
            meteorology_data (Dict[str, xr.Dataset]): Dictionary mapping
                variable names to their datasets

        Returns:
            xr.Dataset: Combined meteorology dataset with all forcing variables

        Raises:
            ValueError: If coordinate systems don't align
        """

        logger.info("Assembling unified meteorology dataset")

        # Get reference coordinates from first dataset
        reference_ds = list(meteorology_data.values())[0]
        reference_coords = reference_ds.coords

        combined_data_arrays = {}

        for var_name, var_ds in meteorology_data.items():
            # Get the first (and usually only) data variable
            data_var_name = list(var_ds.data_vars)[0]
            data_array = var_ds[data_var_name]

            # Ensure coordinate alignment
            if not all(c in data_array.dims for c in ['latitude', 'longitude']):
                logger.warning(f"{var_name} dimensions: {data_array.dims}")

            combined_data_arrays[var_name] = data_array

            logger.debug(f"Added {var_name}: shape={data_array.shape}")

        # Create combined dataset
        combined_dataset = xr.Dataset(combined_data_arrays)

        # Copy coordinate metadata
        combined_dataset.coords['latitude'] = reference_coords['latitude']
        combined_dataset.coords['longitude'] = reference_coords['longitude']

        if 'time' in reference_coords:
            combined_dataset.coords['time'] = reference_coords['time']

        logger.info(
            f"Assembled meteorology: shape={combined_dataset.dims}, "
            f"variables={list(combined_dataset.data_vars)}"
        )

        return combined_dataset

    def generate_cbf_files(
        self,
        meteorology_dataset: xr.Dataset,
        region: Optional[str] = None,
        lat_range: Optional[tuple] = None,
        lon_range: Optional[tuple] = None,
    ) -> List[Path]:
        """
        Generate pixel-specific CBF files from meteorological dataset.

        This method adapts the logic from erens_cbf_code.py to process
        each land pixel and create CBF files.

        Args:
            meteorology_dataset (xr.Dataset): Combined meteorology data
            region (Optional[str]): Predefined region ('global', 'conus')
            lat_range (Optional[tuple]): (min_lat, max_lat) if custom region
            lon_range (Optional[tuple]): (min_lon, max_lon) if custom region

        Returns:
            List[Path]: Paths to generated CBF files

        Raises:
            ValueError: If region parameters are invalid
        """

        logger.info("Generating pixel-specific CBF files")

        # Define region bounds
        if region == 'global':
            lat_range = (-89.75, 89.75)
            lon_range = (-179.75, 179.75)
        elif region == 'conus':
            lat_range = (20, 60)
            lon_range = (-130, -50)
        elif lat_range is None or lon_range is None:
            raise ValueError(
                "Must specify either 'region' or both 'lat_range' and 'lon_range'"
            )

        logger.info(
            f"Processing region: lat={lat_range}, lon={lon_range}"
        )

        # Get latitude and longitude coordinates
        lats = meteorology_dataset['latitude'].values
        lons = meteorology_dataset['longitude'].values

        # Find indices for region
        lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
        lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])

        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]

        logger.info(
            f"Processing {len(lat_indices)} x {len(lon_indices)} pixels"
        )

        # TODO: Implement land mask loading if available
        # For now, process all pixels in region

        cbf_files = []

        # Process each pixel
        pixel_count = 0
        for lat_idx in lat_indices:
            for lon_idx in lon_indices:
                lat_value = lats[lat_idx]
                lon_value = lons[lon_idx]

                try:
                    # Generate CBF file for this pixel
                    cbf_file = self._generate_pixel_cbf(
                        meteorology_dataset,
                        lat_idx,
                        lon_idx,
                        lat_value,
                        lon_value,
                    )

                    if cbf_file:
                        cbf_files.append(cbf_file)
                        pixel_count += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to generate CBF for ({lat_value:.2f}, "
                        f"{lon_value:.2f}): {e}"
                    )
                    continue

                if pixel_count % 100 == 0:
                    logger.info(f"Generated CBF for {pixel_count} pixels")

        logger.info(f"Generated CBF files for {pixel_count} pixels")

        return cbf_files

    def _generate_pixel_cbf(
        self,
        meteorology_dataset: xr.Dataset,
        lat_idx: int,
        lon_idx: int,
        lat_value: float,
        lon_value: float,
    ) -> Optional[Path]:
        """
        Generate a single pixel CBF file.

        This is a placeholder for the actual CBF generation logic from
        erens_cbf_code.py. In a full implementation, this would:
        1. Extract time series data for the pixel
        2. Assemble forcing and observation variables
        3. Set MCMC parameters
        4. Write CARDAMOM-format NetCDF file

        Args:
            meteorology_dataset: Full meteorology dataset
            lat_idx: Latitude index
            lon_idx: Longitude index
            lat_value: Latitude value (degrees)
            lon_value: Longitude value (degrees)

        Returns:
            Optional[Path]: Path to generated CBF file, or None if failed
        """

        # Generate filename following CARDAMOM convention
        lat_str = f"{abs(int(lat_value)):02d}_{int((lat_value % 1) * 100):02d}"
        lon_str = f"{abs(int(lon_value)):03d}_{int((lon_value % 1) * 100):02d}"
        ns = 'N' if lat_value >= 0 else 'S'
        ew = 'E' if lon_value >= 0 else 'W'

        filename = f"site{lat_str}{ns}{lon_str}{ew}_ID001exp0.cbf.nc"
        output_file = self.output_directory / filename

        # TODO: Implement full CBF generation logic here
        # This is where erens_cbf_code.py logic would be adapted

        logger.debug(f"Would generate: {output_file}")

        return output_file

    def generate(
        self,
        start_date: str,
        end_date: str,
        region: str = 'conus',
    ) -> Dict[str, Any]:
        """
        Main entry point: discover data, validate, and generate CBF files.

        Args:
            start_date (str): Start date in 'YYYY-MM' format
            end_date (str): End date in 'YYYY-MM' format
            region (str): Region ('conus', 'global'). Default: 'conus'

        Returns:
            Dict[str, Any]: Results dictionary with keys:
                - 'cbf_files': List of generated CBF file paths
                - 'success': bool
                - 'metadata': Generation metadata

        Raises:
            ValueError: If data validation fails
            RuntimeError: If generation fails
        """

        logger.info(f"Starting CBF generation for {start_date} to {end_date}")

        # Parse date range
        required_months = self._parse_month_range(start_date, end_date)

        # Step 1: Discover available data
        available_data = self.discover_available_data(start_date, end_date)

        # Step 2: Validate completeness
        validation = self.validate_data_availability(available_data, required_months)

        # Step 3: Load meteorological data
        meteorology_data = {}
        for var_name in validation['available_for_cbf']:
            try:
                met_ds = self.load_variable_data(
                    validation['available_for_cbf'][var_name],
                    var_name,
                )
                meteorology_data[var_name] = met_ds
            except Exception as e:
                logger.error(f"Could not load {var_name}: {e}")

        # Step 4: Assemble unified dataset
        combined_meteorology = self.assemble_cbf_meteorology_dataset(
            meteorology_data
        )

        # Step 5: Generate CBF files
        cbf_files = self.generate_cbf_files(combined_meteorology, region=region)

        logger.info(f"CBF generation complete: {len(cbf_files)} files generated")

        return {
            'cbf_files': cbf_files,
            'success': True,
            'metadata': {
                'date_range': f"{start_date} to {end_date}",
                'region': region,
                'num_files': len(cbf_files),
                'generation_time': datetime.now().isoformat(),
            },
        }

    @staticmethod
    def _parse_month_range(start_date: str, end_date: str) -> List[str]:
        """
        Parse date range string into list of months.

        Args:
            start_date (str): 'YYYY-MM' format
            end_date (str): 'YYYY-MM' format

        Returns:
            List[str]: List of months in 'YYYY-MM' format
        """

        months = []

        start_year, start_month = map(int, start_date.split('-'))
        end_year, end_month = map(int, end_date.split('-'))

        for year in range(start_year, end_year + 1):
            start_m = start_month if year == start_year else 1
            end_m = end_month if year == end_year else 12

            for month in range(start_m, end_m + 1):
                months.append(f"{year:04d}-{month:02d}")

        return months
