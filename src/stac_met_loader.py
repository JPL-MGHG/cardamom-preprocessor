"""
STAC Meteorology Data Loader for CARDAMOM CBF Generation

This module provides functions to discover, load, and assemble meteorological
data from STAC catalogs into a unified dataset compatible with the CBF generator.

The loader handles:
- STAC API discovery of monthly meteorological files
- Loading individual NetCDF files for each variable
- Validation of met data completeness (fails if incomplete)
- Assembly into single unified dataset matching AllMet structure

Meteorological Data Requirements (CRITICAL):
- ALL variables MUST be present for ALL required months
- Missing met data causes immediate failure (cannot generate valid CBF)

Scientific Context:
Meteorological forcing (temperature, precipitation, radiation) is essential
for valid CARDAMOM model runs. Temperature, precipitation, and radiation
drive photosynthesis, respiration, and carbon flux calculations.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import xarray as xr
import pystac
from pystac_client import Client

logger = logging.getLogger(__name__)

# CARDAMOM standard meteorological variable requirements
REQUIRED_FORCING_VARIABLES = [
    'VPD',
    'TOTAL_PREC',
    'T2M_MIN',
    'T2M_MAX',
    'STRD',
    'SSRD',
    'SNOWFALL',
    'CO2',
    'BURNED_AREA',
    'SKT',
]

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


def _is_local_catalog(source: str) -> bool:
    """
    Determine if STAC source is a local file path or remote URL.

    Args:
        source (str): STAC source path or URL

    Returns:
        bool: True if source is a local file, False if remote URL
    """
    # Check if it's a file:// URL
    if source.startswith('file://'):
        return True

    # Check if it starts with http:// or https://
    if source.startswith('http://') or source.startswith('https://'):
        return False

    # Check if it's a local file path that exists
    path = Path(source)
    if path.exists():
        return True

    # Default to treating as remote URL if uncertain
    return False


def _item_in_date_range(item: pystac.Item, start_datetime: str, end_datetime: str) -> bool:
    """
    Check if a STAC item falls within the specified date range.

    Args:
        item (pystac.Item): STAC item to check
        start_datetime (str): Start datetime in ISO format (e.g., '2020-01-01T00:00:00Z')
        end_datetime (str): End datetime in ISO format (e.g., '2020-12-31T23:59:59Z')

    Returns:
        bool: True if item falls within date range, False otherwise
    """
    from datetime import datetime

    # Parse date range
    start_dt = datetime.fromisoformat(start_datetime.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end_datetime.replace('Z', '+00:00'))

    # Get item datetime
    item_datetime = item.datetime
    if item_datetime is None:
        # If no datetime, check start_datetime and end_datetime properties
        item_start = item.properties.get('start_datetime')
        item_end = item.properties.get('end_datetime')

        if item_start:
            item_datetime = datetime.fromisoformat(item_start.replace('Z', '+00:00'))
        elif item_end:
            item_datetime = datetime.fromisoformat(item_end.replace('Z', '+00:00'))
        else:
            # Cannot determine item date, skip
            return False

    # Check if item datetime falls within range
    return start_dt <= item_datetime <= end_dt


def discover_stac_items(
    stac_source: str,
    start_date: str,
    end_date: str,
    variable_name: str,
) -> List[Any]:
    """
    Query STAC catalog (local or remote) to discover available items for a single variable.

    This function accepts either a local STAC catalog.json file path or a remote STAC API URL.
    Local catalogs are useful for testing and offline workflows, while remote catalogs
    enable cloud-based data discovery.

    Args:
        stac_source (str): Path to local STAC catalog.json file or URL to remote STAC API
            Examples:
                - Local: 'file:///path/to/catalog.json' or '/path/to/catalog.json'
                - Remote: 'https://stac-api.example.com'
        start_date (str): Start date in 'YYYY-MM' format
        end_date (str): End date in 'YYYY-MM' format
        variable_name (str): Variable name (e.g., 'T2M_MIN')

    Returns:
        List[Any]: STAC Items available for the variable in the date range

    Raises:
        RuntimeError: If STAC catalog query fails
        ValueError: If unknown variable name provided
    """

    if variable_name not in STAC_COLLECTION_MAPPING:
        raise ValueError(f"Unknown variable: {variable_name}")

    collection_id = STAC_COLLECTION_MAPPING[variable_name]

    # Convert date strings to full datetime range for STAC query
    start_datetime = f"{start_date}-01T00:00:00Z"
    end_datetime = f"{end_date}-28T23:59:59Z"  # Conservative end date

    # Determine if source is local file or remote URL
    is_local = _is_local_catalog(stac_source)

    try:
        logger.debug(f"Querying STAC collection: {collection_id}")

        if is_local:
            # Load local STAC catalog
            logger.debug(f"Loading local STAC catalog from: {stac_source}")
            catalog = pystac.Catalog.from_file(stac_source)

            # Find collection in catalog
            collection = catalog.get_child(collection_id)
            if collection is None:
                raise ValueError(
                    f"Collection '{collection_id}' not found in local catalog"
                )

            # Get all items from collection and filter by datetime
            items = []
            for item in collection.get_items():
                # Check if item falls within date range
                if _item_in_date_range(item, start_datetime, end_datetime):
                    items.append(item)

        else:
            # Query remote STAC API
            logger.debug(f"Querying remote STAC API: {stac_source}")
            client = Client.open(stac_source)

            search = client.search(
                collections=[collection_id],
                datetime=f"{start_datetime}/{end_datetime}",
            )

            items = list(search.items())

        logger.info(
            f"Discovered {len(items)} items for {variable_name} "
            f"in {collection_id} ({start_date} to {end_date})"
        )

        return items

    except Exception as e:
        raise RuntimeError(
            f"Failed to query STAC collection {collection_id}: {e}"
        ) from e


def load_variable_from_stac_items(
    stac_items: List[Any],
    variable_name: str,
) -> xr.Dataset:
    """
    Load variable data from STAC Item asset URLs and combine along time.

    Scientific Context:
    STAC items represent monthly meteorological snapshots. This function loads
    all months for a single variable and concatenates them into a time series.
    Each NetCDF file contains one time step (1 month) of data.

    Args:
        stac_items (List[Any]): STAC Items for a variable (one per month)
        variable_name (str): Variable name for logging

    Returns:
        xr.Dataset: Combined dataset with all monthly data concatenated along time

    Raises:
        RuntimeError: If file loading fails or all items fail
    """

    logger.info(
        f"Loading {variable_name} data from {len(stac_items)} STAC items"
    )

    datasets = []
    failed_items = []

    for i, item in enumerate(stac_items):
        try:
            # Get data asset URL
            data_asset = item.get_asset('data')
            if not data_asset:
                logger.warning(
                    f"No 'data' asset found in item {item.id}. Skipping."
                )
                failed_items.append(item.id)
                continue

            file_path = data_asset.href

            # Load NetCDF file
            ds = xr.open_dataset(file_path)
            datasets.append(ds)

            logger.debug(
                f"Loaded {variable_name} item {i + 1}/{len(stac_items)}: "
                f"{file_path}"
            )

        except Exception as e:
            logger.warning(f"Failed to load {item.id}: {e}")
            failed_items.append(item.id)
            continue

    # Check if we loaded any data
    if not datasets:
        raise RuntimeError(
            f"Could not load any data for {variable_name}. "
            f"Failed items: {failed_items}"
        )

    # Warn if some items failed
    if failed_items:
        logger.warning(
            f"{variable_name}: Failed to load {len(failed_items)}/{len(stac_items)} items"
        )

    # Combine along time dimension
    combined = xr.concat(datasets, dim='time')

    logger.info(
        f"Combined {variable_name}: {len(datasets)} items, shape={combined.dims}"
    )

    return combined


def validate_meteorology_completeness(
    met_data: Dict[str, xr.Dataset],
    required_months: List[str],
) -> None:
    """
    Validate that all meteorological variables are complete for all months.

    CRITICAL VALIDATION: Fails if any required variable is missing for any month.
    Complete meteorology is essential for valid CARDAMOM carbon cycle simulations.

    Scientific Rationale:
    Missing forcing variables (temperature, precipitation, radiation) prevent
    accurate calculation of photosynthesis, respiration, and carbon fluxes.
    Forward analysis cannot proceed without complete meteorology.

    Args:
        met_data (Dict[str, xr.Dataset]): Dict mapping variable names to datasets
        required_months (List[str]): List of required months in 'YYYY-MM' format

    Raises:
        ValueError: If any variable is missing for any required month
    """

    logger.info(
        f"Validating meteorology completeness for {len(required_months)} months"
    )

    missing_by_variable = {}

    for var_name in REQUIRED_FORCING_VARIABLES:
        if var_name not in met_data or met_data[var_name] is None:
            missing_by_variable[var_name] = required_months
            continue

        # Get available time steps for this variable
        # Extract year-month from time coordinates
        ds = met_data[var_name]
        if 'time' not in ds.dims:
            logger.warning(f"{var_name} has no time dimension")
            missing_by_variable[var_name] = required_months
            continue

        available_times = ds['time'].values
        available_months = set()

        for t in available_times:
            try:
                # Convert numpy datetime64 to string
                time_str = str(t)
                # Extract YYYY-MM
                year_month = time_str[:7]  # e.g., '2020-01'
                available_months.add(year_month)
            except Exception as e:
                logger.warning(f"Could not parse time value {t}: {e}")

        # Check for missing months
        required_set = set(required_months)
        missing_months = required_set - available_months

        if missing_months:
            missing_by_variable[var_name] = sorted(missing_months)

    # Report validation results
    if missing_by_variable:
        error_msg = "CRITICAL: Meteorological data is incomplete. Cannot generate valid CBF.\n"
        for var_name, missing_months in missing_by_variable.items():
            error_msg += (
                f"  {var_name}: Missing {len(missing_months)} months: "
                f"{missing_months[:3]}...\n"
            )
        logger.error(error_msg)
        raise ValueError(
            f"Incomplete meteorology: {len(missing_by_variable)} variables have gaps"
        )

    logger.info("Meteorology validation passed: All variables complete")


def assemble_unified_meteorology_dataset(
    met_data: Dict[str, xr.Dataset],
) -> xr.Dataset:
    """
    Assemble individual meteorological variables into unified dataset.

    This creates a dataset with all meteorological variables sharing
    identical spatial and temporal coordinates, similar to the AllMet structure.

    Args:
        met_data (Dict[str, xr.Dataset]): Dictionary mapping variable names
            to their datasets

    Returns:
        xr.Dataset: Unified meteorology dataset with all variables and coordinates

    Raises:
        ValueError: If coordinate systems don't align across variables
    """

    logger.info("Assembling unified meteorology dataset")

    if not met_data:
        raise ValueError("No meteorological data to assemble")

    # Get reference coordinates from first variable
    reference_ds = list(met_data.values())[0]
    reference_coords = reference_ds.coords

    combined_data_arrays = {}

    # Extract data arrays for each variable
    for var_name, var_ds in met_data.items():
        # Get the first (and usually only) data variable in the file
        data_var_names = list(var_ds.data_vars)
        if not data_var_names:
            logger.warning(f"{var_name} dataset has no data variables")
            continue

        data_var_name = data_var_names[0]
        data_array = var_ds[data_var_name]

        # Verify coordinate alignment
        if not all(c in data_array.dims for c in ['latitude', 'longitude']):
            logger.warning(
                f"{var_name} has non-standard dimensions: {data_array.dims}"
            )

        # Add to combined dataset with standardized variable name
        combined_data_arrays[var_name] = data_array
        logger.debug(f"Added {var_name}: shape={data_array.shape}")

    # Create unified dataset
    combined_dataset = xr.Dataset(combined_data_arrays)

    # Set coordinates to match reference
    combined_dataset.coords['latitude'] = reference_coords['latitude']
    combined_dataset.coords['longitude'] = reference_coords['longitude']

    if 'time' in reference_coords:
        combined_dataset.coords['time'] = reference_coords['time']

    logger.info(
        f"Assembled meteorology dataset: "
        f"shape={combined_dataset.dims}, "
        f"variables={list(combined_dataset.data_vars)}"
    )

    return combined_dataset


def load_met_data_from_stac(
    stac_source: str,
    start_date: str,
    end_date: str,
) -> xr.Dataset:
    """
    Load meteorological data from STAC catalog and assemble into unified dataset.

    This is the main entry point for loading meteorology. It:
    1. Discovers STAC items for all required variables
    2. Loads each variable's monthly NetCDF files
    3. Validates data completeness (FAILS if incomplete)
    4. Assembles all variables into single unified dataset
    5. Returns dataset ready for CBF generation

    Complete Workflow:
    ```
    STAC Catalog
        ↓ discover items for each variable
    List of STAC Items per variable
        ↓ load monthly files
    Dataset per variable (time, lat, lon)
        ↓ validate completeness (FAIL if gaps)
    Validated variable datasets
        ↓ merge into single dataset
    Unified meteorology dataset
        ↓ ready for pixel extraction
    CBF pixel processing
    ```

    Args:
        stac_source (str): Path to local STAC catalog.json file or URL to remote STAC API
            Examples:
                - Local: 'file:///path/to/catalog.json' or '/path/to/catalog.json'
                - Remote: 'https://stac-api.example.com'
        start_date (str): Start date in 'YYYY-MM' format (e.g., '2020-01')
        end_date (str): End date in 'YYYY-MM' format (e.g., '2020-12')

    Returns:
        xr.Dataset: Unified meteorology dataset with structure:
            Dimensions: time, latitude, longitude
            Variables: VPD, TOTAL_PREC, T2M_MIN, T2M_MAX, STRD, SSRD,
                      SNOWFALL, CO2, BURNED_AREA, SKT
            Coordinates: time, latitude, longitude

    Raises:
        ValueError: If meteorological data is incomplete
        RuntimeError: If STAC catalog query fails or file loading fails

    Example:
        ```python
        # Using local STAC catalog
        met_data = load_met_data_from_stac(
            stac_source='file:///path/to/catalog.json',
            start_date='2020-01',
            end_date='2020-12'
        )

        # Using remote STAC API
        met_data = load_met_data_from_stac(
            stac_source='https://stac-api.example.com',
            start_date='2020-01',
            end_date='2020-12'
        )
        ```
    """

    logger.info(
        f"Loading meteorology from STAC catalog: {start_date} to {end_date}"
    )

    # Generate required month list
    required_months = _parse_month_range(start_date, end_date)
    logger.info(f"Required months: {len(required_months)} ({start_date} to {end_date})")

    # Step 1: Discover STAC items for all variables
    available_items = {}

    for var_name in REQUIRED_FORCING_VARIABLES:
        try:
            items = discover_stac_items(
                stac_source=stac_source,
                start_date=start_date,
                end_date=end_date,
                variable_name=var_name,
            )
            available_items[var_name] = items
        except Exception as e:
            logger.error(f"Failed to discover {var_name}: {e}")
            available_items[var_name] = []

    # Step 2: Load each variable
    met_data = {}

    for var_name, stac_items in available_items.items():
        try:
            if not stac_items:
                logger.warning(f"No STAC items found for {var_name}")
                met_data[var_name] = None
                continue

            ds = load_variable_from_stac_items(stac_items, var_name)
            met_data[var_name] = ds

        except Exception as e:
            logger.error(f"Failed to load {var_name}: {e}")
            met_data[var_name] = None

    # Step 3: Validate completeness (FAIL if incomplete)
    validate_meteorology_completeness(met_data, required_months)

    # Step 4: Assemble into unified dataset
    unified_met_data = assemble_unified_meteorology_dataset(met_data)

    logger.info("Meteorology loading complete")

    return unified_met_data


def _parse_month_range(start_date: str, end_date: str) -> List[str]:
    """
    Parse date range strings into list of months.

    Args:
        start_date (str): 'YYYY-MM' format
        end_date (str): 'YYYY-MM' format

    Returns:
        List[str]: List of months in 'YYYY-MM' format

    Example:
        >>> _parse_month_range('2020-01', '2020-03')
        ['2020-01', '2020-02', '2020-03']
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
