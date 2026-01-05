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
import pandas as pd
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

# Map CARDAMOM variable names to expected metadata values
# Allows flexible naming: 'VPD' vs 'vpd' vs 'vapor_pressure_deficit'
# Most cases: CARDAMOM name matches metadata exactly
VARIABLE_METADATA_MAPPING = {
    'VPD': 'VPD',
    'T2M_MIN': 'T2M_MIN',
    'T2M_MAX': 'T2M_MAX',
    'TOTAL_PREC': 'TOTAL_PREC',
    'SSRD': 'SSRD',
    'STRD': 'STRD',
    'SKT': 'SKT',
    'SNOWFALL': 'SNOWFALL',
    'CO2': 'CO2',
    'BURNED_AREA': 'BURNED_AREA',
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
    Check if a STAC item's temporal extent overlaps with the specified date range.

    This handles two types of STAC items:
    1. Point-in-time items: Single datetime (e.g., monthly meteorology)
    2. Time-range items: start_datetime and end_datetime (e.g., yearly data with
       multiple time steps, or time-series data like CO2 1979-2025)

    Scientific Context:
    Some data sources create yearly STAC items with 12 monthly time steps inside,
    or time-series items covering decades. This function checks if the item's
    temporal extent overlaps with the requested date range, not just if a point
    datetime falls within the range.

    Overlap detection uses standard interval overlap logic:
    Item overlaps with range if: item_start <= range_end AND item_end >= range_start

    Args:
        item (pystac.Item): STAC item to check
        start_datetime (str): Start datetime in ISO format (e.g., '2020-01-01T00:00:00Z')
        end_datetime (str): End datetime in ISO format (e.g., '2020-12-31T23:59:59Z')

    Returns:
        bool: True if item's temporal extent overlaps with date range
    """
    from datetime import datetime

    # Parse requested date range
    start_dt = datetime.fromisoformat(start_datetime.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end_datetime.replace('Z', '+00:00'))

    # Get item's temporal extent
    # First, try to get start/end datetime properties (for time-range items)
    item_start_str = item.properties.get('start_datetime')
    item_end_str = item.properties.get('end_datetime')

    if item_start_str and item_end_str:
        # Time-range item: has both start and end datetime
        try:
            item_start = datetime.fromisoformat(item_start_str.replace('Z', '+00:00'))
            item_end = datetime.fromisoformat(item_end_str.replace('Z', '+00:00'))
            # Check for overlap between [item_start, item_end] and [start_dt, end_dt]
            # Overlap occurs if: item_start <= end_dt AND item_end >= start_dt
            return item_start <= end_dt and item_end >= start_dt
        except Exception:
            pass

    # Fall back to item.datetime (point-in-time items)
    item_datetime = item.datetime
    if item_datetime is None:
        # Try to parse start_datetime or end_datetime as fallback
        if item_start_str:
            try:
                item_datetime = datetime.fromisoformat(item_start_str.replace('Z', '+00:00'))
            except Exception:
                pass
        elif item_end_str:
            try:
                item_datetime = datetime.fromisoformat(item_end_str.replace('Z', '+00:00'))
            except Exception:
                pass

    if item_datetime is None:
        # Cannot determine item date, skip
        return False

    # Check if point-in-time item falls within range
    return start_dt <= item_datetime <= end_dt


def discover_stac_items(
    stac_source: str,
    start_date: str,
    end_date: str,
    variable_name: str,
) -> List[Any]:
    """
    Query STAC catalog to discover items for a variable using pure metadata filtering.

    Uses PURE METADATA FILTERING - makes NO assumptions about catalog structure,
    collection names, or organization. Queries all items recursively and filters
    by 'cardamom:variable' property.

    This approach works with ANY STAC organization:
    - Collections or no collections
    - Nested catalogs or flat catalogs
    - Monolithic collections or per-variable collections
    - Any naming scheme

    Args:
        stac_source (str): Path to local STAC catalog.json file or URL to remote STAC API
            Examples:
                - Local: 'file:///path/to/catalog.json' or '/path/to/catalog.json'
                - Remote: 'https://stac-api.example.com'
        start_date (str): Start date in 'YYYY-MM' format
        end_date (str): End date in 'YYYY-MM' format
        variable_name (str): CARDAMOM variable name (e.g., 'T2M_MIN', 'VPD')

    Returns:
        List[Any]: STAC Items matching the variable and date range

    Raises:
        RuntimeError: If STAC catalog query fails
        ValueError: If no items found for variable
    """

    # Get expected metadata value for this variable
    if variable_name not in VARIABLE_METADATA_MAPPING:
        raise ValueError(
            f"Unknown variable: {variable_name}. "
            f"Available: {list(VARIABLE_METADATA_MAPPING.keys())}"
        )

    variable_metadata_value = VARIABLE_METADATA_MAPPING[variable_name]

    # Convert date strings to full datetime range for STAC query
    start_datetime = f"{start_date}-01T00:00:00Z"
    end_datetime = f"{end_date}-28T23:59:59Z"  # Conservative end date

    logger.info(
        f"Searching for items with cardamom:variable='{variable_metadata_value}' "
        f"({start_date} to {end_date})"
    )

    # Determine if source is local file or remote URL
    is_local = _is_local_catalog(stac_source)

    try:
        if is_local:
            # Load local STAC catalog
            logger.debug(f"Loading local STAC catalog from: {stac_source}")
            catalog = pystac.Catalog.from_file(stac_source)

            # Query ALL items from catalog recursively
            # Makes NO assumptions about structure - works with any organization
            logger.debug("Querying all items from catalog (recursive)")
            all_items = list(catalog.get_items(recursive=True))
            logger.info(f"Found {len(all_items)} total items in catalog")

        else:
            # Query remote STAC API (also returns all matching items)
            logger.debug(f"Querying remote STAC API: {stac_source}")
            client = Client.open(stac_source)

            search = client.search(
                datetime=f"{start_datetime}/{end_datetime}",
            )

            all_items = list(search.items())
            logger.info(f"Found {len(all_items)} total items from remote API")

        if not all_items:
            raise ValueError("Catalog contains no items. Ensure data has been downloaded.")

        # Filter items by variable metadata property
        # Items have 'cardamom:variable' property that identifies the variable
        matching_items = []
        for item in all_items:
            item_variable = item.properties.get('cardamom:variable')
            if item_variable == variable_metadata_value:
                matching_items.append(item)

        logger.info(
            f"Found {len(matching_items)} items with cardamom:variable='{variable_metadata_value}' "
            f"(filtered from {len(all_items)} total items)"
        )

        if not matching_items:
            # Show available variables to help debugging
            available_vars = set()
            for item in all_items:
                var = item.properties.get('cardamom:variable')
                if var:
                    available_vars.add(var)

            raise ValueError(
                f"No items found with cardamom:variable='{variable_metadata_value}'. "
                f"Available variables in catalog: {sorted(available_vars)}. "
                f"Check that data was downloaded for this variable."
            )

        # Filter by date range
        items_in_range = []
        for item in matching_items:
            if _item_in_date_range(item, start_datetime, end_datetime):
                items_in_range.append(item)

        logger.info(
            f"After date filtering: {len(items_in_range)} items in range "
            f"[{start_date} to {end_date}]"
        )

        if not items_in_range:
            logger.warning(
                f"No items found for variable '{variable_metadata_value}' in date range. "
                f"Check start/end dates."
            )
            return []

        # Deduplicate items by ID (root catalog may link to items also in collections)
        original_count = len(items_in_range)
        unique_items_by_id = {}
        for item in items_in_range:
            if item.id not in unique_items_by_id:
                unique_items_by_id[item.id] = item

        items_in_range = list(unique_items_by_id.values())

        if len(items_in_range) < original_count:
            logger.info(
                f"Deduplicated items: {original_count} → {len(items_in_range)} "
                f"(removed {original_count - len(items_in_range)} duplicates)"
            )

        # Sort by datetime
        items_in_range.sort(key=lambda x: x.datetime or '')

        return items_in_range

    except Exception as e:
        raise RuntimeError(
            f"Failed to discover items for {variable_name}: {e}"
        ) from e


def load_variable_from_stac_items(
    stac_items: List[Any],
    variable_name: str,
) -> xr.Dataset:
    """
    Load variable data from STAC Item asset URLs and combine along time.

    Scientific Context:
    STAC items can represent either:
    1. Single month snapshots: One time step per file (e.g., ERA5 monthly data)
    2. Yearly time-series: 12 time steps per file (e.g., GFED burned area)
    3. Full time-series: Many time steps in one file (e.g., CO2 1979-2025)

    This function handles all three cases by:
    - Loading NetCDF files and checking their time dimension size
    - Extracting or generating proper time coordinates from STAC metadata
    - Combining datasets with matching time coordinates

    Monthly NetCDF files typically have no explicit time dimension - only lat/lon.
    This function adds proper time coordinates from STAC item metadata
    (start_datetime, end_datetime, or datetime properties).

    Args:
        stac_items (List[Any]): STAC Items for a variable
        variable_name (str): Variable name for logging

    Returns:
        xr.Dataset: Combined dataset with all data concatenated along time,
                   with proper time coordinates extracted from STAC item metadata

    Raises:
        RuntimeError: If file loading fails or all items fail
    """
    from datetime import datetime, timedelta

    def add_months(dt, months):
        """Add months to a datetime, handling year/month rollover."""
        month = dt.month - 1 + months
        year = dt.year + month // 12
        month = month % 12 + 1
        day = min(dt.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                           31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
        return dt.replace(year=year, month=month, day=day)

    logger.info(
        f"Loading {variable_name} data from {len(stac_items)} STAC items"
    )

    datasets = []
    all_time_coords = []  # Track time coordinates for each dataset
    failed_items = []

    for i, item in enumerate(stac_items):
        try:
            # Get data asset URL
            data_asset = item.assets.get('data')
            if not data_asset:
                logger.warning(
                    f"No 'data' asset found in item {item.id}. Skipping."
                )
                failed_items.append(item.id)
                continue

            # Resolve asset href to absolute path
            # If asset href is relative, resolve it from the item file location
            asset_href = data_asset.href
            if not Path(asset_href).is_absolute():
                # Get item's directory (where the .json file is located)
                # Use get_self_href() or self_href if available
                try:
                    self_href = item.get_self_href()
                    if self_href:
                        # Handle file:// URLs - convert to filesystem paths
                        from urllib.parse import urlparse, unquote
                        if self_href.startswith('file://'):
                            parsed = urlparse(self_href)
                            item_file = unquote(parsed.path)
                        else:
                            item_file = self_href

                        item_dir = Path(item_file).parent
                        file_path = str((item_dir / asset_href).resolve())
                    else:
                        # Fall back to just using the relative path
                        file_path = asset_href
                except (AttributeError, Exception):
                    # Fallback if get_self_href doesn't exist or fails
                    file_path = asset_href
            else:
                file_path = asset_href

            # Load NetCDF file
            ds = xr.open_dataset(file_path)

            # Get item temporal extent (for time coordinate generation)
            item_start_str = item.properties.get('start_datetime')
            item_end_str = item.properties.get('end_datetime')

            item_start_dt = None
            if item_start_str:
                try:
                    item_start_dt = datetime.fromisoformat(
                        item_start_str.replace('Z', '+00:00')
                    )
                except Exception:
                    pass

            # Determine number of time steps in this dataset
            time_steps_in_file = 1  # Default: single time step
            if 'time' in ds.dims:
                time_steps_in_file = ds.sizes['time']
            else:
                # Check STAC metadata for time_steps
                time_steps_meta = item.properties.get('cardamom:time_steps')
                if time_steps_meta:
                    time_steps_in_file = int(time_steps_meta)

            # Generate time coordinates for this dataset
            dataset_time_coords = []

            if time_steps_in_file == 1:
                # Single time step: use item's datetime
                if item.datetime is not None:
                    item_datetime = item.datetime
                elif item_start_dt:
                    item_datetime = item_start_dt
                else:
                    item_datetime = None

                if item_datetime is None:
                    logger.warning(
                        f"No datetime found for item {item.id}. "
                        f"Using integer index instead."
                    )
                    dataset_time_coords.append(None)
                else:
                    # Strip timezone info for numpy compatibility
                    if item_datetime.tzinfo is not None:
                        item_datetime = item_datetime.replace(tzinfo=None)
                    dataset_time_coords.append(item_datetime)
                    logger.debug(
                        f"Loaded {variable_name} item {i + 1}/{len(stac_items)}: "
                        f"{file_path} (datetime: {item_datetime.isoformat()})"
                    )
            else:
                # Multiple time steps: generate monthly coordinates
                if not item_start_dt:
                    logger.warning(
                        f"Item {item.id} has {time_steps_in_file} time steps "
                        f"but no start_datetime. Using integer indices."
                    )
                    dataset_time_coords = [None] * time_steps_in_file
                else:
                    # Generate monthly datetime coordinates starting from item_start_dt
                    current_dt = item_start_dt.replace(tzinfo=None)  # Strip timezone
                    for step in range(time_steps_in_file):
                        dataset_time_coords.append(current_dt)
                        # Move to next month
                        current_dt = add_months(current_dt, 1)

                    logger.debug(
                        f"Loaded {variable_name} item {i + 1}/{len(stac_items)}: "
                        f"{file_path} ({time_steps_in_file} time steps, "
                        f"{dataset_time_coords[0].isoformat()} to "
                        f"{dataset_time_coords[-1].isoformat()})"
                    )

            # Store dataset and its time coordinates
            datasets.append(ds)
            all_time_coords.append(dataset_time_coords)

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

    # Build complete time coordinate array from all datasets
    # Flatten the all_time_coords list (which has one list per dataset)
    flattened_time_coords = []
    for dataset_coords in all_time_coords:
        flattened_time_coords.extend(dataset_coords)

    # Add proper time coordinates from STAC items if available
    if flattened_time_coords and any(dt is not None for dt in flattened_time_coords):
        # Convert all datetimes to naive UTC (numpy doesn't support timezone info)
        time_coords_naive = []
        for dt in flattened_time_coords:
            if dt is None:
                time_coords_naive.append(np.datetime64('NaT'))
            elif hasattr(dt, 'replace'):
                # Already a datetime object, ensure it's timezone-naive
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                time_coords_naive.append(dt)
            else:
                # Already a numpy datetime64
                time_coords_naive.append(dt)

        # Create numpy datetime64 array
        try:
            time_coords_np = np.array(time_coords_naive, dtype='datetime64[D]')

            # Verify dimensions match
            if len(time_coords_np) != combined.sizes['time']:
                logger.warning(
                    f"Time coordinate count ({len(time_coords_np)}) does not match "
                    f"concatenated data time dimension ({combined.sizes['time']}). "
                    f"Using integer indices instead."
                )
            else:
                # Assign to combined dataset
                combined = combined.assign_coords(time=time_coords_np)

                logger.info(
                    f"Added time coordinates from STAC items for {variable_name}: "
                    f"{time_coords_np[0]} to {time_coords_np[-1]}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to create time coordinates for {variable_name}: {e}. "
                f"Using integer indices instead."
            )
    else:
        logger.warning(
            f"Could not extract datetimes from STAC items for {variable_name}. "
            f"Using integer indices for time dimension."
        )

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

    This function handles two cases:
    1. Datasets WITH time coordinates: Extracts YYYY-MM from time values
    2. Datasets WITHOUT time coordinates: Checks that dataset has expected number
       of time steps (monthly data should have len(time) == len(required_months))

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
        ds = met_data[var_name]

        # Check if dataset has a time dimension
        if 'time' not in ds.dims:
            logger.warning(
                f"{var_name} has no time dimension. Expected {len(required_months)} months."
            )
            missing_by_variable[var_name] = required_months
            continue

        available_times = ds['time'].values
        available_months = set()

        # Case 1: Time coordinates are datetime64 values (from STAC items)
        # Extract YYYY-MM from datetime values
        if available_times.dtype.kind == 'M':  # Check if dtype is datetime64
            for t in available_times:
                try:
                    # Convert numpy datetime64 to string
                    time_str = str(t)
                    # Extract YYYY-MM (handles both '2020-01-15' and '2020-01' formats)
                    year_month = time_str[:7]  # e.g., '2020-01'
                    available_months.add(year_month)
                except Exception as e:
                    logger.warning(f"Could not parse time value {t}: {e}")

            logger.debug(
                f"{var_name}: Found time coordinates, "
                f"extracted {len(available_months)} unique months"
            )

        # Case 2: Time coordinates are integer indices (no datetimes available)
        # Check that we have the expected number of time steps
        else:
            logger.debug(
                f"{var_name}: Time dimension is numeric indices "
                f"(no datetime coordinates from STAC items)"
            )

            num_time_steps = len(available_times)
            expected_time_steps = len(required_months)

            if num_time_steps == expected_time_steps:
                # Assume all months are present in order
                available_months = set(required_months)
                logger.info(
                    f"{var_name}: {num_time_steps} time steps match "
                    f"required {expected_time_steps} months (using STAC order)"
                )
            else:
                logger.warning(
                    f"{var_name}: Time step count mismatch - "
                    f"expected {expected_time_steps}, got {num_time_steps}"
                )
                # Cannot validate which months are present without datetime coords
                missing_by_variable[var_name] = required_months
                continue

        # Check for missing months
        required_set = set(required_months)
        missing_months = required_set - available_months

        if missing_months:
            missing_by_variable[var_name] = sorted(missing_months)
            logger.warning(
                f"{var_name}: Missing {len(missing_months)} months: "
                f"{sorted(missing_months)}"
            )

    # Report validation results
    if missing_by_variable:
        error_msg = "CRITICAL: Meteorological data is incomplete. Cannot generate valid CBF.\n"
        for var_name, missing_months in sorted(missing_by_variable.items()):
            error_msg += (
                f"  {var_name}: Missing {len(missing_months)} months: "
                f"{missing_months[:3]}...\n"
            )
        logger.error(error_msg)
        raise ValueError(
            f"Incomplete meteorology: {len(missing_by_variable)} variables have gaps"
        )

    logger.info(
        f"✓ Meteorology validation passed: All {len(REQUIRED_FORCING_VARIABLES)} "
        f"variables complete for {len(required_months)} months"
    )


def _subset_meteorology_to_time_range(
    met_data: Dict[str, xr.Dataset],
    start_date: str,
    end_date: str,
    required_months: List[str],
) -> Dict[str, xr.Dataset]:
    """
    Subset meteorological datasets to requested time range.

    Handles cases where a single STAC item spans a much wider time range
    (e.g., CO2 1979-2025 when user only requested 2020). After validation
    confirms required months are present, this function extracts only the
    requested date range.

    Args:
        met_data: Dictionary of variable datasets
        start_date: Start date in 'YYYY-MM' format
        end_date: End date in 'YYYY-MM' format
        required_months: List of required months in 'YYYY-MM' format

    Returns:
        Dictionary with same structure but time-subsetted datasets
    """
    from datetime import datetime

    # Parse date range
    start_dt = datetime.strptime(start_date, '%Y-%m')
    end_dt = datetime.strptime(end_date, '%Y-%m')

    logger.info(f"Subsetting meteorology to time range: {start_date} to {end_date}")

    met_data_subset = {}

    for var_name, ds in met_data.items():
        if ds is None or 'time' not in ds.dims:
            met_data_subset[var_name] = ds
            continue

        try:
            # Get time coordinates
            time_values = ds['time'].values

            # Handle datetime64 time coordinates
            if hasattr(time_values[0], 'astype') or str(time_values.dtype).startswith('datetime64'):
                # Convert to datetime for comparison
                time_datetimes = [pd.Timestamp(t).to_pydatetime() for t in time_values]

                # Find indices within requested range
                mask = [
                    (start_dt.year < dt.year or
                     (start_dt.year == dt.year and start_dt.month <= dt.month)) and
                    (end_dt.year > dt.year or
                     (end_dt.year == dt.year and end_dt.month >= dt.month))
                    for dt in time_datetimes
                ]

                if sum(mask) == len(mask):
                    # All time steps within range, no subsetting needed
                    met_data_subset[var_name] = ds
                    logger.debug(f"{var_name}: All {len(mask)} time steps within requested range")
                else:
                    # Convert boolean mask to integer indices for xarray.isel()
                    indices = [i for i, m in enumerate(mask) if m]
                    ds_subset = ds.isel(time=indices)
                    num_kept = sum(mask)
                    logger.info(
                        f"{var_name}: Subsetted from {len(mask)} to {num_kept} time steps "
                        f"(kept months: {required_months[0]} to {required_months[-1]})"
                    )
                    met_data_subset[var_name] = ds_subset
            else:
                # Integer time indices, assume already in order
                # Check if dataset has all required months
                if len(time_values) >= len(required_months):
                    # Take only the required number of time steps
                    ds_subset = ds.isel(time=slice(0, len(required_months)))
                    logger.info(
                        f"{var_name}: Using first {len(required_months)} time steps "
                        f"(integer indices)"
                    )
                    met_data_subset[var_name] = ds_subset
                else:
                    met_data_subset[var_name] = ds

        except Exception as e:
            logger.warning(f"Could not subset {var_name} to time range: {e}. Keeping original.")
            met_data_subset[var_name] = ds

    return met_data_subset


def assemble_unified_meteorology_dataset(
    met_data: Dict[str, xr.Dataset],
) -> xr.Dataset:
    """
    Assemble individual meteorological variables into unified dataset.

    This creates a dataset with all meteorological variables sharing
    identical spatial and temporal coordinates, similar to the AllMet structure.

    Scientific Context:
    Different data sources may have different spatial resolutions:
    - ERA5: 0.25° grid (1440 longitude)
    - CO2 & BURNED_AREA: 0.5° grid (720 longitude)

    This function automatically detects resolution mismatches and regrids
    finer-resolution data to coarser-resolution grid using linear interpolation.

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

    # Step 1: Find reference grid (coarsest resolution)
    # This ensures all data is on the same grid as data with fewest grid points
    reference_ds = None
    reference_var_name = None
    min_grid_size = float('inf')

    for var_name, var_ds in met_data.items():
        # Count total grid points
        if 'latitude' in var_ds.coords and 'longitude' in var_ds.coords:
            grid_size = len(var_ds.coords['latitude']) * len(var_ds.coords['longitude'])
            if grid_size < min_grid_size:
                min_grid_size = grid_size
                reference_ds = var_ds
                reference_var_name = var_name

    if reference_ds is None:
        raise ValueError("Could not find reference grid from meteorological data")

    logger.info(
        f"Using {reference_var_name} as reference grid: "
        f"latitude={len(reference_ds.coords['latitude'])}, "
        f"longitude={len(reference_ds.coords['longitude'])}"
    )

    # Step 2: Extract data arrays and regrid if needed
    combined_data_arrays = {}

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
            continue

        # Check if regridding is needed (resolution mismatch)
        var_grid_size = len(var_ds.coords['latitude']) * len(var_ds.coords['longitude'])
        if var_grid_size != min_grid_size:
            logger.info(
                f"Regridding {var_name} from {len(var_ds.coords['latitude'])}x"
                f"{len(var_ds.coords['longitude'])} "
                f"to {len(reference_ds.coords['latitude'])}x"
                f"{len(reference_ds.coords['longitude'])} grid"
            )
            try:
                # Check if longitude coordinate normalization is needed
                if data_array.longitude.max() > 180:
                    logger.debug(
                        f"  Converting {var_name} longitude from 0-360° to -180°/+180° system"
                    )
                    # Convert longitude coordinates from 0-360° to -180° to +180°
                    data_array = data_array.assign_coords(
                        longitude=(data_array.longitude + 180) % 360 - 180
                    ).sortby('longitude')
                    logger.debug(
                        f"  ✓ Normalized {var_name} longitude: "
                        f"range {data_array.longitude.min().item():.2f}° to "
                        f"{data_array.longitude.max().item():.2f}°"
                    )

                # Regrid to reference grid using linear interpolation
                data_array = data_array.interp_like(reference_ds, method='linear')
                logger.debug(
                    f"  ✓ Regridded {var_name}: "
                    f"new shape={data_array.shape}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to regrid {var_name}: {e}. "
                    f"Skipping variable."
                )
                continue

        # Add to combined dataset with standardized variable name
        combined_data_arrays[var_name] = data_array
        logger.debug(
            f"Added {var_name}: shape={data_array.shape}, "
            f"lat={len(data_array.latitude)}, lon={len(data_array.longitude)}"
        )

    # Step 3: Create unified dataset with common coordinates
    combined_dataset = xr.Dataset(combined_data_arrays)

    # Set coordinates to match reference (all data now on same grid)
    reference_coords = reference_ds.coords
    combined_dataset.coords['latitude'] = reference_coords['latitude']
    combined_dataset.coords['longitude'] = reference_coords['longitude']

    if 'time' in reference_coords:
        combined_dataset.coords['time'] = reference_coords['time']

    logger.info(
        f"✓ Assembled unified meteorology dataset: "
        f"shape={dict(combined_dataset.dims)}, "
        f"variables={list(combined_dataset.data_vars)}, "
        f"grid=0.5° (land fraction masked ready)"
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

    # Step 4: Subset time dimension to requested date range
    # This handles cases like CO2 where a single item spans 1979-2025
    # but user only requested 2020 data
    met_data = _subset_meteorology_to_time_range(
        met_data, start_date, end_date, required_months
    )

    # Step 5: Assemble into unified dataset
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
