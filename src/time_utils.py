"""
Time Coordinate Normalization Utilities

Provides standardized time handling across the CARDAMOM preprocessing pipeline.
Ensures consistent time encoding for CBF compatibility and cross-dataset alignment.

Scientific Context:
CARDAMOM requires consistent temporal coordinates across meteorological drivers,
observations, and model outputs. This module standardizes time encoding to use
"days since 2001-01-01" convention matching the CARDAMOM framework expectations.
"""

import numpy as np
import xarray as xr
from typing import List, Union, Tuple
from datetime import datetime
import logging

# Standard CARDAMOM time reference
CARDAMOM_TIME_REFERENCE = "days since 2001-01-01"
CARDAMOM_TIME_CALENDAR = "proleptic_gregorian"


def standardize_time_coordinate(dataset: xr.Dataset,
                                reference: str = CARDAMOM_TIME_REFERENCE,
                                calendar: str = CARDAMOM_TIME_CALENDAR) -> xr.Dataset:
    """
    Standardize time coordinate to CARDAMOM convention.

    Converts time coordinates from any encoding (seconds/minutes/days since various epochs,
    or direct datetime64) to the standardized CARDAMOM format for consistent processing.

    Scientific Background:
    Time coordinate standardization is critical for:
    - Aligning meteorological drivers with observational constraints
    - Ensuring consistent temporal resolution in carbon cycle modeling
    - Enabling accurate interpolation between different data sources

    Args:
        dataset: xarray Dataset with time coordinate to standardize
        reference: Time reference string (default: "days since 2001-01-01")
        calendar: Calendar system (default: "proleptic_gregorian")

    Returns:
        xr.Dataset: Dataset with standardized time coordinate

    Examples:
        >>> ds = xr.open_dataset("era5_monthly.nc")
        >>> ds_standardized = standardize_time_coordinate(ds)
        >>> print(ds_standardized.time.encoding['units'])
        'days since 2001-01-01'
    """
    if 'time' not in dataset.coords and 'time' not in dataset.dims:
        logging.debug("No time coordinate found in dataset, skipping standardization")
        return dataset

    # Make a copy to avoid modifying original
    standardized_ds = dataset.copy()

    # Get time coordinate
    time_coord = standardized_ds['time']

    # Check if time is already in datetime64 format
    if np.issubdtype(time_coord.dtype, np.datetime64):
        # Already decoded, just update encoding
        time_coord.encoding.update({
            'units': reference,
            'calendar': calendar,
            'dtype': 'int64'
        })
        logging.debug(f"Time already datetime64, updated encoding to: {reference}")
    else:
        # Time is numeric, need to decode first
        # Check if it has encoding information
        if 'units' in time_coord.attrs or 'units' in time_coord.encoding:
            # Decode using existing units
            standardized_ds = xr.decode_cf(standardized_ds)
            time_coord = standardized_ds['time']

            # Now set new encoding
            time_coord.encoding.update({
                'units': reference,
                'calendar': calendar,
                'dtype': 'int64'
            })
            logging.debug(f"Decoded and re-encoded time to: {reference}")
        else:
            # No encoding info, assume it's already in the desired format
            logging.warning("Time coordinate has no encoding info, assuming correct format")

    return standardized_ds


def align_time_coordinates(datasets: List[xr.Dataset],
                          method: str = 'nearest') -> List[xr.Dataset]:
    """
    Align time coordinates across multiple datasets.

    Ensures temporal alignment between datasets with potentially different
    time coordinates, enabling proper merging and interpolation.

    Scientific Context:
    Different data sources (ERA5, NOAA CO2, GFED fire) may have:
    - Different temporal resolutions (hourly, daily, monthly)
    - Different time stamps (start-of-month vs mid-month)
    - Different time encodings (various epoch references)

    This function aligns them to a common temporal grid for integration.

    Args:
        datasets: List of xarray Datasets to align
        method: Interpolation method ('nearest', 'linear', 'cubic')

    Returns:
        List[xr.Dataset]: Aligned datasets on common time grid

    Example:
        >>> era5_ds = xr.open_dataset("era5.nc")
        >>> co2_ds = xr.open_dataset("co2.nc")
        >>> aligned = align_time_coordinates([era5_ds, co2_ds])
    """
    if not datasets:
        return []

    if len(datasets) == 1:
        return datasets

    # Standardize all time coordinates first
    standardized_datasets = [
        standardize_time_coordinate(ds) for ds in datasets
    ]

    # Use first dataset's time as reference
    reference_time = standardized_datasets[0]['time']

    # Align all other datasets to reference time
    aligned_datasets = [standardized_datasets[0]]  # Keep first as-is

    for ds in standardized_datasets[1:]:
        if 'time' in ds.dims or 'time' in ds.coords:
            # Interpolate to reference time
            aligned_ds = ds.interp(time=reference_time, method=method)
            aligned_datasets.append(aligned_ds)
        else:
            # No time dimension, keep as-is
            aligned_datasets.append(ds)

    logging.info(f"Aligned {len(datasets)} datasets to common time grid")
    return aligned_datasets


def validate_time_encoding(dataset: xr.Dataset,
                          expected_reference: str = CARDAMOM_TIME_REFERENCE,
                          expected_calendar: str = CARDAMOM_TIME_CALENDAR) -> Tuple[bool, str]:
    """
    Validate that time encoding matches CARDAMOM standards.

    Args:
        dataset: Dataset to validate
        expected_reference: Expected time reference string
        expected_calendar: Expected calendar system

    Returns:
        Tuple[bool, str]: (validation_passed, message)

    Example:
        >>> ds = xr.open_dataset("cbf_met.nc")
        >>> valid, msg = validate_time_encoding(ds)
        >>> if not valid:
        ...     print(f"Time encoding issue: {msg}")
    """
    if 'time' not in dataset.coords and 'time' not in dataset.dims:
        return True, "No time coordinate (OK for time-invariant data)"

    time_coord = dataset['time']

    # Check encoding
    encoding = time_coord.encoding

    # Validate units
    if 'units' in encoding:
        if encoding['units'] != expected_reference:
            return False, f"Time units '{encoding['units']}' != expected '{expected_reference}'"
    else:
        # Check attributes as fallback
        if 'units' in time_coord.attrs:
            if time_coord.attrs['units'] != expected_reference:
                return False, f"Time units '{time_coord.attrs['units']}' != expected '{expected_reference}'"
        else:
            # No units found - check if datetime64
            if not np.issubdtype(time_coord.dtype, np.datetime64):
                return False, "Time coordinate has no units encoding"

    # Validate calendar
    if 'calendar' in encoding:
        if encoding['calendar'] != expected_calendar:
            return False, f"Calendar '{encoding['calendar']}' != expected '{expected_calendar}'"

    return True, "Time encoding valid"


def convert_to_monthly_midpoint(time_values: np.ndarray) -> np.ndarray:
    """
    Convert time values to mid-month timestamps.

    Some data sources use month-start (2001-01-01) while others use mid-month
    (2001-01-16). This function standardizes to mid-month for consistency with
    some CARDAMOM conventions.

    Scientific Context:
    Monthly meteorological data represents averages over the month. Using
    mid-month timestamps (e.g., Jan 15-16) better represents the temporal
    center of the averaging period.

    Args:
        time_values: Array of datetime64 values

    Returns:
        np.ndarray: Time values adjusted to mid-month

    Example:
        >>> times = np.array(['2020-01-01', '2020-02-01'], dtype='datetime64[D]')
        >>> midpoint_times = convert_to_monthly_midpoint(times)
        >>> # Returns ['2020-01-16', '2020-02-15']
    """
    # Convert to datetime64 if not already
    times_dt64 = np.array(time_values, dtype='datetime64[D]')

    # Extract year and month
    times_pd = np.array(time_values, dtype='datetime64[M]')

    # Calculate mid-month (15th or 16th depending on month length)
    # For simplicity, use 15th for all months
    midpoint_times = times_pd.astype('datetime64[D]') + np.timedelta64(15, 'D')

    return midpoint_times


def get_time_range_info(dataset: xr.Dataset) -> dict:
    """
    Extract time range information from dataset for logging/validation.

    Args:
        dataset: Dataset with time coordinate

    Returns:
        dict: Time range information including start, end, count, resolution

    Example:
        >>> ds = xr.open_dataset("era5.nc")
        >>> info = get_time_range_info(ds)
        >>> print(f"Time range: {info['start']} to {info['end']}")
    """
    if 'time' not in dataset.coords and 'time' not in dataset.dims:
        return {'has_time': False}

    time_coord = dataset['time']

    # Decode to datetime if needed
    if not np.issubdtype(time_coord.dtype, np.datetime64):
        if 'units' in time_coord.encoding or 'units' in time_coord.attrs:
            decoded_ds = xr.decode_cf(dataset)
            time_values = decoded_ds['time'].values
        else:
            time_values = time_coord.values
    else:
        time_values = time_coord.values

    # Calculate time resolution (difference between consecutive timesteps)
    if len(time_values) > 1:
        time_diffs = np.diff(time_values)
        median_diff = np.median(time_diffs)

        # Determine resolution type
        if median_diff < np.timedelta64(2, 'h'):
            resolution = 'hourly'
        elif median_diff < np.timedelta64(2, 'D'):
            resolution = 'daily'
        elif median_diff < np.timedelta64(40, 'D'):
            resolution = 'monthly'
        else:
            resolution = 'yearly'
    else:
        resolution = 'single_timestep'

    return {
        'has_time': True,
        'start': str(time_values[0]),
        'end': str(time_values[-1]),
        'count': len(time_values),
        'resolution': resolution,
        'encoding': time_coord.encoding if hasattr(time_coord, 'encoding') else {},
        'dtype': str(time_coord.dtype)
    }


def ensure_monotonic_time(dataset: xr.Dataset) -> xr.Dataset:
    """
    Ensure time coordinate is monotonically increasing.

    Some data downloads may result in unsorted time coordinates, especially
    when merging multiple files. This function sorts the dataset by time.

    Args:
        dataset: Dataset potentially with unsorted time

    Returns:
        xr.Dataset: Dataset with sorted time coordinate

    Example:
        >>> ds = xr.open_dataset("unsorted.nc")
        >>> ds_sorted = ensure_monotonic_time(ds)
    """
    if 'time' not in dataset.coords and 'time' not in dataset.dims:
        return dataset

    # Check if already monotonic
    time_values = dataset['time'].values

    if len(time_values) > 1:
        is_monotonic = np.all(time_values[1:] >= time_values[:-1])

        if not is_monotonic:
            logging.warning("Time coordinate not monotonic, sorting dataset")
            return dataset.sortby('time')

    return dataset
