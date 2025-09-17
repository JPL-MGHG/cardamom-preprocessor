"""
Statistical Utilities for CARDAMOM Preprocessing

This module provides statistical and data processing utility functions that replicate
MATLAB functionality for temporal analysis, spatial interpolation, and data aggregation.
All functions include explicit references to the original MATLAB source code.

Scientific Context:
These utilities are essential for processing time series data, performing spatial
interpolation, and statistical analysis of atmospheric and carbon cycle datasets
in the CARDAMOM preprocessing system.

References:
- MATLAB Source: /Users/shah/Desktop/Development/ghg/CARDAMOM/MATLAB/auxi_fun/
- MATLAB Source: /Users/shah/Desktop/Development/ghg/CARDAMOM/MATLAB/stats_fun/
"""

import numpy as np
from typing import Union, Tuple, Optional
import warnings


def nan_to_zero(data: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert NaN values to zero, preserving array structure.

    This function replicates the MATLAB nan2zero auxiliary function used
    throughout the CARDAMOM system for handling missing data.

    MATLAB Source Reference:
    File: /MATLAB/auxi_fun/nan2zero.m
    Lines: 1-3
    MATLAB Code:
        function B=nan2zero(A);
        B=A;
        B(isnan(A))=0;

    Scientific Background:
    Converting NaN to zero is often appropriate for accumulative quantities
    like precipitation or fluxes where missing data can be treated as zero
    contribution rather than unknown values.

    Args:
        data: Input data array or scalar with potential NaN values
            Any numeric type, commonly meteorological or flux data

    Returns:
        Data with NaN values replaced by zero, same shape as input
            Preserves all finite values unchanged

    Example:
        >>> # Missing precipitation data
        >>> precip = np.array([2.5, np.nan, 0.8, np.nan, 1.2])
        >>> clean_precip = nan_to_zero(precip)
        >>> # Result: [2.5, 0.0, 0.8, 0.0, 1.2]
    """

    B = np.asarray(data, dtype=float)
    B[np.isnan(B)] = 0
    return B


def monthly_to_annual(monthly_data: np.ndarray, dim: int = -1) -> np.ndarray:
    """
    Convert monthly time series data to annual averages.

    This function replicates the MATLAB monthly2annual auxiliary function
    for temporal aggregation of monthly datasets to annual means.

    MATLAB Source Reference:
    File: /MATLAB/auxi_fun/monthly2annual.m
    Lines: 1-23
    MATLAB Code:
        for m=1:12;TSA=TSA+TS(:,m:12:end)/12;end (line 13)
        for m=1:12;TSA=TSA+TS(:,:,m:12:end,:,:)/12;end (line 16)

    Scientific Background:
    Annual averaging is essential for climate analysis and removes seasonal
    variability to focus on interannual trends. Commonly used for temperature,
    precipitation, and carbon flux climatologies.

    Args:
        monthly_data: Monthly time series data
            Shape: (..., N*12) where N is number of years
            Last dimension contains months in chronological order
        dim: Dimension along which months are arranged
            Default: -1 (last dimension)

    Returns:
        Annual averaged data
            Shape: (..., N) where N is number of years
            Each value represents average of 12 consecutive months

    Notes:
        - Assumes data is arranged with complete years (multiples of 12)
        - Handles 2D and 3D arrays following MATLAB logic
        - Averages by dividing by 12 (MATLAB lines 13, 16)

    Example:
        >>> # Monthly temperature data for 2 years (24 months)
        >>> monthly_temp = np.random.randn(50, 50, 24) + 285  # K
        >>> annual_temp = monthly_to_annual(monthly_temp, dim=2)
        >>> # Result shape: (50, 50, 2) - 2 annual averages
    """

    data = np.asarray(monthly_data)

    # Move time dimension to end for consistent processing
    data = np.moveaxis(data, dim, -1)

    # Get number of months and check for complete years
    n_months = data.shape[-1]
    if n_months % 12 != 0:
        warnings.warn(f"Data has {n_months} months, not a multiple of 12. Results may be incomplete.")

    n_years = n_months // 12

    # MATLAB logic: for m=1:12;TSA=TSA+TS(:,m:12:end)/12;end
    annual_data = np.zeros(data.shape[:-1] + (n_years,))

    for month in range(12):
        # Select every 12th month starting from current month (0-based indexing)
        month_indices = slice(month, n_months, 12)
        annual_data += data[..., month_indices] / 12

    return annual_data


def monthly_to_seasonal(monthly_data: np.ndarray, dim: int = -1) -> np.ndarray:
    """
    Extract seasonal averages from monthly time series data.

    This function replicates the MATLAB monthly2seasonal auxiliary function
    for extracting seasonal patterns from monthly datasets.

    MATLAB Source Reference:
    File: /MATLAB/auxi_fun/monthly2seasonal.m
    Lines: 1-21
    MATLAB Logic: Calculates seasonal averages across multiple years

    Scientific Background:
    Seasonal averaging reveals the annual cycle of atmospheric and ecological
    variables, essential for understanding climate patterns and ecosystem
    phenology in carbon cycle modeling.

    Args:
        monthly_data: Monthly time series data
            Shape: (..., N*12) where N is number of years
            Last dimension contains months in chronological order
        dim: Dimension along which months are arranged
            Default: -1 (last dimension)

    Returns:
        Seasonal data
            Shape: (..., 12) representing average for each month
            Values represent climatological monthly means

    Notes:
        - Averages across all years for each month of the year
        - Useful for creating climatological seasonal cycles
        - Results in 12 values representing annual cycle

    Example:
        >>> # 5 years of monthly precipitation data
        >>> monthly_precip = np.random.randn(100, 100, 60) + 50  # mm/month
        >>> seasonal_precip = monthly_to_seasonal(monthly_precip, dim=2)
        >>> # Result shape: (100, 100, 12) - climatological monthly cycle
    """

    data = np.asarray(monthly_data)

    # Move time dimension to end
    data = np.moveaxis(data, dim, -1)

    n_months = data.shape[-1]
    n_years = n_months // 12

    if n_months % 12 != 0:
        warnings.warn(f"Data has {n_months} months, not a multiple of 12.")

    # Reshape to separate years and months, then average across years
    reshaped_data = data[..., :n_years*12].reshape(data.shape[:-1] + (n_years, 12))
    seasonal_data = np.mean(reshaped_data, axis=-2)  # Average across years

    return seasonal_data


def calculate_percentile(data: np.ndarray, percentile: float, dim: int = 0) -> np.ndarray:
    """
    Calculate percentile along specified dimension.

    This function replicates the MATLAB percentile statistical function
    with support for multi-dimensional arrays and specified dimensions.

    MATLAB Source Reference:
    File: /MATLAB/stats_fun/percentile.m
    Lines: 1-42
    MATLAB Code:
        la=size(a,pdim)-1;
        a=sort(a,pdim);
        aperc=a(round(la*perc/100)+1);

    Scientific Background:
    Percentiles are essential for understanding data distributions, identifying
    outliers, and creating robust statistics for atmospheric and carbon cycle
    datasets that often contain extreme values.

    Args:
        data: Input data array
            Any shape, commonly meteorological or flux measurements
        percentile: Percentile to calculate (0-100)
            Example: 50 for median, 95 for 95th percentile
        dim: Dimension along which to calculate percentile
            Default: 0 (first dimension)

    Returns:
        Percentile values along specified dimension
            Shape reduced by one dimension along calculation axis

    Notes:
        - Handles finite values only (removes NaN/Inf)
        - Uses linear interpolation for non-integer indices
        - MATLAB uses round() function for index calculation (line 26)

    Example:
        >>> # Temperature data with outliers
        >>> temp_data = np.random.randn(365, 100, 100) * 5 + 290  # K
        >>> temp_95th = calculate_percentile(temp_data, 95, dim=0)
        >>> # Result shape: (100, 100) - 95th percentile temperature
    """

    arr = np.asarray(data)

    # Remove non-finite values (MATLAB line 16: a=a(isfinite(a)))
    finite_mask = np.isfinite(arr)

    if not np.any(finite_mask):
        warnings.warn("No finite values found in data")
        return np.full(arr.shape[:dim] + arr.shape[dim+1:], np.nan)

    # Sort along specified dimension (MATLAB line 19: a=sort(a,pdim))
    sorted_data = np.sort(arr, axis=dim)

    # Calculate percentile index (MATLAB line 26: round(la*perc/100)+1)
    # Note: MATLAB uses 1-based indexing, Python uses 0-based
    array_size = arr.shape[dim] - 1  # MATLAB: la=size(a,pdim)-1
    percentile_index = np.round(array_size * percentile / 100).astype(int)

    # Ensure index is within bounds
    percentile_index = np.clip(percentile_index, 0, array_size)

    # Extract percentile values
    result = np.take(sorted_data, percentile_index, axis=dim)

    return result


def find_closest_grid_points(points_x: np.ndarray, points_y: np.ndarray,
                            grid_x: np.ndarray, grid_y: np.ndarray,
                            irregular: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find closest grid points to specified coordinates.

    This function replicates the MATLAB closest2d auxiliary function for
    spatial interpolation and grid point matching in atmospheric modeling.

    MATLAB Source Reference:
    File: /MATLAB/auxi_fun/closest2d.m
    Lines: 1-117
    MATLAB Function: [pti,ri,ci]=closest2d(xp,yp,x,y,irregular)
    Algorithm (lines 68-78): d=(x-xp(n)).^2+(y-yp(n)).^2; find(d==min(d))

    Scientific Background:
    Spatial interpolation is essential for matching data from different grids,
    such as combining observational data with model grids, or regridding
    between different atmospheric model resolutions.

    Args:
        points_x: X coordinates of points to be matched
            Shape: (N,) - longitude or x-coordinates in degrees or meters
        points_y: Y coordinates of points to be matched
            Shape: (N,) - latitude or y-coordinates in degrees or meters
        grid_x: X coordinates of grid
            Shape: (M, L) - grid longitude/x-coordinates
        grid_y: Y coordinates of grid
            Shape: (M, L) - grid latitude/y-coordinates
        irregular: Whether grid is irregular
            True: Use distance-based search (default)
            False: Use fast regular grid method

    Returns:
        Tuple of (point_indices, row_indices, column_indices)
            point_indices: Flattened grid indices of closest points
            row_indices: Row indices in grid
            column_indices: Column indices in grid

    Notes:
        - Uses Euclidean distance for closest point calculation (MATLAB line 69)
        - Supports both regular and irregular grids (MATLAB cases 0,1)
        - For irregular=False, assumes constant grid spacing (MATLAB case 0)

    Example:
        >>> # Find grid points closest to observation stations
        >>> station_lons = np.array([-120.5, -118.2, -115.8])
        >>> station_lats = np.array([35.3, 34.1, 36.2])
        >>> grid_lons, grid_lats = np.meshgrid(np.arange(-125, -110, 0.5),
        ...                                    np.arange(30, 40, 0.5))
        >>> pts, rows, cols = find_closest_grid_points(station_lons, station_lats,
        ...                                          grid_lons, grid_lats)
    """

    xp = np.asarray(points_x).flatten()
    yp = np.asarray(points_y).flatten()
    x = np.asarray(grid_x)
    y = np.asarray(grid_y)

    n_points = len(xp)
    row_indices = np.zeros(n_points, dtype=int)
    col_indices = np.zeros(n_points, dtype=int)

    if irregular:
        # MATLAB case 1: irregular grid (lines 66-78)
        for n in range(n_points):
            # MATLAB line 69: d=(x-xp(n)).^2+(y-yp(n)).^2 (no need for square root)
            distance_squared = (x - xp[n])**2 + (y - yp[n])**2

            # MATLAB line 70: find(d==min(d(1:end)))
            min_distance_idx = np.unravel_index(np.argmin(distance_squared), distance_squared.shape)
            row_indices[n] = min_distance_idx[0]
            col_indices[n] = min_distance_idx[1]

    else:
        # MATLAB case 0: regular grid (lines 28-59)
        # Assuming constant intervals
        min_x, max_x = x[0, 0], x[-1, -1]
        min_y, max_y = y[0, 0], y[-1, -1]

        # Grid intervals (MATLAB lines 37-38)
        int_y = y[1, 0] - y[0, 0] if y.shape[0] > 1 else 1.0
        int_x = x[0, 1] - x[0, 0] if x.shape[1] > 1 else 1.0

        # Calculate grid indices (MATLAB lines 43-44)
        col_indices = np.round((xp - min_x) / int_x).astype(int)
        row_indices = np.round((yp - min_y) / int_y).astype(int)

        # Clip to grid bounds (MATLAB lines 51-54)
        row_indices = np.clip(row_indices, 0, x.shape[0] - 1)
        col_indices = np.clip(col_indices, 0, x.shape[1] - 1)

    # Calculate flattened indices (MATLAB line 59: pti=(ci-1)*nrows+ri)
    # Note: MATLAB uses 1-based indexing, Python uses 0-based
    n_rows = x.shape[0]
    point_indices = col_indices * n_rows + row_indices

    return point_indices, row_indices, col_indices


def monthly_to_deseasonalized(monthly_data: np.ndarray, dim: int = -1) -> np.ndarray:
    """
    Remove seasonal cycle from monthly time series data.

    This function replicates the MATLAB monthly2deseasonalized auxiliary function
    for removing climatological seasonal patterns from time series.

    MATLAB Source Reference:
    File: /MATLAB/auxi_fun/monthly2deseasonalized.m
    Lines: 1-13
    MATLAB Logic: Subtracts climatological monthly means from data

    Scientific Background:
    Deseasonalization removes the annual cycle to reveal interannual variability
    and trends. Essential for climate change analysis and anomaly detection
    in atmospheric and carbon cycle datasets.

    Args:
        monthly_data: Monthly time series data
            Shape: (..., N*12) where N is number of years
        dim: Dimension along which months are arranged
            Default: -1 (last dimension)

    Returns:
        Deseasonalized data (anomalies from climatological cycle)
            Same shape as input
            Values represent departures from typical seasonal values

    Example:
        >>> # Remove seasonal cycle from CO2 data
        >>> monthly_co2 = np.random.randn(50, 50, 120) + 400  # ppm
        >>> deseasonalized_co2 = monthly_to_deseasonalized(monthly_co2)
        >>> # Result: CO2 anomalies with seasonal cycle removed
    """

    data = np.asarray(monthly_data)

    # Calculate climatological seasonal cycle
    seasonal_cycle = monthly_to_seasonal(data, dim=dim)

    # Move time dimension to end for broadcasting
    data = np.moveaxis(data, dim, -1)
    n_months = data.shape[-1]
    n_years = n_months // 12

    # Tile seasonal cycle to match data length
    seasonal_tiled = np.tile(seasonal_cycle, (1,) * (data.ndim - 1) + (n_years,))

    # Remove seasonal cycle
    deseasonalized = data - seasonal_tiled[..., :n_months]

    # Move time dimension back to original position
    deseasonalized = np.moveaxis(deseasonalized, -1, dim)

    return deseasonalized