"""
Quality Control and Data Validation for CARDAMOM Preprocessing

This module provides enhanced data validation and quality control functions
that build upon existing validation in scientific_utils.py and extend
validation capabilities for atmospheric and carbon cycle datasets.

Scientific Context:
Robust quality control is essential for atmospheric and carbon cycle modeling
to ensure data integrity, identify outliers, and maintain scientific validity
of preprocessing results used in CARDAMOM analysis.

References:
- WMO Guide to Meteorological Instruments and Methods of Observation
- ICOS Ecosystem Instructions for data quality control
- FluxNet data quality control protocols
"""

import numpy as np
from typing import Union, Dict, List, Tuple, Optional, Any
import warnings
from units_constants import PhysicalConstants


class DataQualityReport:
    """
    Container for data quality assessment results.

    Provides structured reporting of data quality metrics, outliers,
    and validation results for scientific datasets.
    """

    def __init__(self):
        self.passed_checks: List[str] = []
        self.failed_checks: List[str] = []
        self.warnings: List[str] = []
        self.statistics: Dict[str, Any] = {}
        self.outlier_indices: Dict[str, np.ndarray] = {}
        self.data_coverage: Dict[str, float] = {}

    def add_check_result(self, check_name: str, passed: bool, message: str = ""):
        """Add result of a quality control check."""
        if passed:
            self.passed_checks.append(check_name)
        else:
            self.failed_checks.append(f"{check_name}: {message}")

    def add_warning(self, message: str):
        """Add a quality control warning."""
        self.warnings.append(message)

    def is_valid(self) -> bool:
        """Check if data passes all quality control checks."""
        return len(self.failed_checks) == 0

    def summary(self) -> str:
        """Generate summary report of quality control results."""
        summary = f"Quality Control Summary:\n"
        summary += f"  Passed checks: {len(self.passed_checks)}\n"
        summary += f"  Failed checks: {len(self.failed_checks)}\n"
        summary += f"  Warnings: {len(self.warnings)}\n"

        if self.failed_checks:
            summary += f"\nFailed checks:\n"
            for check in self.failed_checks:
                summary += f"  - {check}\n"

        if self.warnings:
            summary += f"\nWarnings:\n"
            for warning in self.warnings:
                summary += f"  - {warning}\n"

        return summary


def validate_temperature_range_extended(temperature_data: Union[float, np.ndarray],
                                       min_temp_kelvin: float = 173.15,
                                       max_temp_kelvin: float = 333.15,
                                       check_units: bool = True) -> DataQualityReport:
    """
    Extended temperature validation with comprehensive quality checks.

    This function builds upon the basic temperature validation in scientific_utils.py
    with enhanced range checking, statistical analysis, and unit detection.

    Scientific Background:
    Temperature data quality is critical for atmospheric modeling. Extreme values
    outside observed Earth temperature ranges indicate data errors or incorrect units.
    Statistical outliers may indicate measurement errors or unusual conditions.

    Args:
        temperature_data: Temperature values (Kelvin or Celsius, auto-detected)
            Expected ranges: 173-333 K (-100 to 60°C) for atmospheric applications
        min_temp_kelvin: Minimum acceptable temperature in Kelvin
            Default: 173.15 K (-100°C, record low on Earth)
        max_temp_kelvin: Maximum acceptable temperature in Kelvin
            Default: 333.15 K (60°C, extreme high temperature)
        check_units: Whether to attempt unit detection and validation
            Default: True

    Returns:
        DataQualityReport containing validation results and statistics

    Example:
        >>> # ERA5 temperature data
        >>> temp_data = np.random.randn(1000) * 10 + 285  # K
        >>> report = validate_temperature_range_extended(temp_data)
        >>> print(report.summary())
    """

    temp = np.asarray(temperature_data)
    report = DataQualityReport()

    # Basic data checks
    if temp.size == 0:
        report.add_check_result("data_exists", False, "No temperature data provided")
        return report

    report.add_check_result("data_exists", True)

    # Check for finite values
    finite_mask = np.isfinite(temp)
    finite_count = np.sum(finite_mask)
    total_count = temp.size

    if finite_count == 0:
        report.add_check_result("finite_values", False, "No finite temperature values found")
        return report

    report.add_check_result("finite_values", True)
    report.data_coverage["finite_fraction"] = finite_count / total_count

    if finite_count < total_count:
        missing_fraction = (total_count - finite_count) / total_count
        report.add_warning(f"{missing_fraction:.1%} of temperature data is missing or non-finite")

    temp_finite = temp[finite_mask]

    # Unit detection
    mean_temp = np.mean(temp_finite)
    if check_units:
        if mean_temp < 100:
            # Likely Celsius
            report.add_warning("Temperature appears to be in Celsius, converting to Kelvin")
            temp_finite = temp_finite + PhysicalConstants.STANDARD_TEMPERATURE
            mean_temp = np.mean(temp_finite)
        elif 150 < mean_temp < 350:
            # Likely Kelvin
            report.add_check_result("units_kelvin", True)
        else:
            report.add_check_result("units_reasonable", False,
                                  f"Unusual temperature range (mean: {mean_temp:.1f}), check units")

    # Range validation
    below_min = np.sum(temp_finite < min_temp_kelvin)
    above_max = np.sum(temp_finite > max_temp_kelvin)

    if below_min > 0:
        report.add_check_result("min_temperature", False,
                              f"{below_min} values below {min_temp_kelvin:.1f} K")
        report.outlier_indices["below_minimum"] = np.where(temp < min_temp_kelvin)[0]

    if above_max > 0:
        report.add_check_result("max_temperature", False,
                              f"{above_max} values above {max_temp_kelvin:.1f} K")
        report.outlier_indices["above_maximum"] = np.where(temp > max_temp_kelvin)[0]

    if below_min == 0 and above_max == 0:
        report.add_check_result("temperature_range", True)

    # Statistical analysis
    report.statistics = {
        "mean_kelvin": float(np.mean(temp_finite)),
        "std_kelvin": float(np.std(temp_finite)),
        "min_kelvin": float(np.min(temp_finite)),
        "max_kelvin": float(np.max(temp_finite)),
        "range_kelvin": float(np.ptp(temp_finite))
    }

    # Outlier detection using z-score
    z_scores = np.abs((temp_finite - report.statistics["mean_kelvin"]) / report.statistics["std_kelvin"])
    outlier_threshold = 4.0  # 4 standard deviations
    outlier_mask = z_scores > outlier_threshold

    if np.any(outlier_mask):
        n_outliers = np.sum(outlier_mask)
        outlier_fraction = n_outliers / finite_count
        report.add_warning(f"{n_outliers} statistical outliers detected ({outlier_fraction:.1%})")
        report.outlier_indices["statistical_outliers"] = np.where(finite_mask)[0][outlier_mask]

    return report


def validate_precipitation_extended(precipitation_data: Union[float, np.ndarray],
                                  max_hourly_mm: float = 300.0,
                                  max_daily_mm: float = 1000.0,
                                  check_negative: bool = True) -> DataQualityReport:
    """
    Extended precipitation validation with physical and statistical checks.

    Validates precipitation data for physical consistency, extreme values,
    and statistical outliers commonly found in meteorological datasets.

    Scientific Background:
    Precipitation data quality is critical for hydrological and carbon cycle
    modeling. Extreme precipitation rates may indicate measurement errors,
    unit conversion problems, or rare meteorological events requiring verification.

    Args:
        precipitation_data: Precipitation values (mm or m, auto-detected)
            Expected ranges: 0-300 mm/hr for extreme events
        max_hourly_mm: Maximum acceptable hourly precipitation in mm
            Default: 300 mm/hr (extreme rainfall rate)
        max_daily_mm: Maximum acceptable daily precipitation in mm
            Default: 1000 mm/day (extreme daily total)
        check_negative: Whether to check for negative precipitation
            Default: True (precipitation cannot be negative)

    Returns:
        DataQualityReport containing validation results

    Example:
        >>> # ERA5 precipitation data (m/s)
        >>> precip_data = np.random.exponential(2e-6, 1000)  # m/s
        >>> report = validate_precipitation_extended(precip_data)
    """

    precip = np.asarray(precipitation_data)
    report = DataQualityReport()

    # Basic data checks
    if precip.size == 0:
        report.add_check_result("data_exists", False, "No precipitation data provided")
        return report

    report.add_check_result("data_exists", True)

    # Check for finite values
    finite_mask = np.isfinite(precip)
    finite_count = np.sum(finite_mask)

    if finite_count == 0:
        report.add_check_result("finite_values", False, "No finite precipitation values found")
        return report

    report.add_check_result("finite_values", True)
    report.data_coverage["finite_fraction"] = finite_count / precip.size

    precip_finite = precip[finite_mask]

    # Unit detection and conversion
    max_value = np.max(precip_finite)
    mean_value = np.mean(precip_finite)

    if max_value < 0.1 and mean_value < 0.001:
        # Likely in m or m/s, convert to mm
        report.add_warning("Precipitation appears to be in meters, converting to mm")
        precip_finite = precip_finite * 1000.0  # Convert m to mm
        max_value = np.max(precip_finite)

    # Check for negative values
    if check_negative:
        negative_count = np.sum(precip_finite < 0)
        if negative_count > 0:
            report.add_check_result("negative_precipitation", False,
                                  f"{negative_count} negative precipitation values found")
            report.outlier_indices["negative_values"] = np.where(precip < 0)[0]
        else:
            report.add_check_result("negative_precipitation", True)

    # Range validation
    extreme_hourly = np.sum(precip_finite > max_hourly_mm)
    extreme_daily = np.sum(precip_finite > max_daily_mm)

    if extreme_hourly > 0:
        report.add_warning(f"{extreme_hourly} values exceed extreme hourly rate ({max_hourly_mm} mm/hr)")
        report.outlier_indices["extreme_hourly"] = np.where(precip > max_hourly_mm)[0]

    if extreme_daily > 0:
        report.add_check_result("extreme_daily", False,
                              f"{extreme_daily} values exceed extreme daily total ({max_daily_mm} mm/day)")
        report.outlier_indices["extreme_daily"] = np.where(precip > max_daily_mm)[0]

    # Statistical analysis
    non_zero_precip = precip_finite[precip_finite > 0]
    if len(non_zero_precip) > 0:
        report.statistics = {
            "total_precipitation": float(np.sum(precip_finite)),
            "mean_all": float(np.mean(precip_finite)),
            "mean_nonzero": float(np.mean(non_zero_precip)),
            "max_value": float(np.max(precip_finite)),
            "zero_fraction": float(np.sum(precip_finite == 0) / len(precip_finite)),
            "wet_days_fraction": float(len(non_zero_precip) / len(precip_finite))
        }
    else:
        report.add_warning("No non-zero precipitation values found")
        report.statistics = {"total_precipitation": 0.0}

    return report


def validate_carbon_flux_extended(carbon_flux_data: Union[float, np.ndarray],
                                flux_type: str,
                                check_mass_balance: bool = True) -> DataQualityReport:
    """
    Extended validation for carbon flux data with ecosystem-specific checks.

    Validates carbon flux data against ecological principles and typical
    ranges observed in ecosystem measurements and modeling.

    Scientific Background:
    Carbon flux validation requires understanding of ecosystem processes
    and typical magnitudes. Different flux types (GPP, respiration, NEE)
    have different expected ranges and physical constraints.

    Args:
        carbon_flux_data: Carbon flux values in gC/m²/s or gC/m²/day
            Sign convention: atmospheric perspective
        flux_type: Type of carbon flux
            Options: 'GPP', 'respiration', 'NEE', 'fire'
        check_mass_balance: Whether to perform mass balance checks
            Default: True

    Returns:
        DataQualityReport containing validation results

    Example:
        >>> # Net ecosystem exchange data
        >>> nee_data = np.random.randn(365) * 5 - 2  # gC/m²/day (slight sink)
        >>> report = validate_carbon_flux_extended(nee_data, 'NEE')
    """

    flux_data = np.asarray(carbon_flux_data)
    report = DataQualityReport()

    # Define expected ranges for different flux types (gC/m²/day)
    flux_ranges = {
        'GPP': {'min': 0, 'max': 50, 'typical_max': 30, 'sign': 'positive'},
        'respiration': {'min': 0, 'max': 30, 'typical_max': 20, 'sign': 'positive'},
        'NEE': {'min': -30, 'max': 20, 'typical_range': (-10, 10), 'sign': 'either'},
        'fire': {'min': 0, 'max': 100, 'typical_max': 10, 'sign': 'positive'}
    }

    if flux_type.lower() not in [k.lower() for k in flux_ranges.keys()]:
        report.add_check_result("valid_flux_type", False, f"Unknown flux type: {flux_type}")
        return report

    # Find matching flux type (case insensitive)
    flux_key = next(k for k in flux_ranges.keys() if k.lower() == flux_type.lower())
    ranges = flux_ranges[flux_key]

    # Basic data checks
    if flux_data.size == 0:
        report.add_check_result("data_exists", False, "No flux data provided")
        return report

    report.add_check_result("data_exists", True)

    # Check for finite values
    finite_mask = np.isfinite(flux_data)
    finite_count = np.sum(finite_mask)

    if finite_count == 0:
        report.add_check_result("finite_values", False, "No finite flux values found")
        return report

    report.add_check_result("finite_values", True)
    flux_finite = flux_data[finite_mask]

    # Sign convention checks
    if ranges['sign'] == 'positive':
        negative_count = np.sum(flux_finite < 0)
        if negative_count > 0:
            report.add_check_result("sign_convention", False,
                                  f"{negative_count} negative values for {flux_type} (should be positive)")
            report.outlier_indices["negative_values"] = np.where(flux_data < 0)[0]
        else:
            report.add_check_result("sign_convention", True)

    # Physical range checks
    below_min = np.sum(flux_finite < ranges['min'])
    above_max = np.sum(flux_finite > ranges['max'])

    if below_min > 0:
        report.add_check_result("physical_minimum", False,
                              f"{below_min} values below physical minimum ({ranges['min']} gC/m²/day)")

    if above_max > 0:
        report.add_check_result("physical_maximum", False,
                              f"{above_max} values above physical maximum ({ranges['max']} gC/m²/day)")

    if below_min == 0 and above_max == 0:
        report.add_check_result("physical_range", True)

    # Typical range checks
    if 'typical_max' in ranges:
        above_typical = np.sum(flux_finite > ranges['typical_max'])
        if above_typical > 0:
            typical_fraction = above_typical / finite_count
            if typical_fraction > 0.05:  # More than 5% above typical
                report.add_warning(f"{above_typical} values above typical maximum "
                                 f"({ranges['typical_max']} gC/m²/day)")

    # Statistical analysis
    report.statistics = {
        "mean": float(np.mean(flux_finite)),
        "std": float(np.std(flux_finite)),
        "min": float(np.min(flux_finite)),
        "max": float(np.max(flux_finite)),
        "median": float(np.median(flux_finite)),
        "range": float(np.ptp(flux_finite))
    }

    # Ecosystem reasonableness checks
    if flux_type.lower() == 'nee':
        # Check for reasonable sink/source balance
        negative_fraction = np.sum(flux_finite < 0) / finite_count
        if negative_fraction < 0.1 or negative_fraction > 0.9:
            report.add_warning(f"NEE shows {negative_fraction:.1%} carbon sink periods, "
                             f"check for systematic bias")

    elif flux_type.lower() == 'gpp':
        # GPP should show diurnal/seasonal patterns
        if np.std(flux_finite) < 0.1 * np.mean(flux_finite):
            report.add_warning("GPP shows unusually low variability, "
                             "check for constant values or unit errors")

    return report


def validate_spatial_grid_consistency(longitude: np.ndarray,
                                     latitude: np.ndarray,
                                     data_array: np.ndarray,
                                     expected_resolution: Optional[float] = None) -> DataQualityReport:
    """
    Validate spatial grid consistency and coordinate alignment.

    Checks spatial grid coordinates for regularity, coverage, and consistency
    with data array dimensions commonly required in atmospheric modeling.

    Args:
        longitude: Longitude coordinates (degrees East)
            Expected range: -180 to 360
        latitude: Latitude coordinates (degrees North)
            Expected range: -90 to 90
        data_array: Data array with spatial dimensions
            Should match coordinate array shapes
        expected_resolution: Expected grid resolution in degrees
            If provided, checks for consistent spacing

    Returns:
        DataQualityReport containing spatial validation results
    """

    lon = np.asarray(longitude)
    lat = np.asarray(latitude)
    data = np.asarray(data_array)
    report = DataQualityReport()

    # Check coordinate ranges
    if np.any(lat < -90) or np.any(lat > 90):
        report.add_check_result("latitude_range", False,
                              f"Latitude outside valid range: {np.min(lat):.2f} to {np.max(lat):.2f}")
    else:
        report.add_check_result("latitude_range", True)

    # Check longitude range (allow both -180/180 and 0/360 conventions)
    if np.any(lon < -180) or np.any(lon > 360):
        report.add_check_result("longitude_range", False,
                              f"Longitude outside valid range: {np.min(lon):.2f} to {np.max(lon):.2f}")
    else:
        report.add_check_result("longitude_range", True)

    # Check grid regularity
    if lon.ndim == 1 and lat.ndim == 1:
        # 1D coordinate arrays
        lon_spacing = np.diff(lon)
        lat_spacing = np.diff(lat)

        if len(lon_spacing) > 1:
            lon_regular = np.allclose(lon_spacing, lon_spacing[0], rtol=0.01)
            if not lon_regular:
                report.add_warning("Longitude grid appears irregular")
            else:
                report.add_check_result("longitude_regular", True)

        if len(lat_spacing) > 1:
            lat_regular = np.allclose(lat_spacing, lat_spacing[0], rtol=0.01)
            if not lat_regular:
                report.add_warning("Latitude grid appears irregular")
            else:
                report.add_check_result("latitude_regular", True)

        # Check expected resolution
        if expected_resolution is not None:
            mean_lon_spacing = np.mean(np.abs(lon_spacing))
            mean_lat_spacing = np.mean(np.abs(lat_spacing))

            if not np.isclose(mean_lon_spacing, expected_resolution, rtol=0.1):
                report.add_warning(f"Longitude spacing ({mean_lon_spacing:.3f}°) "
                                 f"differs from expected ({expected_resolution:.3f}°)")

            if not np.isclose(mean_lat_spacing, expected_resolution, rtol=0.1):
                report.add_warning(f"Latitude spacing ({mean_lat_spacing:.3f}°) "
                                 f"differs from expected ({expected_resolution:.3f}°)")

    # Check dimension consistency
    if data.ndim >= 2:
        expected_shape = (len(lat), len(lon)) if lon.ndim == 1 else lon.shape
        data_spatial_shape = data.shape[-2:] if data.ndim > 2 else data.shape

        if data_spatial_shape != expected_shape:
            report.add_check_result("dimension_consistency", False,
                                  f"Data shape {data_spatial_shape} doesn't match "
                                  f"coordinate shape {expected_shape}")
        else:
            report.add_check_result("dimension_consistency", True)

    # Calculate grid statistics
    report.statistics = {
        "longitude_range": (float(np.min(lon)), float(np.max(lon))),
        "latitude_range": (float(np.min(lat)), float(np.max(lat))),
        "grid_points": int(lon.size * lat.size) if lon.ndim == 1 else int(lon.size),
        "longitude_resolution": float(np.mean(np.abs(np.diff(lon.flatten())))) if lon.size > 1 else 0.0,
        "latitude_resolution": float(np.mean(np.abs(np.diff(lat.flatten())))) if lat.size > 1 else 0.0
    }

    return report