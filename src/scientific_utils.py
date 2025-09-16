"""
Scientific Utility Functions for CARDAMOM Preprocessing

This module provides scientific calculation functions used in CARDAMOM data processing,
including vapor pressure deficit calculations, unit conversions, and physical
validation functions. All functions include clear documentation with scientific
references and expected value ranges.
"""

import numpy as np
from typing import Union, Tuple, Dict, Any


def calculate_vapor_pressure_deficit(temperature_max_kelvin: Union[float, np.ndarray],
                                   dewpoint_temperature_kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate Vapor Pressure Deficit from temperature and dewpoint.

    VPD represents the atmospheric moisture demand and is crucial for
    understanding plant water stress and photosynthesis rates.

    Scientific Background:
    VPD = e_sat(T_max) - e_sat(T_dewpoint)
    where e_sat is saturation vapor pressure calculated using Tetens equation.

    Args:
        temperature_max_kelvin: Daily maximum temperature in Kelvin
            Typical range: 250-320 K (-23 to 47°C)
        dewpoint_temperature_kelvin: Dewpoint temperature in Kelvin
            Typical range: 230-300 K (-43 to 27°C)

    Returns:
        Vapor pressure deficit in hectopascals (hPa)
            Typical range: 0-60 hPa
            - Low VPD (0-10 hPa): High humidity, low atmospheric demand
            - Medium VPD (10-30 hPa): Moderate atmospheric demand
            - High VPD (>30 hPa): Low humidity, high atmospheric demand

    References:
        Tetens, O. (1930). Über einige meteorologische Begriffe.
        Zeitschrift für Geophysik, 6, 297-309.
    """

    # Step 1: Calculate saturation vapor pressure at maximum temperature
    # Using Tetens equation: e_sat = 6.1078 * exp(17.27 * T_c / (T_c + 237.3))
    temp_max_celsius = temperature_max_kelvin - 273.15
    saturation_pressure_at_tmax_hpa = 6.1078 * np.exp(
        17.27 * temp_max_celsius / (temp_max_celsius + 237.3)
    )

    # Step 2: Calculate saturation vapor pressure at dewpoint temperature
    dewpoint_celsius = dewpoint_temperature_kelvin - 273.15
    saturation_pressure_at_dewpoint_hpa = 6.1078 * np.exp(
        17.27 * dewpoint_celsius / (dewpoint_celsius + 237.3)
    )

    # Step 3: Calculate VPD as the difference
    vapor_pressure_deficit_hpa = (saturation_pressure_at_tmax_hpa -
                                 saturation_pressure_at_dewpoint_hpa)

    # Step 4: Validate results are physically reasonable
    if np.any(vapor_pressure_deficit_hpa < 0):
        raise ValueError("VPD cannot be negative. Check that T_max >= T_dewpoint")

    return vapor_pressure_deficit_hpa


def saturation_pressure_water(temperature_kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate saturation pressure of water vapor using Tetens equation.

    This function is equivalent to MATLAB SCIFUN_H2O_SATURATION_PRESSURE
    and provides the fundamental calculation for humidity-related variables.

    Args:
        temperature_kelvin: Temperature in Kelvin
            Typical range: 230-330 K (-43 to 57°C)

    Returns:
        Saturation pressure in hectopascals (hPa)
            Typical range: 0.1-200 hPa

    References:
        Tetens, O. (1930). Über einige meteorologische Begriffe.
        Zeitschrift für Geophysik, 6, 297-309.
    """
    temperature_celsius = temperature_kelvin - 273.15

    # Tetens equation for saturation vapor pressure
    saturation_pressure_hpa = 6.1078 * np.exp(
        17.27 * temperature_celsius / (temperature_celsius + 237.3)
    )

    return saturation_pressure_hpa


def convert_precipitation_units(precip_meters_per_second: Union[float, np.ndarray],
                              scale_factor: float = 86400.0) -> Union[float, np.ndarray]:
    """
    Convert precipitation from meters per second to millimeters per day.

    ERA5 precipitation data is provided in m/s and needs conversion to
    mm/day for CARDAMOM processing.

    Args:
        precip_meters_per_second: Precipitation rate in m/s
            Typical range: 0-1e-4 m/s (0-8.64 mm/day)
        scale_factor: Conversion factor (default: 86400 s/day * 1000 mm/m)

    Returns:
        Precipitation in mm/day
            Typical range: 0-50 mm/day for most regions
    """
    # Convert m/s to mm/day: m/s * (86400 s/day) * (1000 mm/m)
    precipitation_mm_per_day = precip_meters_per_second * scale_factor * 1000

    return precipitation_mm_per_day


def convert_radiation_units(radiation_joules_per_m2: Union[float, np.ndarray],
                          time_period_seconds: float = 86400.0) -> Union[float, np.ndarray]:
    """
    Convert accumulated radiation from J/m² to W/m².

    ERA5 radiation data is often provided as accumulated values in J/m²
    and needs conversion to instantaneous flux in W/m².

    Args:
        radiation_joules_per_m2: Accumulated radiation in J/m²
            Typical range: 0-30,000,000 J/m² per day
        time_period_seconds: Time period for accumulation in seconds (default: 86400 for daily)

    Returns:
        Radiation flux in W/m²
            Typical range: 0-350 W/m² for solar radiation
    """
    # Convert J/m² to W/m²: J/m² / (time_period_s) = W/m²
    radiation_watts_per_m2 = radiation_joules_per_m2 / time_period_seconds

    return radiation_watts_per_m2


def convert_carbon_flux_units(flux_gc_m2_day: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert carbon fluxes to standard CARDAMOM units.

    Converts carbon fluxes from gC/m²/day to KgC/Km²/sec for consistency
    with CARDAMOM internal calculations.

    Args:
        flux_gc_m2_day: Carbon flux in gC/m²/day
            Typical range: -50 to +50 gC/m²/day

    Returns:
        Carbon flux in KgC/Km²/sec
            Typical range: -5.8e-4 to +5.8e-4 KgC/Km²/sec

    Unit Conversion:
        1 gC/m²/day = 1e3 g/kg * 1e6 m²/km² / (24*3600 s/day) KgC/Km²/sec
                    = 1e9 / 86400 KgC/Km²/sec
                    ≈ 11.574 KgC/Km²/sec
    """
    # Convert gC/m²/day to KgC/Km²/sec
    conversion_factor = 1e3 / (24 * 3600)  # 1e3 for unit conversion, divide by seconds per day
    flux_kgc_km2_sec = flux_gc_m2_day * conversion_factor

    return flux_kgc_km2_sec


def calculate_photosynthetically_active_radiation(solar_radiation_wm2: Union[float, np.ndarray],
                                                par_fraction: float = 0.45) -> Union[float, np.ndarray]:
    """
    Calculate Photosynthetically Active Radiation from solar radiation.

    PAR represents the portion of solar radiation (400-700 nm) that plants
    can use for photosynthesis, typically about 45% of total solar radiation.

    Args:
        solar_radiation_wm2: Solar radiation in W/m²
            Typical range: 0-350 W/m²
        par_fraction: Fraction of solar radiation that is PAR (default: 0.45)

    Returns:
        PAR in µmol photons/m²/s
            Typical range: 0-2000 µmol photons/m²/s

    Conversion:
        1 W/m² PAR ≈ 4.6 µmol photons/m²/s (approximate conversion factor)
    """
    # Step 1: Calculate PAR portion of solar radiation
    par_radiation_wm2 = solar_radiation_wm2 * par_fraction

    # Step 2: Convert W/m² to µmol photons/m²/s
    # Approximate conversion: 1 W/m² PAR ≈ 4.6 µmol photons/m²/s
    conversion_factor = 4.6  # µmol photons/m²/s per W/m²
    par_umol_m2_s = par_radiation_wm2 * conversion_factor

    return par_umol_m2_s


def validate_temperature_data(temperature_kelvin: Union[float, np.ndarray],
                            variable_name: str = "temperature") -> Dict[str, Any]:
    """
    Validate temperature data for physical reasonableness.

    Checks temperature values against known physical limits and typical
    meteorological ranges to identify potential data quality issues.

    Args:
        temperature_kelvin: Temperature values in Kelvin
        variable_name: Name of temperature variable for error messages

    Returns:
        Dictionary with validation results including:
        - 'status': 'pass', 'warning', or 'fail'
        - 'min_value': Minimum temperature found
        - 'max_value': Maximum temperature found
        - 'num_below_physical_min': Count of values below physical minimum
        - 'num_above_physical_max': Count of values above physical maximum
        - 'messages': List of validation messages
    """
    validation_results = {
        'variable_name': variable_name,
        'status': 'pass',
        'min_value': float(np.nanmin(temperature_kelvin)),
        'max_value': float(np.nanmax(temperature_kelvin)),
        'num_below_physical_min': 0,
        'num_above_physical_max': 0,
        'messages': []
    }

    # Physical limits for Earth's surface temperature
    absolute_min_kelvin = 173.0  # -100°C (below lowest recorded Earth temperature)
    absolute_max_kelvin = 333.0  # 60°C (above typical meteorological maximum)

    # Check for values below physical minimum
    below_min_mask = temperature_kelvin < absolute_min_kelvin
    num_below_min = np.sum(below_min_mask)
    validation_results['num_below_physical_min'] = int(num_below_min)

    if num_below_min > 0:
        validation_results['status'] = 'fail'
        validation_results['messages'].append(
            f"{num_below_min} {variable_name} values below -100°C detected. "
            f"Check data units - temperature should be in Kelvin."
        )

    # Check for values above physical maximum
    above_max_mask = temperature_kelvin > absolute_max_kelvin
    num_above_max = np.sum(above_max_mask)
    validation_results['num_above_physical_max'] = int(num_above_max)

    if num_above_max > 0:
        validation_results['status'] = 'fail'
        validation_results['messages'].append(
            f"{num_above_max} {variable_name} values above 60°C detected. "
            f"Check data units and spatial domain."
        )

    # Check for reasonable meteorological ranges (warnings only)
    reasonable_min_kelvin = 200.0  # -73°C
    reasonable_max_kelvin = 320.0  # 47°C

    num_below_reasonable = np.sum(temperature_kelvin < reasonable_min_kelvin)
    num_above_reasonable = np.sum(temperature_kelvin > reasonable_max_kelvin)

    if num_below_reasonable > 0 and validation_results['status'] == 'pass':
        validation_results['status'] = 'warning'
        validation_results['messages'].append(
            f"{num_below_reasonable} {variable_name} values below -73°C. "
            f"This is very cold but not impossible."
        )

    if num_above_reasonable > 0 and validation_results['status'] == 'pass':
        validation_results['status'] = 'warning'
        validation_results['messages'].append(
            f"{num_above_reasonable} {variable_name} values above 47°C. "
            f"This is very hot but not impossible."
        )

    return validation_results


def validate_precipitation_data(precipitation_mm_day: Union[float, np.ndarray]) -> Dict[str, Any]:
    """
    Validate precipitation data for physical reasonableness.

    Args:
        precipitation_mm_day: Precipitation values in mm/day

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'variable_name': 'precipitation',
        'status': 'pass',
        'min_value': float(np.nanmin(precipitation_mm_day)),
        'max_value': float(np.nanmax(precipitation_mm_day)),
        'num_negative': 0,
        'num_extreme': 0,
        'messages': []
    }

    # Check for negative values (physically impossible)
    negative_mask = precipitation_mm_day < 0
    num_negative = np.sum(negative_mask)
    validation_results['num_negative'] = int(num_negative)

    if num_negative > 0:
        validation_results['status'] = 'fail'
        validation_results['messages'].append(
            f"{num_negative} negative precipitation values detected. "
            f"Precipitation cannot be negative."
        )

    # Check for extremely high values (>500 mm/day)
    extreme_threshold = 500.0  # mm/day
    extreme_mask = precipitation_mm_day > extreme_threshold
    num_extreme = np.sum(extreme_mask)
    validation_results['num_extreme'] = int(num_extreme)

    if num_extreme > 0:
        validation_results['status'] = 'warning' if validation_results['status'] == 'pass' else validation_results['status']
        validation_results['messages'].append(
            f"{num_extreme} precipitation values above {extreme_threshold} mm/day detected. "
            f"This is extremely high precipitation - verify data quality."
        )

    return validation_results


def calculate_growing_degree_days(daily_temp_max_celsius: Union[float, np.ndarray],
                                daily_temp_min_celsius: Union[float, np.ndarray],
                                base_temperature_celsius: float = 10.0) -> Union[float, np.ndarray]:
    """
    Calculate Growing Degree Days for vegetation phenology.

    Growing Degree Days (GDD) quantify heat accumulation for plant development
    and are used in CARDAMOM for phenology modeling.

    Args:
        daily_temp_max_celsius: Daily maximum temperature in °C
        daily_temp_min_celsius: Daily minimum temperature in °C
        base_temperature_celsius: Base temperature for GDD calculation (default: 10°C)

    Returns:
        Growing degree days (GDD) in degree-days
            Range: 0 to ~40 degree-days per day in warm climates

    Formula:
        GDD = max(0, (T_max + T_min)/2 - T_base)
    """
    # Calculate daily mean temperature
    daily_mean_temperature = (daily_temp_max_celsius + daily_temp_min_celsius) / 2.0

    # Calculate GDD (cannot be negative)
    growing_degree_days = np.maximum(0, daily_mean_temperature - base_temperature_celsius)

    return growing_degree_days