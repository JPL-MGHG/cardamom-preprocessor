"""
Physical Constants and Unit Conversions for CARDAMOM Preprocessing

This module provides physical constants and unit conversion utilities used
throughout the CARDAMOM preprocessing system. Values are consistent with
atmospheric science standards and MATLAB implementations.

Scientific Context:
Accurate physical constants and unit conversions are essential for maintaining
scientific consistency across atmospheric modeling, carbon cycle calculations,
and data processing workflows in CARDAMOM.

References:
- CODATA 2018 internationally recommended values
- MATLAB Source: Various CARDAMOM scripts for conversion factors
- NIST Physical Constants: https://physics.nist.gov/cuu/Constants/
"""

import numpy as np
from typing import Union, Optional


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class PhysicalConstants:
    """
    Collection of physical constants used in atmospheric and carbon cycle modeling.

    All values follow CODATA 2018 recommendations unless otherwise specified
    for compatibility with existing MATLAB implementations.
    """

    # Universal constants
    AVOGADRO_NUMBER = 6.02214076e23  # mol⁻¹
    BOLTZMANN_CONSTANT = 1.380649e-23  # J K⁻¹
    UNIVERSAL_GAS_CONSTANT = 8.314462618  # J mol⁻¹ K⁻¹
    STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8  # W m⁻² K⁻⁴

    # Earth and atmospheric constants
    EARTH_RADIUS = 6.371e6  # m (mean radius)
    STANDARD_GRAVITY = 9.80665  # m s⁻²
    STANDARD_PRESSURE = 101325.0  # Pa (1 atm)
    STANDARD_TEMPERATURE = 273.15  # K (0°C)

    # Gas-specific constants (J kg⁻¹ K⁻¹)
    GAS_CONSTANT_DRY_AIR = 287.04  # Specific gas constant for dry air
    GAS_CONSTANT_WATER_VAPOR = 461.5  # Specific gas constant for water vapor

    # Molecular weights (g mol⁻¹)
    MOLECULAR_WEIGHT_DRY_AIR = 28.9647  # Dry air
    MOLECULAR_WEIGHT_WATER = 18.01528  # H₂O
    MOLECULAR_WEIGHT_CO2 = 44.0095  # CO₂
    MOLECULAR_WEIGHT_CARBON = 12.011  # C
    MOLECULAR_WEIGHT_NITROGEN = 28.014  # N₂
    MOLECULAR_WEIGHT_OXYGEN = 31.998  # O₂

    # Density constants (kg m⁻³ at STP)
    DENSITY_DRY_AIR_STP = 1.2922  # kg m⁻³ at 0°C, 1 atm
    DENSITY_WATER_LIQUID = 1000.0  # kg m⁻³ at 4°C

    # Solar and radiation constants
    SOLAR_CONSTANT = 1361.0  # W m⁻² (total solar irradiance)

    # Carbon cycle constants
    CARBON_ATOMIC_WEIGHT = 12.011  # g mol⁻¹
    CO2_TO_CARBON_RATIO = MOLECULAR_WEIGHT_CO2 / MOLECULAR_WEIGHT_CARBON  # 3.664


# =============================================================================
# UNIT CONVERSION FUNCTIONS
# =============================================================================

def temperature_kelvin_to_celsius(temperature_kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert temperature from Kelvin to Celsius.

    Args:
        temperature_kelvin: Temperature in Kelvin

    Returns:
        Temperature in Celsius
    """
    return np.asarray(temperature_kelvin) - PhysicalConstants.STANDARD_TEMPERATURE


def temperature_celsius_to_kelvin(temperature_celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert temperature from Celsius to Kelvin.

    Args:
        temperature_celsius: Temperature in Celsius

    Returns:
        Temperature in Kelvin
    """
    return np.asarray(temperature_celsius) + PhysicalConstants.STANDARD_TEMPERATURE


def pressure_pa_to_hpa(pressure_pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert pressure from Pascals to hectopascals (millibars).

    Args:
        pressure_pa: Pressure in Pascals

    Returns:
        Pressure in hectopascals (hPa)
    """
    return np.asarray(pressure_pa) / 100.0


def pressure_hpa_to_pa(pressure_hpa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert pressure from hectopascals to Pascals.

    Args:
        pressure_hpa: Pressure in hectopascals (hPa)

    Returns:
        Pressure in Pascals
    """
    return np.asarray(pressure_hpa) * 100.0


def pressure_pa_to_kpa(pressure_pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert pressure from Pascals to kilopascals.

    Used in MATLAB SCIFUN_H2O_SATURATION_PRESSURE function.

    Args:
        pressure_pa: Pressure in Pascals

    Returns:
        Pressure in kilopascals (kPa)
    """
    return np.asarray(pressure_pa) / 1000.0


def precipitation_m_to_mm(precipitation_m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert precipitation from meters to millimeters.

    Common conversion for ERA5 precipitation data which is provided in meters.

    Args:
        precipitation_m: Precipitation in meters

    Returns:
        Precipitation in millimeters
    """
    return np.asarray(precipitation_m) * 1000.0


def precipitation_m_per_s_to_mm_per_day(precip_rate_m_s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert precipitation rate from m/s to mm/day.

    Standard conversion for ERA5 precipitation rate data.

    Args:
        precip_rate_m_s: Precipitation rate in m/s

    Returns:
        Precipitation rate in mm/day
    """
    seconds_per_day = 86400.0  # 24 * 60 * 60
    return np.asarray(precip_rate_m_s) * 1000.0 * seconds_per_day


def radiation_j_m2_to_w_m2(radiation_j_m2: Union[float, np.ndarray],
                          time_period_seconds: float = 3600.0) -> Union[float, np.ndarray]:
    """
    Convert radiation from J/m² to W/m².

    ERA5 radiation data is often provided as accumulated energy (J/m²) over
    a time period, typically hourly (3600 seconds).

    Args:
        radiation_j_m2: Radiation in J/m²
        time_period_seconds: Time period over which energy was accumulated (default: 3600s)

    Returns:
        Radiation power in W/m²
    """
    return np.asarray(radiation_j_m2) / time_period_seconds


def carbon_flux_gc_m2_s_to_umol_m2_s(carbon_flux_gc_m2_s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert carbon flux from gC/m²/s to µmol C/m²/s.

    Standard conversion for carbon flux measurements and modeling.

    Args:
        carbon_flux_gc_m2_s: Carbon flux in gC/m²/s

    Returns:
        Carbon flux in µmol C/m²/s
    """
    # 1 g C = 1000 mg C = 1000000 µg C
    # 1 mol C = 12.011 g C (atomic weight)
    # 1 µmol C = 12.011 µg C
    conversion_factor = 1e6 / PhysicalConstants.MOLECULAR_WEIGHT_CARBON  # µmol/g
    return np.asarray(carbon_flux_gc_m2_s) * conversion_factor


def carbon_flux_gc_m2_day_to_kg_c_km2_s(carbon_flux_gc_m2_day: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert carbon flux from gC/m²/day to kg C/km²/s.

    Common conversion for GeosChem atmospheric model inputs.
    Used in MATLAB diurnal flux processing.

    Args:
        carbon_flux_gc_m2_day: Carbon flux in gC/m²/day

    Returns:
        Carbon flux in kg C/km²/s
    """
    # Convert g to kg: / 1000
    # Convert m² to km²: * 1e6
    # Convert day to seconds: / 86400
    conversion_factor = 1e6 / (1000.0 * 86400.0)  # (km²/m²) / (g/kg) / (s/day)
    return np.asarray(carbon_flux_gc_m2_day) * conversion_factor


def co2_flux_to_carbon_flux(co2_flux: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert CO₂ flux to carbon flux by molecular weight ratio.

    Args:
        co2_flux: CO₂ flux in any mass-based units

    Returns:
        Carbon flux in same units as input
    """
    return np.asarray(co2_flux) / PhysicalConstants.CO2_TO_CARBON_RATIO


def carbon_flux_to_co2_flux(carbon_flux: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert carbon flux to CO₂ flux by molecular weight ratio.

    Args:
        carbon_flux: Carbon flux in any mass-based units

    Returns:
        CO₂ flux in same units as input
    """
    return np.asarray(carbon_flux) * PhysicalConstants.CO2_TO_CARBON_RATIO


def vapor_pressure_pa_to_kpa(vapor_pressure_pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert vapor pressure from Pascals to kilopascals.

    Used for consistency with MATLAB SCIFUN functions.

    Args:
        vapor_pressure_pa: Vapor pressure in Pascals

    Returns:
        Vapor pressure in kilopascals
    """
    return np.asarray(vapor_pressure_pa) / 1000.0


def specific_humidity_to_mixing_ratio(specific_humidity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert specific humidity to mixing ratio.

    Specific humidity (q) = mass of water vapor / total mass of moist air
    Mixing ratio (w) = mass of water vapor / mass of dry air
    Relationship: w = q / (1 - q)

    Args:
        specific_humidity: Specific humidity (kg/kg)

    Returns:
        Mixing ratio (kg water vapor / kg dry air)
    """
    q = np.asarray(specific_humidity)
    return q / (1.0 - q)


def mixing_ratio_to_specific_humidity(mixing_ratio: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert mixing ratio to specific humidity.

    Relationship: q = w / (1 + w)

    Args:
        mixing_ratio: Mixing ratio (kg water vapor / kg dry air)

    Returns:
        Specific humidity (kg/kg)
    """
    w = np.asarray(mixing_ratio)
    return w / (1.0 + w)


def degrees_to_radians(degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert degrees to radians.

    Args:
        degrees: Angle in degrees

    Returns:
        Angle in radians
    """
    return np.asarray(degrees) * np.pi / 180.0


def radians_to_degrees(radians: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert radians to degrees.

    Args:
        radians: Angle in radians

    Returns:
        Angle in degrees
    """
    return np.asarray(radians) * 180.0 / np.pi


# =============================================================================
# ATMOSPHERIC CALCULATIONS WITH UNIT CONVERSIONS
# =============================================================================

def calculate_air_density(temperature_kelvin: Union[float, np.ndarray],
                         pressure_pa: Union[float, np.ndarray],
                         specific_humidity: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
    """
    Calculate air density using ideal gas law with optional humidity correction.

    Args:
        temperature_kelvin: Air temperature in Kelvin
        pressure_pa: Atmospheric pressure in Pascals
        specific_humidity: Specific humidity (kg/kg), optional

    Returns:
        Air density in kg/m³
    """
    T = np.asarray(temperature_kelvin)
    P = np.asarray(pressure_pa)

    if specific_humidity is not None:
        # Virtual temperature correction for moist air
        q = np.asarray(specific_humidity)
        T_virtual = T * (1.0 + 0.61 * q)
        density = P / (PhysicalConstants.GAS_CONSTANT_DRY_AIR * T_virtual)
    else:
        # Dry air density
        density = P / (PhysicalConstants.GAS_CONSTANT_DRY_AIR * T)

    return density


def calculate_scale_height(temperature_kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate atmospheric scale height.

    Scale height is the height over which atmospheric pressure decreases by
    a factor of e (approximately 2.718).

    Args:
        temperature_kelvin: Temperature in Kelvin

    Returns:
        Scale height in meters
    """
    T = np.asarray(temperature_kelvin)

    # H = RT / (Mg) where M is molar mass of air
    scale_height = (PhysicalConstants.UNIVERSAL_GAS_CONSTANT * T) / \
                  (PhysicalConstants.MOLECULAR_WEIGHT_DRY_AIR * 1e-3 * PhysicalConstants.STANDARD_GRAVITY)

    return scale_height