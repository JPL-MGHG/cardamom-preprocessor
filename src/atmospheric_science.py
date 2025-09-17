"""
Atmospheric Science Calculations for CARDAMOM Preprocessing

This module provides atmospheric science utility functions that replicate and extend
MATLAB functionality for water vapor, pressure, humidity, and radiation calculations.
All functions include explicit references to the original MATLAB source code.

Scientific Context:
These calculations are essential for atmospheric modeling and carbon cycle analysis,
providing the meteorological foundations needed for ecosystem modeling in CARDAMOM.

References:
- MATLAB Source: /Users/shah/Desktop/Development/ghg/CARDAMOM/MATLAB/sci_fun/
- MATLAB Source: /Users/shah/Desktop/Development/ghg/CARDAMOM/MATLAB/prototypes/
"""

import numpy as np
from typing import Union, Optional
import warnings


def saturation_pressure_water_matlab(temperature_celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate saturation pressure of water vapor using exact MATLAB SCIFUN formula.

    This function replicates the MATLAB SCIFUN_H2O_SATURATION_PRESSURE function
    with identical numerical results for consistency with existing CARDAMOM workflows.

    MATLAB Source Reference:
    File: /MATLAB/sci_fun/SCIFUN_H2O_SATURATION_PRESSURE.m
    Lines: 19, 34
    Formula: VPSAT=6.11*10.^(7.5*T./(237.3+T))./10

    Scientific Background:
    Uses NOAA method from http://www.srh.noaa.gov/epz/?n=wxcalc_vaporpressure
    This is a simplified Magnus-type formula suitable for meteorological applications
    in the temperature range commonly encountered in atmospheric modeling.

    Args:
        temperature_celsius: Temperature in degrees Celsius (scalar or array)
            Typical range: -40 to 50°C for atmospheric applications
            Source: ERA5 reanalysis or meteorological observations

    Returns:
        Saturation vapor pressure in kilopascals (kPa)
            Typical range: 0.1-12 kPa for atmospheric temperatures
            Physical interpretation: Maximum water vapor pressure at given temperature

    Notes:
        - Formula from NOAA website (MATLAB comment line 6-7)
        - Result in kPa (MATLAB comment line 15-17)
        - For VPD calculations, use relationship on MATLAB lines 22-24

    Example:
        >>> # Standard atmospheric conditions
        >>> temp_c = np.array([0, 10, 20, 30])  # 0°C to 30°C
        >>> vpsat = saturation_pressure_water_matlab(temp_c)
        >>> # Expected: [0.611, 1.228, 2.338, 4.243] kPa approximately
    """

    T = np.asarray(temperature_celsius)

    # Exact MATLAB formula: VPSAT=6.11*10.^(7.5*T./(237.3+T))./10
    saturation_pressure_kpa = 6.11 * np.power(10, 7.5 * T / (237.3 + T)) / 10

    return saturation_pressure_kpa


def calculate_vapor_pressure_deficit_matlab(temperature_max_celsius: Union[float, np.ndarray],
                                          dewpoint_temperature_celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate Vapor Pressure Deficit using exact MATLAB implementation.

    This function replicates the VPD calculation from the main CARDAMOM MATLAB script
    using the SCIFUN saturation pressure function for consistency with existing workflows.

    MATLAB Source Reference:
    File: /MATLAB/prototypes/CARDAMOM_MAPS_05deg_DATASETS_JUL24.m
    Line: 202
    Formula: VPD=(SCIFUN_H2O_SATURATION_PRESSURE(ET2M.datamax) - SCIFUN_H2O_SATURATION_PRESSURE(ED2M.datamax))*10

    Scientific Background:
    VPD represents atmospheric moisture demand and is crucial for understanding
    plant water stress and photosynthesis rates. It quantifies how much additional
    water vapor the air can hold at a given temperature.

    Args:
        temperature_max_celsius: Daily maximum temperature in Celsius
            Typical range: -20 to 50°C for global land areas
            Source: ERA5 2m_temperature maximum
        dewpoint_temperature_celsius: Dewpoint temperature in Celsius
            Typical range: -30 to 30°C for global land areas
            Source: ERA5 2m_dewpoint_temperature

    Returns:
        Vapor pressure deficit in hectopascals (hPa)
            Typical range: 0-60 hPa for terrestrial ecosystems
            Physical interpretation:
            - Low VPD (0-10 hPa): High humidity, low atmospheric demand
            - Medium VPD (10-30 hPa): Moderate atmospheric demand
            - High VPD (>30 hPa): Low humidity, high atmospheric demand

    Notes:
        - Uses MATLAB SCIFUN function for saturation pressure calculation
        - Multiplies by 10 to convert kPa to hPa (MATLAB line 202)
        - Assumes T_max >= T_dewpoint for physical consistency

    Example:
        >>> # Summer conditions: warm and moderately humid
        >>> t_max = np.array([25, 30, 35])  # Maximum temperature °C
        >>> t_dew = np.array([15, 18, 20])  # Dewpoint temperature °C
        >>> vpd = calculate_vapor_pressure_deficit_matlab(t_max, t_dew)
        >>> # Expected: [11.7, 19.4, 29.8] hPa approximately
    """

    T_max = np.asarray(temperature_max_celsius)
    T_dew = np.asarray(dewpoint_temperature_celsius)

    # MATLAB line 202: VPD=(SCIFUN_H2O_SATURATION_PRESSURE(ET2M.datamax) - SCIFUN_H2O_SATURATION_PRESSURE(ED2M.datamax))*10
    vpsat_at_tmax_kpa = saturation_pressure_water_matlab(T_max)
    vpsat_at_tdew_kpa = saturation_pressure_water_matlab(T_dew)

    # Convert kPa to hPa by multiplying by 10 (MATLAB formula)
    vapor_pressure_deficit_hpa = (vpsat_at_tmax_kpa - vpsat_at_tdew_kpa) * 10

    # Validate physical consistency
    if np.any(vapor_pressure_deficit_hpa < 0):
        warnings.warn("VPD cannot be negative. Check that T_max >= T_dewpoint")
        vapor_pressure_deficit_hpa = np.maximum(vapor_pressure_deficit_hpa, 0)

    return vapor_pressure_deficit_hpa


def humidity_ratio_from_vapor_pressure(vapor_pressure_kpa: Union[float, np.ndarray],
                                     atmospheric_pressure_kpa: float = 101.325) -> Union[float, np.ndarray]:
    """
    Calculate humidity ratio (mixing ratio) from vapor pressure.

    This function implements the humidity ratio calculation referenced in the
    MATLAB SCIFUN comments for use in atmospheric modeling applications.

    MATLAB Source Reference:
    File: /MATLAB/sci_fun/SCIFUN_H2O_SATURATION_PRESSURE.m
    Lines: 22-24
    Formula: xs=0.6*VPSAT/(101-VPSAT)
    Reference: http://www.engineeringtoolbox.com/humidity-ratio-air-d_686.html

    Scientific Background:
    Humidity ratio (or mixing ratio) is the mass of water vapor per unit mass
    of dry air. It is useful for atmospheric calculations where mass conservation
    is important, such as in atmospheric transport models.

    Args:
        vapor_pressure_kpa: Water vapor pressure in kPa
            Typical range: 0.1-6 kPa for atmospheric conditions
        atmospheric_pressure_kpa: Atmospheric pressure in kPa
            Default: 101.325 kPa (standard sea level pressure)
            Typical range: 50-105 kPa (high altitude to sea level)

    Returns:
        Humidity ratio in kg water vapor / kg dry air
            Typical range: 0.001-0.030 for atmospheric conditions
            Physical interpretation: Mass of water per mass of dry air

    Notes:
        - MATLAB uses simplified formula with factor 0.6 (line 23)
        - Assumes atmospheric pressure in kPa units
        - Valid for typical atmospheric conditions

    Example:
        >>> # Humid tropical conditions
        >>> vapor_pressure = np.array([2.0, 3.0, 4.0])  # kPa
        >>> humidity_ratio = humidity_ratio_from_vapor_pressure(vapor_pressure)
        >>> # Expected: [0.012, 0.018, 0.025] kg/kg approximately
    """

    vp = np.asarray(vapor_pressure_kpa)

    # MATLAB formula: xs=0.6*VPSAT/(101-VPSAT)
    # Note: MATLAB assumes pressure in kPa, using 101 kPa as reference
    humidity_ratio_kg_per_kg = 0.6 * vp / (atmospheric_pressure_kpa - vp)

    return humidity_ratio_kg_per_kg


def radiation_to_par_conversion(solar_radiation_w_m2: Union[float, np.ndarray],
                              par_fraction: float = 0.45) -> Union[float, np.ndarray]:
    """
    Convert solar radiation to Photosynthetically Active Radiation (PAR).

    This function converts broadband solar radiation to PAR for use in
    photosynthesis calculations within CARDAMOM ecosystem modeling.

    Scientific Background:
    PAR represents the portion of solar radiation (400-700 nm wavelength)
    that plants can use for photosynthesis. This conversion is essential
    for calculating light limitation in ecosystem carbon models.

    Args:
        solar_radiation_w_m2: Downward solar radiation in W/m²
            Typical range: 0-1400 W/m² (varies with latitude, season, clouds)
            Source: ERA5 surface_solar_radiation_downwards
        par_fraction: Fraction of solar radiation that is PAR
            Default: 0.45 (typical value for clear sky conditions)
            Range: 0.4-0.5 depending on atmospheric conditions

    Returns:
        PAR in µmol photons/m²/s
            Typical range: 0-2500 µmol/m²/s
            Light saturation for most plants: ~1500 µmol/m²/s

    Notes:
        - Standard conversion: 1 W/m² PAR ≈ 4.57 µmol photons/m²/s
        - PAR fraction varies with cloud conditions and solar zenith angle
        - Used in CARDAMOM for photosynthesis light response calculations

    Example:
        >>> # Typical diurnal solar radiation cycle
        >>> solar_rad = np.array([0, 200, 800, 1000, 600, 100, 0])  # W/m²
        >>> par = radiation_to_par_conversion(solar_rad)
        >>> # Expected: [0, 411, 1643, 2054, 1232, 206, 0] µmol/m²/s
    """

    solar_rad = np.asarray(solar_radiation_w_m2)

    # Standard conversion factors for atmospheric science
    watts_to_umol_conversion = 4.57  # µmol photons/m²/s per W/m² PAR

    # Calculate PAR
    par_w_m2 = solar_rad * par_fraction
    par_umol_m2_s = par_w_m2 * watts_to_umol_conversion

    # Ensure non-negative values
    par_umol_m2_s = np.maximum(par_umol_m2_s, 0)

    return par_umol_m2_s


def air_density_from_temperature_pressure(temperature_kelvin: Union[float, np.ndarray],
                                        pressure_pa: Union[float, np.ndarray],
                                        humidity_ratio: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
    """
    Calculate air density from temperature and pressure using ideal gas law.

    This function calculates dry or moist air density for atmospheric calculations
    in CARDAMOM preprocessing, including unit conversions and flux calculations.

    Scientific Background:
    Air density is needed for converting between mass and volume flux units,
    and for atmospheric transport calculations. Humidity affects density
    through the lower molecular weight of water vapor compared to dry air.

    Args:
        temperature_kelvin: Air temperature in Kelvin
            Typical range: 200-320 K for atmospheric applications
            Source: ERA5 2m_temperature
        pressure_pa: Atmospheric pressure in Pascals
            Typical range: 50000-105000 Pa (high altitude to sea level)
            Source: ERA5 surface_pressure
        humidity_ratio: Water vapor mixing ratio in kg/kg (optional)
            Typical range: 0.001-0.030 for atmospheric conditions
            If None, calculates dry air density

    Returns:
        Air density in kg/m³
            Typical range: 0.5-1.4 kg/m³ for atmospheric conditions
            Physical interpretation: Mass of air per unit volume

    Notes:
        - Uses ideal gas law: ρ = P/(R*T)
        - Dry air gas constant: R_d = 287.04 J/(kg·K)
        - Accounts for water vapor if humidity_ratio provided
        - Essential for atmospheric flux calculations

    Example:
        >>> # Standard atmospheric conditions
        >>> temp_k = np.array([273.15, 293.15, 313.15])  # 0°C, 20°C, 40°C
        >>> pressure = np.array([101325, 101325, 101325])  # Sea level pressure
        >>> density = air_density_from_temperature_pressure(temp_k, pressure)
        >>> # Expected: [1.29, 1.20, 1.13] kg/m³ approximately
    """

    T = np.asarray(temperature_kelvin)
    P = np.asarray(pressure_pa)

    # Gas constants (J/(kg·K))
    R_dry_air = 287.04  # Specific gas constant for dry air
    R_water_vapor = 461.5  # Specific gas constant for water vapor

    if humidity_ratio is not None:
        q = np.asarray(humidity_ratio)
        # Virtual temperature accounts for water vapor effect
        # R_virtual = R_d * (1 + 0.61*q) approximately
        virtual_temperature_factor = 1 + 0.61 * q
        air_density_kg_m3 = P / (R_dry_air * T * virtual_temperature_factor)
    else:
        # Dry air density
        air_density_kg_m3 = P / (R_dry_air * T)

    return air_density_kg_m3