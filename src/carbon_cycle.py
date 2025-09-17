"""
Carbon Cycle and Ecosystem Modeling Functions for CARDAMOM Preprocessing

This module provides carbon cycle calculation functions for ecosystem modeling,
including photosynthesis, respiration, fire emissions, and carbon balance
calculations used in CARDAMOM preprocessing.

Scientific Context:
These functions implement fundamental carbon cycle processes essential for
ecosystem modeling, including primary productivity, ecosystem respiration,
fire emissions, and net ecosystem exchange calculations.

References:
- MATLAB Source: /Users/shah/Desktop/Development/ghg/CARDAMOM/MATLAB/prototypes/
- CARDAMOM Framework: Bloom, A. A., et al. (2016). Nature Geoscience, 9(10), 796-800
"""

import numpy as np
from typing import Union, Optional, Dict, Any
import warnings
from units_constants import PhysicalConstants, carbon_flux_to_co2_flux, co2_flux_to_carbon_flux


def calculate_photosynthetically_active_radiation(solar_radiation_w_m2: Union[float, np.ndarray],
                                                par_fraction: float = 0.45) -> Union[float, np.ndarray]:
    """
    Calculate Photosynthetically Active Radiation (PAR) from solar radiation.

    PAR represents the spectral range (400-700 nm) of solar radiation that
    photosynthetic organisms can use in the process of photosynthesis.

    Scientific Background:
    PAR is essential for calculating light limitation in photosynthesis models.
    Approximately 45% of solar radiation falls within the PAR range under
    clear sky conditions.

    Args:
        solar_radiation_w_m2: Downward solar radiation in W/m²
            Typical range: 0-1400 W/m² (varies with solar zenith angle, clouds)
            Source: ERA5 surface_solar_radiation_downwards
        par_fraction: Fraction of solar radiation that is PAR
            Default: 0.45 (typical value for clear sky)
            Range: 0.4-0.5 depending on atmospheric conditions

    Returns:
        PAR in µmol photons/m²/s
            Typical range: 0-2500 µmol/m²/s
            Light saturation for C3 plants: ~1000-2000 µmol/m²/s
            Light saturation for C4 plants: ~1500-2500 µmol/m²/s

    Example:
        >>> # Typical diurnal solar radiation cycle
        >>> solar_rad = np.array([0, 200, 800, 1000, 600, 100, 0])  # W/m²
        >>> par = calculate_photosynthetically_active_radiation(solar_rad)
        >>> # Expected: [0, 411, 1643, 2054, 1232, 206, 0] µmol/m²/s
    """

    solar_rad = np.asarray(solar_radiation_w_m2)

    # Standard conversion factors for PAR calculations
    watts_to_umol_conversion = 4.57  # µmol photons/m²/s per W/m² PAR

    # Calculate PAR
    par_w_m2 = solar_rad * par_fraction
    par_umol_m2_s = par_w_m2 * watts_to_umol_conversion

    # Ensure non-negative values
    par_umol_m2_s = np.maximum(par_umol_m2_s, 0)

    return par_umol_m2_s


def calculate_gross_primary_productivity_light_response(par_umol_m2_s: Union[float, np.ndarray],
                                                       par_max_umol_m2_s: float = 2000.0,
                                                       alpha_quantum_efficiency: float = 0.08) -> Union[float, np.ndarray]:
    """
    Calculate GPP light response using rectangular hyperbola model.

    This function implements a simple light response curve for gross primary
    productivity based on photosynthetically active radiation.

    Scientific Background:
    The rectangular hyperbola model describes how photosynthesis responds to
    light availability, with initial linear response at low light and saturation
    at high light levels. This is a fundamental relationship in ecosystem modeling.

    Args:
        par_umol_m2_s: Photosynthetically active radiation in µmol/m²/s
            Typical range: 0-2500 µmol/m²/s for terrestrial ecosystems
        par_max_umol_m2_s: Light saturation point in µmol/m²/s
            Default: 2000 µmol/m²/s (typical for C3 vegetation)
            C3 plants: 1000-2000 µmol/m²/s
            C4 plants: 1500-2500 µmol/m²/s
        alpha_quantum_efficiency: Initial quantum efficiency (dimensionless)
            Default: 0.08 (typical for C3 plants)
            Range: 0.04-0.12 depending on vegetation type

    Returns:
        Relative GPP response (0-1, dimensionless)
            0: No photosynthesis (darkness)
            1: Light-saturated photosynthesis
            Physical interpretation: Fraction of maximum potential GPP

    Notes:
        - Uses rectangular hyperbola: GPP = (α * PAR) / (1 + α * PAR / GPP_max)
        - This is a simplified model; actual ecosystem models use more complex formulations
        - Results should be multiplied by maximum GPP capacity for absolute fluxes

    Example:
        >>> # Light response across day
        >>> par_values = np.array([0, 500, 1000, 1500, 2000, 2500])  # µmol/m²/s
        >>> gpp_response = calculate_gross_primary_productivity_light_response(par_values)
        >>> # Expected: [0.0, 0.67, 0.89, 0.96, 1.0, 1.0] approximately
    """

    par = np.asarray(par_umol_m2_s)

    # Rectangular hyperbola light response model
    # GPP_rel = (α * PAR) / (1 + α * PAR / PAR_max)
    numerator = alpha_quantum_efficiency * par
    denominator = 1.0 + (alpha_quantum_efficiency * par / par_max_umol_m2_s)

    gpp_relative_response = numerator / denominator

    # Ensure values are between 0 and 1
    gpp_relative_response = np.clip(gpp_relative_response, 0.0, 1.0)

    return gpp_relative_response


def calculate_ecosystem_respiration_temperature_response(temperature_celsius: Union[float, np.ndarray],
                                                       base_temperature_celsius: float = 15.0,
                                                       q10_factor: float = 2.0) -> Union[float, np.ndarray]:
    """
    Calculate ecosystem respiration temperature response using Q10 model.

    This function implements the exponential temperature response of ecosystem
    respiration commonly used in carbon cycle modeling.

    Scientific Background:
    Ecosystem respiration includes both autotrophic (plant) and heterotrophic
    (soil microbial) respiration. Both processes show exponential temperature
    dependence, typically doubling every 10°C (Q10 ≈ 2).

    Args:
        temperature_celsius: Air or soil temperature in Celsius
            Typical range: -10 to 40°C for terrestrial ecosystems
            Source: ERA5 2m_temperature or skin_temperature
        base_temperature_celsius: Reference temperature in Celsius
            Default: 15°C (typical temperate ecosystem reference)
            Range: 10-20°C depending on ecosystem type
        q10_factor: Temperature sensitivity coefficient
            Default: 2.0 (doubling every 10°C)
            Typical range: 1.5-3.0 for different ecosystems and seasons

    Returns:
        Relative respiration response (dimensionless)
            1.0 at base temperature
            2.0 at base temperature + 10°C (for Q10 = 2)
            Physical interpretation: Multiplier for base respiration rate

    Notes:
        - Uses Q10 model: R_rel = Q10^((T - T_base) / 10)
        - This is the most common temperature response model in ecosystem science
        - Results should be multiplied by base respiration rate for absolute fluxes

    Example:
        >>> # Temperature response across seasonal range
        >>> temperatures = np.array([-5, 5, 15, 25, 35])  # °C
        >>> resp_response = calculate_ecosystem_respiration_temperature_response(temperatures)
        >>> # Expected: [0.25, 0.5, 1.0, 2.0, 4.0] for Q10=2, base=15°C
    """

    temp = np.asarray(temperature_celsius)

    # Q10 exponential temperature response model
    # R_rel = Q10^((T - T_base) / 10)
    temperature_difference = temp - base_temperature_celsius
    respiration_relative_response = np.power(q10_factor, temperature_difference / 10.0)

    # Ensure non-negative values
    respiration_relative_response = np.maximum(respiration_relative_response, 0.0)

    return respiration_relative_response


def calculate_net_ecosystem_exchange(gpp_flux_gc_m2_s: Union[float, np.ndarray],
                                   ecosystem_respiration_gc_m2_s: Union[float, np.ndarray],
                                   fire_emissions_gc_m2_s: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
    """
    Calculate Net Ecosystem Exchange (NEE) from component fluxes.

    NEE represents the net carbon exchange between terrestrial ecosystems
    and the atmosphere, following atmospheric sign convention.

    Scientific Background:
    NEE = Ecosystem Respiration - GPP + Fire Emissions
    Sign convention (atmospheric perspective):
    - Negative NEE: Net carbon sink (ecosystem uptake)
    - Positive NEE: Net carbon source (ecosystem emission)

    Args:
        gpp_flux_gc_m2_s: Gross Primary Productivity in gC/m²/s
            Should be positive values (carbon uptake)
            Typical range: 0-50 gC/m²/s for peak growing season
        ecosystem_respiration_gc_m2_s: Ecosystem respiration in gC/m²/s
            Should be positive values (carbon emission)
            Typical range: 0-30 gC/m²/s for terrestrial ecosystems
        fire_emissions_gc_m2_s: Fire emissions in gC/m²/s (optional)
            Should be positive values (carbon emission)
            Typical range: 0-100 gC/m²/s during fire events

    Returns:
        NEE in gC/m²/s
            Negative: Net carbon sink (ecosystem uptake)
            Positive: Net carbon source (ecosystem emission)
            Typical range: -30 to +20 gC/m²/s for terrestrial ecosystems

    Notes:
        - Follows atmospheric sign convention used in CARDAMOM
        - GPP is converted to negative (uptake) in calculation
        - Fire emissions are episodic and can dominate during fire events

    Example:
        >>> # Growing season conditions
        >>> gpp = np.array([20, 30, 25])  # gC/m²/s (positive uptake)
        >>> respiration = np.array([12, 15, 18])  # gC/m²/s (positive emission)
        >>> fire = np.array([0, 0, 5])  # gC/m²/s (fire event)
        >>> nee = calculate_net_ecosystem_exchange(gpp, respiration, fire)
        >>> # Expected: [-8, -15, -2] gC/m²/s (net sinks, except during fire)
    """

    gpp = np.asarray(gpp_flux_gc_m2_s)
    respiration = np.asarray(ecosystem_respiration_gc_m2_s)

    # Convert GPP to atmospheric sign convention (negative = uptake)
    gpp_atmospheric_sign = -gpp

    # Calculate NEE: Respiration - GPP + Fire
    nee_flux = respiration + gpp_atmospheric_sign

    if fire_emissions_gc_m2_s is not None:
        fire = np.asarray(fire_emissions_gc_m2_s)
        nee_flux += fire

    return nee_flux


def calculate_fire_emission_factors(burned_area_fraction: Union[float, np.ndarray],
                                  fuel_load_gc_m2: Union[float, np.ndarray],
                                  combustion_completeness: float = 0.8,
                                  emission_factor_co2: float = 1686.0) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate fire emissions from burned area and fuel characteristics.

    This function implements fire emission calculations based on the approach
    used in Global Fire Emissions Database (GFED) and CARDAMOM fire modeling.

    MATLAB Source Reference:
    File: /MATLAB/prototypes/ (various fire emission calculations)
    Used in GFED processing and diurnal fire pattern applications

    Scientific Background:
    Fire emissions = Burned Area × Fuel Load × Combustion Completeness × Emission Factor
    This approach separates the physical process (burning) from the chemical
    process (emissions) for different species.

    Args:
        burned_area_fraction: Fraction of grid cell burned (0-1)
            Typical range: 0-0.1 for most grid cells, up to 1.0 during extreme events
            Source: GFED4 burned area product
        fuel_load_gc_m2: Available fuel load in gC/m²
            Typical range: 100-10000 gC/m² depending on ecosystem type
            Higher for forests, lower for grasslands
        combustion_completeness: Fraction of fuel consumed (0-1)
            Default: 0.8 (typical for mixed ecosystems)
            Range: 0.7-0.95 (grasslands higher, forests lower)
        emission_factor_co2: CO₂ emission per unit dry matter burned (g CO₂/kg dry matter)
            Default: 1686 g CO₂/kg (typical for mixed vegetation)
            Range: 1550-1750 g CO₂/kg for different vegetation types

    Returns:
        Dictionary containing fire emission fluxes:
            'co2_emissions_gc_m2': CO₂ emissions in gC/m²
            'carbon_emissions_gc_m2': Total carbon emissions in gC/m²
            'burned_carbon_gc_m2': Carbon consumed by burning in gC/m²

    Notes:
        - Assumes carbon content of dry matter is ~45%
        - Uses molecular weight conversion for CO₂ to carbon
        - Results represent emissions for the time period of input data

    Example:
        >>> # Moderate burning in grassland
        >>> burned_fraction = np.array([0.01, 0.05, 0.02])  # 1-5% burned
        >>> fuel_load = np.array([500, 800, 600])  # gC/m² (grassland)
        >>> emissions = calculate_fire_emission_factors(burned_fraction, fuel_load)
        >>> # Expected CO₂ emissions: few to tens of gC/m²
    """

    burned_area = np.asarray(burned_area_fraction)
    fuel = np.asarray(fuel_load_gc_m2)

    # Calculate total carbon consumed by burning
    burned_carbon_gc_m2 = burned_area * fuel * combustion_completeness

    # Convert carbon to dry matter (assuming ~45% carbon content)
    carbon_fraction_dry_matter = 0.45
    burned_dry_matter_g_m2 = burned_carbon_gc_m2 / carbon_fraction_dry_matter

    # Calculate CO₂ emissions (g CO₂/m²)
    co2_emissions_g_m2 = burned_dry_matter_g_m2 * emission_factor_co2 / 1000.0  # Convert kg to g

    # Convert CO₂ emissions to carbon equivalent
    co2_emissions_carbon_gc_m2 = co2_flux_to_carbon_flux(co2_emissions_g_m2)

    return {
        'co2_emissions_gc_m2': co2_emissions_carbon_gc_m2,
        'carbon_emissions_gc_m2': burned_carbon_gc_m2,
        'burned_carbon_gc_m2': burned_carbon_gc_m2
    }


def calculate_vegetation_carbon_balance(gpp_annual_gc_m2_yr: Union[float, np.ndarray],
                                      autotrophic_respiration_gc_m2_yr: Union[float, np.ndarray],
                                      allocation_to_wood_fraction: float = 0.3) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate vegetation carbon allocation and balance components.

    This function implements basic vegetation carbon balance calculations
    used in ecosystem modeling and carbon cycle analysis.

    Scientific Background:
    Net Primary Productivity (NPP) = GPP - Autotrophic Respiration
    NPP is then allocated to different plant components (leaves, wood, roots)
    with different turnover rates and contributions to soil carbon.

    Args:
        gpp_annual_gc_m2_yr: Annual gross primary productivity in gC/m²/yr
            Typical range: 100-3000 gC/m²/yr depending on ecosystem type
            Forests: 1000-3000 gC/m²/yr
            Grasslands: 200-1000 gC/m²/yr
        autotrophic_respiration_gc_m2_yr: Annual autotrophic respiration in gC/m²/yr
            Typical range: 50-1500 gC/m²/yr (roughly 30-70% of GPP)
        allocation_to_wood_fraction: Fraction of NPP allocated to wood (0-1)
            Default: 0.3 (30% to wood)
            Forests: 0.2-0.6 depending on age and species
            Grasslands: 0.0 (no wood allocation)

    Returns:
        Dictionary containing carbon balance components:
            'npp_gc_m2_yr': Net primary productivity in gC/m²/yr
            'wood_allocation_gc_m2_yr': Carbon allocated to wood in gC/m²/yr
            'foliage_fine_root_allocation_gc_m2_yr': Carbon to leaves/roots in gC/m²/yr
            'carbon_use_efficiency': Ratio of NPP to GPP (dimensionless)

    Example:
        >>> # Temperate forest conditions
        >>> gpp_annual = np.array([2000, 2500, 1800])  # gC/m²/yr
        >>> autoresp_annual = np.array([1000, 1200, 900])  # gC/m²/yr
        >>> balance = calculate_vegetation_carbon_balance(gpp_annual, autoresp_annual)
        >>> # Expected NPP: [1000, 1300, 900] gC/m²/yr
    """

    gpp = np.asarray(gpp_annual_gc_m2_yr)
    autotrophic_resp = np.asarray(autotrophic_respiration_gc_m2_yr)

    # Calculate Net Primary Productivity
    npp_gc_m2_yr = gpp - autotrophic_resp

    # Ensure NPP is non-negative (cannot allocate more than available)
    npp_gc_m2_yr = np.maximum(npp_gc_m2_yr, 0.0)

    # Calculate carbon allocation to different components
    wood_allocation_gc_m2_yr = npp_gc_m2_yr * allocation_to_wood_fraction
    foliage_fine_root_allocation_gc_m2_yr = npp_gc_m2_yr * (1.0 - allocation_to_wood_fraction)

    # Calculate carbon use efficiency (NPP/GPP)
    carbon_use_efficiency = np.divide(npp_gc_m2_yr, gpp,
                                    out=np.zeros_like(npp_gc_m2_yr),
                                    where=gpp != 0)

    return {
        'npp_gc_m2_yr': npp_gc_m2_yr,
        'wood_allocation_gc_m2_yr': wood_allocation_gc_m2_yr,
        'foliage_fine_root_allocation_gc_m2_yr': foliage_fine_root_allocation_gc_m2_yr,
        'carbon_use_efficiency': carbon_use_efficiency
    }


def validate_carbon_flux_mass_balance(gpp_flux: Union[float, np.ndarray],
                                    respiration_flux: Union[float, np.ndarray],
                                    nee_flux: Union[float, np.ndarray],
                                    fire_flux: Optional[Union[float, np.ndarray]] = None,
                                    tolerance_gc_m2_s: float = 0.1) -> bool:
    """
    Validate carbon flux mass balance for consistency checking.

    This function checks that carbon flux components satisfy mass balance
    constraints according to ecosystem carbon cycle principles.

    Scientific Background:
    Mass balance equation: NEE = Respiration - GPP + Fire
    This fundamental relationship must hold for physically consistent
    carbon flux estimates in ecosystem modeling.

    Args:
        gpp_flux: Gross primary productivity flux
            Should be positive (uptake)
        respiration_flux: Ecosystem respiration flux
            Should be positive (emission)
        nee_flux: Net ecosystem exchange flux
            Negative = net sink, Positive = net source
        fire_flux: Fire emissions flux (optional)
            Should be positive (emission)
        tolerance_gc_m2_s: Tolerance for mass balance check
            Default: 0.1 gC/m²/s

    Returns:
        Boolean indicating whether mass balance is satisfied within tolerance

    Raises:
        ValueError: If mass balance is violated beyond tolerance

    Example:
        >>> # Check flux consistency
        >>> gpp = 20.0  # gC/m²/s
        >>> resp = 12.0  # gC/m²/s
        >>> nee = -8.0  # gC/m²/s (net sink)
        >>> is_valid = validate_carbon_flux_mass_balance(gpp, resp, nee)
        >>> # Expected: True (12 - 20 = -8, mass balance satisfied)
    """

    gpp = np.asarray(gpp_flux)
    respiration = np.asarray(respiration_flux)
    nee = np.asarray(nee_flux)

    # Calculate expected NEE from components
    expected_nee = respiration - gpp

    if fire_flux is not None:
        fire = np.asarray(fire_flux)
        expected_nee += fire

    # Check mass balance
    mass_balance_error = np.abs(nee - expected_nee)
    max_error = np.max(mass_balance_error)

    if max_error > tolerance_gc_m2_s:
        raise ValueError(
            f"Carbon flux mass balance violated. "
            f"Maximum error: {max_error:.3f} gC/m²/s exceeds tolerance: {tolerance_gc_m2_s} gC/m²/s. "
            f"Check that NEE = Respiration - GPP + Fire."
        )

    return True