"""
CARDAMOM Variable Registry - Master Configuration for All Variables

This module provides the single source of truth for all meteorological and carbon
cycle variables used in CARDAMOM preprocessing. It consolidates variable metadata
that was previously scattered across multiple files.

Scientific Context:
Different variables have different spatial characteristics that require appropriate
interpolation methods:
- Continuous fields (temperature, CO2): linear interpolation preserves gradients
- Patchy/categorical data (fire, snow): nearest neighbor preserves sharp boundaries
- Cumulative variables (precipitation): linear interpolation for smooth transitions

All variable names, units, conversions, and processing parameters are centralized here
to prevent duplication and ensure consistency across downloaders and processors.
"""

from typing import Dict, List, Optional, Any, Union


# Master variable registry with complete metadata
CARDAMOM_VARIABLE_REGISTRY: Dict[str, Dict[str, Any]] = {

    # ========== ERA5 Meteorological Variables ==========

    '2m_temperature': {
        'source': 'era5',
        'alternative_names': ['t2m', 'T2M'],
        'cbf_names': ['TMIN', 'TMAX'],  # Derives both min and max
        'units': {'source': 'K', 'cbf': 'K'},
        'interpolation_method': 'linear',
        'spatial_nature': 'continuous',
        'description': 'Air temperature at 2 meters above surface',
        'physical_range': (173, 333),  # -100°C to 60°C in Kelvin
        'essential': True,
        'processing_notes': 'Monthly min/max derived from sub-monthly data',
        'era5_product_type': 'monthly_averaged_reanalysis_by_hour_of_day'
    },

    '2m_dewpoint_temperature': {
        'source': 'era5',
        'alternative_names': ['d2m', 'D2M'],
        'cbf_names': None,  # Used for VPD calculation, not direct CBF output
        'units': {'source': 'K', 'cbf': 'K'},
        'interpolation_method': 'linear',
        'spatial_nature': 'continuous',
        'description': 'Dewpoint temperature at 2 meters (for VPD calculation)',
        'physical_range': (173, 333),
        'essential': True,
        'processing_notes': 'Combined with temperature to calculate VPD',
        'era5_product_type': 'monthly_averaged_reanalysis_by_hour_of_day'
    },

    'total_precipitation': {
        'source': 'era5',
        'alternative_names': ['tp', 'TP'],
        'cbf_names': ['PREC'],
        'units': {'source': 'm', 'cbf': 'mm/month'},
        'unit_conversion': {'factor': 1000, 'method': 'multiply'},
        'interpolation_method': 'linear',
        'spatial_nature': 'continuous',
        'description': 'Total precipitation (rain + snow water equivalent)',
        'physical_range': (0, 1000),  # mm/month
        'enforce_positive': True,
        'essential': True,
        'era5_product_type': 'monthly_averaged_reanalysis'
    },

    'skin_temperature': {
        'source': 'era5',
        'alternative_names': ['skt', 'SKT'],
        'cbf_names': ['SKT'],
        'units': {'source': 'K', 'cbf': 'K'},
        'interpolation_method': 'linear',
        'spatial_nature': 'continuous',
        'description': 'Surface skin temperature',
        'physical_range': (173, 373),
        'enforce_positive': True,
        'essential': True,
        'era5_product_type': 'monthly_averaged_reanalysis'
    },

    'surface_solar_radiation_downwards': {
        'source': 'era5',
        'alternative_names': ['ssrd', 'SSRD'],
        'cbf_names': ['SSRD'],
        'units': {'source': 'J m-2', 'cbf': 'W m-2'},
        'unit_conversion': {'method': 'radiation_monthly'},
        'interpolation_method': 'linear',
        'spatial_nature': 'continuous',
        'description': 'Downward surface solar radiation',
        'physical_range': (0, 1000),  # W m-2
        'enforce_positive': True,
        'essential': True,
        'era5_product_type': 'monthly_averaged_reanalysis'
    },

    'surface_thermal_radiation_downwards': {
        'source': 'era5',
        'alternative_names': ['strd', 'STRD'],
        'cbf_names': ['STRD'],
        'units': {'source': 'J m-2', 'cbf': 'W m-2'},
        'unit_conversion': {'method': 'radiation_monthly'},
        'interpolation_method': 'linear',
        'spatial_nature': 'continuous',
        'description': 'Downward surface thermal radiation',
        'physical_range': (0, 600),  # W m-2
        'enforce_positive': True,
        'essential': True,
        'era5_product_type': 'monthly_averaged_reanalysis'
    },

    'snowfall': {
        'source': 'era5',
        'alternative_names': ['sf', 'SF'],
        'cbf_names': ['SNOW'],
        'units': {'source': 'm of water equivalent', 'cbf': 'mm/month'},
        'unit_conversion': {'factor': 1000, 'method': 'multiply'},
        'interpolation_method': 'nearest',
        'spatial_nature': 'patchy',
        'description': 'Snowfall water equivalent',
        'physical_range': (0, 500),  # mm/month
        'enforce_positive': True,
        'processing_notes': 'Nearest neighbor preserves patchy snow distribution',
        'era5_product_type': 'monthly_averaged_reanalysis'
    },

    # ========== Derived ERA5 Variables ==========

    'VPD': {
        'source': 'derived',
        'derived_from': ['2m_temperature', '2m_dewpoint_temperature'],
        'cbf_names': ['VPD'],
        'units': {'cbf': 'hPa'},
        'interpolation_method': 'linear',
        'spatial_nature': 'continuous',
        'description': 'Vapor Pressure Deficit',
        'physical_range': (0, 60),  # hPa
        'essential': True,
        'processing_notes': 'Calculated using Tetens equation from Tmax and dewpoint'
    },

    # ========== External Data Sources ==========

    'co2': {
        'source': 'noaa',
        'alternative_names': ['CO2', 'co2_concentration', 'mole_fraction', 'co2_mole_fraction'],
        'cbf_names': ['CO2_2'],
        'units': {'source': 'ppm', 'cbf': 'ppm'},
        'interpolation_method': 'linear',
        'spatial_nature': 'continuous',
        'description': 'Atmospheric CO2 concentration',
        'physical_range': (250, 500),  # ppm
        'enforce_positive': True,
        'essential': True,
        'processing_notes': 'Spatially smooth global field, linear interpolation appropriate'
    },

    'burned_area': {
        'source': 'gfed',
        'alternative_names': ['BURNED_AREA', 'burned_fraction', 'ba'],
        'cbf_names': ['BURN_2'],
        'units': {'source': 'fraction', 'cbf': 'fraction'},
        'interpolation_method': 'nearest',
        'spatial_nature': 'patchy',
        'description': 'Fraction of grid cell burned by fires',
        'physical_range': (0, 1),
        'enforce_positive': True,
        'processing_notes': 'Nearest neighbor preserves sharp fire/no-fire boundaries'
    },

    'fire_carbon': {
        'source': 'gfed',
        'cbf_names': None,  # Not directly used in CBF
        'units': {'source': 'gC/m2/month', 'cbf': 'gC/m2/month'},
        'interpolation_method': 'nearest',
        'spatial_nature': 'patchy',
        'description': 'Carbon emissions from fires'
    },

    # ========== Land/Sea Mask Variables ==========

    'land_sea_mask': {
        'source': 'modis',
        'alternative_names': ['land_fraction', 'land_sea_frac', 'fraction', 'data'],
        'cbf_names': None,  # Used for masking only
        'units': {'source': 'fraction', 'cbf': 'fraction'},
        'interpolation_method': 'nearest',
        'spatial_nature': 'categorical',
        'description': 'Land/sea fraction for masking',
        'physical_range': (0, 1),
        'processing_notes': 'Binary land/ocean classification, nearest neighbor required'
    },

    # ========== CBF Framework Variables ==========

    'DISTURBANCE_FLUX': {
        'source': 'framework',
        'cbf_names': ['DISTURBANCE_FLUX'],
        'units': {'cbf': 'gC/m2/day'},
        'description': 'Carbon flux from disturbances (typically zero for met processing)',
        'default_value': 0.0
    },

    'YIELD': {
        'source': 'framework',
        'cbf_names': ['YIELD'],
        'units': {'cbf': 'gC/m2/day'},
        'description': 'Carbon flux from harvesting/yields (typically zero for met processing)',
        'default_value': 0.0
    }
}


# ========== Helper Functions ==========

def get_variable_config(variable_name: str) -> Dict[str, Any]:
    """
    Get complete configuration for a specific variable.

    Args:
        variable_name: Variable name (can be standard name or alternative)

    Returns:
        dict: Variable configuration

    Raises:
        KeyError: If variable not found in registry
    """
    # Try direct lookup first
    if variable_name in CARDAMOM_VARIABLE_REGISTRY:
        return CARDAMOM_VARIABLE_REGISTRY[variable_name]

    # Try alternative names
    for var_name, config in CARDAMOM_VARIABLE_REGISTRY.items():
        alt_names = config.get('alternative_names', [])
        if variable_name in alt_names:
            return config

    # Try CBF names
    for var_name, config in CARDAMOM_VARIABLE_REGISTRY.items():
        cbf_names = config.get('cbf_names', [])
        if cbf_names and variable_name in cbf_names:
            return config

    raise KeyError(f"Variable '{variable_name}' not found in CARDAMOM variable registry")


def get_interpolation_method(variable_name: str) -> str:
    """
    Get interpolation method for a variable.

    Args:
        variable_name: Variable name

    Returns:
        str: Interpolation method ('linear', 'nearest', 'cubic')
    """
    try:
        config = get_variable_config(variable_name)
        return config.get('interpolation_method', 'nearest')
    except KeyError:
        # Default to nearest for unknown variables
        return 'nearest'


def get_cbf_name(variable_name: str) -> Optional[Union[str, List[str]]]:
    """
    Get CBF output name(s) for a variable.

    Args:
        variable_name: Source variable name

    Returns:
        str, list, or None: CBF name(s) or None if not output to CBF
    """
    try:
        config = get_variable_config(variable_name)
        cbf_names = config.get('cbf_names')
        if isinstance(cbf_names, list) and len(cbf_names) == 1:
            return cbf_names[0]
        return cbf_names
    except KeyError:
        return None


def get_unit_conversion(variable_name: str) -> Optional[Dict[str, Any]]:
    """
    Get unit conversion parameters for a variable.

    Args:
        variable_name: Variable name

    Returns:
        dict or None: Unit conversion parameters
    """
    try:
        config = get_variable_config(variable_name)
        return config.get('unit_conversion')
    except KeyError:
        return None


def get_variables_by_source(source: str) -> List[str]:
    """
    Get all variables from a specific data source.

    Args:
        source: Data source name ('era5', 'noaa', 'gfed', 'modis', 'derived', 'framework')

    Returns:
        list: Variable names from that source
    """
    return [
        var_name for var_name, config in CARDAMOM_VARIABLE_REGISTRY.items()
        if config.get('source') == source
    ]


def get_essential_variables(source: Optional[str] = None) -> List[str]:
    """
    Get list of essential variables.

    Args:
        source: Optional data source filter

    Returns:
        list: Essential variable names
    """
    essential = [
        var_name for var_name, config in CARDAMOM_VARIABLE_REGISTRY.items()
        if config.get('essential', False)
    ]

    if source:
        essential = [v for v in essential if get_variable_config(v).get('source') == source]

    return essential


def get_variables_by_product_type(product_type: str) -> List[str]:
    """
    Get all ERA5 variables that require a specific product type for download.

    Args:
        product_type: ERA5 product type ('monthly_averaged_reanalysis' or 
                     'monthly_averaged_reanalysis_by_hour_of_day')

    Returns:
        list: Variable names requiring that product type

    Examples:
        >>> get_variables_by_product_type('monthly_averaged_reanalysis')
        ['total_precipitation', 'skin_temperature', 'surface_solar_radiation_downwards', 'snowfall']
        
        >>> get_variables_by_product_type('monthly_averaged_reanalysis_by_hour_of_day')
        ['2m_temperature', '2m_dewpoint_temperature']
    """
    return [
        var_name for var_name, config in CARDAMOM_VARIABLE_REGISTRY.items()
        if config.get('era5_product_type') == product_type
    ]


def get_cbf_variable_mapping() -> Dict[str, Union[str, List[str]]]:
    """
    Get mapping from source variable names to CBF names.

    Returns:
        dict: Mapping of source variable -> CBF name(s)
    """
    mapping = {}
    for var_name, config in CARDAMOM_VARIABLE_REGISTRY.items():
        cbf_names = config.get('cbf_names')
        if cbf_names:
            mapping[var_name] = cbf_names
    return mapping


def get_variable_alternatives(variable_name: str) -> List[str]:
    """
    Get alternative names for a variable.

    Args:
        variable_name: Variable name

    Returns:
        list: Alternative names (including the standard name)
    """
    try:
        config = get_variable_config(variable_name)
        alternatives = [variable_name] + config.get('alternative_names', [])
        return alternatives
    except KeyError:
        return [variable_name]


def validate_variable(variable_name: str) -> bool:
    """
    Check if a variable exists in the registry.

    Args:
        variable_name: Variable name to validate

    Returns:
        bool: True if variable exists
    """
    try:
        get_variable_config(variable_name)
        return True
    except KeyError:
        return False


def get_all_variables() -> List[str]:
    """
    Get list of all registered variables.

    Returns:
        list: All variable names in registry
    """
    return list(CARDAMOM_VARIABLE_REGISTRY.keys())


def get_variable_physical_range(variable_name: str) -> Optional[tuple]:
    """
    Get physical range for a variable.

    Args:
        variable_name: Variable name

    Returns:
        tuple or None: (min, max) physical range
    """
    try:
        config = get_variable_config(variable_name)
        return config.get('physical_range')
    except KeyError:
        return None
