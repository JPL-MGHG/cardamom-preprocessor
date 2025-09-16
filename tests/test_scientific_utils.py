"""
Basic tests for scientific utility functions.

Tests basic functionality of scientific calculations without
detailed scientific value validation.
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scientific_utils import (
    calculate_vapor_pressure_deficit,
    saturation_pressure_water,
    convert_precipitation_units,
    convert_radiation_units,
    convert_carbon_flux_units,
    validate_temperature_data,
    validate_precipitation_data
)


def test_vapor_pressure_deficit_basic():
    """Test basic VPD calculation functionality"""
    # Test with simple values
    temp_max_k = 298.15  # 25°C
    dewpoint_k = 288.15  # 15°C

    vpd = calculate_vapor_pressure_deficit(temp_max_k, dewpoint_k)

    assert isinstance(vpd, (float, np.ndarray))
    assert vpd > 0  # VPD should be positive


def test_vapor_pressure_deficit_arrays():
    """Test VPD calculation with numpy arrays"""
    temp_max_k = np.array([295.15, 300.15, 305.15])  # 22, 27, 32°C
    dewpoint_k = np.array([285.15, 290.15, 295.15])  # 12, 17, 22°C

    vpd = calculate_vapor_pressure_deficit(temp_max_k, dewpoint_k)

    assert isinstance(vpd, np.ndarray)
    assert len(vpd) == 3
    assert np.all(vpd > 0)


def test_vapor_pressure_deficit_validation():
    """Test VPD validation for impossible conditions"""
    temp_max_k = 288.15  # 15°C
    dewpoint_k = 298.15  # 25°C (higher than temp_max - impossible)

    with pytest.raises(ValueError, match="VPD cannot be negative"):
        calculate_vapor_pressure_deficit(temp_max_k, dewpoint_k)


def test_saturation_pressure_water():
    """Test saturation pressure calculation"""
    temp_k = 298.15  # 25°C

    sat_pressure = saturation_pressure_water(temp_k)

    assert isinstance(sat_pressure, (float, np.ndarray))
    assert sat_pressure > 0


def test_saturation_pressure_water_arrays():
    """Test saturation pressure with arrays"""
    temp_k = np.array([273.15, 298.15, 323.15])  # 0, 25, 50°C

    sat_pressure = saturation_pressure_water(temp_k)

    assert isinstance(sat_pressure, np.ndarray)
    assert len(sat_pressure) == 3
    assert np.all(sat_pressure > 0)


def test_precipitation_unit_conversion():
    """Test precipitation unit conversion"""
    precip_ms = 1e-5  # m/s (typical ERA5 value)

    precip_mm_day = convert_precipitation_units(precip_ms)

    assert isinstance(precip_mm_day, (float, np.ndarray))
    assert precip_mm_day > 0


def test_precipitation_unit_conversion_arrays():
    """Test precipitation conversion with arrays"""
    precip_ms = np.array([1e-6, 1e-5, 1e-4])  # Various m/s values

    precip_mm_day = convert_precipitation_units(precip_ms)

    assert isinstance(precip_mm_day, np.ndarray)
    assert len(precip_mm_day) == 3
    assert np.all(precip_mm_day >= 0)


def test_radiation_unit_conversion():
    """Test radiation unit conversion"""
    radiation_j_m2 = 20000000  # J/m² (typical daily solar radiation)

    radiation_w_m2 = convert_radiation_units(radiation_j_m2)

    assert isinstance(radiation_w_m2, (float, np.ndarray))
    assert radiation_w_m2 > 0


def test_carbon_flux_unit_conversion():
    """Test carbon flux unit conversion"""
    flux_gc_m2_day = 10.0  # gC/m²/day

    flux_kgc_km2_sec = convert_carbon_flux_units(flux_gc_m2_day)

    assert isinstance(flux_kgc_km2_sec, (float, np.ndarray))
    # Don't check sign since carbon flux can be positive or negative


def test_temperature_validation():
    """Test temperature validation function"""
    # Valid temperature data
    valid_temps = np.array([273.15, 298.15, 313.15])  # 0, 25, 40°C

    validation_result = validate_temperature_data(valid_temps)

    assert isinstance(validation_result, dict)
    assert 'status' in validation_result
    assert 'min_value' in validation_result
    assert 'max_value' in validation_result


def test_temperature_validation_failures():
    """Test temperature validation with invalid data"""
    # Invalid temperature data (too cold)
    invalid_temps = np.array([100.0, 200.0, 300.0])  # All below -100°C

    validation_result = validate_temperature_data(invalid_temps)

    assert validation_result['status'] == 'fail'
    assert validation_result['num_below_physical_min'] > 0


def test_precipitation_validation():
    """Test precipitation validation function"""
    # Valid precipitation data
    valid_precip = np.array([0.0, 5.0, 25.0, 100.0])  # mm/day

    validation_result = validate_precipitation_data(valid_precip)

    assert isinstance(validation_result, dict)
    assert 'status' in validation_result
    assert 'min_value' in validation_result
    assert 'max_value' in validation_result


def test_precipitation_validation_failures():
    """Test precipitation validation with invalid data"""
    # Invalid precipitation data (negative values)
    invalid_precip = np.array([-5.0, 10.0, -2.0])

    validation_result = validate_precipitation_data(invalid_precip)

    assert validation_result['status'] == 'fail'
    assert validation_result['num_negative'] > 0


def test_function_with_zero_values():
    """Test functions handle zero values correctly"""
    # Test VPD with zero difference (saturated conditions)
    temp_k = 298.15
    dewpoint_k = 298.15  # Same as temperature

    vpd = calculate_vapor_pressure_deficit(temp_k, dewpoint_k)
    assert abs(vpd) < 0.01  # Should be very close to zero

    # Test precipitation conversion with zero
    zero_precip = convert_precipitation_units(0.0)
    assert zero_precip == 0.0


if __name__ == '__main__':
    pytest.main([__file__])