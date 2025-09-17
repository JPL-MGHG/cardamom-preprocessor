"""
Simple tests for Phase 8 scientific utility functions.

Tests basic functionality of the implemented Phase 8 modules to ensure
they work correctly and produce reasonable results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

# Import Phase 8 modules
from atmospheric_science import (
    saturation_pressure_water_matlab,
    calculate_vapor_pressure_deficit_matlab,
    radiation_to_par_conversion
)
from statistics_utils import (
    nan_to_zero,
    monthly_to_annual,
    find_closest_grid_points
)
from units_constants import (
    PhysicalConstants,
    temperature_celsius_to_kelvin,
    carbon_flux_gc_m2_day_to_kg_c_km2_s
)
from carbon_cycle import (
    calculate_net_ecosystem_exchange,
    validate_carbon_flux_mass_balance
)
from quality_control import (
    validate_temperature_range_extended,
    DataQualityReport
)


def test_atmospheric_science_functions():
    """Test atmospheric science calculations."""

    # Test MATLAB saturation pressure function
    temp_c = np.array([0, 10, 20, 30])
    vpsat = saturation_pressure_water_matlab(temp_c)

    # Check that results are positive and increasing with temperature
    assert np.all(vpsat > 0)
    assert np.all(np.diff(vpsat) > 0)  # Should increase with temperature

    # Test VPD calculation
    t_max = 25.0  # °C
    t_dew = 15.0  # °C
    vpd = calculate_vapor_pressure_deficit_matlab(t_max, t_dew)

    # VPD should be positive and reasonable
    assert vpd > 0
    assert vpd < 100  # Should be less than 100 hPa

    # Test PAR conversion
    solar_rad = np.array([0, 500, 1000])  # W/m²
    par = radiation_to_par_conversion(solar_rad)

    assert np.all(par >= 0)
    assert par[0] == 0  # No PAR when no solar radiation
    assert par[2] > par[1]  # Higher solar rad = higher PAR


def test_statistics_utils():
    """Test statistical utility functions."""

    # Test nan_to_zero
    data_with_nan = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    clean_data = nan_to_zero(data_with_nan)

    assert not np.any(np.isnan(clean_data))
    assert np.sum(clean_data) == 9.0  # 1 + 0 + 3 + 0 + 5

    # Test monthly to annual conversion
    # Create 24 months of data (2 years)
    monthly_data = np.random.randn(50, 50, 24) + 10
    annual_data = monthly_to_annual(monthly_data, dim=2)

    assert annual_data.shape == (50, 50, 2)  # 2 years

    # Test closest grid points
    # Simple regular grid
    grid_x, grid_y = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6))
    point_x = np.array([0.3, 2.7])
    point_y = np.array([0.8, -1.2])

    pts, rows, cols = find_closest_grid_points(point_x, point_y, grid_x, grid_y, irregular=False)

    assert len(pts) == 2
    assert len(rows) == 2
    assert len(cols) == 2


def test_units_constants():
    """Test physical constants and unit conversions."""

    # Test physical constants
    assert PhysicalConstants.GAS_CONSTANT_DRY_AIR == 287.04
    assert PhysicalConstants.STANDARD_PRESSURE == 101325.0

    # Test temperature conversion
    temp_c = 25.0
    temp_k = temperature_celsius_to_kelvin(temp_c)
    assert temp_k == 298.15

    # Test carbon flux unit conversion
    flux_gc_m2_day = 10.0  # gC/m²/day
    flux_kg_km2_s = carbon_flux_gc_m2_day_to_kg_c_km2_s(flux_gc_m2_day)

    assert flux_kg_km2_s > 0
    assert isinstance(flux_kg_km2_s, (float, np.ndarray))


def test_carbon_cycle():
    """Test carbon cycle calculations."""

    # Test NEE calculation
    gpp = 20.0  # gC/m²/s (positive uptake)
    respiration = 12.0  # gC/m²/s (positive emission)

    nee = calculate_net_ecosystem_exchange(gpp, respiration)

    # NEE should be negative (net sink)
    assert nee < 0
    assert abs(nee - (-8.0)) < 0.001  # Should be -8.0

    # Test mass balance validation
    is_valid = validate_carbon_flux_mass_balance(gpp, respiration, nee)
    assert is_valid


def test_quality_control():
    """Test data quality control functions."""

    # Test temperature validation
    temp_data = np.random.randn(100) * 10 + 285  # K, reasonable range
    report = validate_temperature_range_extended(temp_data)

    assert isinstance(report, DataQualityReport)
    assert report.is_valid()  # Should pass all checks
    assert len(report.passed_checks) > 0

    # Test with bad temperature data
    bad_temp_data = np.array([100, 200, 500])  # Unreasonable values
    bad_report = validate_temperature_range_extended(bad_temp_data)

    assert not bad_report.is_valid()  # Should fail checks
    assert len(bad_report.failed_checks) > 0


def test_integration():
    """Test integration between modules."""

    # Create synthetic meteorological data
    temperature_c = np.random.randn(100) * 5 + 20  # °C
    solar_radiation = np.random.exponential(300, 100)  # W/m²

    # Convert temperature
    temperature_k = temperature_celsius_to_kelvin(temperature_c)

    # Calculate PAR
    par = radiation_to_par_conversion(solar_radiation)

    # Validate temperature
    temp_report = validate_temperature_range_extended(temperature_k)

    # All should work together
    assert len(temperature_k) == len(par)
    assert temp_report.is_valid()

    print("All Phase 8 integration tests passed!")


if __name__ == "__main__":
    # Run simple tests
    test_atmospheric_science_functions()
    test_statistics_utils()
    test_units_constants()
    test_carbon_cycle()
    test_quality_control()
    test_integration()

    print("✅ All Phase 8 tests passed successfully!")
    print("Phase 8 implementation is ready for use.")