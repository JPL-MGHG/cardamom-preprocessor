"""
Basic tests for validation module.

Tests basic functionality of validation functions.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from validation import (
    validate_spatial_coverage,
    check_temporal_continuity,
    validate_physical_ranges,
    validate_data_consistency,
    QualityAssurance
)


def test_spatial_coverage_validation():
    """Test spatial coverage validation"""
    # Create test data
    data = np.random.random((180, 360))  # Global 1° grid
    expected_grid = {'shape': (180, 360)}

    result = validate_spatial_coverage(data, expected_grid)

    assert isinstance(result, dict)
    assert 'status' in result
    assert 'coverage_fraction' in result
    assert 'total_grid_points' in result


def test_spatial_coverage_with_missing_data():
    """Test spatial coverage validation with missing data"""
    # Create data with some NaN values
    data = np.random.random((100, 200))
    data[0:10, 0:10] = np.nan  # Add some missing data

    expected_grid = {'shape': (100, 200)}

    result = validate_spatial_coverage(data, expected_grid, min_coverage_fraction=0.9)

    assert isinstance(result, dict)
    assert result['missing_data_points'] > 0
    assert result['coverage_fraction'] < 1.0


def test_temporal_continuity_validation():
    """Test temporal continuity validation"""
    # Create regular time coordinates
    time_coords = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Monthly
    data = np.random.random((50, 100, 12))

    result = check_temporal_continuity(data, time_coords)

    assert isinstance(result, dict)
    assert 'status' in result
    assert 'time_range' in result
    assert 'num_time_steps' in result


def test_temporal_continuity_with_gaps():
    """Test temporal continuity with gaps"""
    # Create time coordinates with gaps
    time_coords = np.array([1, 2, 3, 6, 7, 8, 11, 12])  # Missing months
    data = np.random.random((50, 100, 8))

    result = check_temporal_continuity(data, time_coords)

    assert isinstance(result, dict)
    assert len(result['time_gaps_detected']) > 0


def test_physical_ranges_validation():
    """Test physical ranges validation"""
    # Valid temperature data
    temp_data = np.array([250, 280, 300, 320])  # Kelvin

    result = validate_physical_ranges(temp_data, 'temperature_kelvin')

    assert isinstance(result, dict)
    assert 'status' in result
    assert 'data_range' in result
    assert 'expected_range' in result


def test_physical_ranges_validation_failures():
    """Test physical ranges validation with invalid data"""
    # Invalid temperature data (too hot)
    temp_data = np.array([100, 200, 500])  # Some values too high

    result = validate_physical_ranges(temp_data, 'temperature_kelvin')

    # Should detect out-of-range values
    assert result['num_above_max'] > 0 or result['num_below_min'] > 0


def test_physical_ranges_unknown_variable():
    """Test physical ranges with unknown variable type"""
    data = np.array([1, 2, 3, 4, 5])

    result = validate_physical_ranges(data, 'unknown_variable')

    assert result['status'] == 'warning'
    assert 'No physical range defined' in result['messages'][0]


def test_data_consistency_validation():
    """Test data consistency validation"""
    # Consistent data
    data_dict = {
        'var1': np.random.random((100, 200)),
        'var2': np.random.random((100, 200)),
        'var3': np.random.random((100, 200))
    }

    result = validate_data_consistency(data_dict)

    assert isinstance(result, dict)
    assert result['status'] == 'pass'
    assert result['consistent_shapes'] is True


def test_data_consistency_validation_failures():
    """Test data consistency with inconsistent shapes"""
    # Inconsistent data shapes
    data_dict = {
        'var1': np.random.random((100, 200)),
        'var2': np.random.random((50, 100)),  # Different shape
        'var3': np.random.random((100, 200))
    }

    result = validate_data_consistency(data_dict)

    assert result['status'] == 'fail'
    assert result['consistent_shapes'] is False


def test_data_consistency_empty_dict():
    """Test data consistency with empty dictionary"""
    result = validate_data_consistency({})

    assert result['status'] == 'warning'
    assert 'No data provided' in result['messages'][0]


def test_quality_assurance_initialization():
    """Test QA system initialization"""
    config = {
        'enable_validation': True,
        'physical_range_checks': True,
        'missing_data_tolerance': 0.05
    }

    qa_system = QualityAssurance(config)

    assert qa_system.enable_validation is True
    assert qa_system.physical_range_checks is True
    assert qa_system.missing_data_tolerance == 0.05


def test_quality_assurance_disabled():
    """Test QA system when validation is disabled"""
    config = {'enable_validation': False}

    qa_system = QualityAssurance(config)
    data_dict = {'var1': np.random.random((100, 200))}
    metadata = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        result = qa_system.run_full_qa_suite(data_dict, metadata, temp_dir)

    assert result['status'] == 'skipped'


def test_qa_system_basic_run():
    """Test basic QA system run"""
    config = {
        'enable_validation': True,
        'physical_range_checks': True,
        'spatial_continuity_checks': True,
        'temporal_continuity_checks': False,  # Skip temporal for this test
        'missing_data_tolerance': 0.05
    }

    qa_system = QualityAssurance(config)

    # Create test data
    data_dict = {
        'temperature': np.random.uniform(250, 320, (100, 200))  # Valid temp range
    }

    metadata = {
        'variables': {
            'temperature': {
                'type': 'temperature_kelvin',
                'grid_info': {'shape': (100, 200)}
            }
        }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        result = qa_system.run_full_qa_suite(data_dict, metadata, temp_dir)

    assert isinstance(result, dict)
    assert 'status' in result
    assert 'num_tests' in result
    assert 'validation_results' in result


if __name__ == '__main__':
    pytest.main([__file__])