"""
Basic tests for coordinate systems module.

Tests basic functionality of CoordinateGrid and StandardGrids classes.
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coordinate_systems import CoordinateGrid, StandardGrids


def test_coordinate_grid_initialization():
    """Test basic coordinate grid initialization"""
    grid = CoordinateGrid(resolution=0.5)

    assert grid.resolution == 0.5
    assert len(grid.bounds) == 4
    assert grid.num_longitude_points > 0
    assert grid.num_latitude_points > 0


def test_coordinate_grid_custom_bounds():
    """Test coordinate grid with custom bounds"""
    # Test CONUS bounds
    conus_bounds = [20, -130, 60, -50]  # S, W, N, E
    grid = CoordinateGrid(resolution=1.0, bounds=conus_bounds)

    assert grid.bounds == conus_bounds
    assert len(grid.longitude_coordinates) > 0
    assert len(grid.latitude_coordinates) > 0


def test_coordinate_grid_arrays():
    """Test that coordinate arrays are created correctly"""
    grid = CoordinateGrid(resolution=1.0)

    # Check that arrays exist and have correct properties
    assert isinstance(grid.longitude_coordinates, np.ndarray)
    assert isinstance(grid.latitude_coordinates, np.ndarray)

    # Check that coordinates are in reasonable ranges
    assert np.all(grid.longitude_coordinates >= -180)
    assert np.all(grid.longitude_coordinates <= 180)
    assert np.all(grid.latitude_coordinates >= -90)
    assert np.all(grid.latitude_coordinates <= 90)


def test_coordinate_grid_template_array():
    """Test template array creation"""
    grid = CoordinateGrid(resolution=2.0)

    # Test 2D template
    template_2d = grid.create_template_array()
    expected_shape = (grid.num_latitude_points, grid.num_longitude_points)
    assert template_2d.shape == expected_shape
    assert np.all(np.isnan(template_2d))

    # Test 3D template
    template_3d = grid.create_template_array(num_time_steps=12)
    expected_shape_3d = (grid.num_latitude_points, grid.num_longitude_points, 12)
    assert template_3d.shape == expected_shape_3d


def test_coordinate_grid_info():
    """Test grid information method"""
    grid = CoordinateGrid(resolution=0.5)
    grid_info = grid.get_grid_info()

    assert isinstance(grid_info, dict)
    assert 'resolution_degrees' in grid_info
    assert 'bounds' in grid_info
    assert 'num_longitude_points' in grid_info
    assert 'num_latitude_points' in grid_info
    assert grid_info['resolution_degrees'] == 0.5


def test_standard_grids():
    """Test standard grid creation"""
    # Test global grids
    global_half = StandardGrids.create_global_half_degree()
    assert global_half.resolution == 0.5

    global_quarter = StandardGrids.create_global_quarter_degree()
    assert global_quarter.resolution == 0.25

    # Test CONUS grid
    conus_grid = StandardGrids.create_conus_half_degree()
    assert conus_grid.resolution == 0.5
    # Check that bounds are for CONUS region
    assert conus_grid.bounds[0] >= 20  # South >= 20°N
    assert conus_grid.bounds[2] <= 60  # North <= 60°N


def test_standard_grids_info():
    """Test standard grids information"""
    grids_info = StandardGrids.get_standard_grid_info()

    assert isinstance(grids_info, dict)
    assert 'global_0.5deg' in grids_info
    assert 'global_0.25deg' in grids_info
    assert 'conus_0.5deg' in grids_info

    # Check that each grid info has required fields
    for grid_name, grid_info in grids_info.items():
        assert 'resolution_degrees' in grid_info
        assert 'num_longitude_points' in grid_info
        assert 'num_latitude_points' in grid_info


def test_custom_grid():
    """Test custom grid creation"""
    custom_grid = StandardGrids.create_custom_grid(
        resolution=1.0,
        region_bounds=[40, -120, 50, -100]  # Custom region
    )

    assert custom_grid.resolution == 1.0
    assert custom_grid.bounds == [40, -120, 50, -100]


def test_regional_subset():
    """Test regional subset creation"""
    grid = CoordinateGrid(resolution=1.0)

    # Create subset for a smaller region
    subset_bounds = [30, -120, 40, -110]
    subset_grid = grid.get_regional_subset(subset_bounds)

    assert subset_grid.resolution == grid.resolution
    assert subset_grid.bounds == subset_bounds
    assert subset_grid.num_longitude_points <= grid.num_longitude_points
    assert subset_grid.num_latitude_points <= grid.num_latitude_points


def test_coordinate_grid_validation():
    """Test coordinate grid input validation"""
    # Test invalid bounds (South >= North)
    with pytest.raises(ValueError):
        CoordinateGrid(resolution=1.0, bounds=[50, -120, 30, -100])  # South > North

    # Test invalid bounds (West >= East)
    with pytest.raises(ValueError):
        CoordinateGrid(resolution=1.0, bounds=[30, -100, 40, -120])  # West > East

    # Test invalid bounds format
    with pytest.raises(ValueError):
        CoordinateGrid(resolution=1.0, bounds=[30, -120, 40])  # Too few elements


if __name__ == '__main__':
    pytest.main([__file__])