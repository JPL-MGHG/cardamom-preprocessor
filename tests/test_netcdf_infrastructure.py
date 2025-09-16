"""
Basic tests for NetCDF infrastructure module.

Tests basic functionality of NetCDF writer classes without
creating actual NetCDF files.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Note: These tests may fail if netCDF4 is not installed
# They test basic class initialization and validation

try:
    from netcdf_infrastructure import (
        CARDAMOMNetCDFWriter,
        DimensionManager,
        DataVariableManager,
        MetadataManager,
        TemplateGenerator
    )
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False


@pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
def test_netcdf_writer_initialization():
    """Test NetCDF writer initialization"""
    writer = CARDAMOMNetCDFWriter()

    assert writer.template_type == "3D"
    assert writer.compression is True
    assert isinstance(writer.global_attributes, dict)
    assert hasattr(writer, 'dimension_manager')
    assert hasattr(writer, 'data_variable_manager')
    assert hasattr(writer, 'metadata_manager')


@pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
def test_netcdf_writer_compression_setting():
    """Test NetCDF writer with different compression settings"""
    writer_compressed = CARDAMOMNetCDFWriter(compression=True)
    assert writer_compressed.compression is True

    writer_uncompressed = CARDAMOMNetCDFWriter(compression=False)
    assert writer_uncompressed.compression is False


@pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
def test_dimension_manager():
    """Test dimension manager initialization"""
    dim_manager = DimensionManager()

    assert hasattr(dim_manager, 'standard_dimensions')
    assert isinstance(dim_manager.standard_dimensions, dict)
    assert 'longitude' in dim_manager.standard_dimensions
    assert 'latitude' in dim_manager.standard_dimensions
    assert 'time' in dim_manager.standard_dimensions


@pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
def test_data_variable_manager():
    """Test data variable manager initialization"""
    data_manager = DataVariableManager(compression=True)

    assert data_manager.compression is True
    assert hasattr(data_manager, 'fill_value')
    assert isinstance(data_manager.fill_value, (int, float))


@pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
def test_metadata_manager():
    """Test metadata manager initialization"""
    metadata_manager = MetadataManager()

    assert hasattr(metadata_manager, 'default_global_attrs')
    assert isinstance(metadata_manager.default_global_attrs, dict)
    assert hasattr(metadata_manager, 'cardamom_version')


@pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
def test_template_generator():
    """Test template generator initialization"""
    writer = CARDAMOMNetCDFWriter()
    template_gen = TemplateGenerator(writer)

    assert hasattr(template_gen, 'writer')
    assert template_gen.writer is writer


def test_2d_data_validation():
    """Test 2D data validation without netCDF dependency"""
    if not NETCDF_AVAILABLE:
        pytest.skip("netCDF4 not available")

    writer = CARDAMOMNetCDFWriter()

    # Valid 2D data dictionary
    valid_data_dict = {
        'filename': 'test.nc',
        'x': np.arange(-179.75, 180, 0.5),
        'y': np.arange(89.75, -90, -0.5),
        'data': np.random.random((360, 720)),
        'info': {'name': 'test_var', 'units': 'test_units'}
    }

    # This should not raise an exception
    try:
        writer._validate_2d_input(valid_data_dict)
    except Exception as e:
        pytest.fail(f"Valid 2D data validation failed: {e}")


def test_2d_data_validation_failures():
    """Test 2D data validation failures"""
    if not NETCDF_AVAILABLE:
        pytest.skip("netCDF4 not available")

    writer = CARDAMOMNetCDFWriter()

    # Missing required field
    invalid_data_dict = {
        'filename': 'test.nc',
        'x': np.arange(-179.75, 180, 0.5),
        'y': np.arange(89.75, -90, -0.5),
        # Missing 'data' and 'info'
    }

    with pytest.raises(ValueError, match="Required field"):
        writer._validate_2d_input(invalid_data_dict)

    # Wrong data shape
    wrong_shape_dict = {
        'filename': 'test.nc',
        'x': np.arange(-179.75, 180, 0.5),  # 720 points
        'y': np.arange(89.75, -90, -0.5),   # 360 points
        'data': np.random.random((100, 100)),  # Wrong shape
        'info': {'name': 'test_var', 'units': 'test_units'}
    }

    with pytest.raises(ValueError, match="Data shape"):
        writer._validate_2d_input(wrong_shape_dict)


def test_3d_data_validation():
    """Test 3D data validation"""
    if not NETCDF_AVAILABLE:
        pytest.skip("netCDF4 not available")

    writer = CARDAMOMNetCDFWriter()

    # Valid 3D data dictionary
    valid_data_dict = {
        'filename': 'test.nc',
        'x': np.arange(-179.75, 180, 0.5),
        'y': np.arange(89.75, -90, -0.5),
        't': np.arange(1, 13),  # 12 months
        'data': np.random.random((360, 720, 12)),
        'info': {'name': 'test_var', 'units': 'test_units'}
    }

    # This should not raise an exception
    try:
        writer._validate_3d_input(valid_data_dict)
    except Exception as e:
        pytest.fail(f"Valid 3D data validation failed: {e}")


def test_3d_data_validation_failures():
    """Test 3D data validation failures"""
    if not NETCDF_AVAILABLE:
        pytest.skip("netCDF4 not available")

    writer = CARDAMOMNetCDFWriter()

    # Missing time coordinate
    invalid_data_dict = {
        'filename': 'test.nc',
        'x': np.arange(-179.75, 180, 0.5),
        'y': np.arange(89.75, -90, -0.5),
        'data': np.random.random((360, 720, 12)),
        'info': {'name': 'test_var', 'units': 'test_units'}
        # Missing 't' coordinate
    }

    with pytest.raises(ValueError, match="Time coordinate"):
        writer._validate_3d_input(invalid_data_dict)


if __name__ == '__main__':
    pytest.main([__file__])