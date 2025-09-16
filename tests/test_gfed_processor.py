"""
Basic validation tests for GFED Processing Module.

Tests core functionality of GFEDProcessor against expected behavior
from MATLAB CARDAMOM_MAPS_READ_GFED_NOV24.m implementation.
"""

import pytest
import numpy as np
import tempfile
from unittest.mock import Mock, patch

from gfed_processor import GFEDProcessor, GFEDData
from coordinate_systems import load_land_sea_mask, convert_to_geoschem_grid


class TestGFEDProcessor:
    """Test suite for GFEDProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create GFEDProcessor instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield GFEDProcessor(data_dir=temp_dir)

    @pytest.fixture
    def sample_gfed_data(self):
        """Create sample GFED data arrays for testing."""
        # Create sample 0.25 degree resolution data (720 x 1440 grid)
        n_lat, n_lon = 120, 240  # Smaller for testing
        n_months = 24  # 2 years of data

        # Sample burned area data
        burned_area = np.random.rand(n_lat, n_lon, n_months) * 0.1  # 0-10% burned
        burned_area[burned_area < 0.05] = 0  # Most cells have no fire

        # Sample carbon emissions data
        carbon_emissions = burned_area * np.random.rand(n_lat, n_lon, n_months) * 100

        return burned_area, carbon_emissions

    def test_temporal_coordinates_creation(self, processor):
        """Test temporal coordinate creation matches MATLAB logic."""
        # Test equivalent to MATLAB lines 75-77
        start_year, end_year = 2001, 2003

        years, months = processor._create_temporal_coordinates(start_year, end_year)

        # Expected number of months
        expected_months = (end_year - start_year + 1) * 12
        assert len(years) == expected_months
        assert len(months) == expected_months

        # Check year values
        assert years[0] == start_year
        assert years[-1] == end_year

        # Check month cycling (1-12)
        assert np.all((months >= 1) & (months <= 12))
        assert months[0] == 1  # January
        assert months[11] == 12  # December
        assert months[12] == 1  # January of next year

    def test_land_sea_mask_application(self, processor, sample_gfed_data):
        """Test land-sea mask application logic."""
        burned_area, carbon_emissions = sample_gfed_data

        # Apply land-sea mask
        ba_masked, ce_masked = processor._apply_land_sea_mask(burned_area, carbon_emissions)

        # Check that output shapes match input
        assert ba_masked.shape == burned_area.shape
        assert ce_masked.shape == carbon_emissions.shape

        # Check that some values are NaN (sea regions)
        assert np.any(np.isnan(ba_masked))
        assert np.any(np.isnan(ce_masked))

        # Check that some values are zero (land fire-free regions)
        assert np.any(ba_masked == 0)
        assert np.any(ce_masked == 0)

    def test_resolution_conversion_05deg(self, processor, sample_gfed_data):
        """Test 0.5 degree resolution conversion."""
        burned_area, carbon_emissions = sample_gfed_data

        # Convert to 0.5 degree resolution
        result = processor._convert_resolution(burned_area, carbon_emissions, '05deg')

        # Check output structure
        assert 'BA' in result
        assert 'FireC' in result

        # Check dimension reduction (should be roughly half in each spatial dimension)
        assert result['BA'].shape[0] <= burned_area.shape[0] // 2 + 1
        assert result['BA'].shape[1] <= burned_area.shape[1] // 2 + 1
        assert result['BA'].shape[2] == burned_area.shape[2]  # Time unchanged

    def test_resolution_conversion_025deg(self, processor, sample_gfed_data):
        """Test native 0.25 degree resolution (no conversion)."""
        burned_area, carbon_emissions = sample_gfed_data

        # Keep native resolution
        result = processor._convert_resolution(burned_area, carbon_emissions, '0.25deg')

        # Check that shapes are preserved
        assert result['BA'].shape == burned_area.shape
        # FireC should have unit conversion applied
        assert result['FireC'].shape == carbon_emissions.shape

    def test_gap_filling_logic(self, processor):
        """Test gap-filling for missing years."""
        # Create test data with reference period (2001-2016) and missing years
        n_lat, n_lon = 50, 100
        n_months_ref = 16 * 12  # 2001-2016
        n_months_total = 20 * 12  # 2001-2020

        # Create sample data
        ba_data = np.random.rand(n_lat, n_lon, n_months_total) * 0.1
        firec_data = np.random.rand(n_lat, n_lon, n_months_total) * 50

        # Set post-2016 data to NaN to simulate missing data
        ba_data[:, :, n_months_ref:] = np.nan

        gfed_data = {'BA': ba_data, 'FireC': firec_data}
        years = list(range(2001, 2021))

        # Apply gap-filling
        result = processor._fill_missing_years(gfed_data, years)

        # Check that NaN values were filled
        assert not np.all(np.isnan(result['BA'][:, :, n_months_ref:]))

        # Check that reference period data unchanged
        np.testing.assert_array_equal(
            result['BA'][:, :, :n_months_ref],
            ba_data[:, :, :n_months_ref]
        )

    def test_invalid_resolution_raises_error(self, processor, sample_gfed_data):
        """Test that invalid resolution raises appropriate error."""
        burned_area, carbon_emissions = sample_gfed_data

        with pytest.raises(ValueError, match="Unsupported resolution"):
            processor._convert_resolution(burned_area, carbon_emissions, 'invalid_res')

    @patch('src.gfed_processor.GFEDDownloader')
    @patch('src.gfed_processor.GFEDReader')
    def test_process_gfed_data_integration(self, mock_reader, mock_downloader, processor):
        """Test full processing pipeline integration."""
        # Mock downloader behavior
        mock_downloader_instance = Mock()
        mock_downloader_instance.download_yearly_files.return_value = {
            'downloaded_files': ['test_file.hdf5']
        }
        mock_downloader.return_value = mock_downloader_instance

        # Mock reader behavior
        mock_reader_instance = Mock()
        mock_reader_instance.extract_monthly_data.return_value = {
            'burned_area': np.random.rand(100, 200),
            'total_emission': np.random.rand(100, 200) * 50
        }
        mock_reader.return_value = mock_reader_instance

        # Test processing
        with patch.object(processor, '_apply_land_sea_mask') as mock_mask, \
             patch.object(processor, '_convert_resolution') as mock_convert, \
             patch.object(processor, '_fill_missing_years') as mock_fill:

            # Set up mock returns
            mock_mask.return_value = (np.random.rand(100, 200, 12), np.random.rand(100, 200, 12))
            mock_convert.return_value = {'BA': np.random.rand(50, 100, 12), 'FireC': np.random.rand(50, 100, 12)}
            mock_fill.return_value = {'BA': np.random.rand(50, 100, 12), 'FireC': np.random.rand(50, 100, 12)}

            # Run processing
            result = processor.process_gfed_data(
                target_resolution='05deg',
                start_year=2020,
                end_year=2020
            )

            # Check result type and basic properties
            assert isinstance(result, GFEDData)
            assert result.resolution == '05deg'
            assert 'burned_area' in result.units
            assert 'fire_carbon' in result.units

    def test_netcdf_output_integration(self, processor):
        """Test NetCDF file creation integration."""
        # Create sample GFEDData
        sample_data = GFEDData(
            burned_area=np.random.rand(50, 100, 12),
            fire_carbon=np.random.rand(50, 100, 12) * 50,
            year=np.array([2020] * 12),
            month=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            resolution='05deg',
            units={'burned_area': 'fraction_of_cell', 'fire_carbon': 'g_C_m-2_month-1'}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test NetCDF file creation
            created_files = processor.create_cardamom_netcdf_files(
                sample_data, temp_dir, "monthly"
            )

            # Check that files were created
            assert len(created_files) > 0
            assert len(created_files) == 24  # 12 months × 2 variables

            # Check file naming convention
            for filepath in created_files:
                filename = filepath.split('/')[-1]
                assert filename.startswith('CARDAMOM_GFED_')
                assert filename.endswith('.nc')
                assert '05deg' in filename

    def test_cardamom_naming_convention(self, processor):
        """Test CARDAMOM file naming convention."""
        # Test burned area filename
        ba_filename = processor._apply_cardamom_naming_convention(
            "burned_area", 2020, 3, "05deg", "/test/dir"
        )
        expected_ba = "/test/dir/CARDAMOM_GFED_burned_area_05deg_202003.nc"
        assert ba_filename == expected_ba

        # Test fire carbon filename
        fc_filename = processor._apply_cardamom_naming_convention(
            "fire_carbon", 2021, 12, "0.25deg", "/output"
        )
        expected_fc = "/output/CARDAMOM_GFED_fire_carbon_0.25deg_202112.nc"
        assert fc_filename == expected_fc

    def test_gfed_data_netcdf_export_methods(self):
        """Test GFEDData NetCDF export functionality."""
        # Create sample GFEDData
        sample_data = GFEDData(
            burned_area=np.random.rand(20, 40, 6),
            fire_carbon=np.random.rand(20, 40, 6) * 25,
            year=np.array([2020, 2020, 2020, 2020, 2020, 2020]),
            month=np.array([1, 2, 3, 4, 5, 6]),
            resolution='05deg',
            units={'burned_area': 'fraction_of_cell', 'fire_carbon': 'g_C_m-2_month-1'}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test monthly format
            monthly_files = sample_data.to_netcdf_files(temp_dir, "monthly")
            assert len(monthly_files) == 12  # 6 months × 2 variables

            # Test yearly format
            yearly_files = sample_data.to_netcdf_files(temp_dir, "yearly")
            assert len(yearly_files) == 2  # 1 year × 2 variables

            # Test single file format
            single_files = sample_data.to_netcdf_files(temp_dir, "single")
            assert len(single_files) == 1

            # Test CARDAMOM format (structured directories)
            cardamom_result = sample_data.to_cardamom_format(temp_dir)
            assert 'burned_area_files' in cardamom_result
            assert 'fire_carbon_files' in cardamom_result
            assert len(cardamom_result['burned_area_files']) == 6
            assert len(cardamom_result['fire_carbon_files']) == 6

    def test_process_gfed_data_with_netcdf_output(self, processor):
        """Test full processing pipeline with NetCDF output."""
        with patch('src.gfed_processor.GFEDDownloader') as mock_downloader, \
             patch('src.gfed_processor.GFEDReader') as mock_reader, \
             patch.object(processor, '_apply_land_sea_mask') as mock_mask, \
             patch.object(processor, '_convert_resolution') as mock_convert, \
             patch.object(processor, '_fill_missing_years') as mock_fill, \
             tempfile.TemporaryDirectory() as temp_dir:

            # Set up mocks
            mock_downloader_instance = Mock()
            mock_downloader_instance.download_yearly_files.return_value = {
                'downloaded_files': ['test_file.hdf5']
            }
            mock_downloader.return_value = mock_downloader_instance

            mock_reader_instance = Mock()
            mock_reader_instance.extract_monthly_data.return_value = {
                'burned_area': np.random.rand(50, 100),
                'total_emission': np.random.rand(50, 100) * 30
            }
            mock_reader.return_value = mock_reader_instance

            mock_mask.return_value = (np.random.rand(50, 100, 12), np.random.rand(50, 100, 12))
            mock_convert.return_value = {'BA': np.random.rand(25, 50, 12), 'FireC': np.random.rand(25, 50, 12)}
            mock_fill.return_value = {'BA': np.random.rand(25, 50, 12), 'FireC': np.random.rand(25, 50, 12)}

            # Mock NetCDF creation to avoid actual file I/O in test
            with patch.object(processor, 'create_cardamom_netcdf_files') as mock_netcdf:
                mock_netcdf.return_value = ['file1.nc', 'file2.nc']

                # Test processing with NetCDF creation
                result = processor.process_gfed_data(
                    target_resolution='05deg',
                    start_year=2020,
                    end_year=2020,
                    create_netcdf=True,
                    output_dir=temp_dir
                )

                # Verify result
                assert isinstance(result, GFEDData)
                assert result.resolution == '05deg'

                # Verify NetCDF creation was called
                mock_netcdf.assert_called_once_with(result, temp_dir, "monthly")

    def test_cf_standard_names(self, processor):
        """Test CF-compliant standard names for variables."""
        ba_name = processor._get_cf_standard_name('burned_area')
        assert ba_name == 'burned_area_fraction'

        fc_name = processor._get_cf_standard_name('fire_carbon')
        assert fc_name == 'surface_carbon_emissions_due_to_fires'

        # Test unknown variable
        unknown_name = processor._get_cf_standard_name('unknown_var')
        assert unknown_name == 'unknown_var'


class TestCoordinateSystemFunctions:
    """Test coordinate system utility functions."""

    def test_land_sea_mask_loading(self):
        """Test land-sea mask loading function."""
        land_mask, land_fraction = load_land_sea_mask(resolution=0.5)

        # Check that arrays are returned
        assert isinstance(land_mask, np.ndarray)
        assert isinstance(land_fraction, np.ndarray)

        # Check shapes match
        assert land_mask.shape == land_fraction.shape

        # Check value ranges
        assert np.all((land_mask >= 0) & (land_mask <= 1))
        assert np.all((land_fraction >= 0) & (land_fraction <= 1))

        # Check binary mask relationship
        expected_binary = (land_fraction > 0.5).astype(float)
        np.testing.assert_array_equal(land_mask, expected_binary)

    def test_geoschem_grid_conversion_2d(self):
        """Test GeosChem grid conversion for 2D data."""
        # Create test data at 0.25 degree resolution
        test_data = np.random.rand(120, 240)  # Sample high-res grid

        # Convert to GeosChem grid
        gc_data = convert_to_geoschem_grid(test_data)

        # Check dimension reduction
        assert gc_data.shape[0] < test_data.shape[0]
        assert gc_data.shape[1] < test_data.shape[1]
        assert gc_data.ndim == 2

    def test_geoschem_grid_conversion_3d(self):
        """Test GeosChem grid conversion for 3D data."""
        # Create test data with time dimension
        test_data = np.random.rand(120, 240, 12)

        # Convert to GeosChem grid
        gc_data = convert_to_geoschem_grid(test_data)

        # Check dimensions
        assert gc_data.shape[0] < test_data.shape[0]
        assert gc_data.shape[1] < test_data.shape[1]
        assert gc_data.shape[2] == test_data.shape[2]  # Time preserved
        assert gc_data.ndim == 3

    def test_geoschem_grid_invalid_dimensions(self):
        """Test error handling for invalid data dimensions."""
        test_data = np.random.rand(10, 20, 30, 40)  # 4D data

        with pytest.raises(ValueError, match="Unsupported data dimensions"):
            convert_to_geoschem_grid(test_data)


if __name__ == "__main__":
    pytest.main([__file__])