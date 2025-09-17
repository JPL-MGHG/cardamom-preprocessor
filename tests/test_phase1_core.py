"""
Basic functional tests for Phase 1 core components.

These tests focus on functionality and integration rather than extensive edge cases,
following the project's simple testing approach.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import core Phase 1 components
try:
    from src.cardamom_preprocessor import CARDAMOMProcessor
    from src.config_manager import CardamomConfig
    from src.coordinate_systems import StandardGrids
    from src.netcdf_infrastructure import CARDAMOMNetCDFWriter
except ImportError as e:
    print(f"Warning: Could not import Phase 1 modules: {e}")
    pytest.skip("Phase 1 modules not available", allow_module_level=True)


class TestCARDAMOMProcessor:
    """Test CARDAMOMProcessor main orchestration class"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            'processing': {
                'output_directory': self.temp_dir,
                'temp_directory': f"{self.temp_dir}/temp",
                'log_level': 'INFO'
            }
        }

    def test_processor_initialization(self):
        """Test that CARDAMOMProcessor initializes correctly"""
        with patch('src.cardamom_preprocessor.CardamomConfig') as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                'processing.output_directory': self.temp_dir,
                'processing.log_level': 'INFO',
                'processing.compression': True
            }.get(key, default)
            mock_config.return_value.get_quality_control_config.return_value = {'enable_validation': True}

            processor = CARDAMOMProcessor()

            assert hasattr(processor, 'config')
            assert hasattr(processor, 'coordinate_systems')
            assert hasattr(processor, 'netcdf_writer')
            assert hasattr(processor, 'qa_system')

    def test_validate_inputs(self):
        """Test input validation"""
        with patch('src.cardamom_preprocessor.CardamomConfig'):
            with patch('src.cardamom_preprocessor.setup_cardamom_logging'):
                with patch('src.cardamom_preprocessor.ProcessingLogger'):
                    processor = CARDAMOMProcessor()

                    # Valid inputs
                    assert processor.validate_inputs([2020, 2021], [1, 2, 3]) == True

                    # Invalid month
                    assert processor.validate_inputs([2020], [13]) == False

    @patch('src.cardamom_preprocessor.ECMWFDownloader')
    def test_download_era5_data(self, mock_downloader_class):
        """Test ERA5 data download functionality"""
        # Setup mocks
        mock_downloader = Mock()
        mock_downloader_class.return_value = mock_downloader

        with patch('src.cardamom_preprocessor.CardamomConfig') as mock_config:
            mock_config.return_value.get_downloader_config.return_value = {'api_timeout': 3600}
            mock_config.return_value.get.return_value = self.temp_dir

            with patch('src.cardamom_preprocessor.setup_cardamom_logging'):
                with patch('src.cardamom_preprocessor.ProcessingLogger'):
                    processor = CARDAMOMProcessor()

                    # Create a test file to simulate download
                    test_file = Path(self.temp_dir) / "temp" / "downloads" / "CARDAMOM_MONTHLY_temperature_012020.nc"
                    test_file.parent.mkdir(parents=True, exist_ok=True)
                    test_file.touch()

                    # Test download
                    result = processor._download_era5_data([2020], [1], ['temperature'], 'monthly')

                    assert isinstance(result, list)
                    mock_downloader.download_monthly_data.assert_called()


class TestCardamomConfig:
    """Test CardamomConfig unified configuration system"""

    def test_default_config_creation(self):
        """Test that default configuration is created correctly"""
        config = CardamomConfig()

        # Check that all required sections exist
        assert 'processing' in config.to_dict()
        assert 'global_monthly' in config.to_dict()
        assert 'conus_diurnal' in config.to_dict()
        assert 'downloaders' in config.to_dict()

    def test_config_get_method(self):
        """Test configuration value retrieval"""
        config = CardamomConfig()

        # Test getting nested values
        output_dir = config.get('processing.output_directory')
        assert output_dir == "./DATA/CARDAMOM-MAPS_05deg_MET/"

        # Test default value
        non_existent = config.get('non.existent.key', 'default_value')
        assert non_existent == 'default_value'

    def test_workflow_config_retrieval(self):
        """Test workflow-specific configuration retrieval"""
        config = CardamomConfig()

        # Test global monthly config
        global_config = config.get_workflow_config('global_monthly')
        assert 'resolution' in global_config
        assert 'variables' in global_config

        # Test CONUS diurnal config
        diurnal_config = config.get_workflow_config('conus_diurnal')
        assert 'region' in diurnal_config
        assert 'diurnal_hours' in diurnal_config

    def test_environment_variable_loading(self):
        """Test environment variable configuration override"""
        with patch.dict('os.environ', {'CARDAMOM_OUTPUT_DIR': '/custom/output'}):
            config = CardamomConfig()
            output_dir = config.get('processing.output_directory')
            assert output_dir == '/custom/output'

    def test_template_generation(self):
        """Test configuration template generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test minimal template
            template_path = CardamomConfig.create_template_config(
                template_type='minimal',
                output_path=f"{temp_dir}/test_minimal.yaml"
            )

            assert Path(template_path).exists()

            # Verify template content has expected structure
            with open(template_path, 'r') as f:
                content = f.read()
                assert 'processing:' in content
                assert 'global_monthly:' in content

    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        config = CardamomConfig()
        # Should not raise any exceptions
        assert config.to_dict() is not None

        # Test invalid configuration would be tested with specific invalid configs


class TestCoordinateGrids:
    """Test coordinate system functionality"""

    def test_standard_grids_creation(self):
        """Test that standard grids can be created"""
        # Test global 0.5 degree grid
        global_grid = StandardGrids.create_global_half_degree()
        assert hasattr(global_grid, 'longitude')
        assert hasattr(global_grid, 'latitude')

        # Test CONUS grid
        conus_grid = StandardGrids.create_conus_half_degree()
        assert hasattr(conus_grid, 'longitude')
        assert hasattr(conus_grid, 'latitude')

    def test_grid_bounds_validation(self):
        """Test grid bounds are reasonable"""
        global_grid = StandardGrids.create_global_half_degree()

        # Check global coverage
        assert min(global_grid.longitude) <= -179
        assert max(global_grid.longitude) >= 179
        assert min(global_grid.latitude) <= -89
        assert max(global_grid.latitude) >= 89


class TestCARDAMOMNetCDFWriter:
    """Test NetCDF writing infrastructure"""

    def test_netcdf_writer_initialization(self):
        """Test NetCDF writer initializes correctly"""
        writer = CARDAMOMNetCDFWriter()
        assert hasattr(writer, 'dimension_manager')
        assert hasattr(writer, 'data_variable_manager')
        assert hasattr(writer, 'metadata_manager')

    def test_input_validation(self):
        """Test input validation for NetCDF creation"""
        writer = CARDAMOMNetCDFWriter()

        # Test 2D validation
        valid_2d_data = {
            'filename': 'test.nc',
            'x': [1, 2, 3],
            'y': [1, 2],
            'data': [[1, 2, 3], [4, 5, 6]],
            'info': {'name': 'test_var', 'units': 'test_units'}
        }

        # Should not raise exception
        writer._validate_2d_input(valid_2d_data)

        # Test invalid data shape
        invalid_2d_data = valid_2d_data.copy()
        invalid_2d_data['data'] = [[1, 2]]  # Wrong shape

        with pytest.raises(ValueError):
            writer._validate_2d_input(invalid_2d_data)


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms"""

    def test_retry_operation_success(self):
        """Test successful operation with retry mechanism"""
        with patch('src.cardamom_preprocessor.CardamomConfig'):
            with patch('src.cardamom_preprocessor.setup_cardamom_logging'):
                with patch('src.cardamom_preprocessor.ProcessingLogger'):
                    processor = CARDAMOMProcessor()

                    # Mock operation that succeeds
                    def mock_operation():
                        return "success"

                    result = processor._retry_operation(mock_operation, "test_operation")
                    assert result == "success"

    def test_retry_operation_failure_then_success(self):
        """Test operation that fails once then succeeds"""
        with patch('src.cardamom_preprocessor.CardamomConfig'):
            with patch('src.cardamom_preprocessor.setup_cardamom_logging'):
                with patch('src.cardamom_preprocessor.ProcessingLogger'):
                    processor = CARDAMOMProcessor()
                    processor.error_recovery = {'max_retries': 2, 'retry_delay': 0}

                    # Mock operation that fails once then succeeds
                    call_count = 0
                    def mock_operation():
                        nonlocal call_count
                        call_count += 1
                        if call_count == 1:
                            raise Exception("First attempt fails")
                        return "success"

                    with patch('time.sleep'):  # Skip actual sleep
                        result = processor._retry_operation(mock_operation, "test_operation")
                        assert result == "success"
                        assert call_count == 2


class TestProcessingState:
    """Test processing state management for resumability"""

    def test_processing_state_creation(self):
        """Test processing state initialization"""
        with patch('src.cardamom_preprocessor.CardamomConfig'):
            with patch('src.cardamom_preprocessor.setup_cardamom_logging'):
                with patch('src.cardamom_preprocessor.ProcessingLogger'):
                    processor = CARDAMOMProcessor()

                    assert hasattr(processor, 'processing_state')
                    assert 'completed_operations' in processor.processing_state
                    assert 'failed_operations' in processor.processing_state

    def test_task_completion_tracking(self):
        """Test task completion tracking"""
        with patch('src.cardamom_preprocessor.CardamomConfig'):
            with patch('src.cardamom_preprocessor.setup_cardamom_logging'):
                with patch('src.cardamom_preprocessor.ProcessingLogger'):
                    processor = CARDAMOMProcessor()

                    # Test marking task as completed
                    result = {'status': 'success', 'files_created': ['test.nc']}
                    processor._mark_task_completed('global_monthly', 2020, 1, result)

                    # Test checking if task is completed
                    assert processor._is_task_completed('global_monthly', 2020, 1) == True
                    assert processor._is_task_completed('global_monthly', 2020, 2) == False


# Integration test
class TestPhase1Integration:
    """Test integration between Phase 1 components"""

    def test_processor_with_real_config(self):
        """Test processor initialization with real configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test configuration file
            test_config = {
                'processing': {
                    'output_directory': temp_dir,
                    'log_level': 'INFO'
                },
                'global_monthly': {
                    'resolution': 0.5,
                    'variables': {'era5': ['temperature']}
                },
                'downloaders': {
                    'era5': {'api_timeout': 3600}
                },
                'quality_control': {'enable_validation': True}
            }

            config_file = Path(temp_dir) / "test_config.json"
            with open(config_file, 'w') as f:
                json.dump(test_config, f)

            # Test processor with configuration file
            with patch('src.cardamom_preprocessor.setup_cardamom_logging'):
                with patch('src.cardamom_preprocessor.ProcessingLogger'):
                    processor = CARDAMOMProcessor(config_file=str(config_file))

                    assert processor.output_dir == Path(temp_dir)
                    assert processor.config.get('processing.log_level') == 'INFO'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])