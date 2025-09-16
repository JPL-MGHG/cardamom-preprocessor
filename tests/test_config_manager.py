"""
Basic tests for configuration management module.

Tests basic functionality of the CardamomConfig class including
configuration loading, validation, and access methods.
"""

import pytest
import tempfile
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config_manager import CardamomConfig


def test_config_initialization():
    """Test that configuration initializes with defaults"""
    config = CardamomConfig()

    # Check that basic sections exist
    assert 'processing' in config.to_dict()
    assert 'pipeline' in config.to_dict()
    assert 'global_monthly' in config.to_dict()
    assert 'conus_diurnal' in config.to_dict()
    assert 'downloaders' in config.to_dict()


def test_config_get_method():
    """Test configuration get method with dot notation"""
    config = CardamomConfig()

    # Test basic get
    output_dir = config.get('processing.output_directory')
    assert output_dir is not None

    # Test get with default
    missing_value = config.get('nonexistent.key', 'default_value')
    assert missing_value == 'default_value'


def test_config_sections():
    """Test configuration section access methods"""
    config = CardamomConfig()

    # Test section getters
    processing_config = config.get_processing_config()
    assert isinstance(processing_config, dict)
    assert 'output_directory' in processing_config

    pipeline_config = config.get_pipeline_config()
    assert isinstance(pipeline_config, dict)
    assert 'resume_on_failure' in pipeline_config

    # Test workflow configs
    global_config = config.get_workflow_config('global_monthly')
    assert isinstance(global_config, dict)
    assert 'resolution' in global_config

    conus_config = config.get_workflow_config('conus_diurnal')
    assert isinstance(conus_config, dict)
    assert 'region' in conus_config


def test_config_from_json_file():
    """Test loading configuration from JSON file"""
    test_config = {
        'processing': {
            'output_directory': './test_output/',
            'max_workers': 8
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        temp_file = f.name

    try:
        config = CardamomConfig(config_file=temp_file)

        # Check that file values override defaults
        assert config.get('processing.output_directory') == './test_output/'
        assert config.get('processing.max_workers') == 8

    finally:
        Path(temp_file).unlink()


def test_config_cli_override():
    """Test CLI argument override"""
    cli_args = {
        'processing': {
            'log_level': 'DEBUG'
        }
    }

    config = CardamomConfig(cli_args=cli_args)

    # Check that CLI args override defaults
    assert config.get('processing.log_level') == 'DEBUG'


def test_downloader_config():
    """Test downloader configuration access"""
    config = CardamomConfig()

    era5_config = config.get_downloader_config('era5')
    assert isinstance(era5_config, dict)
    assert 'api_timeout' in era5_config

    # Test non-existent downloader
    unknown_config = config.get_downloader_config('unknown')
    assert unknown_config == {}


def test_config_validation():
    """Test that configuration validation works"""
    # This should not raise an exception
    config = CardamomConfig()

    # Verify required sections exist
    assert config.get('processing') is not None
    assert config.get('pipeline') is not None


if __name__ == '__main__':
    pytest.main([__file__])