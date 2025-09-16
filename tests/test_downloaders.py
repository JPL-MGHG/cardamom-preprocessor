#!/usr/bin/env python3
"""
Basic Tests for CARDAMOM Downloaders

Simple test suite for Phase 2 downloaders focusing on core functionality
and scientific validation without extensive mocking or complex scenarios.
"""

import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
import sys

# Use package imports
from base_downloader import BaseDownloader
from downloader_factory import DownloaderFactory, RetryManager
from data_source_config import DataSourceConfig


class TestBaseDownloader(unittest.TestCase):
    """Test base downloader functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_base_downloader_initialization(self):
        """Test base downloader can be initialized."""
        # Create a concrete implementation for testing
        class TestDownloader(BaseDownloader):
            def download_data(self, **kwargs):
                return {"status": "success"}

        downloader = TestDownloader(self.test_dir)

        self.assertEqual(downloader.output_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_validate_downloaded_data(self):
        """Test file validation functionality."""
        class TestDownloader(BaseDownloader):
            def download_data(self, **kwargs):
                return {"status": "success"}

        downloader = TestDownloader(self.test_dir)

        # Test with non-existent file
        non_existent_file = os.path.join(self.test_dir, 'missing.txt')
        self.assertFalse(downloader.validate_downloaded_data(non_existent_file))

        # Test with empty file
        empty_file = os.path.join(self.test_dir, 'empty.txt')
        open(empty_file, 'w').close()
        self.assertFalse(downloader.validate_downloaded_data(empty_file))

        # Test with valid file
        valid_file = os.path.join(self.test_dir, 'valid.txt')
        with open(valid_file, 'w') as f:
            f.write('test content')
        self.assertTrue(downloader.validate_downloaded_data(valid_file))

    def test_download_status_tracking(self):
        """Test download status tracking functionality."""
        class TestDownloader(BaseDownloader):
            def download_data(self, **kwargs):
                self._record_download_attempt('test_file.nc', 'success')
                return {"status": "success"}

        downloader = TestDownloader(self.test_dir)
        downloader.download_data()

        status = downloader.get_download_status()
        self.assertEqual(status['total_downloads_attempted'], 1)
        self.assertEqual(len(status['successful_downloads']), 1)
        self.assertEqual(len(status['failed_downloads']), 0)


class TestDownloaderFactory(unittest.TestCase):
    """Test downloader factory functionality."""

    def test_get_available_downloaders(self):
        """Test getting available downloaders information."""
        available = DownloaderFactory.get_available_downloaders()

        # Check that all expected downloaders are listed
        expected_sources = ['ecmwf', 'noaa', 'gfed', 'modis']
        for source in expected_sources:
            self.assertIn(source, available)
            self.assertIn('class', available[source])
            self.assertIn('dependencies', available[source])
            self.assertIn('available', available[source])

    def test_check_dependencies(self):
        """Test dependency checking for downloaders."""
        # Test NOAA dependencies (should be available in most Python environments)
        noaa_missing = DownloaderFactory.check_downloader_dependencies('noaa')
        # ftplib should be available in standard library, xarray might not be
        self.assertIsInstance(noaa_missing, list)

        # Test unknown source
        unknown_missing = DownloaderFactory.check_downloader_dependencies('unknown')
        self.assertEqual(unknown_missing, [])

    def test_create_downloader_invalid_source(self):
        """Test creating downloader with invalid source."""
        with self.assertRaises(ValueError):
            DownloaderFactory.create_downloader('invalid_source')


class TestRetryManager(unittest.TestCase):
    """Test retry manager functionality."""

    def setUp(self):
        """Set up retry manager."""
        self.retry_manager = RetryManager(max_retries=2, base_delay=0.1)

    def test_retry_manager_initialization(self):
        """Test retry manager initializes correctly."""
        self.assertEqual(self.retry_manager.max_retries, 2)
        self.assertEqual(self.retry_manager.base_delay, 0.1)
        self.assertEqual(self.retry_manager.retry_stats['total_attempts'], 0)

    def test_successful_download_no_retry(self):
        """Test successful download without retry."""
        def mock_download():
            return {"status": "success", "data": "test"}

        result = self.retry_manager.download_with_retry(mock_download)

        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['retry_info']['attempts_made'], 1)
        self.assertFalse(result['retry_info']['retry_successful'])

    def test_failed_download_with_retry(self):
        """Test failed download with retry attempts."""
        def mock_download():
            return {"status": "failed", "error": "network timeout"}

        result = self.retry_manager.download_with_retry(mock_download)

        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['retry_info']['attempts_made'], 3)  # 1 + 2 retries
        self.assertTrue(result['retry_info']['max_retries_exhausted'])

    def test_error_categorization(self):
        """Test error categorization functionality."""
        # Test network error
        network_category = self.retry_manager._categorize_error("Connection timeout")
        self.assertEqual(network_category, 'network')

        # Test server error
        server_category = self.retry_manager._categorize_error("HTTP 500 server error")
        self.assertEqual(server_category, 'server')

        # Test authentication error
        auth_category = self.retry_manager._categorize_error("401 Unauthorized")
        self.assertEqual(auth_category, 'authentication')

        # Test unknown error
        unknown_category = self.retry_manager._categorize_error("Something unexpected")
        self.assertEqual(unknown_category, 'unknown')

    def test_retry_decision_logic(self):
        """Test retry decision logic."""
        # Network errors should be retried
        self.assertTrue(self.retry_manager._should_retry_error("Connection timeout"))

        # Server errors should be retried
        self.assertTrue(self.retry_manager._should_retry_error("HTTP 503 service unavailable"))

        # Authentication errors should not be retried
        self.assertFalse(self.retry_manager._should_retry_error("401 Unauthorized"))

        # Validation errors should not be retried
        self.assertFalse(self.retry_manager._should_retry_error("Data validation failed"))



class TestDataSourceConfig(unittest.TestCase):
    """Test data source configuration functionality."""

    def test_default_config_loading(self):
        """Test default configuration loading."""
        config = DataSourceConfig()

        # Check that all expected data sources are configured
        expected_sources = ['ecmwf', 'noaa', 'gfed', 'modis', 'common']
        for source in expected_sources:
            self.assertIn(source, config.config)

    def test_get_downloader_config(self):
        """Test getting configuration for specific downloaders."""
        config = DataSourceConfig()

        # Test valid source
        ecmwf_config = config.get_downloader_config('ecmwf')
        self.assertIn('base_url', ecmwf_config)
        self.assertIn('output_dir', ecmwf_config)

        # Test invalid source
        with self.assertRaises(ValueError):
            config.get_downloader_config('invalid_source')

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = DataSourceConfig()
        # Should not raise any exceptions

        # Test configuration with missing required fields
        invalid_config_data = {
            'ecmwf': {
                'base_url': ''  # Empty URL should fail validation
            }
        }

        with self.assertRaises(ValueError):
            DataSourceConfig(custom_config=invalid_config_data)

    def test_output_directory_management(self):
        """Test output directory management."""
        config = DataSourceConfig()

        # Get all output directories
        output_dirs = config.get_all_output_directories()
        self.assertIn('ecmwf', output_dirs)
        self.assertIn('noaa', output_dirs)
        self.assertIn('gfed', output_dirs)
        self.assertIn('modis', output_dirs)

        # Test directory creation (in temp directory to avoid side effects)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = {
                'ecmwf': {'output_dir': os.path.join(temp_dir, 'ecmwf_test')},
                'noaa': {'output_dir': os.path.join(temp_dir, 'noaa_test')}
            }
            test_config = DataSourceConfig(custom_config=temp_config)
            test_config.ensure_output_directories()

            # Check directories were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'ecmwf_test')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'noaa_test')))

    def test_config_summary(self):
        """Test configuration summary generation."""
        config = DataSourceConfig()
        summary = config.get_config_summary()

        self.assertIn('data_sources', summary)
        self.assertIn('output_directories', summary)

        # Check that all expected sources are in summary
        expected_sources = ['ecmwf', 'noaa', 'gfed', 'modis', 'common']
        for source in expected_sources:
            self.assertIn(source, summary['data_sources'])


class TestDownloaderIntegration(unittest.TestCase):
    """Test integration between downloaders and supporting components."""

    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_factory_with_config(self):
        """Test factory with configuration integration."""
        # Create test configuration
        test_config = DataSourceConfig()

        # Try to create downloaders (may fail due to missing dependencies)
        for source in ['ecmwf', 'noaa', 'gfed', 'modis']:
            try:
                downloader_config = test_config.get_downloader_config(source)
                downloader_config['output_dir'] = os.path.join(self.test_dir, source)

                # Check if dependencies are available
                missing_deps = DownloaderFactory.check_downloader_dependencies(source)
                if not missing_deps:
                    downloader = DownloaderFactory.create_downloader(source, downloader_config)
                    self.assertIsInstance(downloader, BaseDownloader)
                else:
                    # Skip if dependencies are missing (expected in test environment)
                    self.assertIsInstance(missing_deps, list)

            except Exception as e:
                # Some downloaders may fail due to missing optional dependencies
                # This is acceptable in the test environment
                self.assertIsInstance(e, (ValueError, ImportError))



if __name__ == '__main__':
    # Run tests with simple output
    unittest.main(verbosity=2)