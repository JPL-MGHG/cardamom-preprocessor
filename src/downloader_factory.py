#!/usr/bin/env python3
"""
Downloader Factory and Retry Management for CARDAMOM

Provides factory pattern for creating downloader instances and
robust retry mechanisms for handling download failures.
"""

import time
import random
from typing import Dict, List, Optional, Any, Callable
import logging
from .base_downloader import BaseDownloader
from .ecmwf_downloader import ECMWFDownloader
from .noaa_downloader import NOAADownloader
from .gfed_downloader import GFEDDownloader
from .modis_downloader import MODISDownloader


class DownloaderFactory:
    """
    Factory class for creating data downloader instances.

    Provides a centralized way to create and configure downloaders for
    different data sources used in CARDAMOM preprocessing. Supports
    dependency checking and configuration validation.
    """

    # Registry of available downloaders
    DOWNLOADERS = {
        'ecmwf': ECMWFDownloader,
        'noaa': NOAADownloader,
        'gfed': GFEDDownloader,
        'modis': MODISDownloader
    }

    # Dependencies required for each downloader
    DEPENDENCIES = {
        'ecmwf': ['cdsapi'],
        'noaa': ['ftplib', 'xarray'],
        'gfed': ['requests', 'h5py'],
        'modis': ['requests', 'xarray', 'scipy']
    }

    @staticmethod
    def create_downloader(source: str, config: Optional[Dict[str, Any]] = None) -> BaseDownloader:
        """
        Create downloader instance for specific data source.

        Args:
            source: Data source identifier ('ecmwf', 'noaa', 'gfed', 'modis')
            config: Configuration dictionary for the downloader

        Returns:
            BaseDownloader: Configured downloader instance

        Raises:
            ValueError: If source is unknown or dependencies are missing
        """
        if source not in DownloaderFactory.DOWNLOADERS:
            available_sources = list(DownloaderFactory.DOWNLOADERS.keys())
            raise ValueError(f"Unknown downloader source: {source}. Available: {available_sources}")

        # Check dependencies
        missing_deps = DownloaderFactory.check_downloader_dependencies(source)
        if missing_deps:
            raise ValueError(f"Missing dependencies for {source}: {missing_deps}")

        # Get downloader class
        downloader_class = DownloaderFactory.DOWNLOADERS[source]

        # Apply configuration
        if config is None:
            config = {}

        # Create downloader instance
        try:
            downloader = downloader_class(**config)
            logging.getLogger(__name__).info(f"Created {source} downloader successfully")
            return downloader

        except Exception as e:
            raise ValueError(f"Failed to create {source} downloader: {e}")

    @staticmethod
    def check_downloader_dependencies(source: str) -> List[str]:
        """
        Check dependencies for specific downloader.

        Args:
            source: Data source identifier

        Returns:
            list: List of missing dependencies (empty if all available)
        """
        if source not in DownloaderFactory.DEPENDENCIES:
            return []

        required_deps = DownloaderFactory.DEPENDENCIES[source]
        missing_deps = []

        for dep in required_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)

        return missing_deps

    @staticmethod
    def get_available_downloaders() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available downloaders and their status.

        Returns:
            dict: Available downloaders with dependency status
        """
        downloaders_info = {}

        for source in DownloaderFactory.DOWNLOADERS:
            missing_deps = DownloaderFactory.check_downloader_dependencies(source)
            downloaders_info[source] = {
                'class': DownloaderFactory.DOWNLOADERS[source].__name__,
                'dependencies': DownloaderFactory.DEPENDENCIES.get(source, []),
                'missing_dependencies': missing_deps,
                'available': len(missing_deps) == 0
            }

        return downloaders_info

    @staticmethod
    def create_all_downloaders(configs: Dict[str, Dict[str, Any]]) -> Dict[str, BaseDownloader]:
        """
        Create multiple downloader instances from configuration.

        Args:
            configs: Dictionary mapping source names to their configurations

        Returns:
            dict: Dictionary of created downloaders
        """
        downloaders = {}

        for source, config in configs.items():
            try:
                downloaders[source] = DownloaderFactory.create_downloader(source, config)
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to create {source} downloader: {e}")

        return downloaders


class RetryManager:
    """
    Handle download failures and retry logic for robust data acquisition.

    Provides configurable retry mechanisms with exponential backoff,
    error categorization, and recovery strategies for different types
    of download failures.
    """

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0, base_delay: float = 1.0):
        """
        Initialize retry manager.

        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            base_delay: Base delay in seconds before first retry
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.base_delay = base_delay
        self.logger = logging.getLogger(self.__class__.__name__)

        # Track retry statistics
        self.retry_stats = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_after_retries': 0,
            'error_types': {}
        }

    def download_with_retry(self, download_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute download function with retry logic.

        Args:
            download_func: Function to execute with retries
            *args: Arguments for download function
            **kwargs: Keyword arguments for download function

        Returns:
            dict: Download results with retry information
        """
        self.retry_stats['total_attempts'] += 1

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Calculate delay for this attempt (exponential backoff with jitter)
                if attempt > 0:
                    delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
                    # Add random jitter to avoid thundering herd
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter

                    self.logger.info(f"Retry attempt {attempt}/{self.max_retries} after {total_delay:.1f}s delay")
                    time.sleep(total_delay)

                # Execute download function
                result = download_func(*args, **kwargs)

                # Check if download was successful
                if isinstance(result, dict) and result.get('status') == 'success':
                    if attempt > 0:
                        self.retry_stats['successful_retries'] += 1
                        self.logger.info(f"Download succeeded on retry attempt {attempt}")

                    result['retry_info'] = {
                        'attempts_made': attempt + 1,
                        'retry_successful': attempt > 0
                    }
                    return result

                # If not successful, treat as error for retry logic
                if isinstance(result, dict) and 'error' in result:
                    last_error = result['error']
                else:
                    last_error = "Unknown error in download result"

            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")

            # Record error type for statistics
            error_type = self._categorize_error(last_error)
            self.retry_stats['error_types'][error_type] = self.retry_stats['error_types'].get(error_type, 0) + 1

            # Check if this error type should be retried
            if not self._should_retry_error(last_error):
                self.logger.error(f"Error type not suitable for retry: {last_error}")
                break

        # All retries exhausted
        self.retry_stats['failed_after_retries'] += 1
        self.logger.error(f"Download failed after {self.max_retries + 1} attempts. Last error: {last_error}")

        return {
            'status': 'failed',
            'error': last_error,
            'retry_info': {
                'attempts_made': self.max_retries + 1,
                'retry_successful': False,
                'max_retries_exhausted': True
            }
        }

    def _categorize_error(self, error_message: str) -> str:
        """
        Categorize error type for retry decision making.

        Args:
            error_message: Error message to categorize

        Returns:
            str: Error category
        """
        error_lower = str(error_message).lower()

        # Network-related errors (usually retriable)
        if any(keyword in error_lower for keyword in ['timeout', 'connection', 'network', 'dns']):
            return 'network'

        # Server errors (retriable)
        if any(keyword in error_lower for keyword in ['500', '502', '503', '504', 'server error']):
            return 'server'

        # Rate limiting (retriable with longer delay)
        if any(keyword in error_lower for keyword in ['rate limit', 'too many requests', '429']):
            return 'rate_limit'

        # Authentication errors (usually not retriable)
        if any(keyword in error_lower for keyword in ['auth', 'credential', 'unauthorized', '401', '403']):
            return 'authentication'

        # File system errors (may be retriable)
        if any(keyword in error_lower for keyword in ['disk space', 'permission', 'file exists']):
            return 'filesystem'

        # Data validation errors (usually not retriable)
        if any(keyword in error_lower for keyword in ['validation', 'corrupt', 'invalid format']):
            return 'validation'

        return 'unknown'

    def _should_retry_error(self, error_message: str) -> bool:
        """
        Determine if an error should be retried.

        Args:
            error_message: Error message to evaluate

        Returns:
            bool: True if error should be retried
        """
        error_category = self._categorize_error(error_message)

        # Retriable error types
        retriable_categories = ['network', 'server', 'rate_limit', 'unknown']

        # Non-retriable error types
        non_retriable_categories = ['authentication', 'validation']

        if error_category in retriable_categories:
            return True
        elif error_category in non_retriable_categories:
            return False
        else:
            # For filesystem errors, retry once
            return error_category == 'filesystem'

    def handle_network_errors(self, error: Exception) -> Dict[str, Any]:
        """
        Handle specific network-related errors.

        Args:
            error: Network-related exception

        Returns:
            dict: Error handling recommendations
        """
        error_str = str(error)
        error_type = type(error).__name__

        recommendations = {
            'error_type': error_type,
            'error_message': error_str,
            'recommended_action': 'retry',
            'suggested_delay': self.base_delay
        }

        # Specific handling for different network errors
        if 'timeout' in error_str.lower():
            recommendations.update({
                'suggested_delay': self.base_delay * 2,
                'recommendation': 'Increase timeout or retry with longer delay'
            })
        elif 'connection refused' in error_str.lower():
            recommendations.update({
                'suggested_delay': self.base_delay * 3,
                'recommendation': 'Server may be temporarily unavailable'
            })
        elif 'dns' in error_str.lower():
            recommendations.update({
                'recommended_action': 'check_configuration',
                'recommendation': 'Verify server URL and DNS settings'
            })

        return recommendations

    def handle_server_errors(self, error: Exception) -> Dict[str, Any]:
        """
        Handle server-side errors (500, 503, etc.).

        Args:
            error: Server-related exception

        Returns:
            dict: Error handling recommendations
        """
        error_str = str(error)

        recommendations = {
            'error_type': type(error).__name__,
            'error_message': error_str,
            'recommended_action': 'retry',
            'suggested_delay': self.base_delay * 2
        }

        # Handle specific HTTP status codes
        if '503' in error_str or 'service unavailable' in error_str.lower():
            recommendations.update({
                'suggested_delay': self.base_delay * 5,
                'recommendation': 'Server temporarily overloaded, retry with longer delay'
            })
        elif '500' in error_str or 'internal server error' in error_str.lower():
            recommendations.update({
                'suggested_delay': self.base_delay * 3,
                'recommendation': 'Server internal error, may resolve automatically'
            })

        return recommendations

    def get_retry_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about retry operations.

        Returns:
            dict: Retry statistics and performance metrics
        """
        total_attempts = self.retry_stats['total_attempts']

        if total_attempts == 0:
            return {'message': 'No retry operations performed yet'}

        success_rate = (total_attempts - self.retry_stats['failed_after_retries']) / total_attempts
        retry_success_rate = (self.retry_stats['successful_retries'] /
                             max(1, self.retry_stats['successful_retries'] + self.retry_stats['failed_after_retries']))

        return {
            'total_download_attempts': total_attempts,
            'successful_on_first_try': total_attempts - self.retry_stats['successful_retries'] - self.retry_stats['failed_after_retries'],
            'successful_after_retry': self.retry_stats['successful_retries'],
            'failed_after_all_retries': self.retry_stats['failed_after_retries'],
            'overall_success_rate': success_rate,
            'retry_success_rate': retry_success_rate,
            'error_type_distribution': self.retry_stats['error_types'].copy(),
            'configuration': {
                'max_retries': self.max_retries,
                'backoff_factor': self.backoff_factor,
                'base_delay': self.base_delay
            }
        }