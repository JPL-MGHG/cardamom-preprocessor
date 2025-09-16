#!/usr/bin/env python3
"""
Base Downloader Abstract Class for CARDAMOM Data Sources

Provides common interface and functionality for all data downloaders used in
CARDAMOM preprocessing pipeline. Each specific downloader inherits from this
base class to ensure consistent behavior and error handling.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging


class BaseDownloader(ABC):
    """
    Abstract base class for all CARDAMOM data downloaders.

    Provides common functionality for directory management, file validation,
    and download status tracking that all data source downloaders require.

    Scientific Context:
    Each downloader handles a specific external data source needed for CARDAMOM
    carbon cycle modeling: meteorological data (ECMWF), atmospheric CO2 (NOAA),
    fire emissions (GFED), and land-sea masks (MODIS).
    """

    def __init__(self, output_dir: str):
        """
        Initialize base downloader with output directory.

        Args:
            output_dir: Directory for final downloaded data files
        """
        self.output_dir = output_dir
        self.setup_directories()

        # Track download operations for status reporting
        self.download_history = []
        self.failed_downloads = []

        # Set up logging for this downloader
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup_directories(self) -> None:
        """Create necessary output directory."""
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def download_data(self, **kwargs) -> Dict[str, Any]:
        """
        Download data from the specific data source.

        This method must be implemented by each downloader to handle
        the specific requirements and API of their data source.

        Returns:
            dict: Download results with status information and file paths
        """
        pass

    def check_existing_files(self, file_pattern: str) -> List[str]:
        """
        Check for existing files matching a pattern to avoid re-downloading.

        Args:
            file_pattern: Pattern to match existing files (can include wildcards)

        Returns:
            list: Paths to existing files that match the pattern
        """
        import glob

        search_path = os.path.join(self.output_dir, file_pattern)
        existing_files = glob.glob(search_path)

        if existing_files:
            self.logger.info(f"Found {len(existing_files)} existing files matching {file_pattern}")

        return existing_files

    def validate_downloaded_data(self, filepath: str) -> bool:
        """
        Validate that downloaded data file is complete and readable.

        Args:
            filepath: Path to downloaded file to validate

        Returns:
            bool: True if file passes validation, False otherwise
        """
        # Basic file existence and size checks
        if not os.path.exists(filepath):
            self.logger.error(f"Downloaded file does not exist: {filepath}")
            return False

        # Check file is not empty
        file_size_bytes = os.path.getsize(filepath)
        if file_size_bytes == 0:
            self.logger.error(f"Downloaded file is empty: {filepath}")
            return False

        # Check minimum reasonable file size (1 KB)
        if file_size_bytes < 1024:
            self.logger.warning(f"Downloaded file suspiciously small ({file_size_bytes} bytes): {filepath}")

        self.logger.info(f"File validation passed: {filepath} ({file_size_bytes} bytes)")
        return True

    def get_download_status(self) -> Dict[str, Any]:
        """
        Get status information about download operations.

        Returns:
            dict: Status information including successful and failed downloads
        """
        return {
            'total_downloads_attempted': len(self.download_history),
            'successful_downloads': [d for d in self.download_history if d['status'] == 'success'],
            'failed_downloads': self.failed_downloads,
            'output_directory': self.output_dir,
        }

    def _record_download_attempt(self, filename: str, status: str, error_message: str = None) -> None:
        """
        Record information about a download attempt for status tracking.

        Args:
            filename: Name of file being downloaded
            status: 'success' or 'failed'
            error_message: Error message if download failed
        """
        download_record = {
            'filename': filename,
            'status': status,
            'timestamp': time.time(),
            'output_path': os.path.join(self.output_dir, filename)
        }

        if error_message:
            download_record['error'] = error_message

        self.download_history.append(download_record)

        if status == 'failed':
            self.failed_downloads.append(download_record)

