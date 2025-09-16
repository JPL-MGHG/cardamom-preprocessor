"""
Error Handling and Logging Infrastructure for CARDAMOM Preprocessing

This module provides standardized logging and error handling capabilities
for CARDAMOM data processing workflows. It includes progress tracking,
error context management, and structured logging for scientific workflows.
"""

import logging
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager


def setup_cardamom_logging(log_level: str = "INFO",
                          log_file: Optional[str] = None,
                          console_output: bool = True) -> logging.Logger:
    """
    Setup standardized logging for CARDAMOM processing.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional path to log file
        console_output: Whether to output logs to console

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('cardamom_preprocessor')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # File logs capture everything
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ProcessingLogger:
    """
    Specialized logger for tracking CARDAMOM processing progress.

    This class provides methods for logging different types of processing
    events with appropriate context and structured information.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize processing logger.

        Args:
            logger: Logger instance to use. If None, creates default logger.
        """
        self.logger = logger or setup_cardamom_logging()
        self.processing_start_time = None
        self.current_workflow = None
        self.processing_stats = {
            'files_processed': 0,
            'files_downloaded': 0,
            'errors_encountered': 0,
            'warnings_issued': 0
        }

    def log_processing_start(self, workflow_type: str, parameters: Dict[str, Any]) -> None:
        """
        Log start of processing workflow.

        Args:
            workflow_type: Type of workflow being started
            parameters: Processing parameters dictionary
        """
        self.processing_start_time = datetime.now()
        self.current_workflow = workflow_type

        self.logger.info("=" * 60)
        self.logger.info(f"Starting CARDAMOM {workflow_type} processing")
        self.logger.info(f"Start time: {self.processing_start_time.isoformat()}")
        self.logger.info("Processing parameters:")

        for param_name, param_value in parameters.items():
            self.logger.info(f"  {param_name}: {param_value}")

        self.logger.info("=" * 60)

    def log_data_download(self, source: str, files_downloaded: List[str]) -> None:
        """
        Log successful data downloads.

        Args:
            source: Data source name (e.g., 'ERA5', 'NOAA')
            files_downloaded: List of downloaded file paths
        """
        num_files = len(files_downloaded)
        self.processing_stats['files_downloaded'] += num_files

        self.logger.info(f"Successfully downloaded {num_files} files from {source}")

        for file_path in files_downloaded:
            self.logger.debug(f"  Downloaded: {file_path}")

        if num_files > 5:  # Only show first few files if many downloaded
            self.logger.info("  (See debug logs for complete file list)")

    def log_file_processing(self, input_file: str, output_file: str, processing_time: float) -> None:
        """
        Log successful file processing.

        Args:
            input_file: Path to input file
            output_file: Path to output file
            processing_time: Processing time in seconds
        """
        self.processing_stats['files_processed'] += 1

        self.logger.info(f"Processed file: {Path(input_file).name}")
        self.logger.info(f"  Output: {Path(output_file).name}")
        self.logger.info(f"  Processing time: {processing_time:.2f} seconds")

    def log_processing_error(self, error_type: str, error_details: str, context: Optional[Dict] = None) -> None:
        """
        Log processing errors with context.

        Args:
            error_type: Type/category of error
            error_details: Detailed error description
            context: Optional context dictionary with additional information
        """
        self.processing_stats['errors_encountered'] += 1

        self.logger.error(f"Processing error ({error_type}): {error_details}")

        if context:
            self.logger.error("Error context:")
            for key, value in context.items():
                self.logger.error(f"  {key}: {value}")

    def log_processing_warning(self, warning_message: str, context: Optional[Dict] = None) -> None:
        """
        Log processing warnings with context.

        Args:
            warning_message: Warning message
            context: Optional context dictionary
        """
        self.processing_stats['warnings_issued'] += 1

        self.logger.warning(f"Processing warning: {warning_message}")

        if context:
            self.logger.warning("Warning context:")
            for key, value in context.items():
                self.logger.warning(f"  {key}: {value}")

    def log_processing_complete(self, summary_stats: Optional[Dict] = None) -> None:
        """
        Log completion of processing with summary statistics.

        Args:
            summary_stats: Optional additional statistics dictionary
        """
        if self.processing_start_time:
            processing_duration = datetime.now() - self.processing_start_time
            self.logger.info("=" * 60)
            self.logger.info(f"CARDAMOM {self.current_workflow} processing completed")
            self.logger.info(f"Total processing time: {processing_duration}")
        else:
            self.logger.info("Processing completed")

        # Log processing statistics
        self.logger.info("Processing statistics:")
        for stat_name, stat_value in self.processing_stats.items():
            self.logger.info(f"  {stat_name}: {stat_value}")

        if summary_stats:
            self.logger.info("Additional statistics:")
            for stat_name, stat_value in summary_stats.items():
                self.logger.info(f"  {stat_name}: {stat_value}")

        self.logger.info("=" * 60)

    def log_qa_results(self, qa_results: Dict[str, Any]) -> None:
        """
        Log quality assurance results.

        Args:
            qa_results: QA results dictionary
        """
        status = qa_results.get('status', 'unknown')

        if status == 'pass':
            self.logger.info("Quality assurance: All tests PASSED")
        elif status == 'warning':
            self.logger.warning("Quality assurance: Tests completed with WARNINGS")
        elif status == 'fail':
            self.logger.error("Quality assurance: Some tests FAILED")

        if 'num_tests' in qa_results:
            self.logger.info(f"QA tests run: {qa_results['num_tests']}")

        if 'qa_report_path' in qa_results:
            self.logger.info(f"Detailed QA report: {qa_results['qa_report_path']}")

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of current processing session.

        Returns:
            Dictionary with processing summary information
        """
        summary = {
            'workflow_type': self.current_workflow,
            'start_time': self.processing_start_time.isoformat() if self.processing_start_time else None,
            'current_time': datetime.now().isoformat(),
            'processing_stats': self.processing_stats.copy()
        }

        if self.processing_start_time:
            duration = datetime.now() - self.processing_start_time
            summary['elapsed_time'] = str(duration)

        return summary


class CARDAMOMError(Exception):
    """Base exception class for CARDAMOM preprocessing errors"""

    def __init__(self, message: str, context: Optional[Dict] = None):
        """
        Initialize CARDAMOM error.

        Args:
            message: Error message
            context: Optional context dictionary with additional information
        """
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.now()

    def get_full_error_info(self) -> Dict[str, Any]:
        """Get complete error information including context"""
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }


class ConfigurationError(CARDAMOMError):
    """Error in configuration or setup"""
    pass


class DataDownloadError(CARDAMOMError):
    """Error in data download process"""
    pass


class DataProcessingError(CARDAMOMError):
    """Error in data processing"""
    pass


class ValidationError(CARDAMOMError):
    """Error in data validation"""
    pass


class NetCDFError(CARDAMOMError):
    """Error in NetCDF file operations"""
    pass


@contextmanager
def error_context(operation_name: str, logger: Optional[ProcessingLogger] = None, **context_info):
    """
    Context manager for wrapping operations with error handling.

    Args:
        operation_name: Name of operation being performed
        logger: Optional ProcessingLogger instance
        **context_info: Additional context information

    Example:
        with error_context("downloading ERA5 data", logger, year=2020, month=1):
            download_era5_data(year, month)
    """
    start_time = datetime.now()

    if logger:
        logger.logger.debug(f"Starting operation: {operation_name}")

    try:
        yield
        duration = datetime.now() - start_time

        if logger:
            logger.logger.debug(f"Completed operation: {operation_name} (duration: {duration})")

    except Exception as e:
        duration = datetime.now() - start_time
        error_context_dict = {
            'operation': operation_name,
            'duration': str(duration),
            **context_info
        }

        if logger:
            logger.log_processing_error(
                error_type=type(e).__name__,
                error_details=str(e),
                context=error_context_dict
            )

        # Re-raise as CARDAMOM error if not already one
        if not isinstance(e, CARDAMOMError):
            if "download" in operation_name.lower():
                raise DataDownloadError(str(e), error_context_dict) from e
            elif "process" in operation_name.lower():
                raise DataProcessingError(str(e), error_context_dict) from e
            elif "config" in operation_name.lower():
                raise ConfigurationError(str(e), error_context_dict) from e
            else:
                raise CARDAMOMError(str(e), error_context_dict) from e
        else:
            # Update context if already a CARDAMOM error
            e.context.update(error_context_dict)
            raise


def create_processing_session_log(output_dir: str, workflow_type: str) -> str:
    """
    Create a new processing session log file.

    Args:
        output_dir: Directory for log files
        workflow_type: Type of workflow being logged

    Returns:
        Path to created log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"cardamom_{workflow_type}_{timestamp}.log"
    log_path = Path(output_dir) / "logs" / log_filename

    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return str(log_path)


def save_processing_session_summary(summary: Dict[str, Any], output_path: str) -> None:
    """
    Save processing session summary to JSON file.

    Args:
        summary: Processing summary dictionary
        output_path: Path for output summary file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Processing summary saved: {output_path}")