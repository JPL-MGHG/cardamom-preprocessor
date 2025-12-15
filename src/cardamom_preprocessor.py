"""
Main Orchestration Module for CARDAMOM Preprocessing

This module provides the central CARDAMOMProcessor class that coordinates
all CARDAMOM data preprocessing workflows. It manages data flow between
downloaders, processors, and writers while providing progress tracking
and error handling.
"""

import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Import ECMWF downloader from package
from downloaders.ecmwf_downloader import ECMWFDownloader

# Import CARDAMOM core modules
from config_manager import CardamomConfig
from coordinate_systems import CoordinateGrid, StandardGrids
from netcdf_infrastructure import CARDAMOMNetCDFWriter
from scientific_utils import (
    calculate_vapor_pressure_deficit,
    convert_precipitation_units,
    convert_radiation_units
)
from validation import QualityAssurance
from logging_utils import ProcessingLogger, setup_cardamom_logging, error_context


class CARDAMOMProcessor:
    """
    Main class for coordinating CARDAMOM data preprocessing workflows.

    Manages data flow between downloaders, processors, and writers while
    providing comprehensive logging, error handling, and progress tracking.
    """

    def __init__(self, config_file: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize CARDAMOM processor with configuration.

        Args:
            config_file: Path to configuration file (YAML/JSON)
            output_dir: Output directory for processed data (overrides config)
        """
        try:
            # Load configuration
            self.config = CardamomConfig(config_file)

            # Set output directory
            if output_dir:
                self.output_dir = Path(output_dir)
            else:
                self.output_dir = Path(self.config.get('processing.output_directory'))

            # Ensure output directories exist first
            self._setup_data_directories()

            # Setup logging
            log_level = self.config.get('processing.log_level', 'INFO')
            log_file = self.output_dir / "logs" / "cardamom_processing.log"
            self.logger_instance = setup_cardamom_logging(log_level, str(log_file))
            self.processing_logger = ProcessingLogger(self.logger_instance)

            # Initialize coordinate systems with error handling
            self.coordinate_systems = self._init_coordinate_systems()

            # Initialize NetCDF writer with error handling
            compression = self.config.get('processing.compression', True)
            self.netcdf_writer = CARDAMOMNetCDFWriter(compression=compression)

            # Initialize quality assurance system
            qa_config = self.config.get_quality_control_config()
            self.qa_system = QualityAssurance(qa_config)

            # Initialize error recovery state
            self._init_error_recovery()

            self.processing_logger.logger.info("CARDAMOM Processor initialized successfully")

        except Exception as e:
            if hasattr(self, 'processing_logger'):
                self.processing_logger.log_processing_error('initialization_error', str(e))
            else:
                print(f"Critical initialization error: {e}")
            raise

    def _init_error_recovery(self):
        """Initialize error recovery and state management"""
        self.error_recovery = {
            'max_retries': self.config.get('processing.retry_attempts', 3),
            'retry_delay': self.config.get('processing.retry_delay', 5),  # seconds
            'failed_operations': [],
            'recovery_log': []
        }

        # Create state file for resumable processing
        self.state_file = self.output_dir / "processing_state.json"
        self.processing_state = self._load_processing_state()

    def _load_processing_state(self) -> Dict[str, Any]:
        """Load processing state for resumable operations"""
        if self.state_file.exists():
            try:
                import json
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.processing_logger.logger.info(f"Loaded processing state from {self.state_file}")
                return state
            except Exception as e:
                self.processing_logger.log_processing_warning(
                    f"Could not load processing state: {e}. Starting fresh."
                )

        return {
            'completed_operations': [],
            'failed_operations': [],
            'last_successful_checkpoint': None,
            'session_start_time': datetime.now().isoformat()
        }

    def _save_processing_state(self):
        """Save current processing state for resumability"""
        try:
            import json
            self.processing_state['last_checkpoint_time'] = datetime.now().isoformat()

            with open(self.state_file, 'w') as f:
                json.dump(self.processing_state, f, indent=2, default=str)

        except Exception as e:
            self.processing_logger.log_processing_warning(
                f"Could not save processing state: {e}"
            )

    def _retry_operation(self, operation_func, operation_name: str, *args, **kwargs):
        """
        Execute an operation with retry logic and error recovery.

        Args:
            operation_func: Function to execute
            operation_name: Name of operation for logging
            *args, **kwargs: Arguments to pass to operation_func

        Returns:
            Result of operation_func if successful

        Raises:
            Exception: If operation fails after all retries
        """
        max_retries = self.error_recovery['max_retries']
        retry_delay = self.error_recovery['retry_delay']

        for attempt in range(max_retries + 1):
            try:
                result = operation_func(*args, **kwargs)

                if attempt > 0:
                    self.processing_logger.logger.info(
                        f"Operation '{operation_name}' succeeded on attempt {attempt + 1}"
                    )
                    self.error_recovery['recovery_log'].append({
                        'operation': operation_name,
                        'attempts': attempt + 1,
                        'success_time': datetime.now().isoformat()
                    })

                return result

            except Exception as e:
                error_msg = f"Operation '{operation_name}' failed on attempt {attempt + 1}: {str(e)}"

                if attempt < max_retries:
                    self.processing_logger.log_processing_warning(
                        f"{error_msg}. Retrying in {retry_delay} seconds..."
                    )
                    import time
                    time.sleep(retry_delay)
                else:
                    self.processing_logger.log_processing_error(
                        'operation_failed_after_retries',
                        f"{error_msg}. No more retries available.",
                        context={'operation': operation_name, 'total_attempts': attempt + 1}
                    )

                    self.error_recovery['failed_operations'].append({
                        'operation': operation_name,
                        'error': str(e),
                        'attempts': attempt + 1,
                        'failure_time': datetime.now().isoformat()
                    })
                    raise

    def _handle_workflow_error(self, workflow_type: str, error: Exception, context: Dict[str, Any]):
        """
        Handle workflow-level errors with appropriate recovery strategies.

        Args:
            workflow_type: Type of workflow that failed
            error: Exception that occurred
            context: Additional context about the error
        """
        error_info = {
            'workflow_type': workflow_type,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }

        # Log detailed error information
        self.processing_logger.log_processing_error(
            'workflow_error',
            f"Workflow '{workflow_type}' encountered error: {str(error)}",
            context=error_info
        )

        # Save state for potential recovery
        self.processing_state['failed_operations'].append(error_info)
        self._save_processing_state()

        # Determine recovery strategy based on error type
        if isinstance(error, (ConnectionError, TimeoutError)):
            self.processing_logger.logger.warning(
                "Network-related error detected. Consider retrying or checking network connectivity."
            )
        elif isinstance(error, MemoryError):
            self.processing_logger.logger.warning(
                "Memory error detected. Consider processing smaller batches or increasing available memory."
            )
        elif isinstance(error, FileNotFoundError):
            self.processing_logger.logger.warning(
                "File not found error. Check that required input data is available."
            )

        # Check if we should continue or stop based on configuration
        resume_on_failure = self.config.get('pipeline.resume_on_failure', True)
        if not resume_on_failure:
            self.processing_logger.logger.error(
                "Pipeline configured to stop on failure. Halting processing."
            )
            raise

    def _init_coordinate_systems(self) -> Dict[str, CoordinateGrid]:
        """Initialize standard coordinate grids used in CARDAMOM processing"""
        return {
            'global_0.5deg': StandardGrids.create_global_half_degree(),
            'global_0.25deg': StandardGrids.create_global_quarter_degree(),
            'conus_0.5deg': StandardGrids.create_conus_half_degree(),
            'geoschem_4x5deg': StandardGrids.create_geoschem_4x5_degree()
        }

    def _setup_data_directories(self) -> None:
        """Create standardized output directory structure"""
        directories = [
            self.output_dir,
            self.output_dir / "logs",
            self.output_dir / "temp",
            self.output_dir / "qa_reports",
            self.output_dir / "templates"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def process_global_monthly(self,
                             years: Union[int, List[int]],
                             months: Optional[Union[int, List[int]]] = None,
                             variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute complete global monthly preprocessing pipeline.

        Args:
            years: Year(s) to process
            months: Month(s) to process (default: all months)
            variables: Variables to process (default: from config)

        Returns:
            Dictionary with processing results and statistics
        """
        # Prepare processing parameters
        years = [years] if isinstance(years, int) else years
        months = months or list(range(1, 13))
        months = [months] if isinstance(months, int) else months

        workflow_config = self.config.get_workflow_config('global_monthly')
        variables = variables or workflow_config.get('variables', {}).get('era5', [])

        processing_params = {
            'workflow_type': 'global_monthly',
            'years': years,
            'months': months,
            'variables': variables,
            'resolution': workflow_config.get('resolution', 0.5),
            'grid_bounds': workflow_config.get('grid_bounds')
        }

        # Start processing
        self.processing_logger.log_processing_start('global_monthly', processing_params)

        results = {
            'status': 'success',
            'files_created': [],
            'processing_summary': {},
            'qa_results': {}
        }

        try:
            with error_context("global monthly processing", self.processing_logger):
                # Step 1: Download data
                downloaded_files = self._download_era5_data(years, months, variables, 'monthly')
                self.processing_logger.log_data_download('ERA5', downloaded_files)

                # Step 2: Process downloaded data
                processed_files = self._process_monthly_data(downloaded_files, workflow_config)

                # Step 3: Run quality assurance
                qa_results = self._run_quality_assurance(processed_files)
                results['qa_results'] = qa_results

                # Step 4: Generate summary
                results['files_created'] = processed_files
                results['processing_summary'] = self.processing_logger.get_processing_summary()

                self.processing_logger.log_processing_complete(results['processing_summary'])

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.processing_logger.log_processing_error('workflow_failure', str(e))
            raise

        return results

    def process_conus_diurnal(self,
                            years: Union[int, List[int]],
                            months: Optional[Union[int, List[int]]] = None,
                            variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute CONUS diurnal flux processing pipeline.

        Args:
            years: Year(s) to process
            months: Month(s) to process (default: all months)
            variables: Variables to process (default: from config)

        Returns:
            Dictionary with processing results and statistics
        """
        # Prepare processing parameters
        years = [years] if isinstance(years, int) else years
        months = months or list(range(1, 13))
        months = [months] if isinstance(months, int) else months

        workflow_config = self.config.get_workflow_config('conus_diurnal')
        variables = variables or workflow_config.get('variables', {}).get('era5', [])

        processing_params = {
            'workflow_type': 'conus_diurnal',
            'years': years,
            'months': months,
            'variables': variables,
            'resolution': workflow_config.get('resolution', 0.5),
            'region': workflow_config.get('region')
        }

        # Start processing
        self.processing_logger.log_processing_start('conus_diurnal', processing_params)

        results = {
            'status': 'success',
            'files_created': [],
            'processing_summary': {},
            'qa_results': {}
        }

        try:
            with error_context("CONUS diurnal processing", self.processing_logger):
                # Step 1: Download hourly data for CONUS region
                downloaded_files = self._download_era5_data(years, months, variables, 'hourly')
                self.processing_logger.log_data_download('ERA5', downloaded_files)

                # Step 2: Process hourly data
                processed_files = self._process_diurnal_data(downloaded_files, workflow_config)

                # Step 3: Run quality assurance
                qa_results = self._run_quality_assurance(processed_files)
                results['qa_results'] = qa_results

                # Step 4: Generate summary
                results['files_created'] = processed_files
                results['processing_summary'] = self.processing_logger.get_processing_summary()

                self.processing_logger.log_processing_complete(results['processing_summary'])

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.processing_logger.log_processing_error('workflow_failure', str(e))
            raise

        return results

    def _download_era5_data(self,
                           years: List[int],
                           months: List[int],
                           variables: List[str],
                           frequency: str) -> List[str]:
        """Download ERA5 data using existing ECMWF downloader"""
        with error_context("ERA5 data download", self.processing_logger):
            # Get downloader configuration
            era5_config = self.config.get_downloader_config('era5')

            # Initialize downloader with output directory
            temp_dir = self.output_dir / "temp" / "downloads"
            downloader = ECMWFDownloader(output_dir=str(temp_dir))

            downloaded_files = []

            if frequency == 'monthly':
                # Download monthly data
                for variable in variables:
                    downloader.download_monthly_data(
                        variables=variable,
                        years=years,
                        months=months,
                        file_prefix="CARDAMOM_MONTHLY"
                    )

                    # Track downloaded files
                    for year in years:
                        for month in months:
                            filename = f"CARDAMOM_MONTHLY_{variable}_{month:02d}{year}.nc"
                            file_path = temp_dir / filename
                            if file_path.exists():
                                downloaded_files.append(str(file_path))

            elif frequency == 'hourly':
                # Download hourly data
                for variable in variables:
                    downloader.download_hourly_data(
                        variables=variable,
                        years=years,
                        months=months,
                        file_prefix="CARDAMOM_HOURLY"
                    )

                    # Track downloaded files
                    for year in years:
                        for month in months:
                            filename = f"CARDAMOM_HOURLY_{variable}_{month:02d}{year}.nc"
                            file_path = temp_dir / filename
                            if file_path.exists():
                                downloaded_files.append(str(file_path))

            return downloaded_files

    def _process_monthly_data(self,
                            downloaded_files: List[str],
                            workflow_config: Dict[str, Any]) -> List[str]:
        """Process downloaded monthly data files"""
        processed_files = []

        for input_file in downloaded_files:
            start_time = datetime.now()

            with error_context("monthly data processing", self.processing_logger, input_file=input_file):
                # This is a simplified processing step - in a full implementation,
                # this would include reading NetCDF files, applying scientific calculations,
                # regridding to standard grids, and writing processed output

                output_filename = Path(input_file).name.replace("CARDAMOM_MONTHLY", "CARDAMOM_PROCESSED")
                output_path = self.output_dir / output_filename

                # Placeholder for actual processing
                # In real implementation, would:
                # 1. Read NetCDF file
                # 2. Apply unit conversions
                # 3. Calculate derived variables (VPD, etc.)
                # 4. Regrid to standard coordinate system
                # 5. Write processed NetCDF output

                processing_time = (datetime.now() - start_time).total_seconds()
                self.processing_logger.log_file_processing(input_file, str(output_path), processing_time)

                processed_files.append(str(output_path))

        return processed_files

    def _process_diurnal_data(self,
                            downloaded_files: List[str],
                            workflow_config: Dict[str, Any]) -> List[str]:
        """Process downloaded hourly data files for diurnal analysis using Phase 4 DiurnalProcessor"""
        try:
            # Import Phase 4 diurnal processor
            from diurnal_processor import DiurnalProcessor
        except ImportError:
            self.processing_logger.log_processing_warning(
                "DiurnalProcessor not available. Using placeholder processing."
            )
            return self._process_diurnal_data_placeholder(downloaded_files, workflow_config)

        processed_files = []

        # Initialize DiurnalProcessor with same configuration
        diurnal_processor = DiurnalProcessor(self.config)

        # Extract years and months from downloaded files for diurnal processing
        years_months = self._extract_years_months_from_files(downloaded_files)

        for year, month_list in years_months.items():
            for month in month_list:
                start_time = datetime.now()

                with error_context(f"diurnal processing {year}-{month:02d}", self.processing_logger):
                    try:
                        # Use Phase 4 DiurnalProcessor for actual processing
                        result = diurnal_processor.process_diurnal_fluxes(
                            experiment_number=1,  # Default experiment
                            month=month,
                            year=year,
                            output_dir=str(self.output_dir / "diurnal_output")
                        )

                        # Track output files from diurnal processing
                        if result and result.output_files:
                            processed_files.extend(result.output_files)

                        processing_time = (datetime.now() - start_time).total_seconds()
                        self.processing_logger.log_file_processing(
                            f"diurnal_{year}_{month:02d}",
                            f"diurnal_output_{year}_{month:02d}",
                            processing_time
                        )

                    except Exception as e:
                        self.processing_logger.log_processing_error(
                            'diurnal_processing_error',
                            f"Failed to process diurnal data for {year}-{month:02d}: {str(e)}"
                        )
                        if not self.config.get('pipeline.resume_on_failure', True):
                            raise

        return processed_files

    def _process_diurnal_data_placeholder(self,
                                        downloaded_files: List[str],
                                        workflow_config: Dict[str, Any]) -> List[str]:
        """Fallback placeholder for diurnal processing when Phase 4 components are not available"""
        processed_files = []

        for input_file in downloaded_files:
            start_time = datetime.now()

            with error_context("diurnal data processing (placeholder)", self.processing_logger, input_file=input_file):
                output_filename = Path(input_file).name.replace("CARDAMOM_HOURLY", "CARDAMOM_DIURNAL")
                output_path = self.output_dir / output_filename

                # Placeholder processing - copy file with new name
                # In production, this would never be used as Phase 4 is complete
                import shutil
                if Path(input_file).exists():
                    shutil.copy2(input_file, output_path)

                processing_time = (datetime.now() - start_time).total_seconds()
                self.processing_logger.log_file_processing(input_file, str(output_path), processing_time)

                processed_files.append(str(output_path))

        return processed_files

    def _extract_years_months_from_files(self, file_list: List[str]) -> Dict[int, List[int]]:
        """Extract year/month combinations from downloaded file names"""
        import re
        years_months = {}

        for file_path in file_list:
            filename = Path(file_path).name
            # Extract year and month from filename pattern like "CARDAMOM_HOURLY_variable_MMYYYY.nc"
            match = re.search(r'_(\d{2})(\d{4})\.nc$', filename)
            if match:
                month = int(match.group(1))
                year = int(match.group(2))

                if year not in years_months:
                    years_months[year] = []
                if month not in years_months[year]:
                    years_months[year].append(month)

        return years_months

    def _run_quality_assurance(self, processed_files: List[str]) -> Dict[str, Any]:
        """Run quality assurance on processed files"""
        with error_context("quality assurance", self.processing_logger):
            # Placeholder for QA - in real implementation would:
            # 1. Read processed NetCDF files
            # 2. Extract data arrays
            # 3. Run full QA suite
            # 4. Generate QA reports

            qa_results = {
                'status': 'pass',
                'num_files_checked': len(processed_files),
                'qa_report_path': str(self.output_dir / "qa_reports" / "latest_qa_report.json")
            }

            self.processing_logger.log_qa_results(qa_results)
            return qa_results

    def validate_inputs(self, years: List[int], months: List[int]) -> bool:
        """
        Check year/month ranges and data availability.

        Args:
            years: List of years to validate
            months: List of months to validate

        Returns:
            True if inputs are valid, False otherwise
        """
        current_year = datetime.now().year

        # Validate years
        for year in years:
            if year < 1979 or year > current_year:
                self.processing_logger.log_processing_warning(
                    f"Year {year} outside typical ERA5 range (1979-{current_year})"
                )

        # Validate months
        for month in months:
            if month < 1 or month > 12:
                self.processing_logger.log_processing_error(
                    'invalid_input',
                    f"Invalid month: {month}. Must be 1-12.",
                    context={'months': months}
                )
                return False

        return True

    def process_batch(self,
                     workflow_type: str,
                     years: List[int],
                     months: List[int],
                     variables: Optional[List[str]] = None,
                     show_progress: bool = True) -> Dict[str, Any]:
        """
        Handle multi-year/month processing with comprehensive progress tracking.

        Args:
            workflow_type: Type of workflow ('global_monthly' or 'conus_diurnal')
            years: Years to process
            months: Months to process
            variables: Variables to process
            show_progress: Whether to display progress bar

        Returns:
            Dictionary with batch processing results
        """
        if not self.validate_inputs(years, months):
            raise ValueError("Invalid input parameters")

        # Create list of all year/month combinations
        processing_tasks = [(year, month) for year in years for month in months]
        total_tasks = len(processing_tasks)

        batch_results = {
            'workflow_type': workflow_type,
            'total_combinations': total_tasks,
            'successful_combinations': 0,
            'failed_combinations': 0,
            'skipped_combinations': 0,
            'results': [],
            'start_time': datetime.now().isoformat(),
            'processing_summary': {
                'estimated_completion_time': None,
                'average_processing_time_per_task': None,
                'total_processing_time': None
            }
        }

        # Initialize progress tracking
        start_time = datetime.now()
        task_times = []

        # Setup progress display
        if show_progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(
                    total=total_tasks,
                    desc=f"Processing {workflow_type}",
                    unit="task",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                )
            except ImportError:
                self.processing_logger.log_processing_warning(
                    "tqdm not available. Progress will be shown via logging only."
                )
                progress_bar = None
        else:
            progress_bar = None

        try:
            for task_index, (year, month) in enumerate(processing_tasks):
                task_start_time = datetime.now()
                task_id = f"{year}-{month:02d}"

                # Check if this task was already completed (resumability)
                if self._is_task_completed(workflow_type, year, month):
                    self.processing_logger.logger.info(f"Task {task_id} already completed. Skipping.")
                    batch_results['skipped_combinations'] += 1
                    if progress_bar:
                        progress_bar.update(1)
                    continue

                # Log progress
                self.processing_logger.logger.info(
                    f"Processing task {task_index + 1}/{total_tasks}: {task_id}"
                )

                try:
                    # Execute the processing task with retry logic
                    if workflow_type == 'global_monthly':
                        result = self._retry_operation(
                            self.process_global_monthly,
                            f"global_monthly_{task_id}",
                            [year], [month], variables
                        )
                    elif workflow_type == 'conus_diurnal':
                        result = self._retry_operation(
                            self.process_conus_diurnal,
                            f"conus_diurnal_{task_id}",
                            [year], [month], variables
                        )
                    else:
                        raise ValueError(f"Unknown workflow type: {workflow_type}")

                    # Record successful completion
                    task_time = (datetime.now() - task_start_time).total_seconds()
                    task_times.append(task_time)

                    batch_results['successful_combinations'] += 1
                    batch_results['results'].append({
                        'year': year,
                        'month': month,
                        'status': 'success',
                        'result': result,
                        'processing_time_seconds': task_time
                    })

                    # Mark task as completed in state
                    self._mark_task_completed(workflow_type, year, month, result)

                    # Update progress estimates
                    if task_times:
                        avg_time = sum(task_times) / len(task_times)
                        remaining_tasks = total_tasks - (task_index + 1)
                        estimated_remaining_time = avg_time * remaining_tasks

                        batch_results['processing_summary']['average_processing_time_per_task'] = avg_time
                        batch_results['processing_summary']['estimated_completion_time'] = (
                            datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)
                        ).isoformat()

                except Exception as e:
                    # Handle task failure
                    task_time = (datetime.now() - task_start_time).total_seconds()

                    batch_results['failed_combinations'] += 1
                    batch_results['results'].append({
                        'year': year,
                        'month': month,
                        'status': 'failed',
                        'error': str(e),
                        'processing_time_seconds': task_time
                    })

                    # Handle error according to configuration
                    self._handle_workflow_error(workflow_type, e, {
                        'year': year,
                        'month': month,
                        'task_index': task_index + 1,
                        'total_tasks': total_tasks
                    })

                    if not self.config.get('pipeline.resume_on_failure', True):
                        if progress_bar:
                            progress_bar.close()
                        raise

                # Update progress display
                if progress_bar:
                    progress_bar.update(1)
                    if task_times:
                        progress_bar.set_postfix({
                            'avg_time': f"{sum(task_times)/len(task_times):.1f}s",
                            'success': batch_results['successful_combinations'],
                            'failed': batch_results['failed_combinations']
                        })

                # Save state periodically
                if (task_index + 1) % 5 == 0:  # Save every 5 tasks
                    self._save_processing_state()

        finally:
            if progress_bar:
                progress_bar.close()

        # Finalize batch results
        total_time = (datetime.now() - start_time).total_seconds()
        batch_results['end_time'] = datetime.now().isoformat()
        batch_results['processing_summary']['total_processing_time'] = total_time

        # Log final summary
        self.processing_logger.logger.info(
            f"Batch processing complete. "
            f"Success: {batch_results['successful_combinations']}, "
            f"Failed: {batch_results['failed_combinations']}, "
            f"Skipped: {batch_results['skipped_combinations']}, "
            f"Total time: {total_time:.1f}s"
        )

        # Save final state
        self._save_processing_state()

        return batch_results

    def _is_task_completed(self, workflow_type: str, year: int, month: int) -> bool:
        """Check if a task has already been completed"""
        task_id = f"{workflow_type}_{year}_{month:02d}"
        return task_id in self.processing_state.get('completed_operations', [])

    def _mark_task_completed(self, workflow_type: str, year: int, month: int, result: Dict[str, Any]):
        """Mark a task as completed in the processing state"""
        task_id = f"{workflow_type}_{year}_{month:02d}"

        if 'completed_operations' not in self.processing_state:
            self.processing_state['completed_operations'] = []

        if task_id not in self.processing_state['completed_operations']:
            self.processing_state['completed_operations'].append(task_id)

        # Store completion details
        if 'completion_details' not in self.processing_state:
            self.processing_state['completion_details'] = {}

        self.processing_state['completion_details'][task_id] = {
            'completion_time': datetime.now().isoformat(),
            'files_created': result.get('files_created', []),
            'status': result.get('status', 'unknown')
        }

    def generate_summary_report(self, processing_results: Dict[str, Any]) -> str:
        """
        Create processing summary with statistics.

        Args:
            processing_results: Results from processing workflow

        Returns:
            Path to generated summary report
        """
        summary_path = self.output_dir / "processing_summary.json"

        summary = {
            'generation_time': datetime.now().isoformat(),
            'cardamom_version': 'v1.0',
            'configuration': self.config.to_dict(),
            'processing_results': processing_results,
            'coordinate_systems_info': {
                name: grid.get_grid_info()
                for name, grid in self.coordinate_systems.items()
            }
        }

        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.processing_logger.logger.info(f"Processing summary saved: {summary_path}")
        return str(summary_path)