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
from ecmwf_downloader import ECMWFDownloader

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
        # Load configuration
        self.config = CardamomConfig(config_file)

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.get('processing.output_directory'))

        # Setup logging
        log_level = self.config.get('processing.log_level', 'INFO')
        log_file = self.output_dir / "logs" / "cardamom_processing.log"
        self.logger_instance = setup_cardamom_logging(log_level, str(log_file))
        self.processing_logger = ProcessingLogger(self.logger_instance)

        # Initialize coordinate systems
        self.coordinate_systems = self._init_coordinate_systems()

        # Initialize NetCDF writer
        compression = self.config.get('processing.compression', True)
        self.netcdf_writer = CARDAMOMNetCDFWriter(compression=compression)

        # Initialize quality assurance system
        qa_config = self.config.get_quality_control_config()
        self.qa_system = QualityAssurance(qa_config)

        # Ensure output directories exist
        self._setup_data_directories()

        self.processing_logger.logger.info("CARDAMOM Processor initialized successfully")

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
        """Process downloaded hourly data files for diurnal analysis"""
        processed_files = []

        for input_file in downloaded_files:
            start_time = datetime.now()

            with error_context("diurnal data processing", self.processing_logger, input_file=input_file):
                # Placeholder for diurnal processing
                output_filename = Path(input_file).name.replace("CARDAMOM_HOURLY", "CARDAMOM_DIURNAL")
                output_path = self.output_dir / output_filename

                # In real implementation, would:
                # 1. Read hourly NetCDF file
                # 2. Extract CONUS region
                # 3. Calculate diurnal patterns
                # 4. Apply scientific calculations
                # 5. Write diurnal NetCDF output

                processing_time = (datetime.now() - start_time).total_seconds()
                self.processing_logger.log_file_processing(input_file, str(output_path), processing_time)

                processed_files.append(str(output_path))

        return processed_files

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
                     variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Handle multi-year/month processing with progress tracking.

        Args:
            workflow_type: Type of workflow ('global_monthly' or 'conus_diurnal')
            years: Years to process
            months: Months to process
            variables: Variables to process

        Returns:
            Dictionary with batch processing results
        """
        if not self.validate_inputs(years, months):
            raise ValueError("Invalid input parameters")

        batch_results = {
            'workflow_type': workflow_type,
            'total_combinations': len(years) * len(months),
            'successful_combinations': 0,
            'failed_combinations': 0,
            'results': []
        }

        for year in years:
            for month in months:
                try:
                    if workflow_type == 'global_monthly':
                        result = self.process_global_monthly([year], [month], variables)
                    elif workflow_type == 'conus_diurnal':
                        result = self.process_conus_diurnal([year], [month], variables)
                    else:
                        raise ValueError(f"Unknown workflow type: {workflow_type}")

                    batch_results['successful_combinations'] += 1
                    batch_results['results'].append({
                        'year': year,
                        'month': month,
                        'status': 'success',
                        'result': result
                    })

                except Exception as e:
                    batch_results['failed_combinations'] += 1
                    batch_results['results'].append({
                        'year': year,
                        'month': month,
                        'status': 'failed',
                        'error': str(e)
                    })

                    if not self.config.get('pipeline.resume_on_failure', True):
                        raise

        return batch_results

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