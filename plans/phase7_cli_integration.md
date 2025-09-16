# Phase 7: Enhanced CLI Integration

## Overview
Extend the existing CLI infrastructure to support comprehensive CARDAMOM preprocessing workflows, including the new pipeline management system, GFED processing, and diurnal flux generation. Maintain backward compatibility while adding powerful new capabilities.

## 7.1 Enhanced Main CLI (`cardamom_cli.py`)

### Extended CLI Structure
```python
class CARDAMOMCLIManager:
    """
    Enhanced CLI manager that integrates all CARDAMOM preprocessing capabilities.
    Extends existing ecmwf_downloader.py CLI with comprehensive workflow support.
    """

    def __init__(self):
        self.pipeline_manager = CARDAMOMPipelineManager()
        self.existing_ecmwf_cli = self._import_existing_cli()

    def create_master_parser(self):
        """
        Create comprehensive argument parser that includes existing and new functionality.
        """
        parser = argparse.ArgumentParser(
            description="CARDAMOM Data Preprocessing Suite",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_usage_examples()
        )

        # Add global options
        parser.add_argument('--config', '-c',
                          help='Configuration file path (default: config/cardamom_config.yaml)')
        parser.add_argument('--workspace', '-w', default='./workspace/',
                          help='Workspace directory for processing (default: ./workspace/)')
        parser.add_argument('--verbose', '-v', action='count', default=0,
                          help='Increase verbosity (use -v, -vv, or -vvv)')
        parser.add_argument('--log-file',
                          help='Log file path (default: logs/cardamom.log)')
        parser.add_argument('--dry-run', action='store_true',
                          help='Show what would be done without executing')

        # Create subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Legacy ECMWF commands (maintain backward compatibility)
        self._add_legacy_ecmwf_commands(subparsers)

        # New pipeline commands
        self._add_pipeline_commands(subparsers)

        # Data source specific commands
        self._add_data_source_commands(subparsers)

        # Utility commands
        self._add_utility_commands(subparsers)

        return parser

    def _get_usage_examples(self):
        """Generate comprehensive usage examples"""
        return """
Examples:

Legacy ECMWF Downloads:
  %(prog)s ecmwf hourly -v skin_temperature -y 2020 -m 6-8 --area 60,-130,20,-50
  %(prog)s ecmwf monthly -v 2m_temperature -y 2020-2022 -m 1-12
  %(prog)s ecmwf cardamom-monthly -y 2020 -o ./monthly_data/
  %(prog)s ecmwf cardamom-hourly -y 2020 -o ./hourly_data/

Complete Pipeline Workflows:
  %(prog)s pipeline global-monthly -y 2020-2022 --config config/global_monthly.yaml
  %(prog)s pipeline conus-diurnal -y 2015-2020 --experiments 1,2
  %(prog)s pipeline resume global-monthly-2020-2022

Data Source Management:
  %(prog)s download era5 -y 2020 -m 1-12 -v 2m_temperature,total_precipitation
  %(prog)s download noaa-co2 -y 2020-2022 --update-cache
  %(prog)s download gfed -y 2020-2022 --resolution 05deg
  %(prog)s download modis --product land-sea-mask --resolution 0.5

Data Processing:
  %(prog)s process gfed -y 2020-2022 --gap-fill --resolution 05deg
  %(prog)s process diurnal -y 2020 --experiment 1 --region conus
  %(prog)s process vpd --era5-data ./era5/ --output ./processed/

Quality Control and Utilities:
  %(prog)s validate outputs ./DATA/CARDAMOM-MAPS_05deg_MET/
  %(prog)s report pipeline-summary ./logs/global-monthly-2020.json
  %(prog)s cleanup --older-than 30d --temp-only
  %(prog)s status --pipeline global-monthly-2020
        """
```

### Legacy Command Integration
```python
def _add_legacy_ecmwf_commands(self, subparsers):
    """
    Add existing ECMWF commands for backward compatibility.
    Wraps existing functionality from ecmwf_downloader.py
    """

    # Main ECMWF command group
    ecmwf_parser = subparsers.add_parser('ecmwf',
                                         help='ECMWF ERA5 data downloads (legacy commands)')
    ecmwf_subparsers = ecmwf_parser.add_subparsers(dest='ecmwf_command')

    # Import existing parser structure from ecmwf_downloader.py
    existing_parser = self.existing_ecmwf_cli.create_parser()

    # Extract and adapt existing subcommands
    for action in existing_parser._subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            for choice, subparser in action.choices.items():
                # Clone existing subparser structure
                new_subparser = ecmwf_subparsers.add_parser(
                    choice,
                    help=subparser.description,
                    parents=[subparser],
                    conflict_handler='resolve'
                )

def _execute_legacy_ecmwf_command(self, args):
    """Execute legacy ECMWF command using existing implementation"""

    # Reconstruct arguments for existing CLI
    legacy_args = [args.ecmwf_command]

    # Map arguments back to legacy format
    if hasattr(args, 'variables'):
        legacy_args.extend(['-v'] + args.variables)
    if hasattr(args, 'years'):
        legacy_args.extend(['-y', self._format_years_for_legacy(args.years)])
    if hasattr(args, 'months'):
        legacy_args.extend(['-m', self._format_months_for_legacy(args.months)])

    # Execute using existing main function
    return self.existing_ecmwf_cli.main_with_args(legacy_args)
```

## 7.2 Pipeline Command Interface (`pipeline_commands.py`)

### Pipeline Management Commands
```python
def _add_pipeline_commands(self, subparsers):
    """Add comprehensive pipeline management commands"""

    pipeline_parser = subparsers.add_parser('pipeline',
                                           help='Execute complete preprocessing pipelines')
    pipeline_subparsers = pipeline_parser.add_subparsers(dest='pipeline_command')

    # Global monthly pipeline
    global_monthly = pipeline_subparsers.add_parser('global-monthly',
                                                   help='Execute global monthly preprocessing pipeline')
    self._add_common_pipeline_args(global_monthly)
    global_monthly.add_argument('--resolution', default='0.5', choices=['0.25', '0.5'],
                               help='Spatial resolution in degrees (default: 0.5)')
    global_monthly.add_argument('--variables', nargs='+',
                               help='Specific variables to process (default: all)')

    # CONUS diurnal pipeline
    conus_diurnal = pipeline_subparsers.add_parser('conus-diurnal',
                                                  help='Execute CONUS diurnal flux processing pipeline')
    self._add_common_pipeline_args(conus_diurnal)
    conus_diurnal.add_argument('--experiments', default='1,2',
                              help='CMS experiment numbers (default: 1,2)')
    conus_diurnal.add_argument('--region',
                              help='Custom region bounds as N,W,S,E (default: CONUS)')

    # Pipeline status and management
    status_parser = pipeline_subparsers.add_parser('status',
                                                  help='Check pipeline execution status')
    status_parser.add_argument('pipeline_id', nargs='?',
                              help='Specific pipeline ID to check')
    status_parser.add_argument('--list', action='store_true',
                              help='List all known pipelines')

    # Resume pipeline
    resume_parser = pipeline_subparsers.add_parser('resume',
                                                  help='Resume interrupted pipeline')
    resume_parser.add_argument('pipeline_id',
                              help='Pipeline ID to resume')
    resume_parser.add_argument('--force', action='store_true',
                              help='Force resume even if errors exist')

    # Cancel pipeline
    cancel_parser = pipeline_subparsers.add_parser('cancel',
                                                  help='Cancel running pipeline')
    cancel_parser.add_argument('pipeline_id',
                              help='Pipeline ID to cancel')

def _add_common_pipeline_args(self, parser):
    """Add common arguments for pipeline commands"""

    parser.add_argument('-y', '--years', required=True,
                       help='Years to process (single: 2020, range: 2020-2022, list: 2020,2021,2022)')
    parser.add_argument('-m', '--months',
                       help='Months to process (default: 1-12, range: 6-8, list: 6,7,8)')
    parser.add_argument('-o', '--output-dir',
                       help='Output directory (default: from config)')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing where possible')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous state if possible')
    parser.add_argument('--force', action='store_true',
                       help='Force execution even if outputs exist')

def execute_pipeline_command(self, args):
    """Execute pipeline commands"""

    if args.pipeline_command == 'global-monthly':
        return self._execute_global_monthly_pipeline(args)
    elif args.pipeline_command == 'conus-diurnal':
        return self._execute_conus_diurnal_pipeline(args)
    elif args.pipeline_command == 'status':
        return self._show_pipeline_status(args)
    elif args.pipeline_command == 'resume':
        return self._resume_pipeline(args)
    elif args.pipeline_command == 'cancel':
        return self._cancel_pipeline(args)
    else:
        raise ValueError(f"Unknown pipeline command: {args.pipeline_command}")

def _execute_global_monthly_pipeline(self, args):
    """Execute global monthly pipeline with CLI arguments"""

    # Parse years and months
    years = self._parse_year_range(args.years)
    months = self._parse_month_range(args.months) if args.months else None

    # Configure pipeline
    pipeline_config = {
        'resolution': args.resolution,
        'output_dir': args.output_dir,
        'parallel': args.parallel,
        'resume': args.resume,
        'force': args.force
    }

    if args.variables:
        pipeline_config['variables'] = args.variables

    # Execute pipeline
    return self.pipeline_manager.execute_global_monthly_pipeline(
        years=years,
        months=months,
        config=pipeline_config
    )
```

## 7.3 Data Source Commands (`data_source_commands.py`)

### Individual Data Source Management
```python
def _add_data_source_commands(self, subparsers):
    """Add commands for individual data source management"""

    # Download commands
    download_parser = subparsers.add_parser('download',
                                           help='Download data from specific sources')
    download_subparsers = download_parser.add_subparsers(dest='download_source')

    # ERA5 downloads (enhanced version of existing)
    era5_parser = download_subparsers.add_parser('era5',
                                                help='Download ERA5 data with enhanced options')
    self._add_era5_download_args(era5_parser)

    # NOAA CO2 downloads
    noaa_parser = download_subparsers.add_parser('noaa-co2',
                                               help='Download NOAA CO2 concentration data')
    self._add_noaa_download_args(noaa_parser)

    # GFED downloads
    gfed_parser = download_subparsers.add_parser('gfed',
                                               help='Download GFED burned area data')
    self._add_gfed_download_args(gfed_parser)

    # MODIS downloads
    modis_parser = download_subparsers.add_parser('modis',
                                                help='Download MODIS land-sea mask data')
    self._add_modis_download_args(modis_parser)

    # Process commands
    process_parser = subparsers.add_parser('process',
                                         help='Process specific datasets')
    process_subparsers = process_parser.add_subparsers(dest='process_type')

    # GFED processing
    gfed_process = process_subparsers.add_parser('gfed',
                                               help='Process GFED data with gap-filling')
    self._add_gfed_process_args(gfed_process)

    # Diurnal processing
    diurnal_process = process_subparsers.add_parser('diurnal',
                                                  help='Process diurnal flux patterns')
    self._add_diurnal_process_args(diurnal_process)

    # VPD calculation
    vpd_process = process_subparsers.add_parser('vpd',
                                              help='Calculate Vapor Pressure Deficit')
    self._add_vpd_process_args(vpd_process)

def _add_era5_download_args(self, parser):
    """Add ERA5-specific download arguments"""

    parser.add_argument('-v', '--variables', required=True, nargs='+',
                       help='ERA5 variables to download')
    parser.add_argument('-y', '--years', required=True,
                       help='Years to download (single, range, or list)')
    parser.add_argument('-m', '--months', required=True,
                       help='Months to download (single, range, or list)')
    parser.add_argument('--resolution', default='0.5/0.5',
                       help='Spatial resolution (default: 0.5/0.5)')
    parser.add_argument('--area',
                       help='Spatial bounds as N,W,S,E (default: global)')
    parser.add_argument('--product-type',
                       choices=['reanalysis', 'monthly_averaged_reanalysis',
                               'monthly_averaged_reanalysis_by_hour_of_day'],
                       help='ERA5 product type')
    parser.add_argument('--dataset',
                       choices=['reanalysis-era5-single-levels',
                               'reanalysis-era5-single-levels-monthly-means'],
                       help='ERA5 dataset name')
    parser.add_argument('-o', '--output-dir', default='./era5_data/',
                       help='Output directory (default: ./era5_data/)')
    parser.add_argument('--parallel', action='store_true',
                       help='Download multiple variables in parallel')

def _add_gfed_download_args(self, parser):
    """Add GFED-specific download arguments"""

    parser.add_argument('-y', '--years', required=True,
                       help='Years to download (handles beta versions automatically)')
    parser.add_argument('--resolution', choices=['0.25deg', '05deg', 'GC4x5'],
                       default='05deg', help='Target resolution (default: 05deg)')
    parser.add_argument('-o', '--output-dir', default='./gfed_data/',
                       help='Output directory (default: ./gfed_data/)')
    parser.add_argument('--verify-integrity', action='store_true',
                       help='Verify downloaded HDF5 file integrity')
    parser.add_argument('--cache-dir',
                       help='Cache directory for large downloads')

def _add_gfed_process_args(self, parser):
    """Add GFED processing arguments"""

    parser.add_argument('-y', '--years', required=True,
                       help='Years to process')
    parser.add_argument('--input-dir', default='./gfed_data/',
                       help='Input directory with GFED HDF5 files')
    parser.add_argument('-o', '--output-dir', default='./processed_gfed/',
                       help='Output directory for processed data')
    parser.add_argument('--resolution', choices=['0.25deg', '05deg', 'GC4x5'],
                       default='05deg', help='Target resolution')
    parser.add_argument('--gap-fill', action='store_true',
                       help='Apply gap-filling for missing years (2017+)')
    parser.add_argument('--reference-period', default='2001-2016',
                       help='Reference period for gap-filling (default: 2001-2016)')
    parser.add_argument('--create-cardamom-files', action='store_true',
                       help='Create CARDAMOM-compliant NetCDF outputs')
```

### Command Execution
```python
def execute_download_command(self, args):
    """Execute data source download commands"""

    if args.download_source == 'era5':
        return self._execute_era5_download(args)
    elif args.download_source == 'noaa-co2':
        return self._execute_noaa_download(args)
    elif args.download_source == 'gfed':
        return self._execute_gfed_download(args)
    elif args.download_source == 'modis':
        return self._execute_modis_download(args)
    else:
        raise ValueError(f"Unknown download source: {args.download_source}")

def execute_process_command(self, args):
    """Execute data processing commands"""

    if args.process_type == 'gfed':
        return self._execute_gfed_processing(args)
    elif args.process_type == 'diurnal':
        return self._execute_diurnal_processing(args)
    elif args.process_type == 'vpd':
        return self._execute_vpd_calculation(args)
    else:
        raise ValueError(f"Unknown process type: {args.process_type}")

def _execute_gfed_processing(self, args):
    """Execute GFED processing with CLI arguments"""

    from ..processors import GFEDProcessor

    # Parse years
    years = self._parse_year_range(args.years)

    # Initialize processor
    processor = GFEDProcessor(
        data_dir=args.input_dir,
        output_dir=args.output_dir
    )

    # Configure processing options
    processing_config = {
        'target_resolution': args.resolution,
        'gap_filling': args.gap_fill,
        'reference_period': self._parse_year_range(args.reference_period) if args.gap_fill else None,
        'create_cardamom_files': args.create_cardamom_files
    }

    # Execute processing
    return processor.process_multi_year_data(years, **processing_config)
```

## 7.4 Utility Commands (`utility_commands.py`)

### Validation and Quality Control Commands
```python
def _add_utility_commands(self, subparsers):
    """Add utility commands for validation, reporting, and maintenance"""

    # Validation commands
    validate_parser = subparsers.add_parser('validate',
                                           help='Validate data files and outputs')
    validate_subparsers = validate_parser.add_subparsers(dest='validate_type')

    # Output validation
    outputs_validate = validate_subparsers.add_parser('outputs',
                                                     help='Validate output files')
    outputs_validate.add_argument('directory',
                                 help='Directory containing output files to validate')
    outputs_validate.add_argument('--format', choices=['netcdf', 'hdf5', 'all'],
                                 default='all', help='File format to validate')
    outputs_validate.add_argument('--cf-compliance', action='store_true',
                                 help='Check CF convention compliance')
    outputs_validate.add_argument('--cardamom-compliance', action='store_true',
                                 help='Check CARDAMOM-specific requirements')

    # Configuration validation
    config_validate = validate_subparsers.add_parser('config',
                                                    help='Validate configuration files')
    config_validate.add_argument('config_file',
                                help='Configuration file to validate')

    # Data validation
    data_validate = validate_subparsers.add_parser('data',
                                                  help='Validate data ranges and quality')
    data_validate.add_argument('input_files', nargs='+',
                              help='Data files to validate')
    data_validate.add_argument('--ranges-config',
                              help='Configuration file with expected data ranges')

    # Reporting commands
    report_parser = subparsers.add_parser('report',
                                         help='Generate reports and summaries')
    report_subparsers = report_parser.add_subparsers(dest='report_type')

    # Pipeline summary report
    pipeline_summary = report_subparsers.add_parser('pipeline-summary',
                                                   help='Generate pipeline execution summary')
    pipeline_summary.add_argument('log_file',
                                 help='Pipeline log file or result JSON')
    pipeline_summary.add_argument('--format', choices=['html', 'pdf', 'json'],
                                 default='html', help='Report format')
    pipeline_summary.add_argument('-o', '--output',
                                 help='Output file path')

    # Data summary report
    data_summary = report_subparsers.add_parser('data-summary',
                                               help='Generate data summary report')
    data_summary.add_argument('data_directory',
                             help='Directory containing data to summarize')
    data_summary.add_argument('--include-plots', action='store_true',
                             help='Include data plots in report')

    # Status commands
    status_parser = subparsers.add_parser('status',
                                         help='Check system and pipeline status')
    status_parser.add_argument('--system', action='store_true',
                              help='Show system status (disk space, dependencies)')
    status_parser.add_argument('--pipelines', action='store_true',
                              help='Show all pipeline statuses')
    status_parser.add_argument('--downloads', action='store_true',
                              help='Show download cache status')

    # Cleanup commands
    cleanup_parser = subparsers.add_parser('cleanup',
                                          help='Clean up temporary files and caches')
    cleanup_parser.add_argument('--temp-only', action='store_true',
                               help='Clean only temporary files')
    cleanup_parser.add_argument('--cache-only', action='store_true',
                               help='Clean only cache files')
    cleanup_parser.add_argument('--older-than',
                               help='Clean files older than specified period (e.g., 30d, 2w)')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be cleaned without actually cleaning')

def execute_validate_command(self, args):
    """Execute validation commands"""

    if args.validate_type == 'outputs':
        return self._validate_outputs(args)
    elif args.validate_type == 'config':
        return self._validate_config(args)
    elif args.validate_type == 'data':
        return self._validate_data(args)
    else:
        raise ValueError(f"Unknown validation type: {args.validate_type}")

def execute_report_command(self, args):
    """Execute reporting commands"""

    if args.report_type == 'pipeline-summary':
        return self._generate_pipeline_summary_report(args)
    elif args.report_type == 'data-summary':
        return self._generate_data_summary_report(args)
    else:
        raise ValueError(f"Unknown report type: {args.report_type}")

def _validate_outputs(self, args):
    """Validate output files"""

    from ..validation import NetCDFValidator, CARDAMOMValidator

    validator = NetCDFValidator()
    results = {
        'directory': args.directory,
        'files_checked': 0,
        'files_passed': 0,
        'files_failed': 0,
        'validation_results': {}
    }

    # Find files to validate
    file_patterns = {
        'netcdf': '*.nc',
        'hdf5': '*.hdf5',
        'all': ['*.nc', '*.hdf5']
    }

    patterns = file_patterns[args.format]
    if not isinstance(patterns, list):
        patterns = [patterns]

    for pattern in patterns:
        files = glob.glob(os.path.join(args.directory, '**', pattern), recursive=True)

        for file_path in files:
            results['files_checked'] += 1

            try:
                # Basic format validation
                file_result = validator.validate_file_structure(file_path)

                # CF compliance check
                if args.cf_compliance:
                    file_result.update(validator.validate_cf_compliance(file_path))

                # CARDAMOM compliance check
                if args.cardamom_compliance:
                    cardamom_validator = CARDAMOMValidator()
                    file_result.update(cardamom_validator.validate_cardamom_compliance(file_path))

                results['validation_results'][file_path] = file_result
                results['files_passed'] += 1

            except Exception as e:
                results['validation_results'][file_path] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['files_failed'] += 1

    return results
```

### System Status and Maintenance
```python
def execute_status_command(self, args):
    """Execute status commands"""

    status_info = {}

    if args.system or not any([args.pipelines, args.downloads]):
        status_info['system'] = self._get_system_status()

    if args.pipelines:
        status_info['pipelines'] = self._get_pipeline_status()

    if args.downloads:
        status_info['downloads'] = self._get_download_status()

    return status_info

def _get_system_status(self):
    """Get system status information"""

    import psutil
    import shutil

    system_status = {
        'timestamp': datetime.now().isoformat(),
        'disk_space': {},
        'memory': {},
        'dependencies': {}
    }

    # Disk space information
    workspace_dir = self.pipeline_manager.workspace_dir
    if os.path.exists(workspace_dir):
        usage = shutil.disk_usage(workspace_dir)
        system_status['disk_space'] = {
            'total_gb': usage.total / (1024**3),
            'used_gb': usage.used / (1024**3),
            'free_gb': usage.free / (1024**3),
            'percent_used': (usage.used / usage.total) * 100
        }

    # Memory information
    memory = psutil.virtual_memory()
    system_status['memory'] = {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'percent_used': memory.percent
    }

    # Check key dependencies
    dependencies = {
        'netcdf4': self._check_import('netCDF4'),
        'xarray': self._check_import('xarray'),
        'h5py': self._check_import('h5py'),
        'cdsapi': self._check_import('cdsapi'),
        'scipy': self._check_import('scipy')
    }

    system_status['dependencies'] = dependencies

    return system_status

def execute_cleanup_command(self, args):
    """Execute cleanup commands"""

    cleanup_results = {
        'timestamp': datetime.now().isoformat(),
        'dry_run': args.dry_run,
        'files_removed': 0,
        'space_freed_mb': 0,
        'directories_cleaned': []
    }

    # Define cleanup targets
    cleanup_targets = []

    if args.temp_only or not any([args.cache_only]):
        cleanup_targets.extend(self._get_temp_directories())

    if args.cache_only or not any([args.temp_only]):
        cleanup_targets.extend(self._get_cache_directories())

    # Apply age filter if specified
    age_limit = self._parse_age_limit(args.older_than) if args.older_than else None

    for target_dir in cleanup_targets:
        if os.path.exists(target_dir):
            cleaned = self._cleanup_directory(target_dir, age_limit, args.dry_run)
            cleanup_results['files_removed'] += cleaned['files_removed']
            cleanup_results['space_freed_mb'] += cleaned['space_freed_mb']
            cleanup_results['directories_cleaned'].append({
                'directory': target_dir,
                'files_removed': cleaned['files_removed'],
                'space_freed_mb': cleaned['space_freed_mb']
            })

    return cleanup_results
```

## 7.5 Configuration File Support (`config_cli.py`)

### Configuration File Integration
```python
class ConfigCLIManager:
    """Manage configuration file integration with CLI"""

    def __init__(self):
        self.default_config_locations = [
            './config/cardamom_config.yaml',
            './cardamom_config.yaml',
            '~/.cardamom/config.yaml',
            '/etc/cardamom/config.yaml'
        ]

    def load_config_for_cli(self, config_path=None, command_args=None):
        """
        Load configuration file and merge with CLI arguments.
        CLI arguments take precedence over config file settings.
        """

        # Find configuration file
        config_file = self._find_config_file(config_path)

        if config_file:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        # Merge with CLI arguments
        if command_args:
            config = self._merge_cli_args_with_config(config, command_args)

        return config

    def _find_config_file(self, explicit_path=None):
        """Find configuration file in standard locations"""

        search_paths = []

        if explicit_path:
            search_paths.append(explicit_path)

        search_paths.extend(self.default_config_locations)

        for path in search_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path

        return None

    def _merge_cli_args_with_config(self, config, args):
        """Merge CLI arguments with configuration file"""

        # CLI arguments override config file settings
        cli_overrides = {
            'workspace_dir': getattr(args, 'workspace', None),
            'output_dir': getattr(args, 'output_dir', None),
            'verbose': getattr(args, 'verbose', None),
            'parallel': getattr(args, 'parallel', None),
            'dry_run': getattr(args, 'dry_run', None)
        }

        # Apply overrides
        for key, value in cli_overrides.items():
            if value is not None:
                config[key] = value

        return config

    def generate_config_template(self, output_path, config_type='full'):
        """Generate configuration file template"""

        if config_type == 'full':
            template = self._get_full_config_template()
        elif config_type == 'minimal':
            template = self._get_minimal_config_template()
        elif config_type == 'global-monthly':
            template = self._get_global_monthly_template()
        elif config_type == 'conus-diurnal':
            template = self._get_conus_diurnal_template()
        else:
            raise ValueError(f"Unknown config type: {config_type}")

        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)

        return output_path
```

### Configuration Templates
```python
def _get_full_config_template(self):
    """Generate comprehensive configuration template"""
    return {
        'general': {
            'workspace_dir': './workspace/',
            'log_level': 'INFO',
            'max_parallel_downloads': 4,
            'max_parallel_processes': 2
        },
        'paths': {
            'output_base': './DATA/',
            'temp_dir': './temp/',
            'cache_dir': './cache/',
            'logs_dir': './logs/'
        },
        'data_sources': {
            'ecmwf': {
                'credentials_file': '~/.cdsapirc',
                'rate_limit_requests_per_minute': 10,
                'timeout_seconds': 3600
            },
            'noaa': {
                'ftp_server': 'aftp.cmdl.noaa.gov',
                'update_frequency_days': 7
            },
            'gfed': {
                'base_url': 'https://www.globalfiredata.org/data_new/',
                'verify_downloads': True
            },
            'modis': {
                'preferred_servers': [
                    'https://e4ftl01.cr.usgs.gov/MOTA/',
                    'https://n5eil01u.ecs.nsidc.org/'
                ]
            }
        },
        'processing': {
            'global_monthly': {
                'default_resolution': 0.5,
                'default_years': [2001, 2024],
                'create_templates': True,
                'quality_control': True
            },
            'conus_diurnal': {
                'default_resolution': 0.5,
                'default_years': [2015, 2020],
                'default_experiments': [1, 2],
                'region_bounds': [60, -130, 20, -50]
            }
        },
        'quality_control': {
            'enable_validation': True,
            'cf_compliance_check': True,
            'data_range_checks': True,
            'generate_reports': True
        }
    }
```

## 7.6 Interactive Mode and Wizards (`interactive_cli.py`)

### Interactive Configuration Wizard
```python
class InteractiveCLI:
    """Provide interactive mode for complex configurations"""

    def __init__(self):
        self.config_wizard = ConfigurationWizard()
        self.pipeline_wizard = PipelineWizard()

    def run_configuration_wizard(self):
        """Run interactive configuration wizard"""

        print("=== CARDAMOM Configuration Wizard ===")
        print("This wizard will help you set up CARDAMOM preprocessing.")
        print()

        config = {}

        # Basic setup
        config['workspace'] = self._prompt_workspace_setup()
        config['data_sources'] = self._prompt_data_source_setup()
        config['processing'] = self._prompt_processing_setup()

        # Save configuration
        config_path = self._prompt_save_config(config)

        print(f"\nConfiguration saved to: {config_path}")
        print("You can now run CARDAMOM preprocessing using this configuration.")

        return config_path

    def run_pipeline_wizard(self):
        """Run interactive pipeline setup wizard"""

        print("=== CARDAMOM Pipeline Wizard ===")
        print("This wizard will help you set up a preprocessing pipeline.")
        print()

        # Choose workflow type
        workflow_type = self._prompt_workflow_type()

        if workflow_type == 'global-monthly':
            return self._setup_global_monthly_pipeline()
        elif workflow_type == 'conus-diurnal':
            return self._setup_conus_diurnal_pipeline()
        elif workflow_type == 'custom':
            return self._setup_custom_pipeline()

    def _prompt_workspace_setup(self):
        """Prompt for workspace configuration"""

        print("Workspace Setup:")
        print("================")

        workspace_dir = input("Workspace directory [./workspace/]: ").strip()
        if not workspace_dir:
            workspace_dir = "./workspace/"

        output_base = input("Output base directory [./DATA/]: ").strip()
        if not output_base:
            output_base = "./DATA/"

        # Check disk space
        if os.path.exists(workspace_dir):
            usage = shutil.disk_usage(workspace_dir)
            free_gb = usage.free / (1024**3)
            print(f"Available disk space: {free_gb:.1f} GB")

            if free_gb < 50:
                print("Warning: Less than 50 GB available. Consider using a different location.")

        return {
            'workspace_dir': workspace_dir,
            'output_base': output_base
        }

    def _prompt_data_source_setup(self):
        """Prompt for data source configuration"""

        print("\nData Source Setup:")
        print("==================")

        sources = {}

        # ECMWF/CDS API setup
        if self._prompt_yes_no("Set up ECMWF CDS API access?"):
            sources['ecmwf'] = self._setup_ecmwf_credentials()

        # Other data sources
        for source in ['noaa', 'gfed', 'modis']:
            if self._prompt_yes_no(f"Enable {source.upper()} data downloads?"):
                sources[source] = {'enabled': True}

        return sources

    def _setup_ecmwf_credentials(self):
        """Setup ECMWF CDS API credentials"""

        print("\nECMWF CDS API Setup:")
        print("For access to ERA5 data, you need a CDS API account.")
        print("Visit: https://cds.climate.copernicus.eu/api-how-to")
        print()

        cds_url = input("CDS API URL [https://cds.climate.copernicus.eu/api/v2]: ").strip()
        if not cds_url:
            cds_url = "https://cds.climate.copernicus.eu/api/v2"

        cds_key = input("CDS API Key: ").strip()

        if cds_key:
            # Offer to save .cdsapirc file
            save_creds = self._prompt_yes_no("Save credentials to ~/.cdsapirc?")
            if save_creds:
                self._save_cdsapi_credentials(cds_url, cds_key)

        return {
            'url': cds_url,
            'key': cds_key
        }

    def _prompt_yes_no(self, question, default=None):
        """Prompt for yes/no response"""

        if default is True:
            prompt = f"{question} [Y/n]: "
        elif default is False:
            prompt = f"{question} [y/N]: "
        else:
            prompt = f"{question} [y/n]: "

        while True:
            response = input(prompt).strip().lower()

            if not response and default is not None:
                return default
            elif response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please answer 'y' or 'n'.")
```

## 7.7 Testing and Integration

### CLI Testing Framework
```
tests/cli/
├── test_main_cli.py
├── test_pipeline_commands.py
├── test_data_source_commands.py
├── test_utility_commands.py
├── test_config_cli.py
├── test_interactive_cli.py
└── fixtures/
    ├── sample_configs/
    ├── mock_responses/
    └── test_outputs/
```

### Integration with Existing CLI
```python
def test_backward_compatibility():
    """Test that existing ECMWF CLI commands still work"""

def test_new_pipeline_commands():
    """Test new pipeline management commands"""

def test_config_file_integration():
    """Test configuration file loading and merging"""

def test_interactive_mode():
    """Test interactive wizards and prompts"""
```

## 7.8 Success Criteria

### Functional Requirements
- [ ] Maintain complete backward compatibility with existing CLI
- [ ] Support comprehensive pipeline management
- [ ] Provide intuitive data source management
- [ ] Include robust validation and reporting capabilities

### Usability Requirements
- [ ] Clear, comprehensive help documentation
- [ ] Intuitive command structure and naming
- [ ] Interactive wizards for complex setup
- [ ] Progress reporting and status tracking

### Integration Requirements
- [ ] Seamless integration with existing ECMWF downloader
- [ ] Consistent configuration management
- [ ] Compatible with MAAP platform workflows
- [ ] Support for both interactive and batch execution

### Quality Requirements
- [ ] Comprehensive error handling and user feedback
- [ ] Extensive help documentation and examples
- [ ] Robust input validation and sanitization
- [ ] Consistent logging and debugging capabilities